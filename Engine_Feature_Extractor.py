import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from argparse import ArgumentParser
from itertools import chain
import GPUtil
from utils.FeatureExtractorDataset import *
from model_feature_extractor import *
from sklearn.metrics import f1_score, precision_score, recall_score
import copy
import setproctitle
setproctitle.setproctitle('EpiGCN@FeatureExtractor')


def setup_args(parser=None):
    """ Set up default arguments

    return:
        python dictionary
    """
    if parser is None:
        parser = ArgumentParser()

    #################################################
    # Data path
    #################################################
    data = parser.add_argument_group('Data')
    data.add_argument('--data', default='Satellite', help='data to use (Street, Satellite)')
    data.add_argument('--street_view_dir', default='./dataset/GoogleStreetView', help='street view image data')

    data.add_argument('--train_label_path', default='./dataset/infection_risk_label_train.csv', help='train label path')
    data.add_argument('--val_label_path', default='./dataset/infection_risk_label_val.csv', help='val label path')
    data.add_argument('--test_label_path', default='./dataset/infection_risk_label_test.csv', help='test label path')
    data.add_argument('--label_path', default='./dataset/infection_risk_label.csv', help='all label path')

    data.add_argument('--network_path', default='./dataset/single_msoa_network', help='dataset folder for single msoa network')
    data.add_argument('--street_network_information', default='./dataset/gcn_street_info.csv', help='street view network information')


    #################################################
    # Model and learning parameters
    #################################################
    model = parser.add_argument_group('Model')
    model.add_argument('--cv_base_arch', default='ResNet18', help='base architecture of CV model (ResNet18, ViTb32)')

    model.add_argument('--embedding_size', default=128, help='size of embedding', type=int)
    model.add_argument('--input_size', default=224, help='size of image to the model', type=int)
    model.add_argument('--k', default=4, help='number of edges for feature graph', type=int)

    # training setting
    model.add_argument('--mini_batch_update', default=10, type=int, help='batches for update model parameters')
    model.add_argument('--cv_model_batch', default=10, type=int, help='number of images for the CV model in a single feed')

    model.add_argument('--max_epoch', default=80, type=int, help='maximum epoch')
    model.add_argument('--warm_up_steps', default=10, type=int, help='warm-up steps for automatically update learning rate')
    model.add_argument('--early_stop', default=10, type=int, help='early stop criterion')
    model.add_argument('--optimizer', default='Adam', help='model optimizer (Adam, RAdam)')
    model.add_argument('--init_learning_rate', default=3e-5, type=float, help='initial learning rate')
    model.add_argument('--lambda', default=3e-5, type=float, help='lambda for weight decay')
    model.add_argument('--beta', default=0.999, type=float, help='hyperparameter for loss balance (see arxiv 1901.05555)')
    model.add_argument('--scheduler', default=False, action='store_true', help='Use LR scheduler')

    # evaluate setting
    model.add_argument('--epochs_per_val', default=1,  type=int, help='how many epochs between val')

    #################################################
    # Helpers
    #################################################
    helpers = parser.add_argument_group('Helpers')
    helpers.add_argument('--device', default='Auto', help='GPU to use. Auto to select most suitable GPU.')
    return parser


class Engine():
    def __init__(self, opt):
        self.opt = opt
        # load some frequent options

        self.lr = self.opt['init_learning_rate']
        self.lamda = self.opt['lambda']
        self.warm_up = self.opt['warm_up_steps']

        #self.init_save_path()

        # set GPU device
        if self.opt['device'] == 'Auto':
            tmp = GPUtil.getAvailable(order='load', limit=1, maxLoad=0.7, maxMemory=0.6)
            if len(tmp) > 0:
                self.device = torch.device('cuda:' + str(tmp[0]))
                print('Using GPU' + str(tmp[0]))
            else:
                self.device = torch.device('cuda:0')
                print('Using GPU0')
        else:
            self.device = torch.device(self.opt['device'])

        # set loss weights
        train_data = pd.read_csv(self.opt['train_label_path'])
        train_class_number = train_data['RtCluster'].value_counts().sort_index().to_numpy()
        self.loss_weight = (1-self.opt['beta']) / (1-np.power(self.opt['beta'],train_class_number))
        self.loss_weight /= self.loss_weight.sum()
        self.loss_weight = torch.tensor(self.loss_weight).type(torch.float).to(self.device)

    def init_save_path(self, repeat_index=None):
        append_str = ''

        if self.opt['scheduler']:
            append_str += '_scheduler'

        if repeat_index is None:
            self.output_dir = './result/'+ 'feature_extractor_' + self.opt['data'] + '_Emb_' + str(self.opt['embedding_size']) + '_Lr_' + str(self.lr) + '_Lamb_' + str(self.lamda) + '_beta_' + str(self.opt['beta']) + append_str
        else:
            self.output_dir = './result/'+ 'feature_extractor_' + self.opt['data'] + '_Emb_' + str(self.opt['embedding_size']) + '_Lr_' + str(self.lr) + '_Lamb_' + str(self.lamda) + '_beta_' + str(self.opt['beta']) + append_str + '_' + str(repeat_index).zfill(2)


        print(self.output_dir)
        self.writer = SummaryWriter(log_dir=self.output_dir)


    def generate_dataset(self):
        begin_time = datetime.now()

        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(self.opt['input_size'],antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(self.opt['input_size'],antialias=True),
                transforms.CenterCrop(self.opt['input_size']),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(self.opt['input_size'],antialias=True),
                transforms.CenterCrop(self.opt['input_size']),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        if self.opt['data'] == 'Street':
            train_dataset = StreetDataset(label_path=self.opt['train_label_path'],
                                          meta_file_path=self.opt['street_network_information'],
                                          st_img_dir=self.opt['street_view_dir'],
                                          st_graph_dir=self.opt['network_path'],
                                          transform=data_transforms['train']
                                         )
            val_dataset = StreetDataset(label_path=self.opt['val_label_path'],
                                          meta_file_path=self.opt['street_network_information'],
                                          st_img_dir=self.opt['street_view_dir'],
                                          st_graph_dir=self.opt['network_path'],
                                          transform=data_transforms['val']
                                         )
            test_dataset = StreetDataset(label_path=self.opt['test_label_path'],
                                          meta_file_path=self.opt['street_network_information'],
                                          st_img_dir=self.opt['street_view_dir'],
                                          st_graph_dir=self.opt['network_path'],
                                          transform=data_transforms['test']
                                         )

            self.network_info = pd.read_csv(self.opt['street_network_information']).sort_values('MSOACode',ascending=True)
            all_dataset = StreetDataset(label_path=self.opt['label_path'],
                                          meta_file_path=self.opt['street_network_information'],
                                          st_img_dir=self.opt['street_view_dir'],
                                          st_graph_dir=self.opt['network_path'],
                                          transform=data_transforms['test']
                                         )


        self.train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=16)
        self.val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=16)
        self.test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=16)
        self.all_dataloader = DataLoader(all_dataset, batch_size=1, shuffle=True, num_workers=16)

        end_time = datetime.now()
        print("Complete Data Loader Initialization in " + str((end_time - begin_time).total_seconds()) + " seconds" + '\n')


    def init_model(self):
        if self.opt['data'] == 'Street':
            self.model = FeatureExtractorModel(self.opt['embedding_size'], self.opt['cv_base_arch'], self.opt['k'], self.opt['cv_model_batch']).to(self.device)


    def metrics(self, batch_pred, batch_y):
        y = batch_y.view(-1).to('cpu').detach().numpy()
        pred = batch_pred.view(-1).to('cpu').detach().numpy()
        F1_micro = f1_score(y, pred, average='micro')
        F1_macro = f1_score(y, pred, average='macro')
        F1_weighted = f1_score(y, pred, average='weighted')
        precision_micro = precision_score(y, pred, average='micro')
        precision_macro = precision_score(y, pred, average='macro')
        precision_weighted = precision_score(y, pred, average='weighted')
        recall_micro = recall_score(y, pred, average='micro')
        recall_macro = recall_score(y, pred, average='macro')
        recall_weighted = recall_score(y, pred, average='weighted')

        return F1_micro, F1_macro, F1_weighted, precision_micro, precision_macro, precision_weighted, recall_micro, recall_macro, recall_weighted

    def inf_risk_loss(self, y_pred, y_true):
        loss = F.cross_entropy(y_pred, y_true.view(-1), weight=self.loss_weight, reduction='mean')
        return loss


    def train(self):
        self.model.train()
        if self.opt['optimizer'] == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.lamda)
        elif self.opt['optimizer'] == "RAdam":
            optimizer = torch.optim.RAdam(self.model.parameters(), lr=self.lr, weight_decay=self.lamda)

        if self.opt['scheduler']:
            update_func = lambda epoch: 1 if epoch <= self.warm_up else 0.8**int((epoch - self.warm_up) // 10)
            scheduler = LambdaLR(optimizer, lr_lambda=update_func)

        total_begin_time = datetime.now()
        early_stop_indicator = 0
        best_metric = 9999999999
        for epoch in range(self.opt['max_epoch']):
            begin_time_epoch = datetime.now()
            mini_batch_idx = 0
            mini_batch_pred = []
            mini_batch_label = []
            if self.opt['data'] == 'Street':
                for g, w, images, label,  _ in iter(self.train_dataloader):
                    g = torch.squeeze(g).to(self.device)
                    w = torch.squeeze(w).to(self.device)
                    images = torch.squeeze(images).to(self.device)
                    label = label.reshape(-1).to(self.device).type(torch.long)
                    pred_label = self.model(g, w, images)
                    mini_batch_label.append(label)
                    mini_batch_pred.append(pred_label)
                    mini_batch_idx += 1
                    # backward
                    if mini_batch_idx >= self.opt['mini_batch_update']:
                        mini_batch_idx = 0
                        optimizer.zero_grad()
                        loss1 = self.inf_risk_loss(torch.cat(mini_batch_pred, dim=0),
                                                torch.cat(mini_batch_label))
                        output = loss1
                        output.backward()
                        optimizer.step()
                        mini_batch_pred = []
                        mini_batch_label = []


            if self.opt['scheduler']:
                scheduler.step()

            train_end_time = datetime.now()
            if epoch % self.opt['epochs_per_val'] == 0:
                train_loss, train_f1_micro, train_f1_macro, train_f1_weighted, train_precision_micro, train_precision_macro, \
                    train_precision_weighted, train_recall_micro, train_recall_macro, train_recall_weighted = self.train_eval()
                val_loss, val_f1_micro, val_f1_macro, val_f1_weighted, val_precision_micro, val_precision_macro, \
                    val_precision_weighted, val_recall_micro, val_recall_macro, val_recall_weighted = self.val()
                curr_metric = val_loss

                # writer log
                self.writer.add_scalar('train/loss', train_loss, epoch)
                self.writer.add_scalar('train/F1_micro', train_f1_micro, epoch)
                self.writer.add_scalar('train/F1_macro', train_f1_macro, epoch)
                self.writer.add_scalar('train/F1_weighted', train_f1_weighted, epoch)
                self.writer.add_scalar('train/precision_micro', train_precision_micro, epoch)
                self.writer.add_scalar('train/precision_macro', train_precision_macro, epoch)
                self.writer.add_scalar('train/precision_weighted', train_precision_weighted, epoch)
                self.writer.add_scalar('train/recall_micro', train_recall_micro, epoch)
                self.writer.add_scalar('train/recall_macro', train_recall_macro, epoch)
                self.writer.add_scalar('train/recall_weighted', train_recall_weighted, epoch)

                self.writer.add_scalar('val/loss', val_loss, epoch)
                self.writer.add_scalar('val/F1_micro', val_f1_micro, epoch)
                self.writer.add_scalar('val/F1_macro', val_f1_macro, epoch)
                self.writer.add_scalar('val/F1_weighted', val_f1_weighted, epoch)
                self.writer.add_scalar('val/precision_micro', val_precision_micro, epoch)
                self.writer.add_scalar('val/precision_macro', val_precision_macro, epoch)
                self.writer.add_scalar('val/precision_weighted', val_precision_weighted, epoch)
                self.writer.add_scalar('val/recall_micro', val_recall_micro, epoch)
                self.writer.add_scalar('val/recall_macro', val_recall_macro, epoch)
                self.writer.add_scalar('val/recall_weighted', val_recall_weighted, epoch)

                training_time = (train_end_time - begin_time_epoch).total_seconds() / 60
                evaluation_time = (datetime.now() - train_end_time).total_seconds() / 60
                print('=============================================================================')
                print('Training Time: {%(t)f} minutes per epoch, Evaluation Time: {%(e)f} minutes per epoch'%{'t':training_time, 'e':evaluation_time})
                print('Epoch {%(epoch)d}: Train Loss is {%(loss1)f}, F1 weighted during train is {%(train_f1)f}, precision weighted during train is {%(train_precision)f}, recall weighted during train is {%(train_recall)f}' % {
                        'epoch': epoch, 'loss1': train_loss,  'train_f1': train_f1_weighted, 'train_precision': train_precision_weighted, 'train_recall': train_recall_weighted})
                print('Epoch {%(epoch)d}: Val Loss is {%(loss1)f}, F1 weighted during val is {%(val_f1)f}, precision weighted during val is {%(val_precision)f}, recall weighted during val is {%(val_recall)f}' % {
                        'epoch': epoch, 'loss1': val_loss,  'val_f1': val_f1_weighted, 'val_precision': val_precision_weighted,'val_recall': val_recall_weighted})
                print('=============================================================================')

                if curr_metric < best_metric:
                    early_stop_indicator = 0
                    best_metric = curr_metric
                    torch.save(self.model.state_dict(), self.output_dir + '/best_model.pth')
                else:
                    if epoch >= self.opt['warm_up_steps']:
                        early_stop_indicator += 1

                if early_stop_indicator > self.opt['early_stop']:
                    print('Early Stop at Epoch: ' + str(epoch))
                    print('Best Performance is: ' + str(best_metric))
                    break

        total_time = (datetime.now() - total_begin_time).total_seconds() / 3600
        print('Total Training Time: {} hours.'.format(total_time))

    def train_eval(self):
        self.model.eval()
        with torch.no_grad():
            label_total = []
            pred_label_total = []
            if self.opt['data'] == 'Street':
                for g, w, images, label,  _ in iter(self.train_dataloader):
                    g = torch.squeeze(g).to(self.device)
                    w = torch.squeeze(w).to(self.device)
                    images = torch.squeeze(images).to(self.device)
                    label = label.reshape(-1).to(self.device).type(torch.long)
                    pred_label = self.model(g, w, images)
                    # aggreagate all the results
                    label_total.append(label)
                    pred_label_total.append(pred_label)

                label_total = torch.cat(label_total, dim=0)

                pred_label_total = torch.cat(pred_label_total, dim=0)


            loss1 = self.inf_risk_loss(pred_label_total, label_total).to('cpu').detach()
            pred_class = torch.argmax(pred_label_total, dim=-1).type(torch.long)

            train_f1_micro, train_f1_macro, train_f1_weighted, train_precision_micro, train_precision_macro, \
                train_precision_weighted, train_recall_micro, train_recall_macro, train_recall_weighted = self.metrics(pred_class, label_total)

        return loss1,  train_f1_micro, train_f1_macro, train_f1_weighted, train_precision_micro, train_precision_macro, \
                train_precision_weighted, train_recall_micro, train_recall_macro, train_recall_weighted

    def val(self):
        self.model.eval()
        with torch.no_grad():
            label_total = []
            pred_label_total = []

            if self.opt['data'] == 'Street':
                for g, w, images, label, _ in iter(self.val_dataloader):
                    g = torch.squeeze(g).to(self.device)
                    w = torch.squeeze(w).to(self.device)
                    images = torch.squeeze(images).to(self.device)
                    label = label.reshape(-1).to(self.device).type(torch.long)
                    pred_label = self.model(g, w, images)
                    # aggreagate all the results
                    label_total.append(label)
                    pred_label_total.append(pred_label)

                label_total = torch.cat(label_total, dim=0)
                pred_label_total = torch.cat(pred_label_total, dim=0)

            loss1 = self.inf_risk_loss(pred_label_total, label_total).to('cpu').detach()
            pred_class = torch.argmax(pred_label_total, dim=-1).type(torch.long)

            val_f1_micro, val_f1_macro, val_f1_weighted, val_precision_micro, val_precision_macro, \
                val_precision_weighted, val_recall_micro, val_recall_macro, val_recall_weighted = self.metrics(pred_class, label_total)
        return loss1, val_f1_micro, val_f1_macro, val_f1_weighted, val_precision_micro, val_precision_macro, \
                val_precision_weighted, val_recall_micro, val_recall_macro, val_recall_weighted

    def test(self):
        print('Loading Best Model')
        self.model.load_state_dict(torch.load(self.output_dir + '/best_model.pth'))
        self.model.eval()
        with torch.no_grad():
            label_total = []
            pred_label_total = []
            if self.opt['data'] == 'Street':
                for g, w, images, label,  _ in iter(self.test_dataloader):
                    g = torch.squeeze(g).to(self.device)
                    w = torch.squeeze(w).to(self.device)
                    images = torch.squeeze(images).to(self.device)
                    label = label.reshape(-1).to(self.device).type(torch.long)
                    pred_label = self.model(g, w, images)
                    # aggreagate all the results
                    label_total.append(label)
                    pred_label_total.append(pred_label)
                label_total = torch.cat(label_total, dim=0)
                pred_label_total = torch.cat(pred_label_total, dim=0)


            loss1 = self.inf_risk_loss(pred_label_total, label_total).to('cpu').detach()

            pred_class = torch.argmax(pred_label_total, dim=-1).type(torch.long)
            test_f1_micro, test_f1_macro, test_f1_weighted, test_precision_micro, test_precision_macro, \
                test_precision_weighted, test_recall_micro, test_recall_macro, test_recall_weighted = self.metrics(pred_class, label_total)
            print('Test Loss is {%(loss1)f}, F1 weighted for test set is {%(test_f1)f}, precision weighted for test set is {%(test_precision)f}, recall weighted for test set is {%(test_recall)f}' %
                  {'loss1': loss1, 'test_f1': test_f1_weighted, 'test_precision': test_precision_weighted, 'test_recall': test_recall_weighted})
        return loss1.numpy(), test_f1_micro, test_f1_macro, test_f1_weighted, test_precision_micro, test_precision_macro, \
                test_precision_weighted, test_recall_micro, test_recall_macro, test_recall_weighted


    def save_embedding(self):
        iterable_dataset = iter(self.all_dataloader)
        self.model.eval()
        with torch.no_grad():
            if self.opt['data'] == 'Street':
                for g, w, images, _, msoa_code in iterable_dataset:
                    g = torch.squeeze(g).to(self.device)
                    w = torch.squeeze(w).to(self.device)
                    images = torch.squeeze(images).to(self.device)
                    emb = self.model.get_embedding(g, w, images).to('cpu').detach()
                    # aggreagate all the results
                    out_path = Path(self.output_dir).joinpath('msoa_embedding')
                    out_path.mkdir(parents=True, exist_ok=True)
                    torch.save(emb, out_path.joinpath(msoa_code[0] + '_' + self.opt['cv_base_arch'] +'.pt'))


    def repeat_train(self, repeat_num):
        loss1_list = []
        f1_micro_list = []
        f1_macro_list = []
        f1_weighted_list = []
        precision_micro_list = []
        precision_macro_list = []
        precision_weighted_list = []
        recall_micro_list = []
        recall_macro_list = []
        recall_weighted_list = []

        for i in range(repeat_num):
            self.init_save_path(i)
            self.init_model()
            eg.train()
            loss1, f1_micro, f1_macro, f1_weighted, precision_micro, precision_macro, precision_weighted, \
                recall_micro, recall_macro, recall_weighted = eg.test()
            loss1_list.append(loss1)
            f1_micro_list.append(f1_micro)
            f1_macro_list.append(f1_macro)
            f1_weighted_list.append(f1_weighted)
            precision_micro_list.append(precision_micro)
            precision_macro_list.append(precision_macro)
            precision_weighted_list.append(precision_weighted)
            recall_micro_list.append(recall_micro)
            recall_macro_list.append(recall_macro)
            recall_weighted_list.append(recall_weighted)
            eg.save_embedding()

            result_df = pd.DataFrame({'Loss':loss1_list,'F1_weighted':f1_weighted_list, 'Precision_weighted':precision_weighted_list,'Recall_weighted':recall_weighted_list,
                                      'F1_micro':f1_micro_list,'Precision_micro':precision_micro_list,'Recall_micro':recall_micro_list,
                                      'F1_macro':f1_macro_list,'Precision_macro':precision_macro_list,'Recall_macro':recall_macro_list
                                      })

            result_df.to_csv(self.output_dir[:-3]+'.csv', index=False)

if __name__ == '__main__':
    parser = setup_args()
    opt = vars(parser.parse_args())
    eg = Engine(opt)
    eg.generate_dataset()
    eg.repeat_train(5)