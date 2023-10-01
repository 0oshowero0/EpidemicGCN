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
import GPUtil
from model_epigcn import *
from sklearn.metrics import f1_score, precision_score, recall_score
import setproctitle
setproctitle.setproctitle('EpiGCN@GCN')


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
    data.add_argument('--data', default='Street', help='data to use (Street, Satellite, Both)')

    data.add_argument('--train_label_path', default='./dataset/infection_risk_label_train.csv', help='train label path')
    data.add_argument('--val_label_path', default='./dataset/infection_risk_label_val.csv', help='val label path')
    data.add_argument('--test_label_path', default='./dataset/infection_risk_label_test.csv', help='test label path')
    data.add_argument('--label_path', default='./dataset/infection_risk_label.csv', help='label path')

    data.add_argument('--street_feature_path', default='./dataset/msoa_street_embedding', help='street feature path')
    data.add_argument('--msoa_influence_matrix', default='./dataset/msoa_influence_matrix.pt', help='MSOA influence matrix')
    #################################################
    # Model and learning parameters
    #################################################
    model = parser.add_argument_group('Model')
    model.add_argument('--model', default='EpiGCN', help='model to use (EpiGCN)')
    model.add_argument('--feature_size', default=512, help='size of embedding', type=int)
    model.add_argument('--embedding_size', default=512, help='size of embedding', type=int)
    model.add_argument('--k', default=5, help='number of edges for msoa graph', type=int)
    model.add_argument('--heads', default=4, help='number of attention heads', type=int)

    # training setting
    model.add_argument('--max_epoch', default=200, type=int, help='maximum epoch')
    model.add_argument('--warm_up_steps', default=20, type=int, help='warm-up steps for automatically update learning rate')
    model.add_argument('--early_stop', default=20, type=int, help='early stop criterion')
    model.add_argument('--optimizer', default='Adam', help='model optimizer (Adam, RAdam)')
    model.add_argument('--init_learning_rate', default=1e-4, type=float, help='initial learning rate')
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
        if repeat_index is None:
            if self.opt['scheduler']:
                self.output_dir = './result/'+ 'epigcn_' + self.opt['data'] + '_'  + self.opt['model'] + '_Emb_' + str(self.opt['embedding_size']) + '_Lr_' + str(self.lr) + '_Lamb_' + str(self.lamda) + '_beta_' + str(self.opt['beta']) + '_k_' + str(self.opt['k']) + '_scheduler'
            else:
                self.output_dir = './result/'+ 'epigcn_' + self.opt['data'] + '_'  + self.opt['model'] + '_Emb_' + str(self.opt['embedding_size']) + '_Lr_' +  str(self.lr) + '_Lamb_' + str(self.lamda) + '_beta_' + str(self.opt['beta']) + '_k_' + str(self.opt['k'])
        else:
            if self.opt['scheduler']:
                self.output_dir = './result/'+ 'epigcn_' + self.opt['data'] + '_'  + self.opt['model'] + '_Emb_' + str(self.opt['embedding_size']) + '_Lr_' + str(self.lr) + '_Lamb_' + str(self.lamda) + '_beta_' + str(self.opt['beta']) + '_k_' + str(self.opt['k']) + '_scheduler_' + str(repeat_index).zfill(2)
            else:
                self.output_dir = './result/' + 'epigcn_' + self.opt['data'] + '_' + self.opt['model'] + '_Emb_' + str(self.opt['embedding_size']) + '_Lr_' + str(self.lr) + '_Lamb_' + str(self.lamda)  + '_beta_' + str(self.opt['beta']) + '_k_' + str(self.opt['k']) +'_' +str(repeat_index).zfill(2)

        print(self.output_dir)
        self.writer = SummaryWriter(log_dir=self.output_dir)

    def generate_dataset(self):
        begin_time = datetime.now()

        msoa_code_list = pd.read_csv(self.opt['label_path']).sort_values('MSOACode', ascending=True)['MSOACode'].to_list()
        self.msoa_node_id_df = pd.DataFrame({'MSOACode': msoa_code_list, 'NodeID': [i for i in range(len(msoa_code_list))]})

        if self.opt['data'] == 'Street':
            self.st_feature = [torch.load(self.opt['street_feature_path'] + '/' + self.msoa_node_id_df.iloc[i]['MSOACode'] + '_ResNet18.pt').to(self.device) for i in range(self.msoa_node_id_df.shape[0])]
            self.sa_feature = None

        # generate network
        msoa_influence_matrix = torch.load(self.opt['msoa_influence_matrix'])
        msoa_influence_matrix /= msoa_influence_matrix.max()
        inf_topk, inf_topk_idx = torch.topk(msoa_influence_matrix, k=self.opt['k'] + 1, dim=1, largest=True)

        msoa_start_node = torch.repeat_interleave(torch.arange(0, msoa_influence_matrix.shape[0]), self.opt['k'] + 1)
        msoa_end_node = inf_topk_idx.reshape(-1)

        self.g = torch.cat([msoa_start_node.reshape(1, -1), msoa_end_node.reshape(1, -1)], dim=0).to(self.device)
        self.w = inf_topk.reshape(-1).to(self.device)

        # generate label mask
        train_information = pd.read_csv(self.opt['train_label_path']).sort_values('MSOACode',ascending=True)
        self.train_msoa = train_information['MSOACode'].to_list()
        self.train_mask = self.msoa_node_id_df['MSOACode'].isin(self.train_msoa).to_list()
        self.train_label = torch.tensor(train_information['RtCluster'].to_list()).to(self.device)

        val_information = pd.read_csv(self.opt['val_label_path']).sort_values('MSOACode',ascending=True)
        self.val_msoa = val_information['MSOACode'].to_list()
        self.val_mask = self.msoa_node_id_df['MSOACode'].isin(self.val_msoa).to_list()
        self.val_label = torch.tensor(val_information['RtCluster'].to_list()).to(self.device)

        test_information = pd.read_csv(self.opt['test_label_path']).sort_values('MSOACode',ascending=True)
        self.test_msoa = test_information['MSOACode'].to_list()
        self.test_mask = self.msoa_node_id_df['MSOACode'].isin(self.test_msoa).to_list()
        self.test_label = torch.tensor(test_information['RtCluster'].to_list()).to(self.device)

        end_time = datetime.now()
        print("Complete Data Loader Initialization in " + str((end_time - begin_time).total_seconds()) + " seconds" + '\n')


    def init_model(self):
        if self.opt['model'] == 'EpiGCN':
            self.model = EpiGCN(self.g, self.w, st_feature=self.st_feature, sa_feature=self.sa_feature, feature_size=self.opt['feature_size']).to(self.device)

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

    def loss_func(self, y_pred, y_true):
        loss = F.cross_entropy(y_pred, y_true.view(-1), weight=self.loss_weight, reduction='mean')
        return loss

    def train(self):
        self.model.train()

        if self.opt['optimizer'] == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.lamda)
        elif self.opt['optimizer'] == "RAdam":
            optimizer = torch.optim.RAdam(self.model.parameters(), lr=self.lr, weight_decay=self.lamda)

        if self.opt['scheduler']:
            update_func = lambda epoch: 1 if epoch <= self.warm_up else 0.8**int((epoch - self.warm_up) // 50)
            scheduler = LambdaLR(optimizer, lr_lambda=update_func)


        early_stop_indicator = 0
        best_metric = 9999999999

        total_begin_time = datetime.now()

        for epoch in range(self.opt['max_epoch']):
            begin_time_epoch = datetime.now()
            # with torch.autograd.set_detect_anomaly(True):
            pred = self.model()
            optimizer.zero_grad()
            output = self.loss_func(pred[self.train_mask], self.train_label)
            output.backward()
            optimizer.step()

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
                print('Epoch {%(epoch)d}: Train Loss is {%(loss)f}, F1 weighted during train is {%(train_f1)f}, precision weighted during train is {%(train_precision)f}, recall weighted during train is {%(train_recall)f}' % {
                        'epoch': epoch, 'loss': train_loss, 'train_f1': train_f1_weighted, 'train_precision': train_precision_weighted, 'train_recall': train_recall_weighted})
                print('Epoch {%(epoch)d}: Val Loss is {%(loss)f}, F1 weighted during val is {%(val_f1)f}, precision weighted during val is {%(val_precision)f}, recall weighted during val is {%(val_recall)f}' % {
                        'epoch': epoch, 'loss': val_loss, 'val_f1': val_f1_weighted, 'val_precision': val_precision_weighted,'val_recall': val_recall_weighted})
                print('=============================================================================')

                if curr_metric < best_metric:
                    early_stop_indicator = 0
                    best_metric = curr_metric
                    torch.save(self.model.state_dict(), self.output_dir + '/best_gcn_model.pth')
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
            pred = self.model()[self.train_mask]
            loss = self.loss_func(pred, self.train_label).to('cpu').detach()
            pred_class = torch.argmax(pred, dim=-1).type(torch.long)
            train_f1_micro, train_f1_macro, train_f1_weighted, train_precision_micro, train_precision_macro, \
                train_precision_weighted, train_recall_micro, train_recall_macro, train_recall_weighted = self.metrics(pred_class, self.train_label)

        return loss, train_f1_micro, train_f1_macro, train_f1_weighted, train_precision_micro, train_precision_macro, \
                train_precision_weighted, train_recall_micro, train_recall_macro, train_recall_weighted

    def val(self):
        self.model.eval()
        with torch.no_grad():
            pred = self.model()[self.val_mask]
            loss = self.loss_func(pred, self.val_label).to('cpu').detach()
            pred_class = torch.argmax(pred, dim=-1).type(torch.long)
            val_f1_micro, val_f1_macro, val_f1_weighted, val_precision_micro, val_precision_macro, \
                val_precision_weighted, val_recall_micro, val_recall_macro, val_recall_weighted = self.metrics(pred_class, self.val_label)
        return loss, val_f1_micro, val_f1_macro, val_f1_weighted, val_precision_micro, val_precision_macro, \
                val_precision_weighted, val_recall_micro, val_recall_macro, val_recall_weighted

    def test(self):
        print('Loading Best Model')
        self.model.load_state_dict(torch.load(self.output_dir + '/best_gcn_model.pth'))
        self.model.eval()
        with torch.no_grad():
            pred = self.model()[self.test_mask]
            loss = self.loss_func(pred, self.test_label).to('cpu').detach()
            pred_class = torch.argmax(pred, dim=-1).type(torch.long)
            test_f1_micro, test_f1_macro, test_f1_weighted, test_precision_micro, test_precision_macro, \
                test_precision_weighted, test_recall_micro, test_recall_macro, test_recall_weighted = self.metrics(pred_class, self.test_label)
            print('Test Loss is {%(loss)f}, F1 weighted for test set is {%(test_f1)f}, precision weighted for test set is {%(test_precision)f}, recall weighted for test set is {%(test_recall)f}' %
                  {'loss': loss, 'test_f1': test_f1_weighted, 'test_precision': test_precision_weighted, 'test_recall': test_recall_weighted})
        return loss.numpy(), test_f1_micro, test_f1_macro, test_f1_weighted, test_precision_micro, test_precision_macro, \
                test_precision_weighted, test_recall_micro, test_recall_macro, test_recall_weighted

    def predict(self, type='test'):
        self.model.eval()
        with torch.no_grad():
            pred = self.model()

            if type == 'train':
                pred = pred[self.train_mask]
                true_label = self.train_label
                msoa_code = self.msoa_node_id_df['MSOACode'][self.train_mask].to_list()
            elif type == 'val':
                pred = pred[self.val_mask]
                true_label = self.val_label
                msoa_code = self.msoa_node_id_df['MSOACode'][self.val_mask].to_list()
            elif type == 'test':
                pred = pred[self.test_mask]
                true_label = self.test_label
                msoa_code =self.msoa_node_id_df['MSOACode'][self.test_mask].to_list()
            else:
                true_label = torch.zeros(self.msoa_node_id_df['MSOACode'].shape[0]) - 1
                true_label[self.train_mask] = self.train_label
                true_label[self.val_mask] = self.val_label
                true_label[self.test_mask] = self.test_label
                msoa_code = self.msoa_node_id_df['MSOACode'].to_list()

            pred = pred.to('cpu')
            pred_class = torch.argmax(pred, dim=-1).to('cpu').type(torch.long).numpy()

            true_label = true_label.to('cpu').numpy().astype(int)
            pred = pred.numpy()

        return pred_class, true_label, msoa_code, pred

    def save_prediction(self):
        pd_tr, y_tr, msoa_tr, pd_ori_tr = eg.predict('train')
        pd_val, y_val, msoa_val, pd_ori_val = eg.predict('val')
        pd_test, y_test, msoa_test, pd_ori_test = eg.predict('test')
        train_csv = pd.DataFrame({'MSOACode':msoa_tr,'RawPred_0':pd_ori_tr[:,0],'RawPred_1':pd_ori_tr[:,1],'Pred': pd_tr, 'True': y_tr})
        val_csv = pd.DataFrame({'MSOACode':msoa_val,'RawPred_0':pd_ori_val[:,0],'RawPred_1':pd_ori_val[:,1],'Pred': pd_val, 'True': y_val})
        test_csv = pd.DataFrame({'MSOACode':msoa_test,'RawPred_0':pd_ori_test[:,0], 'RawPred_1':pd_ori_test[:,1],'Pred': pd_test, 'True': y_test})
        train_csv.to_csv('./output/' + self.output_dir[9:] + '_train_pred.csv', index=False)
        val_csv.to_csv('./output/' + self.output_dir[9:] + '_val_pred.csv', index=False)
        test_csv.to_csv('./output/' + self.output_dir[9:] + '_test_pred.csv', index=False)

    def repeat_train(self, repeat_num):
        loss_list = []
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
            loss, f1_micro, f1_macro, f1_weighted, precision_micro, precision_macro, precision_weighted, \
                recall_micro, recall_macro, recall_weighted = eg.test()
            loss_list.append(loss)
            f1_micro_list.append(f1_micro)
            f1_macro_list.append(f1_macro)
            f1_weighted_list.append(f1_weighted)
            precision_micro_list.append(precision_micro)
            precision_macro_list.append(precision_macro)
            precision_weighted_list.append(precision_weighted)
            recall_micro_list.append(recall_micro)
            recall_macro_list.append(recall_macro)
            recall_weighted_list.append(recall_weighted)
            eg.save_prediction()

            result_df = pd.DataFrame({'Loss':loss_list,'F1_weighted':f1_weighted_list, 'Precision_weighted':precision_weighted_list,'Recall_weighted':recall_weighted_list,
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

