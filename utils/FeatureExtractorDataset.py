import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.io import read_image
from pathlib import Path
import pandas as pd

STREET_IMG_DIR = Path('./dataset/GoogleStreetView')
STREET_GRAPH_DIR = Path('./dataset/single_msoa_network')
STREET_NETWORK_INFORMATION_PATH = Path('./dataset/gcn_street_info.csv')

TRAIN_LABEL_PATH = Path('./dataset/infection_risk_label_train.csv')
VAL_LABEL_PATH = Path('./dataset/infection_risk_label_val.csv')
TEST_LABEL_PATH = Path('./dataset/infection_risk_label_test.csv')


class StreetDataset(Dataset):
    def __init__(self, label_path = TRAIN_LABEL_PATH, meta_file_path=STREET_NETWORK_INFORMATION_PATH, st_img_dir=STREET_IMG_DIR, st_graph_dir=STREET_GRAPH_DIR, st_aux_path=None, transform=None):
        # read label file
        self.label_file = pd.read_csv(label_path).drop(['Rt'], axis=1)
        self.msoa_list = self.label_file['MSOACode'].tolist()
        # read street view segmentation
        self.st_aux_path = st_aux_path
        if self.st_aux_path:
            self.aux_file = pd.read_csv(self.st_aux_path)
            self.aux_file = self.aux_file.loc[self.aux_file['MSOACode'].isin(self.msoa_list)]
        # read street view network meta file
        self.meta_file = pd.read_csv(meta_file_path)[['MSOACode','FileName','NodeID','RelativeNodeID']]
        self.meta_file = self.meta_file.loc[self.meta_file['MSOACode'].isin(self.msoa_list)]

        self.img_dir = Path(st_img_dir)
        self.graph_dir = Path(st_graph_dir)
        self.transform = transform

    def __len__(self):
        return len(self.msoa_list)

    def __getitem__(self, idx):
        msoa_code = self.msoa_list[idx]
        meta_file = self.meta_file.loc[self.meta_file['MSOACode'] == msoa_code]
        if self.st_aux_path:
            aux = self.aux_file.loc[self.aux_file['MSOACode'] == msoa_code].drop(['MSOACode'], axis=1)
            meta_file_with_aux = meta_file.merge(aux, left_on=['FileName'], right_on=['file_name'], how='left').drop('file_name', axis=1)
            meta_file_with_aux = meta_file_with_aux.sort_values(by=['RelativeNodeID'],ascending=True).reset_index(drop=True)
        else:
            meta_file_with_aux = meta_file.sort_values(by=['RelativeNodeID'], ascending=True).reset_index(drop=True)

        # load graph
        g = torch.load(self.graph_dir.joinpath(msoa_code).joinpath('street_network.pt'))

        # load edge weight
        w = torch.load(self.graph_dir.joinpath(msoa_code).joinpath('street_network_weight.pt'))

        # load images
        img_path = [self.img_dir.joinpath(msoa_code).joinpath(f) for f in meta_file_with_aux['FileName']]
        if self.transform:
            # 归一化像素到0-1
            images = [self.transform(read_image(str(i)).type(torch.float)/255).unsqueeze(0) for i in img_path]
            images = torch.cat(images)
        else:
            # 归一化像素到0-1
            images = torch.cat([(read_image(str(img)).type(torch.float)/255).unsqueeze(0) for img in img_path])

        # load true label
        label = self.label_file.iloc[idx]['RtCluster']

        return g, w, images, label, msoa_code