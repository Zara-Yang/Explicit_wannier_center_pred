from distutils.command.config import config
from tkinter.tix import Tree
from matplotlib.pyplot import axes
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
from Memmap import Memmap

device = torch.device("cpu")

class BPNet(nn.Module):
    def __init__(self):
        super(BPNet, self).__init__()
        n_first = 38
        n_second = 25
        n_third = 25
        self.w1 = nn.Parameter(torch.randn((n_first, n_second))/1e3)
        self.b1 = nn.Parameter(torch.randn(n_second)/1e3)
        self.w2 = nn.Parameter(torch.randn((n_second, n_third))/1e3)
        self.b2 = nn.Parameter(torch.randn(n_third)/1e3)
        self.w3 = nn.Parameter(torch.randn((n_third, 1))/1e3)
        self.b3 = nn.Parameter(torch.randn(1)/1e3)

    def forward(self, x_W,dx_WW):
        z1 = torch.matmul(x_W, self.w1) + self.b1
        z2 = torch.matmul(torch.tanh(z1), self.w2) + self.b2
        dev_buffer1 = torch.matmul(dx_WW, self.w1) / torch.cosh(z1) ** 2
        dev_buffer2 = torch.matmul(dev_buffer1,self.w2) / torch.cosh(z2) ** 2
        dev_WW = torch.matmul(dev_buffer2,self.w3)

        y_W = torch.sum(dev_WW, axis=(-1, -2))

        return y_W

class Memmap_loader():
    def __init__(self,memmap_data_path,memmap_dev_path,memmap_label_path):
        self.features = Memmap.Memmap_read(memmap_data_path)
        self.features_dev = Memmap.Memmap_read(memmap_dev_path)
        self.label = Memmap.Memmap_read(memmap_label_path)
        self.label_dim = 3
    def Generator(self,shuffle = True):
        if not shuffle:
            config_list = np.arange(self.label.shape[0])
        else:
            config_list = np.random.permutation(self.label.shape[0])

        for iconfig in config_list:
            batch_features = torch.from_numpy(self.features[iconfig,:,:])
            batch_features_dev = torch.from_numpy(self.features_dev[iconfig,:,:,:,:])
            batch_labels = torch.from_numpy(self.label[:,:,iconfig])
            yield(batch_features,batch_features_dev,batch_labels)

def Train_network(train_data,valid_data,epoch_number,info_save_path,model_save_path):
    net = BPNet()

    optimizer = optim.Adam(net.parameters())
    info_file = open(info_save_path,"w")

    optimize_model = None
    optimize_loss = 1000
    optimize_index = 0

    for epoch_index in range(epoch_number):
        train_generator = train_data.Generator(True)
        valid_generator = valid_data.Generator(False)
        epoch_train_loss = 0
        batch_number = 0
        net.train()
        for features,features_dev,label in train_generator:
            optimizer.zero_grad()
            pred = net(features,features_dev) 
            loss = torch.sum(torch.abs((pred - label)))
            loss.backward()
            optimizer.step()
            epoch_train_loss += float(loss)
            batch_number = batch_number + 1
    
        average_train_loss = epoch_train_loss / (batch_number * 256 * train_data.label_dim)
        average_valid_loss = 0
        batch_number = 0

        for valid_features,valid_features_dev,valid_label in valid_generator:
            net.eval()
            valid_pred = net(valid_features,valid_features_dev) 
            valid_loss = float(torch.sum(torch.abs((valid_pred - valid_label))))
            average_valid_loss += valid_loss
            batch_number += 1
        average_valid_loss = average_valid_loss / (batch_number * 256 * valid_data.label_dim)

        info_file.write("{}\t{}\t{}\n".format(epoch_index,average_train_loss,average_valid_loss))
        info_file.flush()
        print("{}\t{}\t{}".format(epoch_index,average_train_loss,average_valid_loss))

        if average_valid_loss < optimize_loss:
            optimize_loss = average_valid_loss
            optimize_index = epoch_index
            optimize_model = net.state_dict()
        if epoch_index % 10 == 0 and epoch_index != 0:
            torch.save(optimize_model,"{}/Network_wanner_force.pth".format(model_save_path))

    torch.save(optimize_model,"{}/Network_wanner_force.pth".format(model_save_path))
    info_file.close()


if __name__ == "__main__":

    SAVE_MODEL_PATH = "/DATA/users/yanghe/projects/Wannier_center_pred/Data/Model_saved"
    TRAIN_DATA_PATH = "/DATA/users/yanghe/projects/Wannier_center_pred/Data/Train_data"
    VALID_DATA_PATH = "/DATA/users/yanghe/projects/Wannier_center_pred/Data/Valid_data"

    train_xW_path = "{}/xW_978-256-38.mapdat".format(TRAIN_DATA_PATH)
    train_xWW_d_path = "{}/xWWd_978-256-3-256-38.mapdat".format(TRAIN_DATA_PATH)
    train_label_path = "{}/Wforce_256-3-978.mapdat".format(TRAIN_DATA_PATH)

    valid_xW_path = "{}/xW_499-256-38.mapdat".format(VALID_DATA_PATH)
    valid_xWW_d_path = "{}/xWWd_499-256-3-256-38.mapdat".format(VALID_DATA_PATH)
    valid_label_path = "{}/Wforce_256-3-499.mapdat".format(VALID_DATA_PATH)


    train_data = Memmap_loader(train_xW_path,train_xWW_d_path,train_label_path)
    valid_data = Memmap_loader(valid_xW_path,valid_xWW_d_path,valid_label_path)

    Train_network(train_data,valid_data,2000,"{}/train_info_3-17.txt".format(SAVE_MODEL_PATH),SAVE_MODEL_PATH)

    # print(np.mean(train_data.features,axis = (0,1)))
    # print(np.mean(valid_data.features,axis = (0,1)))

    # net = BPNet()
    # net.load_state_dict(torch.load("/DATA/users/yanghe/projects/Wannier_center_pred/Data/Model_saved/Network_wanner_force.pth"))
    # valid_generator = valid_data.Generator()

    # average_valid_loss = 0
    # batch_number = 0
    # net.eval()
    # for features,features_dev,label in valid_generator:
    #     pred = net(features,features_dev) 

    #     loss = torch.sum(torch.abs((pred - label)))
    #     average_valid_loss += loss
    #     batch_number += 1
    # average_valid_loss = average_valid_loss / (batch_number * 256 * valid_data.label_dim)


    # print(average_valid_loss)
    # print(np.mean(valid_data.features,axis = (0,1)))