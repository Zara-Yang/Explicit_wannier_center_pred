from xml.dom import VALIDATION_ERR
import numpy as np
from tqdm import tqdm
from glob import glob
import subprocess
from Ewald_sum import *
from Data_loader import loading_data
from Memmap import Memmap
import os

natoms = 192
noxygen = 64
nhydrogen = 128
nwannier = 256
nfolders = 1

sigma = 8  # the smoothing length sigma for GT cutoff
qO = 6  # charge on the oxygen
qH = 1  # charge on the hydrogen
qW = -2  # charge on the wannier center


def Produce_split_features(data_path,script_path):
    config_path_list = list(i for i in glob("{}/*".format(data_path))[:])
    print("{}/*".format(data_path))
    for config_path in tqdm(config_path_list):
        print(config_path)
        subprocess.call(["cp",script_path,config_path])
        subprocess.call(["./Produce_features.o"],cwd=config_path)

class Assemble_config():
    @staticmethod
    def Assemble_feature(config_folder_path,config_list,save_folder_path,feature_name,id_center):
        if id_center == "O":
            ncenter = noxygen
        elif(id_center == "H"):
            ncenter = nhydrogen
        elif(id_center == "W"):
            ncenter = nwannier
        
        features_all = ()
        features_d_all = ()
        for config_folder_name in tqdm(config_list):
            feature_path = "{}/{}/features_{}.txt".format(config_folder_path,config_folder_name,feature_name)
            feature_d_path = "{}/{}/features_d{}.txt".format(config_folder_path,config_folder_name,feature_name)
            features = np.loadtxt(feature_path, dtype=np.float32)
            dfeatures = np.loadtxt(feature_d_path, dtype=np.float32)
            features_all += (features, )
            features_d_all += (dfeatures.reshape((features.shape[0], ncenter, natoms + nwannier, 3)), )    
        features_all = np.transpose(np.stack(features_all, axis=0), axes=(0, 2, 1))  # now it is (nconfig, ncenter, nfeatures)
        features_d_all = np.transpose(np.stack(features_d_all, axis=0), axes=(0, 2, 3, 4, 1))  # now it is (nconfig, ncenter, natoms, 3, nfeatures)

        np.save("{}/features_{}".format(save_folder_path,feature_name), features_all)
        np.save("{}/features_d{}".format(save_folder_path,feature_name), features_d_all)   
    @staticmethod
    def Assemble_features_further(feature_file_path,feature_name_tuple):  # assemble features that have the same center atom
        features_all = ()
        features_d_all = ()
        for feature_name in tqdm(feature_name_tuple):
            features = np.load("{}/features_{}".format(feature_file_path,feature_name) + ".npy")
            features_d = np.load("{}/features_d{}".format(feature_file_path,feature_name) + ".npy")
            features_all += (features,)
            features_d_all += (features_d,)
            print(features_d.shape)
        features_all = np.concatenate(features_all, axis=-1)  # stack along the nfeatures axis
        features_d_all = np.transpose(np.concatenate(features_d_all, axis=-1), axes=(0, 2, 3, 1, 4))
        # stack along the nfeatures axis, then make sure the number of center atoms is at the second last axis

        return features_all, features_d_all

    @staticmethod
    def Generate_train_data(config_folder_path, train_config_list,save_folder_path):
        feature_path = "{}/features".format(save_folder_path)
        if not os.path.basename(feature_path):
            os.makedirs(feature_path)
        # Assemble_config.Assemble_basic_data(config_folder_path,train_config_list,save_folder_path)
        # Assemble_config.Assemble_feature(config_folder_path,train_config_list,save_folder_path,"G2WW","W")
        # Assemble_config.Assemble_feature(config_folder_path,train_config_list,save_folder_path,"G2WH","W")
        # Assemble_config.Assemble_feature(config_folder_path,train_config_list,save_folder_path,"G2WO","W")

        # Assemble_config.Assemble_feature(config_folder_path,train_config_list,save_folder_path,"G4WWW","W")
        # Assemble_config.Assemble_feature(config_folder_path,train_config_list,save_folder_path,"G4WOO","W")
        # Assemble_config.Assemble_feature(config_folder_path,train_config_list,save_folder_path,"G4WHH","W")
        print("Furter assemble !")
        xW,xW_d = Assemble_config.Assemble_features_further(feature_path,("G2WW", "G2WH", "G2WO","G4WWW", "G4WOO", "G4WHH"))

        xWW_d = xW_d[:, natoms : natoms + nwannier, :, :, :]
        print("Calculate scale factor")
        xW_av = np.mean(xW, axis=(0, 1))
        xW_min = np.min(xW, axis=(0, 1))
        xW_max = np.max(xW, axis=(0, 1))
        np.savetxt("{}/xW_scalefactor.txt".format(save_folder_path), np.stack((xW_av, xW_min, xW_max), axis=-1))
        print("Eescale data")
        xW = (xW - xW_av) / (xW_max - xW_min)
        xWW_d = xWW_d / (xW_max - xW_min)

        print("Start Saving memmap : ")
        Memmap.Memmap_save(save_folder_path,"xW.dat",xW)
        Memmap.Memmap_save(save_folder_path,"xWW_d.dat",xWW_d)
        print("Start saving npy : ")
        np.save("{}/xW".format(save_folder_path), xW)
        np.save("{}/xWW_d".format(save_folder_path), xWW_d)
    @staticmethod
    def Generate_valid_data(config_folder_path, valid_config_list,save_folder_path,xW_scale_path):

        feature_path = "{}/features".format(save_folder_path)
        if not os.path.basename(feature_path):
            os.makedirs(feature_path)

        Assemble_config.Assemble_feature(config_folder_path,valid_config_list,feature_path,"G2WW","W")
        Assemble_config.Assemble_feature(config_folder_path,valid_config_list,feature_path,"G2WH","W")
        Assemble_config.Assemble_feature(config_folder_path,valid_config_list,feature_path,"G2WO","W")

        Assemble_config.Assemble_feature(config_folder_path,valid_config_list,feature_path,"G4WWW","W")
        Assemble_config.Assemble_feature(config_folder_path,valid_config_list,feature_path,"G4WOO","W")
        Assemble_config.Assemble_feature(config_folder_path,valid_config_list,feature_path,"G4WHH","W")
        print("Furter assemble !")
        xW,xW_d = Assemble_config.Assemble_features_further(feature_path,("G2WW", "G2WH", "G2WO","G4WWW", "G4WOO", "G4WHH"))

        xWW_d = xW_d[:, natoms : natoms + nwannier, :, :, :]

        xW_scale = np.loadtxt(xW_scale_path,dtype = np.float32)

        xW = (xW - xW_scale[:,0]) / (xW_scale[:,2] - xW_scale[:,1])
        
        xWW_d = xWW_d / (xW_scale[:,2] - xW_scale[:,1])

        print("Start Saving memmap : ")
        Memmap.Memmap_save(save_folder_path,"xW.dat",xW)
        Memmap.Memmap_save(save_folder_path,"xWW_d.dat",xWW_d)
        print("Start Saving npy:")
        np.save("{}/xW".format(save_folder_path), xW)
        np.save("{}/xWW_d".format(save_folder_path), xWW_d)

def Produce_labels(data_path,config_list,save_path,chargeW):
    Box_list = ()
    Hxyz_list = ()
    Oxyz_list = ()
    Wxyz_list = ()
    for iconfig,config_name in tqdm(enumerate(config_list)):
        config_path = "{}/{}".format(data_path,config_name)
        Box = np.loadtxt("{}/Box.txt".format(config_path))
        Hxyz = np.loadtxt("{}/Hxyz.txt".format(config_path))
        Oxyz = np.loadtxt("{}/Oxyz.txt".format(config_path))
        Wxyz = np.loadtxt("{}/Wxyz.txt".format(config_path))

        Box_extend = np.expand_dims(Box,axis = 1)
        Hxyz_extend = np.expand_dims(Hxyz,axis = (2,3))
        Oxyz_extend = np.expand_dims(Oxyz,axis = (2,3))
        Wxyz_extend = np.expand_dims(Wxyz,axis = (2,3))

        Box_list += (Box_extend,)
        Hxyz_list += (Hxyz_extend,)
        Oxyz_list += (Oxyz_extend,)
        Wxyz_list += (Wxyz_extend,)

    Box = np.concatenate(Box_list,axis = 1)
    Hxyz = np.concatenate(Hxyz_list,axis = 3)
    Oxyz = np.concatenate(Oxyz_list,axis = 3)
    Wxyz = np.concatenate(Wxyz_list,axis = 3)

    EO,EH,EW = Ewald_sum(Oxyz,Hxyz,Wxyz,Box,qH,qO,qW,sigma,nkmax = 5)
    
    Wannier_force = - chargeW * EW

    Wannier_force = Wannier_force[:,:,0,:]
    
    Memmap.Memmap_save(save_path,"Wforce",Wannier_force)

    # np.save("{}/Wannier_force".format(save_path),Wannier_force)

if __name__ == "__main__":
    bad_configuration = loading_data.Load_bad_config("/DATA/users/yanghe/projects/Wannier_center_pred/Code/bad_configurations.txt")
    train_configuration = [i for i in range(1,1000+1) if i not in bad_configuration]
    valid_configuration = [i for i in range(1001,1500) if i not in bad_configuration]

    TRAIN_CONFIG_DATA_PATH = "/DATA/users/yanghe/projects/Wannier_center_pred/Data/Data_splited/Train_data"
    VALID_CONFIG_DATA_PATH = "/DATA/users/yanghe/projects/Wannier_center_pred/Data/Data_splited/Valid_data"
    TRAIN_DATA_PATH = "/DATA/users/yanghe/projects/Wannier_center_pred/Data/Train_data"
    VALID_DATA_PATH = "/DATA/users/yanghe/projects/Wannier_center_pred/Data/Valid_data"
    SCRIPT_PATH = "/DATA/users/yanghe/projects/Wannier_center_pred/Code/Produce_features.o"



    # Produce_split_features(TRAIN_CONFIG_DATA_PATH,SCRIPT_PATH)
    # Produce_split_features(VALID_CONFIG_DATA_PATH,SCRIPT_PATH)

    # Assemble_config.Generate_train_data(TRAIN_CONFIG_DATA_PATH,train_configuration,TRAIN_DATA_PATH)
    # Assemble_config.Generate_valid_data(VALID_CONFIG_DATA_PATH,valid_configuration,VALID_DATA_PATH,"{}/xW_scalefactor.txt".format(TRAIN_DATA_PATH))
    Produce_labels(TRAIN_CONFIG_DATA_PATH,train_configuration,TRAIN_DATA_PATH,qW)
    Produce_labels(VALID_CONFIG_DATA_PATH,valid_configuration,VALID_DATA_PATH,qW)
#
