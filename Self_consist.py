from ast import iter_child_nodes
from locale import NOEXPR
import os
from tracemalloc import start
import torch    
import numpy as np
from Ewald_sum import *
from Coord_rotation import Coord_rotate
from Train_network import BPNet
from Produce_features import Calculate_features
import time

class Wannier_coord_calculate():
    def __init__(self,Box_path,Hxyz_path,Oxyz_path,Local_wxyz_path,module_path,scalefactor_path):
        self.Box = np.loadtxt(Box_path)
        self.Hxyz = np.loadtxt(Hxyz_path)
        self.Oxyz = np.loadtxt(Oxyz_path)
        # Load network data 
        self.net = BPNet()
        self.net.load_state_dict(torch.load(module_path))
        self.scale_factor = np.loadtxt(scalefactor_path)
        # Check Box type
        if (self.Box.shape == ()):
            self.Box = np.array([self.Box,self.Box,self.Box])
        # Calculate atom number
        self.nOxygen = self.Oxyz.shape[0]
        self.nHydrogen = self.Hxyz.shape[0]
        self.nAtom = self.nOxygen + self.nHydrogen
        self.nWannier = 4 * self.nOxygen
        # Wannier center init
        temp_Wxyz = np.zeros((self.nOxygen,4,3))
        local_wxyz = np.loadtxt(Local_wxyz_path)
        for imol in range(self.nOxygen):
            temp_Wxyz[imol,:,:] = local_wxyz
        # data extend
        self.Box_extend = np.expand_dims(self.Box,axis = 1)
        self.Hxyz_extend = np.expand_dims(self.Hxyz, axis = (2,3))
        self.Oxyz_extend = np.expand_dims(self.Oxyz, axis = (2,3))
        temp_Wxyz_extend = np.expand_dims(temp_Wxyz,axis = (3,4))
        # Back rotate wannier center
        _, _, OH_vec_stack = Coord_rotate.Get_Oxygen_neighbour(self.Oxyz_extend,self.Hxyz_extend,self.Box_extend)
        rotatorO = Coord_rotate.Get_rotamer(OH_vec_stack)
        self.Wxyz_extend = Coord_rotate.Wannier_backrotate_shift(temp_Wxyz_extend,rotatorO,self.Oxyz_extend)
        self.Wxyz = np.squeeze(self.Wxyz_extend,axis = (3,4))
        self.Wxyz_extend = np.reshape(self.Wxyz_extend,(self.nWannier,3,1,1))

    def Selfconsist_iter(self,alpha):
        iter_number = 0
        while True:
            EO,EH,EW = Ewald_sum(self.Oxyz_extend,self.Hxyz_extend,self.Wxyz_extend,self.Box_extend,2,6,-2,8)
            EW = np.squeeze(EW,axis = (2,3))
            xW,xWWd = Calculate_features(self.Hxyz,self.Oxyz,self.Wxyz_extend[:,:,0,0],self.Box,self.nOxygen,self.scale_factor)
            xW = torch.tensor(xW,dtype=torch.float32)
            xWWd = torch.tensor(xWWd,dtype=torch.float32)
            wannier_force = self.net(xW,xWWd).detach().numpy()
            wannier_diff = np.mean(np.abs(wannier_force + EW * (-2) ) )            
            self.Wxyz_extend[:,:,0,0] = self.Wxyz_extend[:,:,0,0] + alpha * ( wannier_force + EW * (-2) )
            iter_number += 1
            print(iter_number,wannier_diff)
            if iter_number % 10:
                self.Generate_file("./Coord_{}.txt".format(iter_number))

    def Generate_file(self,save_path):
        """
            Generate coord file which can be reading by OVITO
        """
        with open(save_path,"w") as f:
            f.write("\t{}\n".format(self.nAtom + self.nWannier))
            f.write("  config_in_3d_box\t{}\t{}\t{}\n".format(self.Box[0],self.Box[1],self.Box[2]))

            for imol in range(self.nOxygen):
                f.write("O\t{}\t{}\t{}\n".format(self.Oxyz[imol,0],self.Oxyz[imol,1],self.Oxyz[imol,2]))
                f.write("H1\t{}\t{}\t{}\n".format(self.Hxyz[2 * imol,0],self.Hxyz[2 * imol,1],self.Hxyz[2 * imol,2]))
                f.write("H2\t{}\t{}\t{}\n".format(self.Hxyz[2 * imol + 1,0],self.Hxyz[2 * imol + 1,1],self.Hxyz[2 * imol + 1,2]))
            for iwannier in range(self.nWannier):
                f.write("W\t{}\t{}\t{}\n".format(self.Wxyz_extend[iwannier,0,:,:],self.Wxyz_extend[iwannier,1,:,:],self.Wxyz_extend[iwannier,2,:,:]))

if __name__ == "__main__":
    CONFIG_PATH = "/DATA/users/yanghe/projects/SCFNN/Data/D0/1593"
    Box_path = "{}/box.txt".format(CONFIG_PATH)
    Hxyz_path = "{}/Hxyz.txt".format(CONFIG_PATH)
    Oxyz_path = "{}/Oxyz.txt".format(CONFIG_PATH)
    Wxyz_path = "{}/wxyz.txt".format(CONFIG_PATH)
    Wxyz_init_path = "/DATA/users/yanghe/projects/Wannier_center_pred/Code/Wannier_center_mean.txt"
    Neuralnetwork_path = "/DATA/users/yanghe/projects/Wannier_center_pred/Data/Model_saved/Network_wanner_force.pth"
    Scale_factor_path = "/DATA/users/yanghe/projects/Wannier_center_pred/Data/Train_data/xW_scalefactor.txt"
    a = Wannier_coord_calculate(Box_path,Hxyz_path,Oxyz_path,Wxyz_init_path,Neuralnetwork_path,Scale_factor_path)
    
    a.Selfconsist_iter(0.7)
    # H_index_stack, OH_norm_stack, OH_vec_stack = Coord_rotate.Get_Oxygen_neighbour(Oxyz_extend,Hxyz_extend,Box_extend)
    # rotamerO = Coord_rotate.Get_rotamer(OH_vec_stack)

    # Backrotate_mapped_Wxyz = Coord_rotate.Wannier_backrotate_shift(Wxyz_first_guess_extend,rotamerO,Oxyz_extend)
    # Backrotate_Wxyz = Backrotate_mapped_Wxyz.reshape((nOxygen * 4,3))

