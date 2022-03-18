import os
import shutil
from glob import glob
import numpy as np

"""
    This code is used to loading SCFNN train data and do some analysis 
        1. Load and split data for train and valid process
        2. Creat copy for each config for create GT feature(.cpp)
        3. Assemble wannnier center to each H2O molecular

                                                ZaraYang
"""

class loading_data():
    @staticmethod
    def Map_wannier_center(wannier_coord_rotated):
        """
        Cluster wannier center for each molecular
        input :
                name                         description                            dimension
            -------------------------------------------------------------------------------------------------
            wannier_coord_rotated    wannier center in local frame      [nCenter,nWannier,3,nFolder,nConfig]

        output :
                name                         description                            dimension
            -------------------------------------------------------------------------------------------------
            wannier_norm_min          wannier center norm               [nCenter,nWannier,nFolder,nConfig]
            wannier_final             wannier center vec sorted         [nCenter,nWannier,3,nFolder,nConfig]
        """
        nMol,nWannier,_,nfolder,nconfig = wannier_coord_rotated.shape
        wannier_norm = np.linalg.norm(wannier_coord_rotated,axis=2)
        argmin_wannier_norm = np.argsort(wannier_norm,axis=1)

        mol_index = np.expand_dims(np.arange(nMol),axis = (1,2,3))
        folder_index = np.expand_dims(np.arange(nfolder),axis = (0,1,3))
        config_index = np.expand_dims(np.arange(nconfig),axis = (0,1,2))

        argmin_wannier_norm_2 = np.expand_dims(argmin_wannier_norm,axis = 2)
        mol_index_2 = np.expand_dims(np.arange(nMol),axis = (1,2,3,4))
        folder_index_2 = np.expand_dims(np.arange(nfolder),axis = (0,1,2,4))
        config_index_2 = np.expand_dims(np.arange(nconfig),axis = (0,1,2,3))
        xyz_index_2 = np.expand_dims(np.arange(3),axis = (0,1,3,4))

        wannier_norm_min = wannier_norm[mol_index,argmin_wannier_norm,folder_index,config_index]
        wannier_vec_min = wannier_coord_rotated[mol_index_2,argmin_wannier_norm_2,xyz_index_2,folder_index_2,config_index_2]

        mapped_wannier_rotated = wannier_vec_min[:,:4,:,:,:]
        # Sorted wannier center
        wannier_z_index = np.expand_dims(np.argsort(mapped_wannier_rotated[:,:,2,:,:],axis= 1),axis = 2)
        wannier_zmin = mapped_wannier_rotated[mol_index_2,wannier_z_index,xyz_index_2,folder_index_2,config_index_2]

        wannier_zmin_lower = wannier_zmin[:,:3,:,:,:]
        wannier_x_index = np.expand_dims(np.argsort(wannier_zmin_lower[:,:,0,:,:],axis = 1 ),axis = 2)
        wannier_xmin = wannier_zmin_lower[mol_index_2,wannier_x_index,xyz_index_2,folder_index_2,config_index_2]

        wannier_final = np.concatenate((wannier_xmin,wannier_zmin[:,3:,:,:,:]),axis = 1) 
        return(wannier_norm_min[:,:4,:,:],wannier_final)

    @staticmethod
    def Load_bad_config(bad_config_path):
        """
        Load bad config from bad_configurations.txt
        """
        bad_config = []
        with open(bad_config_path,"r") as f:
            for line in f:
                bad_config.append(int(line.replace("\n","")))
        return(bad_config)

    @staticmethod
    def load_data(data_path,nmol,load_config,GT_buffer_path = None):
        """
        Loading coord file to matrix
            nfolder : types of external field
            nconfig : H2O data configs of each external field
        """
        nOxygen = nmol
        nHydrogen = 2 * nmol
        nWannier = 4 * nmol

        nfolder = 0
        nconfig = 0

        if (GT_buffer_path != None) and (not os.path.exists(GT_buffer_path)):
            os.makedirs(GT_buffer_path)

        config_path_list = {os.path.basename(i) : [] for i in glob("{}/*".format(data_path))}
        for key in config_path_list:
            for config_index in load_config:
                config_path = "{}/{}/{}".format(data_path,key,config_index)
                config_path_list[key].append(config_path)
            nconfig = len(config_path_list[key])
        nfolder = len(config_path_list)

        Oxygen_coord = np.zeros((nOxygen,3,nfolder,nconfig))        # Oxygen coord matrix 
        Hydrogen_coord = np.zeros((nHydrogen,3,nfolder,nconfig))    # Hydrogen coord matrix
        Wannier_coord = np.zeros((nWannier,3,nfolder,nconfig))      # Wannier center coord matrix
        Oxygen_force = np.zeros((nOxygen,3,nfolder,nconfig))
        Hydrogen_force = np.zeros((nHydrogen,3,nfolder,nconfig))
        Box_length = np.zeros((3,nfolder,nconfig))
        for folder_index,folder_name in enumerate(config_path_list):
            for config_index,config_path in enumerate(config_path_list[folder_name]):
                Oxygen_coord[:,:,folder_index,config_index],Hydrogen_coord[:,:,folder_index,config_index],Wannier_coord[:,:,folder_index,config_index] = loading_data.load_coord_file("{}/W64-bulk-HOMO_centers_s1-1_0.xyz".format(config_path),nmol)
                Box_length[:,folder_index,config_index] = loading_data.load_init_file("{}/init.xyz".format(config_path))
                Oxygen_force[:,:,folder_index,config_index],Hydrogen_force[:,:,folder_index,config_index] = loading_data.load_force_file("{}/W64-bulk-W64-forces-1_0.xyz".format(config_path),nmol)
                if GT_buffer_path != None:
                    temp_path = config_path.replace(data_path,GT_buffer_path)
                    if not os.path.exists(temp_path):
                        os.makedirs(temp_path)
                    else:
                        shutil.rmtree(temp_path)
                        os.makedirs(temp_path)
                    np.savetxt("{}/Oxyz.txt".format(temp_path),Oxygen_coord[:,:,folder_index,config_index])
                    np.savetxt("{}/Hxyz.txt".format(temp_path),Hydrogen_coord[:,:,folder_index,config_index])
                    np.savetxt("{}/Box.txt".format(temp_path),Box_length[:,folder_index,config_index] )
                    np.savetxt("{}/Oforce.txt".format(temp_path),Oxygen_force[:,:,folder_index,config_index])
                    np.savetxt("{}/Hforce.txt".format(temp_path),Hydrogen_force[:,:,folder_index,config_index])
                    np.savetxt("{}/Wxyz.txt".format(temp_path),Wannier_coord[:,:,folder_index,config_index])
        return(Box_length[:,0,:],Oxygen_coord,Hydrogen_coord,Wannier_coord,Oxygen_force,Hydrogen_force)

    @staticmethod
    def load_coord_file(data_path,nMol,atom_unit = True):
        """
        load single coord data file to matrix
        ----------------------------
        input:
            coord_data_path
        output:
            O_coord = [nOxygen, 3]
            H_coord = [nHydrogen, 3]
            w_coord = [4 * nOxygen, 3]
        """
        O_coord = []
        H_coord = []
        W_coord = []

        data_file = open(data_path,"r")
        lines = [i.replace("\n","").split() for i in data_file.readlines()]
        lines = lines[2:]
        
        for mol_index in range(nMol):
            O_coord.append([float(i) for i in lines[mol_index * 3][1:]])
            H_coord.append([float(i) for i in lines[mol_index * 3 + 1][1:]])
            H_coord.append([float(i) for i in lines[mol_index * 3 + 2][1:]])
        
        for mol_index in range(nMol):
            for temp_index in range(4):
                W_coord.append([float(i) for i in lines[nMol * 3 + mol_index * 4 + temp_index][1:]])
        
        O_coord = np.array(O_coord)
        H_coord = np.array(H_coord)
        W_coord = np.array(W_coord)

        if atom_unit :
            O_coord /= 0.529177
            H_coord /= 0.529177
            W_coord /= 0.529177
        
        return(np.array(O_coord),np.array(H_coord),np.array(W_coord))

    @staticmethod
    def load_init_file(data_path):
        """
        Reading box range from init.xyz
        -----------------
        input :
            init.xyz file path
        output :
            box range [3]
        """
        init_file = open(data_path)
        init_file.readline()
        box_range_line = init_file.readline()
        box_range_line = box_range_line.split("Lattice=")[1].split()
        box_range = [box_range_line[0],box_range_line[4],box_range_line[8]]
        box_range = np.array([float(i.replace("\"","")) for i in box_range])
        return(box_range)
    
    @staticmethod
    def data_assemble(data_path,save_path,data_config,nmol):
        Box_length,Oxygen_coord,Hydrogen_coord,Wannier_coord,Oxygen_force,Hydrogen_force = loading_data.load_data(  data_path=data_path,
                                                                                                                    nmol = nmol,
                                                                                                                    load_config=data_config)
        np.save("{}/Box_size".format(save_path),Box_length)
        np.save("{}/Oxygen_xyz".format(save_path),Oxygen_coord)
        np.save("{}/Hydrogen_xyz".format(save_path),Hydrogen_coord)
        np.save("{}/Wannier_xyz".format(save_path),Wannier_coord)
        np.save("{}/Oxygen_force".format(save_path),Oxygen_force)
        np.save("{}/Hydrogen_force".format(save_path),Hydrogen_force)
    
    @staticmethod
    def load_force_file(data_path,nMol):
        O_force = []
        H_force = []

        data_file = open(data_path,"r")
        lines = [i.replace("\n","").split() for i in data_file.readlines()]
        lines = lines[4:]
        lines = lines[:-1]

        for mol_index in range(nMol):
            O_force.append([float(i) for i in lines[mol_index * 3][3:]])
            H_force.append([float(i) for i in lines[mol_index * 3 + 1][3:]])
            H_force.append([float(i) for i in lines[mol_index * 3 + 2][3:]])
        
        O_force = np.array(O_force)
        H_force = np.array(H_force)
        return(np.array(O_force),np.array(H_force))


if __name__ == "__main__":
    # Configuration list
    bad_configuration = loading_data.Load_bad_config("/DATA/users/yanghe/projects/Wannier_center_pred/Code/bad_configurations.txt")
    train_configuration = [i for i in range(1,1000+1) if i not in bad_configuration]
    valid_configuration = [i for i in range(1001,1500) if i not in bad_configuration]

    data_path = "/DATA/users/yanghe/projects/Wannier_center_pred/Data/Origin_data"
    train_path = "/DATA/users/yanghe/projects/Wannier_center_pred/Data/Train_data"
    valid_path = "/DATA/users/yanghe/projects/Wannier_center_pred/Data/Valid_data"

    loading_data.load_data(data_path,64,train_configuration,GT_buffer_path = "/DATA/users/yanghe/projects/Wannier_center_pred/Data/Data_splited/Train_data")
    loading_data.load_data(data_path,64,valid_configuration,GT_buffer_path = "/DATA/users/yanghe/projects/Wannier_center_pred/Data/Data_splited/Valid_data")

    loading_data.data_assemble(data_path = data_path,
                                save_path = train_path,
                                data_config = train_configuration,
                                nmol = 64)


    loading_data.data_assemble(data_path = data_path,
                                save_path = valid_path,
                                data_config = valid_configuration,
                                nmol = 64) 

