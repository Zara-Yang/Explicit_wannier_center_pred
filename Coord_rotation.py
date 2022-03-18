from matplotlib.pyplot import box, minorticks_off
import numpy as np

class Coord_rotate():
    @staticmethod
    def Get_neighbour(coord_i,coord_j,box_length,neighbour_number):
        """
        for any type atom i, find closest atom j 

        input:
                name                   description              dimension
            --------------------------------------------------------------------
            coord_i             : all coord for type atom i : [n_i,3,nfolder,nconfig]
            coord_j             : all coord for type atom j : [n_j,3,nfolder,nconfig]
            box_length          : box size for pbc          : [3]
            neighbour_number    : neighbour atom number     : int
        
        output:
                name                   description              dimension
            --------------------------------------------------------------------
            sorted_index        : sorted index of atom j    : [n_i,n_j]
            rij_norm_min        : sorted |r_{ij}| of atom j : [n_i,n_j]
            rij_vec_min         : sorted vec(r_{ij}) of j   : [n_i,n_j,3]
        """
        coord_i = np.expand_dims(coord_i,axis=1)                                                # [n_i,1,3]
        coord_j = np.expand_dims(coord_j,axis=0)                                                # [1,n_j,3]
        rij_vec = coord_i - coord_j - np.round((coord_i - coord_j) / box_length) * box_length   # [n_i,n_j,3]
        rij_norm = np.linalg.norm(rij_vec,axis = 2)                                             # [n_i,n_j]
        sorted_rij_index = np.argsort(rij_norm,axis=1)                                          # [n_i,n_j]
        sorted_rij_index2 = np.expand_dims(sorted_rij_index,axis = 2)
        arange_index_i = np.expand_dims(np.arange(coord_i.shape[0]),axis = 1)
        arange_index_i2 = np.expand_dims(np.arange(coord_i.shape[0]),axis = (1,2))
        arange_xyz = np.expand_dims(np.arange(3),axis = (0,1))
        
        rij_norm_min = rij_norm[arange_index_i,sorted_rij_index]
        rij_vec_min = rij_vec[arange_index_i2,sorted_rij_index2,arange_xyz]
        return(sorted_rij_index[:,0:neighbour_number],rij_norm_min[:,0:neighbour_number],rij_vec_min[:,0:neighbour_number,:])

    @staticmethod
    def Get_Oxygen_neighbour(coord_O_all,coord_H_all,box_length_all):
        """
        Find two closest Hydrogen atom for any Oxygen atom

        input:
                name                   description              dimension
            ------------------------------------------------------------------------------
            coord_O_all             All Oxygen atom coord       [nOxygen,3,nfolfer,nconfig]
            coord_H_all             All Hydrogen atom coord     [nHydrogen,3,nfolfer,nconfig]
            box_length_all          All config box range        [3,nfolder,nconfig]
        output:
                name                   description              dimension
            --------------------------------------------------------------------
            H_index_stack       index of neighbour Hydrogen     [nOxygen,2,nfolders,nconfigs]
            OH_norm_stack           norm of vector OH           [nOxygen,2,nfolders,nconfigs]
            OH_vec_stack            vec of vector OH            [nOxygen,3,2,nfolders,nconfigs]
        """
        nOxygen,_,nfolders,nconfigs = coord_O_all.shape
        nHydrogen,_,_,_ = coord_H_all.shape

        H_index = ()
        OH_norm = ()
        OH_vec = ()

        for ifolder in range(nfolders):
            for iconfig in range(nconfigs):
                neighbour_OH = Coord_rotate.Get_neighbour(coord_i=coord_O_all[:,:,ifolder,iconfig],
                                                            coord_j=coord_H_all[:,:,ifolder,iconfig],
                                                            box_length=box_length_all[:,iconfig],
                                                            neighbour_number=2)
                H_index += (neighbour_OH[0],)
                OH_norm += (neighbour_OH[1],)
                OH_vec += (neighbour_OH[2],)

        H_index_stack = np.stack(H_index, axis=-1).reshape((nOxygen, 2, nfolders, nconfigs))
        OH_norm_stack = np.stack(OH_norm, axis=-1).reshape((nOxygen, 2, nfolders, nconfigs))
        OH_vec_stack = np.stack(OH_vec, axis=-1).reshape((nOxygen, 2, 3, nfolders, nconfigs))
        return H_index_stack, OH_norm_stack, OH_vec_stack

    @staticmethod
    def Get_Hydrogen_neighbour(coord_O_all,coord_H_all,box_length_all):
        """
        Find One Hydrogen and One Oxygen atom  for every Hydrogen in every folder and every config

        input:
                name                   description              dimension
            ------------------------------------------------------------------------------
            coord_O_all             All Oxygen atom coord       [nOxygen,3,nfolfer,nconfig]
            coord_H_all             All Hydrogen atom coord     [nHydrogen,3,nfolfer,nconfig]
            box_length_all          All config box range        [3,nfolder,nconfig]
        output:
                name                   description              dimension
            ------------------------------------------------------------------------------
            neighbour_index_all  Index of Oxygen and Hydrogen   [nHydrogen,2,nfolder,nconfig]
            neighbour_norm_all      Norm of HO and HH           [nHydrogen,2,nfolder,nconfig]
            neighbour_vec_all       Vec of HO and OO            [nHydrogen,3,2,nfolder,nconfig]
        """
        nOxygen,_,nfolders,nconfigs = coord_O_all.shape
        nHydrogen,_,_,_ = coord_H_all.shape

        O_index = ()
        HO_norm = ()
        HO_vec = ()

        H_index = ()
        HH_norm = ()
        HH_vec = ()
        
        for ifolder in range(nfolders):
            for iconfig in range(nconfigs):
                neighbour_HO = Coord_rotate.Get_neighbour(coord_i=coord_H_all[:,:,ifolder,iconfig],
                                                            coord_j=coord_O_all[:,:,ifolder,iconfig],
                                                            box_length=box_length_all[:,iconfig],
                                                            neighbour_number=1)
                O_index += (neighbour_HO[0],)
                HO_norm += (neighbour_HO[1],)
                HO_vec += (neighbour_HO[2],)

                neighbour_HH = Coord_rotate.Get_neighbour(coord_i=coord_H_all[:,:,ifolder,iconfig],
                                                            coord_j=coord_H_all[:,:,ifolder,iconfig],
                                                            box_length=box_length_all[:,iconfig],
                                                            neighbour_number=2)
                H_index += (neighbour_HH[0][:,1],)
                HH_norm += (neighbour_HH[1][:,1],)
                HH_vec += (neighbour_HH[2][:,1,:],)
        
        O_index_stack = np.stack(O_index,axis = -1).reshape((nHydrogen,1,nfolders,nconfigs))
        HO_norm_stack = np.stack(HO_norm,axis = -1).reshape((nHydrogen,1,nfolders,nconfigs))
        HO_vec_stack = np.stack(HO_vec,axis=-1).reshape((nHydrogen,1,3,nfolders,nconfigs))

        H_index_stack = np.stack(H_index,axis = -1).reshape((nHydrogen,1,nfolders,nconfigs))
        HH_norm_stack = np.stack(HH_norm,axis = -1).reshape((nHydrogen,1,nfolders,nconfigs))
        HH_vec_stack = np.stack(HH_vec,axis=-1).reshape((nHydrogen,1,3,nfolders,nconfigs))

        neighbour_index_all = np.concatenate((O_index_stack, H_index_stack), axis=1)
        neighbour_norm_all = np.concatenate((HO_norm_stack, HH_norm_stack), axis=1)
        neighbour_vec_all = np.concatenate((HO_vec_stack, HH_vec_stack), axis=1)

        return neighbour_index_all, neighbour_norm_all, neighbour_vec_all

    @staticmethod
    def Get_rotamer(neighbour_rij_vec):
        """
        Calculate rotate matrix for local frame

        input:
            name                   description              dimension
        ------------------------------------------------------------------------------
        neighbour_rij_vec       vec rij for local frame     [ncenter,2,3,nfolder,nconfig]
        
        output:
            name                   description              dimension
        ------------------------------------------------------------------------------
        rotamer_matrix            rotate matrixs            [ncenter,3,3,nfolder,nconfig]
        """
        ncenter,_,_,nfolder,nconfig = neighbour_rij_vec.shape
        rotamer_matrix = np.zeros((ncenter,3,3,nfolder,nconfig))
        vec_1 = np.zeros(3)
        vec_2 = np.zeros(3)
        matrix_buffer = np.zeros((3,3))
        for ifolder in range(nfolder):
            for iconfig in range(nconfig):
                for center_index in range(neighbour_rij_vec.shape[0]):
                    vec_1 = neighbour_rij_vec[center_index,0,:,ifolder,iconfig]
                    vec_2 = neighbour_rij_vec[center_index,1,:,ifolder,iconfig]
                    axis_x = np.cross(vec_1,vec_2)
                    axis_x = axis_x / np.linalg.norm(axis_x)
                    axis_z = vec_1 / np.linalg.norm(vec_1)
                    axis_y = np.cross(axis_x,axis_z)
                    axis_y = axis_y / np.linalg.norm(axis_y)
                    matrix_buffer[0,:] = axis_x
                    matrix_buffer[1,:] = axis_y
                    matrix_buffer[2,:] = axis_z
                    rotamer_matrix[center_index,:,:,ifolder,iconfig] = - matrix_buffer
        return(rotamer_matrix)

    @staticmethod
    def Rotate(coord_matrix,rotate_matrix):
        """
        Rotate 3d coord from global frame to local frame by rotate matrix

        input:
            name                   description              dimension
        ------------------------------------------------------------------------------
        coord_matrix      The coord waited to be rotated    [ncenter,3,nfolder,nconfig]
        rotate_matrix             Rotate matrixs            [ncenter,3,3,nfolder,nconfig]

        output:
            name                   description              dimension
        ------------------------------------------------------------------------------
        rotated_coord         Coords in local frame         [ncenter,3,nfolder,nconfig]
        """
        ncenter,_,_,nfolder,nconfig = rotate_matrix.shape
        ncoord,_,_,_ = coord_matrix.shape
        rotated_coord = np.zeros((ncenter,ncoord,3,nfolder,nconfig))

        for ifolder in range(nfolder):
            for iconfig in range(nconfig):
                for icenter in range(ncenter):
                    for icoord in range(ncoord):
                        rotated_coord[icenter,icoord,:,ifolder,iconfig] = rotate_matrix[icenter,:,:,ifolder,iconfig] @ coord_matrix[icoord,:,ifolder,iconfig]
        return(rotated_coord)
    
    @staticmethod
    def Back_rotate(coord_matrix,rotate_matrix):
        """
        Rotate 3d coord from local frame to global frame by rotate matrix

        input:
            name                   description              dimension
        ------------------------------------------------------------------------------
        coord_matrix      The coord waited to be rotated    [ncenter,3,nfolder,nconfig]
        rotate_matrix             Rotate matrixs            [ncenter,3,3,nfolder,nconfig]

        output:
            name                   description              dimension
        ------------------------------------------------------------------------------
        rotated_coord         Coords in local frame         [ncenter,3,nfolder,nconfig]
        """
        ncenter,_,_,nfolder,nconfig = rotate_matrix.shape
        ncoord,_,_,_ = coord_matrix.shape
        rotated_coord = np.zeros((ncenter,ncoord,3,nfolder,nconfig))

        for ifolder in range(nfolder):
            for iconfig in range(nconfig):
                for icenter in range(ncenter):
                    for icoord in range(ncoord):
                        rotated_coord[icenter,icoord,:,ifolder,iconfig] = rotate_matrix[icenter,:,:,ifolder,iconfig].T @ coord_matrix[icoord,:,ifolder,iconfig]
        return(rotated_coord)

    @staticmethod
    def Wannier_backrotate_shift(coord_matrix,rotate_matrix,rotate_center):
        """
        Back rotate and reshift wannier center
        input:
                name                   description              dimension
            --------------------------------------------------------------------
            coord_matrix           : All Wannier center coord  : [nMol,4,3,nfolder,nconfig]
            rotate_matrix          : All rotate matrix         : [nMol,3,3,nfolder,nconfig]
            rotate_center          : Shift center(Oxygen coord): [nOxygen,3,nfolder,nconfig]
        
        output:
                name                   description              dimension
            --------------------------------------------------------------------
            backrotated_coord  : Wannier center in global frame : [nWannier,4,3,nfolder,nconfig] 
        """
        ncenter,_,_,nfolder,nconfig = rotate_matrix.shape
        backrotated_coord = np.zeros((ncenter,4,3,nfolder,nconfig))

        for ifolder in range(nfolder):
            for iconfig in range(nconfig):
                for icenter in range(ncenter):
                    for icoord in range(4):
                        backrotated_coord[icenter,icoord,:,ifolder,iconfig] = rotate_matrix[icenter,:,:,ifolder,iconfig].T @ coord_matrix[icenter,icoord,:,ifolder,iconfig] + rotate_center[icenter,:,ifolder,iconfig]
        return(backrotated_coord)

    @staticmethod
    def Shift_rotate(coord_matrix,rotate_matrix,origin_coord,box_length):
        """
        Shift origin point and rotate to local frame

        """
        ncenter,_,_,nfolder,nconfig = rotate_matrix.shape
        ncoord,_,_,_ = coord_matrix.shape
        coord_xyz = np.expand_dims(coord_matrix,axis = 0)
        origin_xyz = np.expand_dims(origin_coord,axis = 1)
        box_length = np.expand_dims(box_length,axis = 1)
        # ====================================================
        #  Box_xyz's expand_dims might have some problem
        #  Do check again please !
        # ====================================================
        shifted_xyz = coord_xyz - origin_xyz - np.round((coord_xyz - origin_xyz)/box_length) * box_length

        for ifolder in range(nfolder):
            for iconfig in range(nconfig):
                for icenter in range(ncenter):
                    for icoord in range(ncoord):
                        shifted_xyz[icenter,icoord,:,ifolder,iconfig] = rotate_matrix[icenter,:,:,ifolder,iconfig] @ shifted_xyz[icenter,icoord,:,ifolder,iconfig]
        return(shifted_xyz)

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












