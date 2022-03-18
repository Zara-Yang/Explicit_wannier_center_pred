import numpy as np
import math
import multiprocessing
import time
from tqdm import tqdm

class Symmetry_func():
    @staticmethod
    def Trunc_func(x, rc):
        """
            set a smooth truncated function for every r_ij in symmetry function 
        """
        y = 0
        if (x < rc):
            y = pow(math.tanh(1 - (x/rc)),3)
        return(y)
    @staticmethod
    def Trunc_func_dev(x, rc):
        """
            Derivative of truncated function
        """
        y = 0
        if (x < rc):
            y = -3 * pow(math.tanh(1 - (x / rc)),2) / pow(math.cosh(1 - (x / rc)),2) /rc
        return(y)
    @staticmethod
    def G2_func(rij, yeta, rs, rc):
        """
        G2 Symmetry function for any r_ij
        """
        y = math.exp(-yeta * pow((rij - rs),2)) * Symmetry_func.Trunc_func(rij, rc)
        return(y)
    @staticmethod
    def G2_dev_func(rij, yeta, rs, rc):
        """
            Derivative of G2 symmetry function ( except vector part )
            
            Math : 
                \partial G2(r_ij)        -yeta(r_ij - rs)^2                 -yeta(r_ij - rs)^2
                -------------------  =  e                    fc_dev(r_ij) + e                   (-2 yeta (rij - rs)) * fc(r_ij)
                \partial r_ij
        """
        y = math.exp(- yeta * (rij - rs)**2) * Symmetry_func.Trunc_func_dev(rij, rc)
        y+= math.exp(- yeta * (rij - rs)**2) * Symmetry_func.Trunc_func(rij, rc) * (-2 * yeta * (rij - rs))
        return(y)
    @staticmethod
    def G4_func(rij,rjk,rki,cos_alpla,zeta,yeta,lam,rc):
        """
        G4 symmetry function for any r_ij,r_jk,r_ki
        """
        y = math.exp(- yeta * (rij**2 + rki**2 + rjk**2)) * math.pow((1 + lam * cos_alpla),zeta)
        y *= Symmetry_func.Trunc_func(rij,rc)
        y *= Symmetry_func.Trunc_func(rjk,rc)
        y *= Symmetry_func.Trunc_func(rki,rc)
        y *= pow(2,(1 - zeta))
        return(y)
    @staticmethod
    def G4_dev_func(rij, rjk, rki, cos_alpha, zeta, yeta, lam, rc):
        result = np.zeros(3)

        fc_rij = Symmetry_func.Trunc_func(rij, rc)
        fc_rjk = Symmetry_func.Trunc_func(rjk, rc)
        fc_rki = Symmetry_func.Trunc_func(rki, rc)

        fc_rij_dev = Symmetry_func.Trunc_func_dev(rij, rc)
        fc_rjk_dev = Symmetry_func.Trunc_func_dev(rjk, rc)
        fc_rki_dev = Symmetry_func.Trunc_func_dev(rki, rc)

        common_value = pow(2, 1-zeta) * math.exp(-yeta * (rij**2 + rjk**2 + rki**2)) * pow((1 + lam * cos_alpha),(zeta - 1))
        
        rij_dev = zeta * lam * (1 / rki - (rij**2 + rki**2 - rjk**2) / (2 * rij * rij * rki)) * fc_rij * fc_rjk * fc_rki
        rij_dev+= -2 * rij * yeta * (1 + lam * cos_alpha) * fc_rij * fc_rjk * fc_rki
        rij_dev+= fc_rij_dev * (1 + lam * cos_alpha) * fc_rki * fc_rjk

        rjk_dev = zeta * lam * (1 / rij - (rij**2 + rki**2 - rjk**2) / (2 * rki * rki * rij)) * fc_rij * fc_rjk * fc_rki
        rjk_dev+= -2 * rki * yeta * (1 + lam * cos_alpha) * fc_rij * fc_rjk * fc_rki
        rjk_dev+= fc_rki_dev * (1 + lam * cos_alpha) * fc_rij * fc_rjk

        rki_dev = zeta * lam * (- rjk / (rij * rki)) * fc_rij * fc_rjk * fc_rki
        rki_dev+= -2 * rjk * yeta * (1 + lam * cos_alpha) * fc_rij * fc_rjk * fc_rki
        rki_dev+= fc_rjk_dev * (1 + lam * cos_alpha) * fc_rij * fc_rki

        rij_dev *= common_value
        rjk_dev *= common_value
        rki_dev *= common_value

        result[0] = rij_dev
        result[1] = rjk_dev
        result[2] = rki_dev

        return(result)

class Calculate_func():
    @staticmethod
    def Pairwise_matrix(Oxyz,Hxyz,Wxyz,Box):
        total_matrix = np.concatenate((Oxyz,Hxyz,Wxyz),axis = 0)
        total_dimension = total_matrix.shape[0]
        norm_matrix = np.zeros((total_dimension,total_dimension),dtype=np.float32)
        vec_matrix = np.zeros((total_dimension,total_dimension,3),dtype=np.float32)

        for index_x in range(total_dimension):
            for index_y in range(total_dimension):
                vec_matrix[index_x,index_y,:] = total_matrix[index_y] - total_matrix[index_x] - np.round((total_matrix[index_y] - total_matrix[index_x])/Box)* Box
                norm_matrix[index_x,index_y] = np.linalg.norm(vec_matrix[index_x,index_y,:])
        return(norm_matrix,vec_matrix)
    @staticmethod
    def Calculate_G2_features(norm_matrix, nMol, atom_type_1, atom_type_2, features_number, params, rc):
        # Set index list 1 & 2
        """
        Already Checked with Ang Gao's benchmark
        """
        index_list_1 = [0, nMol]
        if (atom_type_1 == 1):
            index_list_1 = [nMol, 3 * nMol]
        elif(atom_type_1 == 2):
            index_list_1 = [3 * nMol, 3 * nMol + 4 * nMol ]

        index_list_2 = [0, nMol]
        if (atom_type_2 == 1):
            index_list_2 = [nMol, 3 * nMol]
        elif(atom_type_2 == 2):
            index_list_2 = [3 * nMol, 3 * nMol + 4 * nMol ]

        G2_result = np.zeros((features_number,index_list_1[1] - index_list_1[0]), dtype=np.float32)

        for index_x in range(index_list_1[0],index_list_1[1]):
            for index_y in range(index_list_2[0],index_list_2[1]):
                if (index_x == index_y or norm_matrix[index_x,index_y] >= rc): continue
                for ip in range(features_number):
                    G2_result[ip,index_x- index_list_1[0]] += Symmetry_func.G2_func(norm_matrix[index_x, index_y],params[ip,1],params[ip,0],rc)
        return(G2_result)
    @staticmethod
    def Calculate_G4_features(norm_matrix, nMol, atom_type_1, atom_type_2, atom_type_3, features_number, params, rc):
        """
        Already Checked with Ang Gao's benchmark
        """

        index_list_1 = [0, nMol]
        if (atom_type_1 == 1):
            index_list_1 = [nMol, 3 * nMol]
        elif(atom_type_1 == 2):
            index_list_1 = [3 * nMol, 3 * nMol + 4 * nMol ]

        index_list_2 = [0, nMol]
        if (atom_type_2 == 1):
            index_list_2 = [nMol, 3 * nMol]
        elif(atom_type_2 == 2):
            index_list_2 = [3 * nMol, 3 * nMol + 4 * nMol ]

        index_list_3 = [0, nMol]
        if (atom_type_3 == 1):
            index_list_3 = [nMol, 3 * nMol]
        elif(atom_type_3 == 2):
            index_list_3 = [3 * nMol, 3 * nMol + 4 * nMol ]

        G4_result = np.zeros((features_number,index_list_1[1] - index_list_1[0]), dtype=np.float32)

        for index_x in range(index_list_1[0],index_list_1[1]):
            for index_y in range(index_list_2[0],index_list_2[1]):
                if (index_x == index_y or norm_matrix[index_x,index_y] >= rc): continue
                for index_z in range(index_list_3[0],index_list_3[1]):
                    if (index_z == index_x or index_z == index_y,norm_matrix[index_x,index_z] >= rc or norm_matrix[index_z,index_y] >= rc): continue
                    cos_alpha = (norm_matrix[index_x,index_y]**2 + norm_matrix[index_x,index_z]**2 - norm_matrix[index_y,index_z]**2) / (2 * norm_matrix[index_x,index_y] * norm_matrix[index_x,index_z])
                    for ip in range(features_number):
                        G4_result[ip,index_x - index_list_1[0]] += Symmetry_func.G4_func(   norm_matrix[index_x, index_y], 
                                                                                            norm_matrix[index_y, index_z],
                                                                                            norm_matrix[index_z, index_x],
                                                                                            cos_alpha,
                                                                                            params[ip,3],params[ip,1],params[ip,2],
                                                                                            rc)
        return(G4_result)

        pass
    @staticmethod
    def Calculate_dG2_features(norm_matrix, vec_matrix, nMol, atom_type_1, atom_type_2, features_number, params, rc):
        index_list_1 = [0, nMol]
        if (atom_type_1 == 1):
            index_list_1 = [nMol, 3 * nMol]
        elif(atom_type_1 == 2):
            index_list_1 = [3 * nMol, 3 * nMol + 4 * nMol ]

        index_list_2 = [0, nMol]
        if (atom_type_2 == 1):
            index_list_2 = [nMol, 3 * nMol]
        elif(atom_type_2 == 2):
            index_list_2 = [3 * nMol, 3 * nMol + 4 * nMol ]

        dG2_result = np.zeros((features_number, index_list_1[1] - index_list_1[0], 7 * nMol, 3), dtype=np.float32)

        for index_x in range(index_list_1[0],index_list_1[1]):
            for index_y in range(index_list_2[0],index_list_2[1]):
                if (index_x == index_y or norm_matrix[index_x,index_y] >= rc): continue
                for ip in range(features_number):
                    dev_G2 = Symmetry_func.G2_dev_func(norm_matrix[index_x,index_y],params[ip,1],params[ip,0],rc)
                    dG2_result[ip,index_x - index_list_1[0],index_y,:] += dev_G2 * vec_matrix[index_x,index_y] / norm_matrix[index_x, index_y]
                    dG2_result[ip,index_x - index_list_1[0],index_x,:] -= dev_G2 * vec_matrix[index_x,index_y] / norm_matrix[index_x, index_y]
        return(dG2_result)
    @staticmethod
    def Calculate_dG4_features(norm_matrix, vec_matrix, nMol, atom_type_1, atom_type_2, atom_type_3, features_number, params, rc):
        index_list_1 = [0, nMol]
        if (atom_type_1 == 1):
            index_list_1 = [nMol, 3 * nMol]
        elif(atom_type_1 == 2):
            index_list_1 = [3 * nMol, 3 * nMol + 4 * nMol ]

        index_list_2 = [0, nMol]
        if (atom_type_2 == 1):
            index_list_2 = [nMol, 3 * nMol]
        elif(atom_type_2 == 2):
            index_list_2 = [3 * nMol, 3 * nMol + 4 * nMol ]

        index_list_3 = [0, nMol]
        if (atom_type_3 == 1):
            index_list_3 = [nMol, 3 * nMol]
        elif(atom_type_3 == 2):
            index_list_3 = [3 * nMol, 3 * nMol + 4 * nMol ]

        dG4_result = np.zeros((features_number,index_list_1[1] - index_list_1[0],7 * nMol, 3), dtype=np.float32)

        for index_x in range(index_list_1[0],index_list_1[1]):
            for index_y in range(index_list_2[0],index_list_2[1]):
                if (index_x == index_y or norm_matrix[index_x,index_y] >= rc): continue
                for index_z in range(index_list_3[0],index_list_3[1]):
                    if (index_z == index_x or index_z == index_y,norm_matrix[index_x,index_z] >= rc or norm_matrix[index_z,index_y] >= rc): continue
                    cos_alpha = (norm_matrix[index_x,index_y]**2 + norm_matrix[index_x,index_z]**2 - norm_matrix[index_y,index_z]**2) / (2 * norm_matrix[index_x,index_y] * norm_matrix[index_x,index_z])

                    for ip in range(features_number):
                        dG4 = Symmetry_func.G4_dev_func(norm_matrix[index_x, index_y],
                                                        norm_matrix[index_y, index_z],
                                                        norm_matrix[index_z, index_x],
                                                        cos_alpha,
                                                        params[ip,3],params[ip,1],params[ip,2],
                                                        rc)
                        dG4_ij_ij = dG4[0]
                        dG4_ij_ik = dG4[1]
                        dG4_jk_jk = dG4[2]

                        dG4_result[ip,index_x - index_list_1[0],index_y,:] += dG4_ij_ij * ( vec_matrix[index_x,index_y] / norm_matrix[index_x,index_y] )
                        dG4_result[ip,index_x - index_list_1[0],index_y,:] += dG4_jk_jk * ( vec_matrix[index_z,index_y] / norm_matrix[index_y,index_z] )

                        dG4_result[ip,index_x - index_list_1[0],index_z,:] += dG4_ij_ik * ( vec_matrix[index_x,index_z] / norm_matrix[index_x,index_z] )
                        dG4_result[ip,index_x - index_list_1[0],index_z,:] += dG4_jk_jk * ( vec_matrix[index_y,index_z] / norm_matrix[index_z,index_y] )

                        dG4_result[ip,index_x - index_list_1[0],index_x,:] -= dG4_ij_ij * ( vec_matrix[index_x,index_y] / norm_matrix[index_x,index_y] )
                        dG4_result[ip,index_x - index_list_1[0],index_x,:] -= dG4_ij_ik * ( vec_matrix[index_x,index_z] / norm_matrix[index_x,index_z])

        return(dG4_result)
    @staticmethod
    def Calculate_G2_features_multthread(norm_matrix, nMol, atom_type_1, atom_type_2, features_number, params, rc):
        # Set index list 1 & 2
        """
        Already Checked with Ang Gao's benchmark
        """
        index_list_1 = [0, nMol]
        if (atom_type_1 == 1):
            index_list_1 = [nMol, 3 * nMol]
        elif(atom_type_1 == 2):
            index_list_1 = [3 * nMol, 3 * nMol + 4 * nMol ]

        index_list_2 = [0, nMol]
        if (atom_type_2 == 1):
            index_list_2 = [nMol, 3 * nMol]
        elif(atom_type_2 == 2):
            index_list_2 = [3 * nMol, 3 * nMol + 4 * nMol ]

        G2_result = np.zeros((features_number,index_list_1[1] - index_list_1[0]), dtype=np.float32)

        pool = multiprocessing.Pool(processes = 10)
        future_list = []
        for index_x in range(index_list_1[0],index_list_1[1]):
            for index_y in range(index_list_2[0],index_list_2[1]):
                if (index_x == index_y or norm_matrix[index_x,index_y] >= rc): continue
                for ip in range(features_number):
                    res=pool.apply_async(Symmetry_func.G2_func,args=(norm_matrix[index_x, index_y],params[ip,1],params[ip,0],rc,))
                    future_list.append(res)
        pool.close()
        pool.join()
        return(G2_result)

# t
def Calculate_features(Hxyz,Oxyz,Wxyz,Box,nMol,scale_factor):
    G2WW_params = np.loadtxt("/DATA/users/yanghe/projects/Wannier_center_pred/Code/parameters/G2_parameters_OO.txt")
    G2WH_params = np.loadtxt("/DATA/users/yanghe/projects/Wannier_center_pred/Code/parameters/G2_parameters_OH.txt")
    G2WO_params = np.loadtxt("/DATA/users/yanghe/projects/Wannier_center_pred/Code/parameters/G2_parameters_HO.txt")

    G4WWW_params = np.loadtxt("/DATA/users/yanghe/projects/Wannier_center_pred/Code/parameters/G4_parameters_OOO.txt")
    G4WHH_params = np.loadtxt("/DATA/users/yanghe/projects/Wannier_center_pred/Code/parameters/G4_parameters_OHH.txt")
    G4WOO_params = np.loadtxt("/DATA/users/yanghe/projects/Wannier_center_pred/Code/parameters/G4_parameters_HOO.txt")
    
    norm_matrix, vec_matrix = Calculate_func.Pairwise_matrix(Oxyz, Hxyz, Wxyz, Box)

    G2WW_features = Calculate_func.Calculate_G2_features(norm_matrix,nMol,2,2,8,G2WW_params,12)
    G2WH_features = Calculate_func.Calculate_G2_features(norm_matrix,nMol,2,1,8,G2WH_params,12)
    G2WO_features = Calculate_func.Calculate_G2_features(norm_matrix,nMol,2,0,8,G2WO_params,12)

    G4WWW_features = Calculate_func.Calculate_G4_features(norm_matrix,nMol,2,2,2,4,G4WWW_params,12)
    G4WHH_features = Calculate_func.Calculate_G4_features(norm_matrix,nMol,2,1,1,6,G4WHH_params,12)
    G4WOO_features = Calculate_func.Calculate_G4_features(norm_matrix,nMol,2,0,0,4,G4WOO_params,12)

    dG2WW_features = Calculate_func.Calculate_dG2_features(norm_matrix,vec_matrix,nMol,2,2,8,G2WW_params,12)
    dG2WH_features = Calculate_func.Calculate_dG2_features(norm_matrix,vec_matrix,nMol,2,1,8,G2WH_params,12)
    dG2WO_features = Calculate_func.Calculate_dG2_features(norm_matrix,vec_matrix,nMol,2,0,8,G2WO_params,12)

    dG4WWW_features = Calculate_func.Calculate_dG4_features(norm_matrix,vec_matrix,nMol,2,2,2,4,G4WWW_params,12)
    dG4WHH_features = Calculate_func.Calculate_dG4_features(norm_matrix,vec_matrix,nMol,2,1,1,6,G4WHH_params,12)
    dG4WOO_features = Calculate_func.Calculate_dG4_features(norm_matrix,vec_matrix,nMol,2,0,0,4,G4WOO_params,12)

    features_all = np.concatenate((G2WW_features, G2WH_features, G2WO_features,G4WWW_features, G4WOO_features, G4WHH_features), axis=0)  # stack along the nfeatures axis
    dfeatures_all = np.concatenate((dG2WW_features, dG2WH_features, dG2WO_features,dG4WWW_features, dG4WOO_features, dG4WHH_features), axis=0)  # stack along the nfeatures axis
    
    features_all = np.transpose(features_all,axes = (1,0))
    dfeatures_all = np.transpose(dfeatures_all[:,:,192:,:],axes = (1,3,2,0))

    features_all = (features_all - scale_factor[:,0]) / (scale_factor[:,2] - scale_factor[:,1])
    dfeatures_all = dfeatures_all / (scale_factor[:,2] - scale_factor[:,1])

    return(features_all,dfeatures_all)
    
if __name__ == "__main__":
    nMol = 64

    Hxyz = np.loadtxt("/DATA/users/yanghe/projects/SCFNN/Data/D0/2/Hxyz.txt")
    Oxyz = np.loadtxt("/DATA/users/yanghe/projects/SCFNN/Data/D0/2/Oxyz.txt")
    Wxyz = np.loadtxt("/DATA/users/yanghe/projects/SCFNN/Data/D0/2/wxyz.txt")
    Box = np.loadtxt("/DATA/users/yanghe/projects/SCFNN/Data/D0/2/box.txt")
    scale_factor = np.loadtxt("/DATA/users/yanghe/projects/Wannier_center_pred/Data/Train_data/xW_scalefactor.txt")
    Box = np.array([Box,Box,Box])

    # Calculate_features(Hxyz,Oxyz,Wxyz,Box,nMol,scale_factor)

    norm_matrix, vec_matrix = Calculate_func.Pairwise_matrix(Oxyz, Hxyz, Wxyz, Box)
    G2WW_params = np.loadtxt("/DATA/users/yanghe/projects/Wannier_center_pred/Code/parameters/G2_parameters_OO.txt")

    begin = time.time()
    G2WW_features = Calculate_func.Calculate_G2_features(norm_matrix,nMol,2,2,8,G2WW_params,12)
    end_1 = time.time()
    print(end_1 - begin)

    G2WW_features = Calculate_func.Calculate_G2_features_multthread(norm_matrix,nMol,2,2,8,G2WW_params,12)
    print(time.time() - end_1)