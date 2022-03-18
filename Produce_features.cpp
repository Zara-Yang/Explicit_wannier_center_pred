#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <assert.h>
#include "omp.h"

/*
    Create features for Wannier center BPNN
    
    This code is refer to Ang Gao's code
        
        Webset : https://github.com/andy90/SCFNN

    Already check with Gao's benchmark (2022/3/14)
*/


#define natoms 192          // the total number of nucleus
#define noxygen 64          // the number of oxygen
#define nhydrogen 128       // the number of hydrogen
#define nwannier 256        // wannier center number
#define ntotal 448          // wannier center number + atom number

using namespace std;

double Trunc_func(double input,double rc){
    /*
        set a smooth truncated function for every r_ij in symmetry function 
    */
    double result = 0;
    if (input < rc){
        result = pow(tanh(1. - input/rc),3);
    }
    return(result);
}

double Trunc_func_dev(double r_ij, double rc){
    /*
        Derivative of truncated function
    */
    double result = 0;
    if (r_ij < rc){
        result = -3 * pow(tanh(1 - r_ij/rc), 2) / pow(cosh(1 - r_ij / rc), 2) / rc;
    }
    return(result);
}

double G2_func(double r_ij, double yeta, double rs, double rc){
    /*
        G2 Symmetry function for any r_ij
    */
    double result = exp(-yeta * pow((r_ij - rs),2)) * Trunc_func(r_ij, rc);
    return(result);
}

double G4_func(double r_ij, double r_ki, double r_jk, double cos_alpha, double zeta, double yeta , double lambda, double rc){
    /*
        G4 symmetry function for any r_ij,r_jk,r_ki
    */
    double result = exp(- yeta * (r_ij * r_ij + r_ki * r_ki + r_jk * r_jk)) * Trunc_func(r_ij,rc) * Trunc_func(r_jk,rc) * Trunc_func(r_ki,rc) * pow((1 + lambda * cos_alpha),zeta);
    result = result * pow(2,(1 - zeta));
    return(result);
}

double G2_func_dev(double r_ij, double yeta, double rs, double rc){
    /*
        Derivative of G2 symmetry function ( except vector part )
        
        Math : 
            \partial G2(r_ij)        -yeta(r_ij - rs)^2                 -yeta(r_ij - rs)^2
            -------------------  =  e                    fc_dev(r_ij) + e                   (-2 yeta (rij - rs)) * fc(r_ij)
            \partial r_ij
    */
    // double result = exp(-yeta * pow((r_ij - rs), 2)) * Trunc_func_dev(r_ij, rc) + exp(-yeta * pow((r_ij - rs),2)) * (-2 * yeta * (r_ij - rs)) * Trunc_func(r_ij, rc);
    double result = -2 * yeta * (r_ij - rs) * Trunc_func(r_ij, rc) * exp( - yeta * pow((r_ij - rs),2));
    result += exp(- yeta * pow((r_ij - rs),2)) * Trunc_func_dev(r_ij, rc); 
    
    return(result);
}

vector<double> G4_func_dev(double r_ij, double r_ki, double r_jk, double cos_alpha, double zeta, double yeta, double lambda, double rc){
    /*
        Derivative of G4 symmetry function ( except vector part )

        Math is too complete OMG.
    */
    double fc_rij = Trunc_func(r_ij, rc);
    double fc_rjk = Trunc_func(r_jk, rc);
    double fc_rki = Trunc_func(r_ki, rc);

    double fc_rij_dev = Trunc_func_dev(r_ij, rc);
    double fc_rjk_dev = Trunc_func_dev(r_jk, rc);
    double fc_rki_dev = Trunc_func_dev(r_ki, rc);

    double common_value = pow(2, 1 - zeta) * exp(-yeta * (r_ij * r_ij + r_jk * r_jk + r_ki * r_ki)) * pow((1 + lambda * cos_alpha), (zeta-1)) ;

    double rij_dev = zeta * lambda * (1. / r_ki - (r_ij * r_ij + r_ki * r_ki - r_jk * r_jk) / (2 * r_ij * r_ij * r_ki)) * fc_rij * fc_rjk * fc_rki;
    rij_dev += - 2 * r_ij * yeta * (1 + lambda * cos_alpha) * fc_rij * fc_rki * fc_rjk;
    rij_dev += fc_rij_dev * (1 + lambda * cos_alpha) * fc_rki * fc_rjk;

    double rjk_dev = zeta * lambda * (1. /r_ij - (r_ij * r_ij + r_ki * r_ki - r_jk * r_jk) / (2 * r_ki * r_ki * r_ij)) * fc_rij * fc_rjk * fc_rki;
    rjk_dev += - 2 * r_ki * yeta * (1 + lambda * cos_alpha) * fc_rij * fc_rjk * fc_rki;
    rjk_dev += fc_rki_dev * (1 + lambda * cos_alpha) * fc_rij * fc_rjk;
    
    double rki_dev = zeta * lambda * (- r_jk /(r_ij *r_ki)) * fc_rij * fc_rjk * fc_rki;
    rki_dev += - 2 * r_jk * yeta * (1 + lambda * cos_alpha) * fc_rij * fc_rjk * fc_rki;
    rki_dev += fc_rjk_dev * (1 + lambda * cos_alpha) * fc_rij * fc_rki;

    rij_dev *= common_value;
    rjk_dev *= common_value;
    rki_dev *= common_value;

    vector<double> r_dev(3);
    r_dev[0] = rij_dev;
    r_dev[1] = rjk_dev;
    r_dev[2] = rki_dev;
    return r_dev;
}

void Loading_coord_data(vector<vector<double>>& coord_matrix, string data_path, int dim_1, int dim_2){
    /*
        Loading coord from coord file to matrix
    */
    ifstream fxyz(data_path);
    for (int i = 0 ; i < dim_1 ; i++)
    {
        for (int j = 0; j < dim_2 ; j++)
        {
            fxyz >> coord_matrix[i][j];
        }
    }
}

void Loading_box_data(vector<double>& box_vector, string data_path, bool cube = false){
    ifstream fbox(data_path);
    if (! cube){
        for (int i = 0 ; i < 3 ; i++){
            fbox >> box_vector[i];
        }
    }
    else{
        double length = 0;
        fbox >> length;
        for (int i = 0 ; i < 3 ; i++){
            box_vector[i] = length;
        }
    }
}

void Merge_coord(vector<vector<double>>& Oxygen_coord, vector<vector<double>>& Hydrogen_coord, vector<vector<double>>& Wannier_coord, vector<vector<double>>& total_coord){
    for (int index = 0; index < noxygen ; index ++){
        for (int dim = 0 ; dim < 3 ; dim++){
            total_coord[index][dim] = Oxygen_coord[index][dim];
        }
    }

    for (int index = 0; index < nhydrogen ; index ++){
        for (int dim = 0 ; dim < 3 ; dim++){
            total_coord[noxygen + index][dim] = Hydrogen_coord[index][dim];
        }
    }

    for (int index = 0; index < nwannier ; index ++){
        for (int dim = 0 ; dim < 3 ; dim++){
            total_coord[natoms + index][dim] = Wannier_coord[index][dim];
        }
    }
}

void Calculate_distance_matrix(vector<vector<double>>& coord_matrix,vector<double>& box_range,  vector<vector<double>>& norm_matrix, vector<vector<vector<double>>>& vec_matrix){
    float norm_buffer = 0;
    vector<double> vec_buffer(3);
    for (int i = 0; i < ntotal ; i++){
        for (int j = 0 ; j < ntotal ; j++){
            double disx = coord_matrix[j][0] - coord_matrix[i][0] - round((coord_matrix[j][0] - coord_matrix[i][0]) / box_range[0]) * box_range[0];
            double disy = coord_matrix[j][1] - coord_matrix[i][1] - round((coord_matrix[j][1] - coord_matrix[i][1]) / box_range[1]) * box_range[1];
            double disz = coord_matrix[j][2] - coord_matrix[i][2] - round((coord_matrix[j][2] - coord_matrix[i][2]) / box_range[2]) * box_range[2];

            double dis = sqrt(disx * disx + disy * disy + disz * disz);
            norm_matrix[i][j] = dis;
            vec_matrix[i][j][0] = disx;
            vec_matrix[i][j][1] = disy;
            vec_matrix[i][j][2] = disz;
        }
    }
}

void Calculate_G2_features(vector<vector<double> > &features, vector<vector<double> > &params, vector<vector<double> > &distance_matrix, int mol_id_1, int mol_id_2, int feature_dimension, string save_path, double rc){
    int rangei[2] = {0, noxygen};
    if (mol_id_1 == 1)
    {
        rangei[0] = noxygen;
        rangei[1] = natoms;
    }
    else if (mol_id_1 == 2)
    {
        rangei[0] = natoms;
        rangei[1] = ntotal;
    }
   
    int rangej[2] = {0, noxygen};
    if (mol_id_2 == 1)
    {
        rangej[0] = noxygen;
        rangej[1] = natoms;
    }
    else if (mol_id_2 == 2)
    {
        rangej[0] = natoms;
        rangej[1] = ntotal;
    }

    #pragma omp parallel for 
    for (int i = rangei[0]; i < rangei[1]; i++)
    {
        for (int j = rangej[0]; j < rangej[1]; j++)
        {
            if ( i == j ){continue;}
            for (int ip = 0; ip < feature_dimension; ip++)
            {
                features[ip][i - rangei[0]] += G2_func(distance_matrix[i][j], params[ip][1], params[ip][0], rc);
            }
        }
    }

    ofstream fp(save_path);
    for (int ip = 0; ip < feature_dimension; ip++)
    {
        for (int i = rangei[0]; i < rangei[1]; i++)
        {
            fp << features[ip][i - rangei[0]] << "\t";
        }
        fp << "\n";
    }
    fp.close();
}

void Calculate_G4_features(vector<vector<double> > &features, vector<vector<double> > &params, vector<vector<double> > &distance_matrix, int mol_id_1, int mol_id_2, int mol_id_3, int feature_dimension, string save_path, double rc){ 
    int rangei[2] = {0, noxygen};
    if (mol_id_1 == 1)
    {
        rangei[0] = noxygen;
        rangei[1] = natoms;
    }
    else if (mol_id_1 == 2)
    {
        rangei[0] = natoms;
        rangei[1] = ntotal;
    }
    
    int rangej[2] = {0, noxygen};
    if (mol_id_2 == 1)
    {
        rangej[0] = noxygen;
        rangej[1] = natoms;
    }
    else if (mol_id_2 == 2)
    {
        rangej[0] = natoms;
        rangej[1] = ntotal;
    }

    int rangek[2] = {0, noxygen};
    if (mol_id_3 == 1)
    {
        rangek[0] = noxygen;
        rangek[1] = natoms;
    }
    else if (mol_id_3 == 2)
    {
        rangek[0] = natoms;
        rangek[1] = ntotal;
    }

    #pragma omp parallel for 
    for (int i = rangei[0]; i < rangei[1]; i++)
    {
        for (int j = rangej[0]; j < rangej[1]; j++)
        {
            if (j != i)
            {
                for (int k = rangek[0]; k < rangek[1]; k++)
                {
                    if ((k != j ) && ( k != i ))
                    {
                        double cosijk = (distance_matrix[i][j] * distance_matrix[i][j] + distance_matrix[i][k] * distance_matrix[i][k] - distance_matrix[j][k] * distance_matrix[j][k]) / (2 * distance_matrix[i][j] * distance_matrix[i][k]);
                        for (int ip = 0; ip < feature_dimension; ip++)
                        {
                            features[ip][i - rangei[0]] += G4_func(distance_matrix[i][j], distance_matrix[i][k], distance_matrix[j][k], cosijk, params[ip][3], params[ip][1], params[ip][2], rc);
                        }
                    }
                }
            }
        }
    }

    ofstream fp(save_path);
    for (int ip = 0; ip < feature_dimension; ip++)
    {
        for (int i = rangei[0]; i < rangei[1]; i++)
        {
            fp << features[ip][i - rangei[0]] << "\t";
        }
        fp << "\n";
    }
    fp.close();
}

void Calculate_dev_G2_features(vector<vector<vector<vector<double>>>> &features, vector<vector<double>> &params, vector<vector<double> > &distance_matrix, vector<vector<vector<double> > > &vector_matrix, int mol_id_1, int mol_id_2, int feature_dimension, string save_path, double rc){
    int rangei[2] = {0, noxygen};
    if (mol_id_1 == 1)
    {
        rangei[0] = noxygen;
        rangei[1] = natoms;
    }
    else if (mol_id_1 == 2)
    {
        rangei[0] = natoms;
        rangei[1] = ntotal;
    }
   
    int rangej[2] = {0, noxygen};
    if (mol_id_2 == 1)
    {
        rangej[0] = noxygen;
        rangej[1] = natoms;
    }
    else if (mol_id_2 == 2)
    {
        rangej[0] = natoms;
        rangej[1] = ntotal;
    }
    
    #pragma omp parallel for 
    for ( int i = rangei[0] ; i < rangei[1] ; i++){
        for ( int j = rangej[0] ; j < rangej[1] ; j++ ){
            if( i == j) { continue; }
            for (int ip = 0 ; ip < feature_dimension ; ip++){
                double dev_G2 = G2_func_dev(distance_matrix[i][j],params[ip][1],params[ip][0],rc); 
                for (int dim = 0 ; dim < 3 ; dim ++){
                    features[ip][i - rangei[0]][j][dim] += dev_G2 * vector_matrix[i][j][dim] / distance_matrix[i][j];
                    features[ip][i - rangei[0]][i][dim] -= dev_G2 * vector_matrix[i][j][dim] / distance_matrix[i][j];
                }
            }
            // exit(-1);
        }
    }
    ofstream fp(save_path);
    for (int ip = 0; ip < feature_dimension; ip++)
    {
        for (int i = rangei[0]; i < rangei[1]; i++)
        {
            for (int j = 0; j < ntotal ; j++)
            {
                for (int ix = 0; ix < 3; ix++)
                {
                    fp << features[ip][i - rangei[0]][j][ix] << " ";
                }
            }
        }
    }
    fp << endl;
    fp.close();
}

void Calculate_dev_G4_features(vector<vector<vector<vector<double>>>> &features, vector<vector<double>> &params, vector<vector<double> > &distance_matrix, vector<vector<vector<double> > > &vector_matrix, int mol_id_1, int mol_id_2, int mol_id_3, int feature_dimension, string save_path, double rc){
    int rangei[2] = {0, noxygen};
    if (mol_id_1 == 1)
    {
        rangei[0] = noxygen;
        rangei[1] = natoms;
    }
    else if (mol_id_1 == 2)
    {
        rangei[0] = natoms;
        rangei[1] = ntotal;
    }
    
    int rangej[2] = {0, noxygen};
    if (mol_id_2 == 1)
    {
        rangej[0] = noxygen;
        rangej[1] = natoms;
    }
    else if (mol_id_2 == 2)
    {
        rangej[0] = natoms;
        rangej[1] = ntotal;
    }

    int rangek[2] = {0, noxygen};
    if (mol_id_3 == 1)
    {
        rangek[0] = noxygen;
        rangek[1] = natoms;
    }
    else if (mol_id_3 == 2)
    {
        rangek[0] = natoms;
        rangek[1] = ntotal;
    }
    
    #pragma omp parallel for 
    for (int i = rangei[0]; i < rangei[1]; i++)
    {
        for (int j = rangej[0]; j < rangej[1]; j++)
        {
            if (j == i){continue;}
            for (int k = rangek[0]; k < rangek[1]; k++)
            {
                if ((k == j) | (k == i)){continue;}
                double cosijk = (distance_matrix[i][j] * distance_matrix[i][j] + distance_matrix[i][k] * distance_matrix[i][k] - distance_matrix[j][k] * distance_matrix[j][k]) / (2 * distance_matrix[i][j] * distance_matrix[i][k]);
                
                for (int ip = 0; ip < feature_dimension; ip++)
                {
                    vector<double> Gs = G4_func_dev(distance_matrix[i][j], distance_matrix[i][k], distance_matrix[j][k], cosijk, params[ip][3], params[ip][1], params[ip][2], rc);
                    double dG4_ij_ij = Gs[0];
                    double dG4_ij_ik = Gs[1];
                    double dG4_jk_jk = Gs[2];
                    for (int ix = 0; ix < 3; ix++)
                    {
                        features[ip][i - rangei[0]][j][ix] += dG4_ij_ij * vector_matrix[i][j][ix] / distance_matrix[i][j];
                        features[ip][i - rangei[0]][j][ix] += dG4_jk_jk * vector_matrix[k][j][ix] / distance_matrix[j][k];

                        features[ip][i - rangei[0]][k][ix] += dG4_ij_ik * vector_matrix[i][k][ix] / distance_matrix[i][k];
                        features[ip][i - rangei[0]][k][ix] += dG4_jk_jk * vector_matrix[j][k][ix] / distance_matrix[k][j];
                        
                        features[ip][i - rangei[0]][i][ix] -= dG4_ij_ij * vector_matrix[i][j][ix] / distance_matrix[i][j];
                        features[ip][i - rangei[0]][i][ix] -= dG4_ij_ik * vector_matrix[i][k][ix] / distance_matrix[i][k];
                    }
                }

            }
        }
    }
    ofstream fp(save_path);
    for (int ip = 0; ip < feature_dimension; ip++)
    {
        for (int i = rangei[0]; i < rangei[1]; i++)
        {
            for (int j = 0; j < ntotal; j++)
            {
                for (int ix = 0; ix < 3; ix++)
                {
                    fp << features[ip][i - rangei[0]][j][ix] << " ";
                }
            }
        }
    }
    fp << "\n";
    fp.close();
}

int main(){
    vector<vector<double> > Wxyz(natoms * 4, vector<double>(3));
    vector<vector<double> > Oxyz(noxygen , vector<double>(3));
    vector<vector<double> > Hxyz(nhydrogen, vector<double>(3));
    vector<vector<double> > Coord(natoms + nwannier, vector<double>(3));
    vector<double> Box(3);
    // Loading params
    vector<vector<double>> G2WH_params(8,vector<double>(4));
    vector<vector<double>> G2WW_params(8,vector<double>(4));
    vector<vector<double>> G2WO_params(8,vector<double>(4));

    vector<vector<double>> G4WWW_params(4,vector<double>(4));
    vector<vector<double>> G4WOO_params(4,vector<double>(4));
    vector<vector<double>> G4WHH_params(6,vector<double>(4));
    // Calculate distance matrix 
    vector<vector<double>> norm_matrix(ntotal,vector<double>(ntotal));
    vector<vector<vector<double>>> vec_matrix(ntotal,vector<vector<double>>(ntotal,vector<double>(3)));
    // Features data matrix
    vector<vector<double>> G2WH_features(8, vector<double>(nwannier));
    vector<vector<double>> G2WO_features(8, vector<double>(nwannier));
    vector<vector<double>> G2WW_features(8, vector<double>(nwannier));

    vector<vector<double>> G4WWW_features(4, vector<double>(nwannier));
    vector<vector<double>> G4WOO_features(4, vector<double>(nwannier));
    vector<vector<double>> G4WHH_features(6, vector<double>(nwannier));

    vector<vector<vector<vector<double> > > > features_dG2WH(8, vector<vector<vector<double> > >(nwannier, vector<vector<double> >(ntotal, vector<double>(3))));
    vector<vector<vector<vector<double> > > > features_dG2WW(8, vector<vector<vector<double> > >(nwannier, vector<vector<double> >(ntotal, vector<double>(3))));
    vector<vector<vector<vector<double> > > > features_dG2WO(8, vector<vector<vector<double> > >(nwannier, vector<vector<double> >(ntotal, vector<double>(3))));

    vector<vector<vector<vector<double> > > > features_dG4WWW(4, vector<vector<vector<double> > >(nwannier, vector<vector<double> >(ntotal, vector<double>(3))));
    vector<vector<vector<vector<double> > > > features_dG4WOO(4, vector<vector<vector<double> > >(nwannier, vector<vector<double> >(ntotal, vector<double>(3))));
    vector<vector<vector<vector<double> > > > features_dG4WHH(6, vector<vector<vector<double> > >(nwannier, vector<vector<double> >(ntotal, vector<double>(3))));
    // Loading basic data
    Loading_coord_data(Wxyz, "./Wxyz.txt", noxygen * 4, 3);
    Loading_coord_data(Oxyz, "./Oxyz.txt", noxygen , 3);
    Loading_coord_data(Hxyz, "./Hxyz.txt", nhydrogen, 3);

    Loading_box_data(Box,"./Box.txt",true);
    // loading parameters
    Loading_coord_data(G2WH_params, "/DATA/users/yanghe/projects/Wannier_center_pred/Code/parameters/G2_parameters_OH.txt", 8 , 4);
    Loading_coord_data(G2WW_params, "/DATA/users/yanghe/projects/Wannier_center_pred/Code/parameters/G2_parameters_HH.txt", 8 , 4);
    Loading_coord_data(G2WO_params, "/DATA/users/yanghe/projects/Wannier_center_pred/Code/parameters/G2_parameters_HO.txt", 8 , 4);

    Loading_coord_data(G4WWW_params, "/DATA/users/yanghe/projects/Wannier_center_pred/Code/parameters/G4_parameters_OOO.txt", 4 , 4);
    Loading_coord_data(G4WOO_params, "/DATA/users/yanghe/projects/Wannier_center_pred/Code/parameters/G4_parameters_HOO.txt", 4 , 4);
    Loading_coord_data(G4WHH_params, "/DATA/users/yanghe/projects/Wannier_center_pred/Code/parameters/G4_parameters_OHH.txt", 6 , 4);
    // merge three coode matrix to one matrix
    Merge_coord(Oxyz, Hxyz, Wxyz, Coord);
    // Calculate distance matrix and vector matrix
    Calculate_distance_matrix(Coord,Box,norm_matrix,vec_matrix);
    // Calculate features
    Calculate_G2_features(G2WW_features, G2WW_params, norm_matrix,2,2,8,"./features_G2WW.txt",12);
    Calculate_G2_features(G2WH_features, G2WH_params, norm_matrix,2,1,8,"./features_G2WH.txt",12);
    Calculate_G2_features(G2WO_features, G2WO_params, norm_matrix,2,0,8,"./features_G2WO.txt",12);

    Calculate_G4_features(G4WWW_features, G4WWW_params, norm_matrix,2,2,2,4,"./features_G4WWW.txt",12);
    Calculate_G4_features(G4WOO_features, G4WOO_params, norm_matrix,2,0,0,4,"./features_G4WOO.txt",12);
    Calculate_G4_features(G4WHH_features, G4WHH_params, norm_matrix,2,1,1,6,"./features_G4WHH.txt",12);

    // Calculate dev features
    Calculate_dev_G2_features(features_dG2WH, G2WH_params, norm_matrix, vec_matrix, 2, 1, 8, "./features_dG2WH.txt", 12);
    Calculate_dev_G2_features(features_dG2WO, G2WO_params, norm_matrix, vec_matrix, 2, 0, 8, "./features_dG2WO.txt", 12);
    Calculate_dev_G2_features(features_dG2WW, G2WW_params, norm_matrix, vec_matrix, 2, 2, 8, "./features_dG2WW.txt", 12);

    Calculate_dev_G4_features(features_dG4WWW, G4WWW_params, norm_matrix, vec_matrix, 2, 2, 2, 4, "./features_dG4WWW.txt", 12);
    Calculate_dev_G4_features(features_dG4WOO, G4WOO_params, norm_matrix, vec_matrix, 2, 0, 0, 4, "./features_dG4WOO.txt", 12);
    Calculate_dev_G4_features(features_dG4WHH, G4WHH_params, norm_matrix, vec_matrix, 2, 1, 1, 6, "./features_dG4WHH.txt", 12);
    
    // ===========================================================
}

