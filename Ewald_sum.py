from cmath import cos
import numpy as np
import math
import os

def Ewald_sum(Oxygen_coord,Hydrogen_coord,Wannier_coord,box_length_all,charge_H,charge_O,charge_W,sigma,nkmax = 5):
    """
    Calculate electric field in position of every atom and wannier center, without external field

    math :      

        E(r_i) = \Delta_{r_i} U

                                                -4k^2/alpha^2
                  2 * pi                       e             ikr_i                    ikr_j                      ikr_j                      ikr_j   
        E(r_i) = --------- * \sum_{k!=0} *  -------- * ik * e      * { \sum_{j}^{nO} e     * q_j + \sum_{j}^{nH} e     * q_j + \sum_{j}^{nW} e    * q_j } 
                    V                          k^2
    input:
            name                   description                    dimension
        ------------------------------------------------------------------------------
        Oxygen_coord       Coord of Oxygen atom unrotate     [nOxygen,3,nfolder,nconfig]
        Hydrogen_coord     Coord of Hydrogen atom unrotate   [nHydrogen,3,nfolder,nconfig]
        Wannier_coord      Coord of Wannier atom unrotate    [nWannier,3,nfolder,nconfig]
        box_length_all     box range for each system         [3,nconfig]
        charge_H/O/W       charge of atom O/H/W              float
        sigma                     ?                          float
        nkmax              max index of k                    float
    
    output:
            name                   description                    dimension
        ------------------------------------------------------------------------------
        EO_sum          electric field in Oxygen position       [nOxygen,3]
        EH_sum          electric field in Hydrogen position     [nHydrogen,3]
        Ew_sum          electric field in Wannier position      [nWannier,3]
        """

    sigmaE = sigma / np.sqrt(2)

    # Electric field     
    EO_all = ()
    EH_all = ()
    Ew_all = ()

    nOxygen,_,nfoloder,nconfig = Oxygen_coord.shape
    nHydrogen,_,_,_ = Hydrogen_coord.shape
    nWannier,_,_,_ = Wannier_coord.shape


    for ifolder in range(nfoloder):
        for iconfig in range(nconfig):
            boxlength = box_length_all[:, iconfig]
            Oxyz = Oxygen_coord[:, :, ifolder, iconfig]
            Hxyz = Hydrogen_coord[:, :, ifolder, iconfig]
            wcxyz = Wannier_coord[:, :, ifolder, iconfig]

            k_unit = 2 * math.pi / boxlength
            k_coord = np.zeros(((2 * nkmax - 1)**3-1, 3), dtype=np.float32)

            ik = 0
            for nx in range(- nkmax + 1, nkmax):
                for ny in range(- nkmax + 1, nkmax):
                    for nz in range(- nkmax + 1, nkmax):
                        if ((nx != 0) | (ny != 0) | (nz != 0)):
                            k_coord[ik, 0] = nx * k_unit[0]
                            k_coord[ik, 1] = ny * k_unit[1]
                            k_coord[ik, 2] = nz * k_unit[2]
                            ik = ik + 1
            Oxyz_expand = np.expand_dims(Oxyz,axis = 1)         # [nOxygen,1,3]
            Hxyz_expand = np.expand_dims(Hxyz,axis = 1)         # [nHydrogen,1,3]
            Wxyz_expand = np.expand_dims(wcxyz,axis = 1)        # [nWannier,1,3]
            kxyz_expand = np.expand_dims(k_coord,axis = 0)      # [1,728,3]

            Sk = 0
            Sk += np.sum(charge_O * np.exp(1j * np.sum(kxyz_expand * Oxyz_expand,axis=-1)),axis = 0)
            Sk += np.sum(charge_H * np.exp(1j * np.sum(Hxyz_expand * kxyz_expand,axis = -1)),axis = 0)
            Sk += np.sum(charge_W * np.exp(1j * np.sum(Wxyz_expand * kxyz_expand,axis = -1)),axis = 0)      # [nk]

            coeff = 2 * math.pi / (boxlength[0] * boxlength[1] * boxlength[2])

            dSkO = 1j * kxyz_expand * np.expand_dims(np.exp(1j * np.sum(kxyz_expand * Oxyz_expand,axis = -1)),axis = 2) # [64,728,3]
            dSkH = 1j * kxyz_expand * np.expand_dims(np.exp(1j * np.sum(kxyz_expand * Hxyz_expand,axis = -1)),axis = 2) # [128,728,3]
            dSkW = 1j * kxyz_expand * np.expand_dims(np.exp(1j * np.sum(kxyz_expand * Wxyz_expand,axis = -1)),axis = 2) # [256,728,3]     

            Sk_expand = np.expand_dims(Sk,axis = 1)             # [728,1]

            knorm = np.linalg.norm(k_coord,axis = -1)
            knorm_expand = np.expand_dims(knorm,axis = -1)    # [728,1]

            EO = - coeff * np.sum(np.exp(- (sigmaE * knorm_expand) ** 2 / 2) / knorm_expand ** 2 * 2 * (Sk_expand.conjugate() * dSkO).real, axis=1)
            EH = - coeff * np.sum(np.exp(- (sigmaE * knorm_expand) ** 2 / 2) / knorm_expand ** 2 * 2 * (Sk_expand.conjugate() * dSkH).real, axis=1)
            Ew = - coeff * np.sum(np.exp(- (sigmaE * knorm_expand) ** 2 / 2) / knorm_expand ** 2 * 2 * (Sk_expand.conjugate() * dSkW).real, axis=1)

            EO_all += (EO,)
            EH_all += (EH,)
            Ew_all += (Ew,)
    
    EO_all_stack = np.stack(EO_all, axis=2)
    EH_all_stack = np.stack(EH_all, axis=2)
    Ew_all_stack = np.stack(Ew_all, axis=2)

    EO_all_reshape = EO_all_stack.reshape((nOxygen, 3, nfoloder, nconfig))
    EH_all_reshape = EH_all_stack.reshape((nHydrogen, 3, nfoloder, nconfig))
    Ew_all_reshape = Ew_all_stack.reshape((nWannier, 3, nfoloder, nconfig))
    # a z-direction external field 

    # Eexternal = np.array([0, 0, 0, 0, 0, 0, 0, 0.1/51.4, 0.2/51.4]).reshape((3, nfoloder, 1))  # the field is applied to the z-direction, 51.4 is the factor that converts the field from V/A to atomic unit

    EO_sum = EO_all_reshape# + Eexternal
    EH_sum = EH_all_reshape# + Eexternal
    Ew_sum = Ew_all_reshape# + Eexternal

    return EO_sum, EH_sum, Ew_sum







