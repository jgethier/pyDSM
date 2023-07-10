import numpy as np
import random as rng
import math

class p_cd(object):
    '''
    Class to store discrete modes from the entanglement lifetime distribution for constraint dynamics
    '''
    def __init__(self,tauArr,gArr,nmodes):
        self.nmodes = nmodes
        self.g = gArr
        self.tau = tauArr

        total = 0
        for j in range(0,nmodes):
            total += self.g[j]

        for j in range(0,nmodes):
            self.g[j] = self.g[j]/total

        self.ptau_sum = 0
        for j in range(0,nmodes):
            self.ptau_sum += self.g[j]*self.tau[j]


    def tau_CD_f_t(self):
        
        p = rng.uniform(0.0,1.0)
        total = 0
        for i in range(0,self.nmodes+1):
            if total>p:
                break
            else:
                total += self.g[i]*self.tau[i]/self.ptau_sum

        return self.tau[i-1]


    def W_CD_destroy_aver(self):

        return 1.0/self.ptau_sum


class p_cd_linear(object):
    '''
    Class to store analytic expressions for calculating probability densities for entanglement lifetimes due to constraint dynamics (linear polymers only)
    '''

    def __init__(self,NK,beta):

        self.g = 0.667
        self.z = (NK + beta) / (beta + 1.0)

        if beta != 1.0:

            self.alpha = (0.053 * np.log(beta) + 0.31) * math.pow(self.z,(-0.012 * np.log(beta) - 0.024))
            self.tau_0 = 0.285 * math.pow((beta + 2.0),0.515)
            if NK < 2:
                self.tau_max = self.tau_0
                self.tau_D = self.tau_0 
            else: 
                self.tau_max = 0.025 * math.pow((beta+2.0),2.6) * math.pow(self.z,2.83)
                self.tau_D = 0.036 * math.pow((beta+2.0),3.07) * math.pow((self.z - 1),3.02)

        else:

            self.alpha = 0.267096 - 0.375571 * np.exp(-0.0838237 * NK)
            self.tau_0 = 0.460277 + 0.298913 * np.exp(-0.0705314 * NK)
            if NK < 4:
                self.tau_max = self.tau_0
                self.tau_D = self.tau_0
            else:
                self.tau_max = 0.0156137 * math.pow(NK, 3.18849)
                self.tau_D = 0.0740131 * math.pow(NK, 3.18363)


        self.tau_alpha = math.pow(self.tau_max,self.alpha) - math.pow(self.tau_0,self.alpha)
        self.tau_alpha_m1 = math.pow(self.tau_max,(self.alpha-1.0)) - math.pow(self.tau_0,(self.alpha - 1.0))
        if self.tau_alpha == 0.0:
            self.ratio_tau_alpha = (self.alpha - 1.0)/self.alpha/self.tau_0
        else:
            self.ratio_tau_alpha = self.tau_alpha_m1 / self.tau_alpha 

        self.At = (1.0 - self.g)
        self.Adt = self.At * self.alpha / (self.alpha - 1.0)
        self.Bdt = self.Adt * self.ratio_tau_alpha
        self.normdt = self.Bdt + self.g / self.tau_D 

    def tau_CD_f_t(self):

        p = rng.uniform(0.0,1.0)
        while p == 0:
            p = rng.uniform(0.0,1.0)
        if p < (1.0 - self.g):
            return math.pow(p * self.tau_alpha / self.At + math.pow(self.tau_0, self.alpha), 1.0 / self.alpha)
        else:
            return self.tau_D


    def W_CD_destroy_aver(self):
        return self.normdt 


