import numpy as np
import math
import core.random_gen as rng
from core.pcd_tau import p_cd


class ensemble_chains(object):

    def __init__(self, config):

        self.beta = config['beta']
        self.CD_flag = config['CD_flag']
        self.QN = []
        self.tau_CD = []
        self.Z = []

        return

    
    def z_dist(self,tNk):

        y = float(rng.genrand_real3()/(1+self.beta)*math.pow(1+(1/self.beta),tNk))
        z = 1
        sum1 = 0.0
        si = float(1.0/self.beta)
        while sum1 < y:
            sum1 += si
            si = si/self.beta*(tNk-z)/z
            z += 1

        return z-1

    
    def z_dist_truncated(self,tNk, z_max):

        tz = self.z_dist(tNk)
        while (tz > z_max):
            tz = z_dist(tNk)

        return tz


    def ratio(self, A, n, i):
        '''Calculates ratio of two binomail coefficients:
                ratio = (i-1)(A-N)!(A-i+1)!/((A-N-i+2)!A!)'''

        r = float(i-1)/float(A-n+1)
        if n > 1:
            for j in range(0, n-1):
                r *= (float(A - i + 1 - j) / float(A - j))

        return r


    def N_dist(self, ztmp, tNk):

        tN = [0]*ztmp

        if ztmp == 1:
            tN[0] = tNk

        else:
            A = tNk-1
            for i in range(ztmp,1,-1):
                p = rng.genrand_real3()
                Ntmp = 0
                sumres = 0.0
                while p>=sumres and Ntmp+1 != A-i+2:
                    Ntmp+=1
                    sumres += self.ratio(A, Ntmp, i)
                tN[i-1] = Ntmp
                A = A - Ntmp
            tN[0] = A + 1
        return tN


    def Q_dist(self, tz, Ntmp, dangling_begin=True):

        Qx = [0.0]*tz
        Qy = [0.0]*tz
        Qz = [0.0]*tz

        if tz>2: #dangling ends not part of distribution
            rng.use_last=False
            for j in range(1,tz-1):
                Qx[j] = rng.gauss_distr()*np.sqrt(float(Ntmp[j])/3.0)
                Qy[j] = rng.gauss_distr()*np.sqrt(float(Ntmp[j])/3.0)
                Qz[j] = rng.gauss_distr()*np.sqrt(float(Ntmp[j])/3.0)

        return Qx,Qy,Qz


    def chain_init(self, Nk, z_max, pcd=None, dangling_begin=True, PD_flag=False):

        tz = self.z_dist_truncated(Nk,z_max)

        tau_CD = [0]*(z_max)

        for k in range(0,tz-1):
            if self.CD_flag !=0:
                tau_CD[k] = pcd.tau_CD_f_t() 
            
            else:
                tau_CD[k] = 0.0

        tN = self.N_dist(tz,Nk)
        Qx,Qy,Qz = self.Q_dist(tz, tN)

        tmpQN = [[0,0,0,0] for j in range(0,z_max)]
        for k in range(0,tz-1):
            tmpQN[k] = [Qx[k], Qy[k], Qz[k], tN[k]]
            if tau_CD[k]==0:
                tau_CD[k]=np.inf
            else:
                tau_CD[k] = 1.0/tau_CD[k]

        if dangling_begin:
            tmpQN[0] = [0.0,0.0,0.0,tN[0]]
            tmpQN[tz-1] = [0.0,0.0,0.0,tN[tz-1]]

        self.QN = np.append(self.QN,tmpQN)
        self.Z = np.append(self.Z,tz)
        self.tau_CD = np.array(np.append(self.tau_CD,tau_CD))


        return

