import numpy as np
import math
import core.random_gen as rng
from core.pcd_tau import p_cd


class ensemble_chains(object):

    def __init__(self, config, seed):

        self.beta = config['beta']
        self.CD_flag = config['CD_flag']
        self.QN = np.zeros(shape=(config['Nchains'],config['NK'],4),dtype=float)
        self.tau_CD = np.zeros(shape=(config['Nchains'],config['NK']),dtype=float)
        self.Z = np.zeros(shape=config['Nchains'],dtype=float)
        rng.initialize_generator(seed)

        return

    
    def z_dist(self,tNk):
        
        p = rng.genrand_real3()  
        y = p/(1+self.beta)*math.pow(1+(1/self.beta),tNk)
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
            tz = self.z_dist(tNk)
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
                Ntmp = 1
                sumres = 0.0
                while (p>=sumres) and ((Ntmp) != (A-i+2)):
                    sumres += self.ratio(A, Ntmp, i)
                    Ntmp+=1
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


    def chain_init(self, chainIdx, Nk, z_max, pcd=None, dangling_begin=True, PD_flag=False):

        tz = self.z_dist_truncated(Nk,z_max)
        self.Z[chainIdx] = tz

        for k in range(0,tz-1):
            if self.CD_flag !=0:
                self.tau_CD[chainIdx,k] = pcd.tau_CD_f_t() 
            
            else:
                self.tau_CD[chainIdx,k] = 0.0

        tN = self.N_dist(tz,Nk)
        Qx,Qy,Qz = self.Q_dist(tz, tN)

        #tmpQN = [[0,0,0,0] for j in range(0,z_max)]
        for k in range(0,tz-1):
            self.QN[chainIdx,k] = [Qx[k], Qy[k], Qz[k], tN[k]]
            if self.tau_CD[chainIdx,k]==0:
                self.tau_CD[chainIdx,k]=np.inf
            else:
                self.tau_CD[chainIdx,k] = 1.0/self.tau_CD[chainIdx,k]

        if dangling_begin:
            self.QN[chainIdx,0] = [0.0,0.0,0.0,tN[0]]
            self.QN[chainIdx,tz-1] = [0.0,0.0,0.0,tN[tz-1]]

        #self.QN = np.append(self.QN,tmpQN)
        #self.Z = np.append(self.Z,tz)
        #self.tau_CD = np.array(np.append(self.tau_CD,tau_CD))


        return

