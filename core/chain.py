import numpy as np
import math
import core.random_gen as rng
from core.polydispersity import froot
from scipy.optimize import root_scalar

class ensemble_chains(object):

    def __init__(self, config, seed):
        
        self.beta = config['beta']
        self.CD_flag = config['CD_flag']
        self.PD_input = config['polydisperse']
        self.QN = np.zeros(shape=(config['Nchains'],config['NK'],4),dtype=float)
        self.tau_CD = np.zeros(shape=(config['Nchains'],config['NK']),dtype=float)
        self.Z = np.zeros(shape=config['Nchains'],dtype=float)

        if self.PD_input['flag']:
            Mw = self.PD_input['Mw']*1000
            Mn = self.PD_input['Mn']*1000
            self.MK = self.PD_input['MK']
            self.mean_ = np.log(Mn**(3/2)/np.sqrt(Mw))
            self.sigma_ = np.sqrt(2*np.log(np.sqrt(Mw)/np.sqrt(Mn)))
            self.Mmax = root_scalar(froot, args=(Mn,Mw), bracket=[50000,1000000], method='bisect').root
        
        rng.initialize_generator(seed)

        return

    
    def z_dist(self,tNk):
        '''
        Function to determine the number of entangled strands, Z for each chain drawn randomly from the equilibrium distribution function.

        Args:
            tNk - total number of Kuhn steps in the chain
        Returns:
            Z - number of entangled strands in the chain
        '''
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
        '''
        Determine the number of entangled strands in the chain and 
        resample from the distribution if total Z is greater than the number of Kuhn steps in the chain
        '''
        tz = self.z_dist(tNk)
        while (tz > z_max):
            tz = self.z_dist(tNk)
        return tz


    def ratio(self, A, n, i):
        '''
        Function to calculate the ratio of two binomail coefficients:
        ratio = (i-1)(A-n)!(A-i+1)!/((A-n-i+2)!A!)
        '''

        r = float(i-1)/float(A-n+1)
        if n > 1:
            for j in range(0, n-1):
                r *= (float(A - i + 1 - j) / float(A - j))

        return r


    def N_dist(self, ztmp, tNk):
        '''
        Function to set the distribution of NK in the chain for each entangled strand, drawn randomly from an equilibrium distribution. 

        Args: 
            ztmp - Number of entangled strands in the chain from Z_dist
            tNk - total number of Kuhn steps in the chain

        Returns:
            tN - number of Kuhn steps in each entangled strand (array) 
        '''
        tN = [0]*ztmp

        if ztmp == 1:
            tN[0] = tNk

        else:
            A = tNk-1
            for i in range(ztmp,1,-1):
                p = rng.genrand_real3()
                Ntmp = 0
                sumres = 0.0
                while (p>=sumres) and (Ntmp != (A-i+2)):
                    Ntmp+=1
                    sumres += self.ratio(A, Ntmp, i)
                tN[i-1] = Ntmp
                A = A - Ntmp
            tN[0] = A + 1
        return tN


    def Q_dist(self, tz, Ntmp, dangling_begin=True):
        '''
        Function to calculate the distribution of slip link orientations Q drawn randomly from an equilibrium distribution. 

        Args: 
            tz - total number of entangled strands in the chain
            Ntmp - number of Kuhn steps in each entangled strand
            dangling_begin - boolean to determine whether the end is dangling or not (used for network structures) #TODO: add network strands
        
        Returns: 
            Qx, Qy, Qz - orientation of slip links in the chain
        '''
        Qx = [0.0]*tz
        Qy = [0.0]*tz
        Qz = [0.0]*tz

        if tz>2: #dangling ends not part of distribution
            for j in range(1,tz-1):
                Qx[j] = rng.gauss_distr()*np.sqrt(float(Ntmp[j])/3.0)
                Qy[j] = rng.gauss_distr()*np.sqrt(float(Ntmp[j])/3.0)
                Qz[j] = rng.gauss_distr()*np.sqrt(float(Ntmp[j])/3.0)

        return Qx,Qy,Qz


    def tau_CD_dist(self,chainIdx,tz,pcd=None):

        rng.use_last = False
        for k in range(0,tz-1):
            if self.CD_flag !=0:
                if self.PD_input['flag']:
                    Mlognorm = (np.exp(rng.gauss_distr()*self.sigma_ + self.mean_))
                    NK_PD = Mlognorm/self.MK
                    while (Mlognorm > self.Mmax) or (NK_PD < 1):
                        Mlognorm = (np.exp(rng.gauss_distr()*self.sigma_ + self.mean_))
                        NK_PD = Mlognorm/self.MK
                    pcd.__init__(NK_PD,self.beta)
                    self.tau_CD[chainIdx,k] = pcd.tau_CD_f_t() 
                else:
                    self.tau_CD[chainIdx,k] = pcd.tau_CD_f_t() 
            else:
                self.tau_CD[chainIdx,k] = 0.0

        return


    def chain_init(self, chainIdx, Nk, z_max, pcd=None, dangling_begin=True):
        '''
        Initialize all chains in the ensemble

        Args:
            chainIdx - index of the chain in the ensemble for array handling
            Nk - total number of Kuhn steps in each chain
            z_max - maximum number of entangled strands each chain can have (currently, set to Nk)
            pcd - probability density for the entanglement to have a characteristic CD lifetime
            dangling_begin - boolean to determine whether the chain ends are free or not (#TODO: currently not implemented)

        Returns:
            None - sets the initialized class objects for each chain including slip-link orientations Q, Kuhn steps N, entangled strands Z, probability densities for CD, etc.

        '''

        tz = self.z_dist_truncated(Nk,z_max)
        self.Z[chainIdx] = tz

        self.tau_CD_dist(chainIdx,tz,pcd=pcd)

        tN = self.N_dist(tz,Nk)
        Qx,Qy,Qz = self.Q_dist(tz, tN)

        for k in range(0,tz-1):
            self.QN[chainIdx,k] = [Qx[k], Qy[k], Qz[k], tN[k]]
            if self.tau_CD[chainIdx,k]==0:
                self.tau_CD[chainIdx,k]=np.inf
            else:
                self.tau_CD[chainIdx,k] = 1.0/self.tau_CD[chainIdx,k]

        if dangling_begin:
            self.QN[chainIdx,0] = [0.0,0.0,0.0,tN[0]]
            self.QN[chainIdx,tz-1] = [0.0,0.0,0.0,tN[tz-1]]

        return

