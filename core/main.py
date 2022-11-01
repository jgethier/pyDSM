import os
import sys
import time
import warnings
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
from alive_progress import alive_bar 
import yaml
import math
import pickle
import random as rng
from core.chain import ensemble_chains
from core.pcd_tau import p_cd, p_cd_linear
import core.ensemble_kernel as ensemble_kernel
import core.gpu_random as gpu_rand 
import core.correlation as correlation
from core.fit import CURVE_FIT

warnings.filterwarnings('ignore')

class FSM_LINEAR(object):

    def __init__(self,sim_ID,device_ID,output_dir,correlator,fit=False,distr=False,load_file=None,save_file=None):
        
        #simulation ID from run argument (>>python gpu_dsm sim_ID)
        self.sim_ID = sim_ID

        #determine correlator (on the fly or post-process)
        if correlator == 'otf':
            self.postprocess = False
        else:
            self.postprocess = True 

        #set fit to True if G(t) will be fit 
        self.fit = fit
        
        self.distr=distr
        
        #read in input file and parameters
        if os.path.isfile('input.yaml'):
            with open('input.yaml') as f:
                self.input_data = yaml.load(f, Loader=yaml.FullLoader)
                
            #get results path ready
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            self.output_dir = output_dir
                
        else:
            sys.exit("No input file found.")

        #check for GPU device
        num_devices = len(cuda.gpus)
        if num_devices == 0:
            sys.exit("No GPU found.")

        #select gpu device
        cuda.select_device(device_ID)
        device_name = str(cuda.cudadrv.driver.Device(device_ID).name).replace('b','')
        print("Using device: %s"%(device_name))

        #set seed number based on num argument
        SEED = sim_ID*self.input_data['Nchains']
        self.seed = SEED
        print("Simulation ID: %d"%(sim_ID))
        print("Using %d*Nchains as a seed for the random number generator."%(sim_ID))
        
        rng.seed(self.seed)

        return


    def save_distributions(self,distr,QN,Z):
        '''
        Function to save Q distributions to file
        Inputs: distr - name of distribution (initial or final) 
                QN - distribution of strand orientations and number of Kuhn steps in cluster
                Z - number of entanglements in each chain
        '''
        Q = []
        L = []
        for i in range(0,self.input_data['Nchains']):
            L_value = 0.0
            for j in range(1,int(Z[i])-1):
                x2 = QN[i,j,0]**2
                y2 = QN[i,j,1]**2
                z2 = QN[i,j,2]**2
                Q_value = np.sqrt(x2+y2+z2)
                L_value += Q_value
                Q.append(Q_value)
                
            L.append(L_value)

        Q_sorted = np.sort(Q)
        L_sorted = np.sort(L)
        
        Q_file = os.path.join(self.output_dir,'distr_Q_%s_%d.txt'%(distr,self.sim_ID))
        L_file = os.path.join(self.output_dir,'distr_L_%s_%d.txt'%(distr,self.sim_ID))
        
        np.savetxt(Q_file, Q_sorted)
        np.savetxt(L_file, L_sorted)

        return

    def write_stress(self,num_sync,time,stress_array):
        '''
        Write the stress of all chains in ensemble
        Inputs: num_sync - sync number for simulation
                time - simulation time
                stress_array - array of stress values to write
        Outputs: stress over time for each chain in stress.txt file
        '''
   
        #if first time sync, need to include 0 and initialize file path
        if num_sync == 1:
            if self.flow or self.turn_flow_off:
                self.stress_output = os.path.join(self.output_dir,'stress_%d.txt'%self.sim_ID)
            else:
                self.stress_output = os.path.join(self.output_dir,'stress_%d.dat'%self.sim_ID)
            self.old_sync_time = 0
            
            if self.flow:
                time_index = 1
            else:
                time_index = 0
        
        else: #do not include the t=0 spot of array (only used for first time sync)
            time_index = 1
        
        #set time array depending on num_sync
        time_resolution = self.input_data['tau_K']
        time_array = np.arange(self.old_sync_time,time+time_resolution/2.0,time_resolution)
        if not self.flow and self.turn_flow_off:
            time_array = np.arange(self.old_sync_time,(time+self.input_data['flow_time'])+time_resolution/2.0,time_resolution)
        time_array = np.reshape(time_array[time_index:],(1,len(time_array[time_index:])))
        len_array = len(time_array[0])+1
        
        #if flow, take average stress tensor over all chains, otherwise write out only tau_xy stress of all chains
        if self.flow or self.turn_flow_off:
            if self.flow:
                stress = np.array([np.mean(stress_array[:,0,i],axis=0) for i in range(0,8)])
                error = np.array([np.std(stress_array[:,0,i],axis=0)/np.sqrt(self.input_data['Nchains']) for i in range(0,8)])
                stress = np.reshape(stress,(8,1))
                error = np.reshape(error,(8,1))
            else:
                stress = np.array([np.mean(stress_array[:,:,i],axis=0) for i in range(0,8)])
                error = np.array([np.std(stress_array[:,:,i],axis=0)/np.sqrt(self.input_data['Nchains']) for i in range(0,8)])
                stress = np.reshape(stress[:,time_index:len_array],(8,len(stress_array[0,time_index:len_array,0])))
                error = np.reshape(error[:,time_index:len_array],(8,len(stress_array[0,time_index:len_array,0])))
            combined = np.hstack((time_array.T, stress.T, error.T))
        else:
            stress = np.reshape(stress_array[:,time_index:len_array,0],(self.input_data['Nchains'],len(stress_array[0,time_index:len_array,0])))    
            combined = np.hstack((time_array.T, stress.T))
        
        #write stress to file
        if num_sync == 1:
            if self.flow or self.turn_flow_off:
                with open(self.stress_output,'w') as f:
                    f.write('time, tau_xx, tau_yy, tau_zz, tau_xy, tau_yz, tau_xz, Z, new_Q, stderr_xx, stderr_yy, stderr_zz, stderr_xy, stderr_yz, stderr_xz, stderr_Z, stderr_newQ\n')
                    np.savetxt(f, combined, delimiter=',', fmt='%.8f')
            else:
                with open(self.stress_output, "wb") as f:
                   pickle.dump(combined,f)
        else:
            if self.flow or self.turn_flow_off:
                with open(self.stress_output,'a') as f:
                    np.savetxt(f, combined, delimiter=',', fmt='%.8f')
            else:
                with open(self.stress_output, 'ab') as f:
                    pickle.dump(combined,f)
        
        #keeping track of the last simulation time for beginning of next array
        if not self.flow and self.turn_flow_off:
            self.old_sync_time = time + self.input_data['flow_time']
        else:
            self.old_sync_time = time
        
        return 


    def write_com(self,num_sync,time,com_array):
        '''
        Write the CoM of all chains in ensemble
        Inputs: num_sync - sync number for simulation
                time - simulation time
                com_array - array of center of mass (CoM) values to write
        Outputs: CoM over time for each chain and dimension (x,y,z) in CoM.txt file
        '''
        
        if num_sync == 1: #if first time sync, need to include 0 and initialize file path
            self.com_output_x = os.path.join(self.output_dir, 'CoM_%d_x.dat'%self.sim_ID)
            self.com_output_y = os.path.join(self.output_dir, 'CoM_%d_y.dat'%self.sim_ID)
            self.com_output_z = os.path.join(self.output_dir, 'CoM_%d_z.dat'%self.sim_ID)
            self.old_sync_time = 0
            time_index = 0
        
        else:
            time_index = 1
        
        #set time array depending on num_sync
        time_resolution = self.input_data['tau_K']
        time_array = np.arange(self.old_sync_time,time+time_resolution/2.0,time_resolution)
        time_array = np.reshape(time_array[time_index:],(1,len(time_array[time_index:])))
        len_array = len(time_array[0])+1
        
        #reshape arrays
        com_x = np.reshape(com_array[:,time_index:len_array,0],(self.input_data['Nchains'],len(com_array[0,time_index:len_array,0])))    
        com_y = np.reshape(com_array[:,time_index:len_array,1],(self.input_data['Nchains'],len(com_array[0,time_index:len_array,1])))    
        com_z = np.reshape(com_array[:,time_index:len_array,2],(self.input_data['Nchains'],len(com_array[0,time_index:len_array,2])))    
        
        #combine time and CoM for each dimension
        combined_x = np.hstack((time_array.T, com_x.T))
        combined_y = np.hstack((time_array.T, com_y.T))
        combined_z = np.hstack((time_array.T, com_z.T))
        
        if num_sync == 1: #write data to files and overwrite if file exists
            with open(self.com_output_x,"wb") as f:
                pickle.dump(combined_x,f)
            with open(self.com_output_y,"wb") as f:
                pickle.dump(combined_y,f)
            with open(self.com_output_z,"wb") as f:
                pickle.dump(combined_z,f)
        else: #append file for num_sync > 1
            with open(self.com_output_x,"ab") as f:
                pickle.dump(combined_x,f)
            with open(self.com_output_y,"ab") as f:
                pickle.dump(combined_y,f)
            with open(self.com_output_z,"ab") as f:
                pickle.dump(combined_z,f)
        
        #keeping track of the last simulation time for beginning of next array
        self.old_sync_time = time

    
    def load_results(self,filename,block_num,block_size,num_chains):
        '''
        Load in part of the binary .dat file into the result array
        Inputs: filename - filename of the data file
                block_num - chain block number (total chains split into n blocks of size block_size)
                num_chains - block_num*block_size
        Returns: an array of data read from filename
        '''
        result_array = []
        with open(filename,'rb') as f:
            try:
                while True:
                    data = pickle.load(f)
                    for j in data:
                        if block_num == 0:
                            first_idx = 1
                        else:
                            first_idx = block_num*block_size + 1
                        last_idx = num_chains+1
                        result_array.append(j[first_idx:last_idx])
            except EOFError:
                pass

        return result_array


    def run(self):
        #set variables and start simulation (also any post-processing after simulation is completed)

        #set cuda  grid dimensions
        dimBlock = (32,32)
        dimGrid_x = (self.input_data['Nchains']+dimBlock[0]-1)//dimBlock[0]
        dimGrid_y = (self.input_data['NK']+dimBlock[1]-1)//dimBlock[1]
        dimGrid = (dimGrid_x,dimGrid_y)
        
        #flattened grid dimensions
        threadsperblock = 256
        blockspergrid = (self.input_data['Nchains'] + threadsperblock - 1)//threadsperblock


        #if CD_flag is set (constraint dynamics is on), set probability of CD parameters with analytic expression
        if self.input_data['CD_flag']==1 and self.input_data['architecture']=='linear':
            analytic=True 
            
            #initialize constants for constraint dynamics probability calculation
            pcd = p_cd_linear(self.input_data['NK'],self.input_data['beta'])
            d_CD_create_prefact = cuda.to_device([pcd.W_CD_destroy_aver()/self.input_data['beta']])
            
            #array of constants used for tau_CD probability calculation
            pcd_array = np.array([pcd.g,                            #0
                                  pcd.alpha,                        #1
                                  pcd.tau_0,                        #2
                                  pcd.tau_max,                      #3
                                  pcd.tau_D,                        #4
                                  1.0/pcd.tau_D,                    #5, inverse tau_D 
                                  1.0*pcd.tau_alpha/pcd.At,         #6, d_At
                                  math.pow(pcd.tau_0,pcd.alpha),    #7, d_Dt
                                  -1.0/pcd.alpha,                   #8, d_Ct
                                  pcd.normdt*pcd.tau_alpha/pcd.Adt, #9, d_Adt
                                  pcd.Bdt/pcd.normdt,               #10, d_Bdt 
                                  -1.0/(pcd.alpha - 1.0),           #11, d_Cdt
                                  pcd.tau_0**(pcd.alpha - 1.0)      #12, d_Ddt
                                  ],dtype=float)
            
            discrete = False #set discrete variable to False to implement analytic expression for p(tau_CD) probability
            
            #set discrete modes to 0 (this may be better handled since it's waste of memory space)
            pcd_table_tau = np.zeros(1)
            pcd_table_cr = np.zeros(1)
            pcd_table_eq = np.zeros(1)

        #if not linear chains, use discrete fit for tau_CD
        elif self.input_data['CD_flag']==1 and self.input_data['architecture']!='linear':
            analytic=False

            #check to see if multi-mode maxwell fit param file exists, read and set modes
            if os.path.isfile("pcd_MMM.txt"):
                with open("pcd_MMM.txt",'r') as pcd_file:
                    nmodes = int(pcd_file.readline())

                    lines = pcd_file.readlines()
                    tauArr = [float(line.split()[0]) for line in lines]
                    gArr = [float(line.split()[1]) for line in lines]
            else:
                sys.exit("No file named pcd_MMM.txt found. Fit f_d(t) statistics to multi-mode Maxwell for pcd statistics.")
            
            #initialize modes and time constants
            pcd = p_cd(tauArr, gArr, nmodes)
            d_CD_create_prefact = cuda.to_device([pcd.W_CD_destroy_aver()/self.input_data['beta']])

            pcd_table_cr = np.zeros(pcd.nmodes,dtype=float)
            pcd_table_eq = np.zeros(pcd.nmodes,dtype=float)
            pcd_table_tau = np.zeros(pcd.nmodes,dtype=float)
            sum_eq = 0.0
            sum_cr = 0.0
            for i in range(0,pcd.nmodes):
                sum_cr += pcd.g[i]
                pcd_table_cr[i] = sum_cr

                sum_eq += pcd.g[i]*pcd.tau[i]/pcd.ptau_sum
                pcd_table_eq[i] = sum_eq

                pcd_table_tau[i] = pcd.tau[i]
            
            pcd_array = np.zeros(1)
            discrete = True

        
        #else, set probability factors to 0
        else:
            analytic=False
            pcd=None
            discrete = False 
            pcd_table_tau = np.zeros(1)
            pcd_table_cr = np.zeros(1)
            pcd_table_eq = np.zeros(1)
            pcd_array = np.zeros(1)
            d_CD_create_prefact = cuda.to_device([0.0])

            
        #generate initial chain conformations on host CPU
        print('Generating initial chain conformations on host...',end="",flush=True)
        chain = ensemble_chains(self.input_data)
        
        #initialize chains
        for m in range(0,self.input_data['Nchains']):
            chain.chain_init(m,self.input_data['NK'],z_max=self.input_data['NK'],pcd=pcd)
        
        print('Done.')
        
        #simulation in flow if kappa strain tensor is set
        if np.any(np.array(self.input_data['kappa'])!=0.0):
            print("")
            print("Strain tensor is non-zero. Simulating polymers in flow...")
            d_kappa = cuda.to_device(np.array(self.input_data['kappa'],dtype=float))
            self.flow = True
            calc_type = 1
            self.fit = False 
            if self.input_data['flow_time']>0 and self.input_data['flow_time']<self.input_data['sim_time']:
                self.turn_flow_off = True
            else:
                self.turn_flow_off = False
        else:
            self.flow = False
            self.turn_flow_off = False
            #determine system size for post-processing
            postprocess = True
            d_kappa = cuda.to_device(np.array(self.input_data['kappa'],dtype=float))
        
        postprocess = self.postprocess

        #if not flow, check for equilibrium calculation type (G(t) or MSD)
        if not self.flow:
            block_size = self.input_data['Nchains'] #blocks of chains to be post-processed simultanously
            
            if self.input_data['EQ_calc']=='stress':
                print("Equilibrium calculation specified: G(t).")
                calc_type = 1
                if self.postprocess:
                    while block_size*self.input_data['sim_time']/self.input_data['tau_K']>1e8: #check for memory limits for post processing calculations 
                        block_size -= 100
                        if block_size < 100:
                            postprocess = False 
                            break 
                
            elif self.input_data['EQ_calc']=='msd': #check for memory limits for post processing calculations
                print("Equilbrium calculation specified: MSD.")
                calc_type = 2
                if self.postprocess:
                    while block_size*self.input_data['sim_time']/self.input_data['tau_K']*3>1e8:
                        block_size -= 100
                        if block_size < 100: 
                            postprocess = False 
                            break 
            
            else:
                sys.exit('Incorrect EQ_calc specified in input file. Please choose "stress" or "msd" for G(t) or MSD for the equilibrium calculation.')

        if not postprocess and self.postprocess: #if simulation time * 100 > 1e9, use on the fly correlator (but will not report uncertainties)
            print("")
            print("Warning: simulation time is too large for postprocessing. Using on the fly correlator for equilibrium calculation. Uncertainty will not be reported.")
            print("")
        elif not postprocess:
            print("Using on the fly correlator for equilibrium calculation. Uncertainty in the correlation values will not be reported.")

        #keep track of first entanglement for MSD
        QN_first = np.zeros(shape=(chain.QN.shape[0],3)) 

        #save initial chain conformation distributions\
        if self.distr:
            self.save_distributions('initial',chain.QN,chain.Z)

        #save initial Z distribution to file
        if self.distr:
            np.savetxt(os.path.join(self.output_dir,'Z_initial_%d.txt'%self.sim_ID),chain.Z,fmt='%d')

        #initialize arrays for finding jump index
        found_index = np.zeros(shape=(chain.QN.shape[0]),dtype=int)
        found_shift = np.zeros(shape=(chain.QN.shape[0]),dtype=int)
        sum_W_sorted = np.zeros(shape=(chain.QN.shape[0]),dtype=float)
        shift_probs = np.zeros(shape=(chain.QN.shape[0],chain.QN.shape[1]+1,4),dtype=float)
        
        #intitialize arrays for random numbers used
        rand_used = np.zeros(shape=chain.QN.shape[0],dtype=int)
        tau_CD_used_SD = np.zeros(shape=chain.QN.shape[0],dtype=int)
        tau_CD_used_CD = np.zeros(shape=chain.QN.shape[0],dtype=int)
        tau_CD_gauss_rand_SD = np.zeros(shape=(chain.QN.shape[0],250,4),dtype=float)
        tau_CD_gauss_rand_CD = np.zeros(shape=(chain.QN.shape[0],250,4),dtype=float)
        uniform_rand = np.zeros(shape=(chain.QN.shape[0],250),dtype=float)
        add_rand = np.zeros(shape=(chain.QN.shape[0]),dtype=float)
        
        #move random number arrays and pcd statistics to device
        d_tau_CD_gauss_rand_SD=cuda.to_device(tau_CD_gauss_rand_SD)
        d_tau_CD_gauss_rand_CD=cuda.to_device(tau_CD_gauss_rand_CD)
        d_uniform_rand=cuda.to_device(uniform_rand)
        d_pcd_array = cuda.to_device(pcd_array)
        d_pcd_table_eq = cuda.to_device(pcd_table_eq)
        d_pcd_table_cr = cuda.to_device(pcd_table_cr)
        d_pcd_table_tau = cuda.to_device(pcd_table_tau)
        
        #initialize random state and fill random variable arrays
        random_state = self.seed
        self.rng_states = create_xoroshiro128p_states(threadsperblock*blockspergrid,seed=random_state)
        
        gpu_rand.fill_gauss_rand_tauCD[blockspergrid,threadsperblock](self.rng_states, discrete, self.input_data['Nchains'], 250, True, self.input_data['CD_flag'],d_tau_CD_gauss_rand_SD, d_pcd_array, d_pcd_table_eq, 
                                      d_pcd_table_cr,d_pcd_table_tau)
        
        #if CD flag is 1 (constraint dynamics is on), fill random gaussian array for new strands created by CD
        if self.input_data['CD_flag'] == 1:
            gpu_rand.fill_gauss_rand_tauCD[blockspergrid,threadsperblock](self.rng_states, discrete, self.input_data['Nchains'], 250, False, 
                                                                         self.input_data['CD_flag'],d_tau_CD_gauss_rand_CD, d_pcd_array, 
                                                                         d_pcd_table_eq, d_pcd_table_cr,d_pcd_table_tau)
        
        gpu_rand.fill_uniform_rand[blockspergrid,threadsperblock](self.rng_states, self.input_data['Nchains'], 250, d_uniform_rand)

        #initialize arrays for chain time and entanglement lifetime
        chain_time = np.zeros(shape=self.input_data['Nchains'],dtype=float)
        time_resolution = self.input_data['tau_K']
        time_compensation = np.zeros(shape=self.input_data['Nchains'],dtype=float)
        tdt = np.zeros(shape=self.input_data['Nchains'],dtype=float)
        t_cr = np.zeros(shape=(chain.QN.shape[0],chain.QN.shape[1]),dtype=float)
        f_t = np.zeros(shape=self.input_data['Nchains'],dtype=float)
        write_time = np.zeros(shape=self.input_data['Nchains'],dtype=int)
        reach_flag = np.zeros(shape=self.input_data['Nchains'],dtype=int)
                
        #initialize arrays for which chains create a slip link from sliding and/or constraint dynamics
        QN_create_SDCD = np.zeros(shape=(chain.QN.shape[0],chain.QN.shape[1]+1,4))
        new_t_cr = np.zeros(shape=(chain.QN.shape[0],chain.QN.shape[1]+1))
        new_tau_CD = np.zeros(shape=(chain.QN.shape[0],chain.QN.shape[1]+1))

        #correlator parameters for both block transformation or on-the-fly
        p = correlation.p
        m = correlation.m
        S_corr = math.ceil(np.log(self.input_data['sim_time']/self.input_data['tau_K']/p)/np.log(m)) + 1 #number of correlator levels
        
        if not postprocess:
            #initialize arrays for correlator
            D_array = np.zeros(shape=(chain.QN.shape[0],S_corr,p,3),dtype=float)
            D_shift_array = np.zeros(shape=(chain.QN.shape[0],S_corr,p,3),dtype=float)
            C_array = np.zeros(shape=(chain.QN.shape[0],S_corr,p),dtype=float)
            N_array = np.zeros(shape=(chain.QN.shape[0],S_corr,p),dtype=float)
            A_array = np.zeros(shape=(chain.QN.shape[0],S_corr,3),dtype=float)
            M_array = np.zeros(shape=(chain.QN.shape[0],S_corr),dtype=int)

            #move correlator arrays to device
            d_D = cuda.to_device(D_array)
            d_D_shift = cuda.to_device(D_shift_array)
            d_C = cuda.to_device(C_array)
            d_N = cuda.to_device(N_array)
            d_A = cuda.to_device(A_array)
            d_M = cuda.to_device(M_array)

        #move arrays to device
        d_QN_create_SDCD = cuda.to_device(QN_create_SDCD)
        d_reach_flag = cuda.to_device(reach_flag) 
        d_new_t_cr = cuda.to_device(new_t_cr)
        d_new_tau_CD = cuda.to_device(new_tau_CD)

        #move arrays to device
        d_QN = cuda.to_device(chain.QN)
        d_NK = cuda.to_device([self.input_data['NK']])
        d_CDflag = cuda.to_device([self.input_data['CD_flag']])
        d_beta = cuda.to_device([self.input_data['beta']])
        d_QN_first = cuda.to_device(QN_first)
        d_tau_CD = cuda.to_device(chain.tau_CD)
        d_Z = cuda.to_device(chain.Z)
        d_found_shift = cuda.to_device(found_shift)
        d_found_index = cuda.to_device(found_index)
        d_shift_probs = cuda.to_device(shift_probs)
        d_sum_W_sorted = cuda.to_device(sum_W_sorted)
        d_add_rand = cuda.to_device(add_rand)
        d_t_cr = cuda.to_device(t_cr)
        d_f_t = cuda.to_device(f_t)
        d_chain_time = cuda.to_device(chain_time)
        d_time_compensation = cuda.to_device(time_compensation)
        d_time_resolution = cuda.to_device([time_resolution])
        d_write_time = cuda.to_device(write_time)
        d_tdt = cuda.to_device(tdt)
        d_rand_used = cuda.to_device(rand_used)
        d_tau_CD_used_SD=cuda.to_device(tau_CD_used_SD)
        d_tau_CD_used_CD=cuda.to_device(tau_CD_used_CD)
        
        #set simulation time and entanglement lifetime array
        self.step_count = 0                                #used to calculate number of jump processes for checking random number arrays
        enttime_bins = np.zeros(shape=(20000),dtype=int) #bins to hold entanglement lifetime distributions
        
        #initialize some time constants used for simulation
        #sync time is the time for a chain to sync with other chains and write data to file
        if self.flow: #if flow, set sync time to tau_K
            max_sync_time = self.input_data['tau_K']
            res = np.zeros(shape=(chain.QN.shape[0],1,8),dtype=float) #initialize result array (stress or CoM)
            if self.turn_flow_off:
                num_time_syncs_flow = int(math.ceil(self.input_data['flow_time'] / max_sync_time))
                max_sync_time_afterflow = 250*self.input_data['tau_K']
                num_time_syncs_afterflow = int(math.ceil((self.input_data['sim_time']-self.input_data['flow_time'])/max_sync_time_afterflow))
        else:
            if postprocess:
                max_sync_time = 250*self.input_data['tau_K'] #setting sync time to t = 250*tau_K
                #set max sync time to simulation time if simulation time is less than sync time
                if max_sync_time > self.input_data['sim_time']:
                    max_sync_time = self.input_data['sim_time']
                if calc_type == 1: #result array (G(t) or MSD) dimensions are set based on EQ_calc
                    res = np.zeros(shape=(chain.QN.shape[0],251,1),dtype=float) #initialize result array (stress or CoM) to always hold 250 stress values per chain
                elif calc_type == 2:
                    res = np.zeros(shape=(chain.QN.shape[0],251,3),dtype=float) 
            else:
                max_sync_time = self.input_data['sim_time']
                res = np.zeros(shape=(chain.QN.shape[0],250,4),dtype=float) 
        
        #move result array, calc_type, and flow variables to device
        d_res = cuda.to_device(res) 
        d_calc_type = cuda.to_device([calc_type])
        d_flow = cuda.to_device([self.flow])
        d_flow_off = cuda.to_device([self.turn_flow_off])

        #calculate number of time syncs based on max_sync_time
        if not self.turn_flow_off:
            num_time_syncs = int(math.ceil(self.input_data['sim_time'] / max_sync_time))
        else:
            num_time_syncs = num_time_syncs_flow + num_time_syncs_afterflow
        
        
        #SIMULATION STARTS -------------------------------------------------------------------------------------------------------------------------------
        
        #timer start
        t0 = time.time()

        
        progress_bar = {'total': None,'bar': 'smooth', 'spinner':None,'manual':True}
        with alive_bar(**progress_bar) as bar:

            #defer memory deallocation until after simulation is done
            with cuda.defer_cleanup():
                
                #start loop over number of times chains are synced
                for x_sync in range(1,num_time_syncs+1):

                    if x_sync == num_time_syncs:
                        next_sync_time = self.input_data['sim_time']
                    else:
                        next_sync_time = x_sync*max_sync_time
                    
                    #if simulating shear flow and flow time is less than total simulation time, turn off flow when flow time is reached
                    if self.flow and self.turn_flow_off:
                        if next_sync_time>self.input_data['flow_time'] and self.flow:
                            print('Turning off flow, equilibrium variables will now be tracked.')
                            self.flow=False
                            max_sync_time = max_sync_time_afterflow
                            d_flow = cuda.to_device([self.flow])
                            res = np.zeros(shape=(chain.QN.shape[0],251,8),dtype=float)
                            d_res = cuda.to_device(res)
                            ensemble_kernel.reset_chain_time[blockspergrid, threadsperblock](d_chain_time,d_write_time,self.input_data['flow_time'])
                            # new_Q = np.zeros(shape=chain.QN.shape[0],dtype=int)
                            # d_new_Q = cuda.to_device(new_Q)
                            new_Z = np.zeros(shape=(chain.QN.shape[0],chain.QN.shape[1]),dtype=int)
                            d_new_Z = cuda.to_device(new_Z)
                            temp_Z = np.zeros(shape=(chain.QN.shape[0],chain.QN.shape[1]+1),dtype=int)
                            d_temp_Z = cuda.to_device(temp_Z)
                    
                    if not self.flow and self.turn_flow_off:
                        if x_sync==num_time_syncs:
                            next_sync_time = self.input_data['sim_time'] - self.input_data['flow_time']
                        else:
                            next_sync_time = (x_sync-num_time_syncs_flow)*max_sync_time

                    
                    #initialize flags for chain sync (if chain times reach the sync time, flag goes up)
                    reach_flag_all = False
                    sum_reach_flags = 0
                    ensemble_kernel.reset_chain_flag[blockspergrid,threadsperblock](d_reach_flag)
                    
                    while not reach_flag_all:
                        
                        #calculate probabilities for entangled strand of a chain (create, destroy, or shuffle)
                        ensemble_kernel.calc_strand_prob[dimGrid, dimBlock](d_Z,d_QN,d_flow,d_tdt,d_kappa,d_tau_CD,d_shift_probs,
                                                                              d_CDflag,d_CD_create_prefact,d_beta,d_NK)
                        
                        #calculate probabilities for chain ends
                        ensemble_kernel.calc_chainends_prob[blockspergrid, threadsperblock](d_Z, d_QN, d_shift_probs, d_CDflag, d_CD_create_prefact, d_beta, d_NK)

                        #control chain time and stress calculation
                        ensemble_kernel.time_control_kernel[blockspergrid, threadsperblock](d_Z,d_QN,d_QN_first,d_NK,d_chain_time,
                                                                                            d_tdt,d_res,d_calc_type,d_flow,d_flow_off,d_reach_flag,next_sync_time,
                                                                                            max_sync_time,d_write_time,d_time_resolution,self.step_count%250,postprocess)
                        
                        #find jump type and location
                        ensemble_kernel.choose_step_kernel[blockspergrid, threadsperblock](d_Z, d_shift_probs, d_sum_W_sorted, d_uniform_rand, d_rand_used, 
                                                                                            d_found_index, d_found_shift,d_add_rand, d_CDflag)
                        
                        # ensemble_kernel.choose_kernel[blockspergrid, threadsperblock](d_Z, d_shift_probs, d_sum_W_sorted, d_uniform_rand, d_rand_used, 
                        #                                                                     d_found_index, d_found_shift,d_add_rand, d_CDflag, d_NK)

                        #if flow is turned off, track fraction of new entanglements
                        if not self.flow and self.turn_flow_off:
                            ensemble_kernel.calc_new_Q_fraction[blockspergrid,threadsperblock](d_Z,d_new_Z,d_temp_Z,d_found_shift,d_found_index,d_res,d_chain_time,
                                                                                               max_sync_time,d_time_resolution,d_write_time)

                            
                        #apply jump move for each chain and update time of chain
                        ensemble_kernel.apply_step_kernel[blockspergrid,threadsperblock](d_Z, d_QN, d_QN_first, d_QN_create_SDCD,
                                                                                            d_chain_time,d_time_compensation,d_sum_W_sorted,
                                                                                            d_found_shift,d_found_index,d_reach_flag, d_tdt,
                                                                                            d_t_cr, d_new_t_cr, d_f_t, d_tau_CD, d_new_tau_CD,
                                                                                            d_rand_used, d_add_rand, d_tau_CD_used_SD,
                                                                                            d_tau_CD_used_CD,d_tau_CD_gauss_rand_SD,
                                                                                            d_tau_CD_gauss_rand_CD)
                        

                        #update step counter for arrays and array positions
                        self.step_count+=1

                        #if OTF correlator, update progress bar based on chain times
                        if (self.step_count%50==0) and (not postprocess):
                            check_time = d_write_time.copy_to_host()
                            sum_time = 0
                            for i in range(0,len(check_time)):
                                sum_time += int(check_time[i])
                            total_progress = round(sum_time/self.input_data['Nchains']/self.input_data['sim_time'],2)
                            bar(total_progress)
                    
                        #record entanglement lifetime distribution
                        if analytic==False:
                            ft = d_f_t.copy_to_host()
                            for k in range(0,self.input_data['Nchains']):
                                if ft[k] > 0.0 and ft[k] < 20:
                                    enttime_bins[math.floor(ft[k]*1000)]+=1

                       
                        #if random numbers are used (max array size is 250), change out the used values with new random numbers and advance the random seed number
                        if self.step_count % 250 == 0:
                            gpu_rand.refill_gauss_rand_tauCD[blockspergrid,threadsperblock](self.rng_states, discrete, self.input_data['Nchains'], d_tau_CD_used_SD, True, self.input_data['CD_flag'], 
                                                                                            d_tau_CD_gauss_rand_SD, d_pcd_array,d_pcd_table_eq,d_pcd_table_cr, d_pcd_table_tau)
                            if self.input_data['CD_flag'] == 1:
                                gpu_rand.refill_gauss_rand_tauCD[blockspergrid,threadsperblock](self.rng_states, discrete, self.input_data['Nchains'], d_tau_CD_used_CD, False, self.input_data['CD_flag'], 
                                                                                            d_tau_CD_gauss_rand_CD, d_pcd_array,d_pcd_table_eq,d_pcd_table_cr, d_pcd_table_tau)
                           
                            gpu_rand.refill_uniform_rand[blockspergrid,threadsperblock](self.rng_states, self.input_data['Nchains'], d_rand_used, d_uniform_rand)
                        
                            if (not postprocess) and (not self.flow):
                                correlation.update_correlator[blockspergrid,threadsperblock](250,d_res,d_D,d_D_shift,d_C,d_N,d_A,d_M,d_calc_type)
                            
                            self.step_count = 0
                        
                        #check if chains have reached sim_time or time_sync
                        if self.flow or self.turn_flow_off:
                            reach_flag_host = d_reach_flag.copy_to_host()
                            sum_reach_flags = int(np.sum(reach_flag_host)) 
                        elif (not self.flow) and (not self.turn_flow_off) and (self.step_count%250==0):
                            reach_flag_host = d_reach_flag.copy_to_host()
                            sum_reach_flags = int(np.sum(reach_flag_host))

                        #if all reach_flags are 1, sum should equal number of chains and all chains are synced
                        reach_flag_all = (sum_reach_flags == int(self.input_data['Nchains'])) 
                    
                    if postprocess:
                        #write result of all chains to file if postprocess correlator is used
                        if self.flow: #if flow, calculate flow stress tensor for each chain
                            ensemble_kernel.calc_flow_stress[blockspergrid,threadsperblock](d_Z,d_QN,d_res)

                        res_host = d_res.copy_to_host()
                        if calc_type == 1: #if G(t), write tau_xy
                            self.write_stress(x_sync,next_sync_time,res_host)
                        elif calc_type == 2: #if MSD, write CoM 
                            self.write_com(x_sync,next_sync_time,res_host)
                  
                    #if not using OTF correlator, update progress bar
                    if postprocess:
                        bar(x_sync/num_time_syncs)
            
        #SIMULATION ENDS---------------------------------------------------------------------------------------------------------------------------
        
        t1 = time.time()
        print('')
        print("Total simulation time: %.2f minutes."%((t1-t0)/60.0))
        
        #copy final chain conformations and entanglement numbers from device to host
        QN_final = d_QN.copy_to_host()
        Z_final = d_Z.copy_to_host()
        
        #calculate entanglement lifetime distribution
        if analytic == False:
            enttime_run_sum = 0
            filename = os.path.join(self.output_dir,'./f_dt_%d.txt'%self.sim_ID)
            with open(filename, 'w') as f:
                for k in range(0,len(enttime_bins)):
                    if enttime_bins[k]!=0:
                        enttime_run_sum += enttime_bins[k]
                        f.write('%d  %d\n'%(k,enttime_bins[k]))
                
        #save final distributions to file
        if self.distr:
            np.savetxt(os.path.join(self.output_dir,'Z_final_%d.txt'%self.sim_ID),Z_final,fmt='%d')
            self.save_distributions('final',QN_final,Z_final)
        
        
        if not self.flow and not self.turn_flow_off:
            if not postprocess:
                #get OTF correlator results
                C_array = d_C.copy_to_host()
                N_array = d_N.copy_to_host()

                corr_time = []
                corr_aver = []
                
                for corrLevel in range(0,S_corr):
                    if corrLevel == 0:
                        for j in range(0,p):
                            corr_time.append(j*(m**corrLevel)*self.input_data['tau_K'])
                            corr_aver.append(np.sum(C_array[:,corrLevel,j]/N_array[:,corrLevel,j])/self.input_data['Nchains'])
                    else:
                        for j in range(int(p/m),p):
                            if j*(m**corrLevel)*self.input_data['tau_K'] <= self.input_data['sim_time']:
                                corr_time.append(j*(m**corrLevel)*self.input_data['tau_K'])
                                corr_aver.append(np.sum(C_array[:,corrLevel,j]/N_array[:,corrLevel,j])/self.input_data['Nchains'])
            else:
                #read in data files for autocorrelation function and split into blocks of block_size chains (helps prevent reaching maximum memory)

                num_chain_blocks = math.ceil(self.input_data['Nchains']/block_size)
                num_times = math.ceil(self.input_data['sim_time']/self.input_data['tau_K'])+1

                #counter for initializing final array size and set the correlated times in corr_time array
                count = 0
                corr_time = [] #array to hold correlated times (log scale) 
                for corrLevel in range(0,S_corr):
                    if corrLevel == 0:
                        for j in range(0,p):
                            count+= 1
                            corr_time.append(j*(m**corrLevel)*self.input_data['tau_K'])
                    
                    else:
                        for j in range(int(p/m),p):
                            if j*(m**corrLevel)*self.input_data['tau_K'] <= self.input_data['sim_time']:
                                count += 1
                                corr_time.append(j*(m**corrLevel)*self.input_data['tau_K'])

                average_corr = np.zeros(shape=len(corr_time))
                average_error = np.zeros(shape=len(corr_time))
                
                print("Loading stress data and calculating correlation function, this may take some time...",end="",flush=True)
                for n in range(0,num_chain_blocks):
                    if n == num_chain_blocks-1:
                        num_chains = self.input_data['Nchains']
                    else:
                        num_chains = n*block_size + block_size

                    if calc_type == 1: #stress data if EQ_calc is 'stress'
                        stress_array = np.array(self.load_results(self.stress_output,block_num=n,block_size=block_size,num_chains=num_chains)) 
                        rawdata = np.array([stress_array])
                        
                    elif calc_type == 2: #CoM data if EQ_calc is 'msd' (this is a little messy, since each dimension is stored separately)
                        com_array_x = np.array(self.load_results(self.com_output_x,block_num=n,block_size=block_size,num_chains=num_chains)) #load center of mass in x file
                        com_array_y = np.array(self.load_results(self.com_output_y,block_num=n,block_size=block_size,num_chains=num_chains)) #load center of mass in y file
                        com_array_z = np.array(self.load_results(self.com_output_z,block_num=n,block_size=block_size,num_chains=num_chains)) #load center of mass in z file

                        rawdata = np.array([com_array_x,com_array_y,com_array_z])

                    #initialize arrays for output
                    data_corr = np.zeros(shape=(block_size,count,2),dtype=float) #hold average chain stress/com correlations 
                    corr_array =np.zeros(shape=(num_times,block_size),dtype=float) #array to store correlation values for averaging each chain inside kernel

                    #transfer to device
                    d_data_corr = cuda.to_device(data_corr)
                    d_rawdata = cuda.to_device(rawdata)
                    d_corr_array = cuda.to_device(corr_array)

                    #run the block transformation and calculate correlation with error
                    correlation.calc_corr[blockspergrid,threadsperblock](d_rawdata,calc_type,S_corr,d_data_corr,d_corr_array)

                    #copy results to host and calculate average over all chains 
                    data_corr_host = d_data_corr.copy_to_host()
                    
                    #running sum of time correlation averages for chains in block
                    average_corr += np.sum(data_corr_host[:,:,0],axis=0)
                    average_error += np.sum(data_corr_host[:,:,1],axis=0)

                #divide running sum by number of chains
                corr_aver = average_corr/self.input_data['Nchains']  #average correlation
                corr_error = average_error/(self.input_data['Nchains']*np.sqrt(self.input_data['Nchains'])) #average error from correlation
            
            #write equilibrium calculation results to file
            if calc_type == 1:
                #make combined result array and write to file
                with open(os.path.join(self.output_dir,'Gt_result_%d.txt'%self.sim_ID), "w") as f:
                    if not postprocess:
                        f.write('Time, G(t)\n')
                        for m in range(0,len(corr_time)):
                                f.write("%d, %.4f \n"%(corr_time[m],corr_aver[m]))
                    else:
                        f.write('Time, G(t), Error\n')
                        for m in range(0,len(corr_time)):
                            f.write("%d, %.4f, %.4f \n"%(corr_time[m],corr_aver[m],corr_error[m]))
                if postprocess:
                    print('Done.')
                print('G(t) results written to Gt_result_%d.txt'%self.sim_ID)

            if calc_type == 2:
                #make combined result array and write to file
                with open(os.path.join(self.output_dir,'MSD_result_%d.txt'%self.sim_ID), "w") as f:
                    if not postprocess:
                        f.write('Time, MSD\n')
                        for m in range(0,len(corr_time)):
                            f.write("%d, %.4f \n"%(corr_time[m],corr_aver[m]))
                    else:
                        f.write('Time, MSD, Error')
                        for m in range(0,len(corr_time)):
                            f.write("%d, %.4f, %.4f \n"%(corr_time[m],corr_aver[m],corr_error[m]))
                if postprocess:
                    print('Done.')
                print('MSD results written to MSD_result_%d.txt'%self.sim_ID)

            #if postprocess correlator used, calculate time after calculation is finished
            if postprocess:
                t2 = time.time()
                print("Total computational time: %.2f minutes."%((t2-t0)/60.0))
            
        if self.fit:
            print("")
            print("Fitting G(t)...")
            gt_fit = CURVE_FIT(os.path.join(self.output_dir,'Gt_result_%d.txt'%self.sim_ID),os.path.join(self.output_dir,'fit_results'))
            gt_fit.fit()
            print("")
            print("G* predictions saved to file.")



if __name__ == "__main__":
    run_dsm = FSM_LINEAR(1,0,None)
    run_dsm.main()


