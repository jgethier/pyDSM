import os
import sys
import time
import numpy as np
from numba import cuda
import yaml
import math
from core.chain import ensemble_chains
from core.pcd_tau import p_cd, p_cd_linear
import core.random_gen as rng
import core.ensemble_kernel as ensemble_kernel
import core.gpu_random as gpu_rand 
import core.stress_acf as stress_acf


class FSM_LINEAR(object):

    def __init__(self,sim_ID,device_ID):
        
        #simulation ID from run argument (>>python gpu_dsm sim_ID)
        self.sim_ID = sim_ID
        
        #read in input file and parameters
        if os.path.isfile('input.yaml'):
            with open('input.yaml') as f:
                self.input_data = yaml.load(f, Loader=yaml.FullLoader)
            
            #get results path ready
            if not os.path.exists('./DSM_results/'):
                os.mkdir('DSM_results')
            self.output_path = './DSM_results/'
                
        else:
            sys.exit("No input file found.")

        #set seed number based on num argument
        if sim_ID != 0:
            SEED = sim_ID*self.input_data['Nchains']
            print("Simulation ID: %d"%(sim_ID))
            print("Using %d as a seed for the random number generator"%(sim_ID))
        else:
            SEED = self.input_data['Nchains']

        #initialize random number generator
        rng.initialize_generator(SEED)
        self.rs = np.random.RandomState(SEED)
        self.SEED=SEED
        num_devices = len(cuda.gpus)
        if num_devices == 0:
            sys.exit("No GPU found.")

        #select gpu device
        cuda.select_device(device_ID)
        
        return


    def save_distributions(self,filename,QN,Z):
        '''
        Function to save Q distributions to file
        Inputs: filename - name of file with extension (.txt, .dat, etc.)
                QN - distribution of strand orientations and number of Kuhn steps in cluster
                Z - number of entanglements in each chain
        '''
        Q = []
        for i in range(0,QN.shape[0]):
            for j in range(1,int(Z[i])-1):
                x2 = QN[i,j,0]*QN[i,j,0]
                y2 = QN[i,j,1]*QN[i,j,1]
                z2 = QN[i,j,2]*QN[i,j,2]
                Q.append(np.sqrt(x2 + y2 + z2))

                if np.sqrt(x2 + y2 + z2) == 0.0:
                    print('Chain %d'%(i))

        Q_sorted = np.sort(Q)

        np.savetxt(filename, Q_sorted)

        return


    def progbar(self, curr, total, full_progbar):
        '''
        Function to update simulation progress bar
        '''
        frac = curr/total
        filled_progbar = round(frac*full_progbar)
        print('\r', '#'*filled_progbar + '-'*(full_progbar-filled_progbar), '[{:>7.2%}]'.format(frac), end='')
        
        return

    
    def calc_avg_stress(self,num_sync,time,stress_array):
        '''
        Output the stress of all chains in ensemble
        Inputs: num_sync - sync number for simulation
                time - simulation time
                stress_array - array of stress values to write
        Outputs: stress over time for each chain in stress.txt file
        '''
        
        #avg_stress = np.mean(stress_array,axis=2) #no longer used for averaging
        #stdev_stress = np.std(stress_array,axis=2) #no longer used for averaging
        
        #get output stress file ready
        self.stress_output = os.path.join(self.output_path,'stress_%d.txt'%self.sim_ID)
        
        if num_sync == 1:
            with open(self.stress_output,'w') as f:
                f.write('time, stress_1, stress_2, ..., stress_nchains\n')     
            self.old_sync_time = 0
            time_index = 0
        
        else:
            time_index = 1
            
        time_resolution = self.input_data['tau_K']
        time_array = np.arange(self.old_sync_time,time+0.5,time_resolution)
        time_array = np.reshape(time_array[time_index:],(1,len(time_array[time_index:])))
        stress_array = np.reshape(stress_array[:,:,0],(self.input_data['Nchains'],len(stress_array[0,:,0])))
        
        #combined = np.hstack((time_array.T,avg_stress, stdev_stress))
        combined = np.hstack((time_array.T, stress_array.T))
        
        with open(self.stress_output, "a") as f:
            np.savetxt(f, combined, delimiter=',', fmt='%.4f')

        self.old_sync_time = time
        
        return 

            
    def main(self):
           
            

        #if CD_flag is set (constraint dynamics is on), set probability of CD parameters with analytic expression
        if self.input_data['CD_flag']==1 and self.input_data['architecture']=='linear':
            
            #initialize constants for constraint dynamics probability calculation
            pcd = p_cd_linear(self.input_data['Nc'],self.input_data['beta'])
            d_CD_create_prefact = cuda.to_device([pcd.W_CD_destroy_aver()/self.input_data['beta']])
            
            #array of constants used for tau_CD probability calculation
            pcd_array = np.array([pcd.g, #0
                        pcd.alpha, #1
                        pcd.tau_0, #2
                        pcd.tau_max, #3
                        pcd.tau_D, #4
                        1.0/pcd.tau_D, #5, inverse tau_D 
                        1.0*pcd.tau_alpha/pcd.At, #6, d_At
                        pcd.tau_0**pcd.alpha, #7, d_Dt
                        -1.0/pcd.alpha, #8, d_Ct
                        pcd.normdt*pcd.tau_alpha/pcd.Adt, #9, d_Adt
                        pcd.Bdt/pcd.normdt, #10, d_Bdt 
                        -1.0/(pcd.alpha - 1.0), #11, d_Cdt
                        pcd.tau_0**(pcd.alpha - 1.0) #12, d_Ddt
                        ])
            
            discrete = False #set discrete variable to False to implement analytic expression for p(tau_CD) probability
            
            #set discrete modes to 0 (this may be better handled since it's waste of memory space)
            pcd_table_tau = np.zeros(1)
            pcd_table_cr = np.zeros(1)
            pcd_table_eq = np.zeros(1)

        #if not linear chains, use discrete fit for tau_CD
        elif self.input_data['CD_flag']==1 and self.input_data['architecture']!='linear':

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

            pcd_table_cr = np.zeros(pcd.nmodes)
            pcd_table_eq = np.zeros(pcd.nmodes)
            pcd_table_tau = np.zeros(pcd.nmodes)
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
        z_max = self.input_data['Nc']
        
        #initialize chains
        for m in range(0,self.input_data['Nchains']):
            chain.chain_init(self.input_data['Nc'],z_max=z_max,pcd=pcd)

        print('Done.')

        #reshape arrays for CUDA kernels
        chain.QN = chain.QN.reshape(self.input_data['Nchains'],self.input_data['Nc'],4)
        chain.tau_CD = chain.tau_CD.reshape(self.input_data['Nchains'],self.input_data['Nc'])
        chain.Z = chain.Z.reshape(self.input_data['Nchains'])

        #save initial chain conformation distributions
        self.save_distributions(os.path.join(self.output_path,'distr_Q_initial_%d.txt'%self.sim_ID),chain.QN,chain.Z)

        #save initial Z distribution to file
        np.savetxt(os.path.join(self.output_path,'Z_initial_%d.txt'%self.sim_ID),chain.Z,fmt='%d')

        #initialize arrays for finding jump index
        found_index = np.zeros(shape=(chain.QN.shape[0]))
        found_shift = np.zeros(shape=(chain.QN.shape[0]))
        sum_W_sorted = np.zeros(shape=(chain.QN.shape[0]),dtype=int)
        shift_probs = np.zeros(shape=(chain.QN.shape[0],chain.QN.shape[1]+1,chain.QN.shape[0]),dtype=int)
        
        #intitialize arrays for random numbers used
        rand_used = np.zeros(shape=chain.QN.shape[0],dtype=int)
        tau_CD_used_SD = np.zeros(shape=chain.QN.shape[0],dtype=int)
        tau_CD_used_CD = np.zeros(shape=chain.QN.shape[0],dtype=int)
        tau_CD_gauss_rand_SD = np.zeros(shape=(chain.QN.shape[0],250,4),dtype=float)
        tau_CD_gauss_rand_CD = np.zeros(shape=(chain.QN.shape[0],250,4),dtype=float)
        uniform_rand = np.zeros(shape=(chain.QN.shape[0],256),dtype=float)
        add_rand = np.zeros(shape=(chain.QN.shape[0]),dtype=int)
        
        #fill random number arrays (random seed number is always advanced by +1)
        d_tau_CD_gauss_rand_SD=cuda.to_device(tau_CD_gauss_rand_SD)
        d_tau_CD_gauss_rand_CD=cuda.to_device(tau_CD_gauss_rand_CD)
        d_uniform_rand=cuda.to_device(uniform_rand)

        random_state = 1
        gpu_rand.gpu_tauCD_gauss_rand(seed=random_state, discrete=discrete, nchains=self.input_data['Nchains'], count=250, SDtoggle=True,
                                      CDflag=self.input_data['CD_flag'],gauss_rand=d_tau_CD_gauss_rand_SD, pcd_array = pcd_array, pcd_table_eq=pcd_table_eq, 
                                      pcd_table_cr=pcd_table_cr,pcd_table_tau=pcd_table_tau,refill=False)

        #if CD flag is 1 (constraint dynamics is on), fill random gaussian array for new strands created by CD
        if self.input_data['CD_flag'] == 1:
            random_state += 1
            gpu_rand.gpu_tauCD_gauss_rand(seed=random_state, discrete=discrete, nchains=self.input_data['Nchains'], count=250, SDtoggle=False, 
                                          CDflag=self.input_data['CD_flag'],gauss_rand=d_tau_CD_gauss_rand_CD, pcd_array = pcd_array, pcd_table_eq=pcd_table_eq, 
                                          pcd_table_cr=pcd_table_cr,pcd_table_tau=pcd_table_tau,refill=False)
        
        #advance random seed number and fill uniform random array
        random_state += 1
        gpu_rand.gpu_uniform_rand(seed=random_state, nchains=self.input_data['Nchains'], count=250, uniform_rand=d_uniform_rand,refill=False)
        
        
        #initialize arrays for chain time and entanglement lifetime
        chain_time = np.zeros(shape=chain.Z.shape[0],dtype=float)
        time_resolution = self.input_data['tau_K']
        time_compensation = np.zeros(shape=chain.Z.shape[0],dtype=float)
        stall_flag = np.zeros(shape=chain.Z.shape[0],dtype=int)
        tdt = np.zeros(shape=chain.Z.shape[0],dtype=float)
        t_cr = np.zeros(shape=(chain.QN.shape[0],chain.QN.shape[1]),dtype=float)
        f_t = np.zeros(shape=chain.QN.shape[0],dtype=float)
        write_time = np.zeros(shape=chain.QN.shape[0],dtype=int)

        #move arrays to device
        d_QN = cuda.to_device(chain.QN)
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
        d_write_time = cuda.to_device(write_time)
        d_stall_flag = cuda.to_device(stall_flag)
        d_tdt = cuda.to_device(tdt)
        d_rand_used = cuda.to_device(rand_used)
        d_tau_CD_used_SD=cuda.to_device(tau_CD_used_SD)
        d_tau_CD_used_CD=cuda.to_device(tau_CD_used_CD)
        

        #set timesteps and begin simulation
        simulation_time = self.input_data['sim_time']
        enttime_bins = np.zeros(shape=(20000),dtype=int)
        
        #initialize some time constants used for simulation and calculate number of time syncs based on sync time resolution
        step_count = 0
        max_sync_time = 100 #arbitrary, just setting every chain to sync at t = 100
        num_time_syncs = int(math.floor(self.input_data['sim_time'] / max_sync_time) + 1)
        
        #set grid dimensions
        dimBlock = (32,8)
        dimGrid_x = (self.input_data['Nchains']+dimBlock[0]-1)//dimBlock[0]
        dimGrid_y = (self.input_data['Nc']+dimBlock[1]-1)//dimBlock[1]
        dimGrid = (dimGrid_x,dimGrid_y)
        
        #flattened grid dimensions
        threadsperblock = 256
        blockspergrid = (chain.QN.shape[0] + threadsperblock - 1)//threadsperblock
        
        #set max sync time to simulation time if simulation time is less than sync time
        if max_sync_time > simulation_time:
            max_sync_time = simulation_time
        
        #initialize stress array
        stress = np.zeros(shape=(chain.QN.shape[0],int(max_sync_time/time_resolution)+1,3),dtype=float)
        d_stress = cuda.to_device(stress)
        
        #timer start
        t0 = time.time()
        
        #initialize cuda streams
        stream1 = cuda.stream()
        stream2 = cuda.stream()
        stream3 = cuda.stream()
        
        
        #SIMULATION STARTS -------------------------------------------------------------------------------------------------------------------------------
        
        #start loop over number of time syncs
        for x_sync in range(1,num_time_syncs+1):
            
            if x_sync == num_time_syncs:
                next_sync_time = simulation_time
            else:
                next_sync_time = x_sync*max_sync_time
            
            #initialize flags for chain sync (if chain times reach the sync time, flag goes up)
            reach_flag_all = False
            reach_flag = np.zeros(shape=self.input_data['Nchains'],dtype=int)
            d_reach_flag = cuda.to_device(reach_flag,stream=stream3)  
            
            
            while reach_flag_all != True:
                
                #calculate Kun step shuffle probabilities
                ensemble_kernel.calc_probs_shuffle[dimGrid, dimBlock, stream1](d_Z,d_QN,d_tau_CD,d_shift_probs,self.input_data['CD_flag'],d_CD_create_prefact)

                #calculate probabilities at chain ends (create, destroy, or shuffle at ends)
                ensemble_kernel.calc_probs_chainends[blockspergrid, threadsperblock, stream2](d_Z,d_QN,d_shift_probs,self.input_data['CD_flag'],
                                                                                             d_CD_create_prefact,self.input_data['beta'])
                stream2.synchronize()

                #control chain time and stress calculation
                ensemble_kernel.chain_control_kernel[blockspergrid, threadsperblock, stream3](d_Z,d_QN,d_chain_time,d_stress,d_reach_flag,next_sync_time,
                                                                                              max_sync_time,d_write_time,time_resolution)

                #find jump type and location
                ensemble_kernel.scan_kernel[blockspergrid, threadsperblock,stream1](d_Z, d_shift_probs, d_sum_W_sorted, d_uniform_rand, d_rand_used, 
                                                                                     d_found_index, d_found_shift,d_add_rand, self.input_data['CD_flag'])
                
                #if chain has reached its sync time, set reach_flag of chain to 1
                reach_flag_host = d_reach_flag.copy_to_host(stream=stream3)
                
                stream1.synchronize()
                stream3.synchronize()

                #find which chains create slip link (from SD or CD) so that shifted arrays can be stored
                found_shift_SDCD = d_found_shift.copy_to_host(stream=stream1)
                shared_size = np.argwhere((found_shift_SDCD==6) | (found_shift_SDCD==4))
                create_SDCD_chains = np.ones(shape=(chain.QN.shape[0]),dtype=int)*-1
                for j in range(0,len(shared_size)):
                    create_SDCD_chains[shared_size[j][0]] = j
                
################################DEBUGGING CODE#######################################               
#                 try:
#                     CD_create = np.argwhere(found_shift_SD==4)[0]
#                 except:
#                     CD_create = None
#                 if CD_create is not None:
#                     check_index = d_found_index.copy_to_host()
#                     check_Z = d_Z.copy_to_host()
#                     if check_index[CD_create] != check_Z[CD_create]-1 and check_Z[CD_create]!=0:
#                         print(step_count)
#                         print(num_refills)
#                         print(CD_create)
#                         print(check_index[CD_create])
                
#                 if step_count==62 and x_sync==1 and num_refills==0:
#                     initial_QN = d_QN.copy_to_host()
#                     print(initial_QN[445])
#                     stress_calc = d_stress.copy_to_host()
#                     print(stress_calc[1,:,2])
#                     for j in range(0,len(stress_calc[1,1,:])):
#                         if stress_calc[1,1,j]< -15:
#                             print(stress_calc[1,:,j])
#                             print(initial_QN[j])
#                     avg_stress = np.mean(stress_calc,axis=2)
#                     print(avg_stress)
#                     shift_prob = d_shift_probs.copy_to_host()
#                     print(shift_prob[445])
#                     print(found_shift_SD[445])
#                     check_index = d_found_index.copy_to_host()
#                     print(check_index[445])
#                     Z = d_Z.copy_to_host()
#                     print(Z[445])
#                     tcd = d_tau_CD.copy_to_host()
#                     print(tcd[445])
#                     tcr = d_t_cr.copy_to_host()
#                     print(tcr[445])
#####################################################################################
                
                #initialize arrays for which chains create a slip link from sliding and/or constraint dynamics
                QN_create_SDCD = np.zeros(shape=(len(shared_size),chain.QN.shape[1]+1,4))
                new_t_cr = np.zeros(shape=(len(shared_size),chain.QN.shape[1]))
                new_tau_CD = np.zeros(shape=(len(shared_size),chain.QN.shape[1]))
                
                #move arrays to device
                d_QN_create_SDCD = cuda.to_device(QN_create_SDCD,stream=stream1)
                d_create_SDCD_chains = cuda.to_device(create_SDCD_chains,stream=stream1)
                d_new_t_cr = cuda.to_device(new_t_cr,stream=stream1)
                d_new_tau_CD = cuda.to_device(new_tau_CD,stream=stream1)
                
                #apply jump move for each chain and update time of chain
                ensemble_kernel.chain_kernel[blockspergrid,threadsperblock,stream1](d_Z, d_QN, d_create_SDCD_chains, d_QN_create_SDCD,
                                                                                    d_chain_time,d_time_compensation,
                                                                                    d_reach_flag, d_found_shift, d_found_index,d_tdt, d_sum_W_sorted,
                                                                                    d_t_cr, d_new_t_cr, d_f_t, d_tau_CD, d_new_tau_CD,
                                                                                    d_rand_used, d_add_rand, d_tau_CD_used_SD,
                                                                                    d_tau_CD_used_CD,d_tau_CD_gauss_rand_SD,
                                                                                    d_tau_CD_gauss_rand_CD)
                
                stream1.synchronize()
                stream3.synchronize()
                
                
##################DEBUGGING CODE######################################
#                 if num_refills==0 and step_count==62:
#                     check_QN = d_QN.copy_to_host()
#                     print(check_QN[445])
#                     Z = d_Z.copy_to_host()
#                     print(Z[445])
#                     tcd = d_tau_CD.copy_to_host()
#                     print(tcd[445])
#                     tcr = d_t_cr.copy_to_host()
#                     print(tcr[445])
#################################################################                    
                    
                
                #check if all chains reached time for sync
                sum_reach_flags = 0 
                for flag in reach_flag_host:
                    sum_reach_flags += int(flag) #sum the reach flags equal to 1 or 0
              
                
                #if all reach_flags are 1, sum should equal number of chains and all chains are synced
                reach_flag_all = (sum_reach_flags == int(self.input_data['Nchains'])) 
                
                #update step counter for arrays and array positions
                step_count+=1

                #record entanglement lifetime distribution
                ft = d_f_t.copy_to_host(stream=stream1)
                stream1.synchronize()
                for k in range(0,self.input_data['Nchains']):
                    if ft[k] > 0.0 and ft[k] < 20:
                        enttime_bins[math.floor(ft[k]*1000)]+=1


                #if random numbers are used (max array size is 250), change out the used values with new random numbers and advance the random seed number
                if step_count % 250 == 0:
                    
                    random_state += 1
                    gpu_rand.gpu_tauCD_gauss_rand(seed=random_state, discrete=discrete, nchains=self.input_data['Nchains'], count=d_tau_CD_used_SD,
                                                  SDtoggle=True,CDflag=self.input_data['CD_flag'], gauss_rand=d_tau_CD_gauss_rand_SD, pcd_array=pcd_array,
                                                  pcd_table_eq=pcd_table_eq,pcd_table_cr=pcd_table_cr, pcd_table_tau=pcd_table_tau, refill=True)

                    if self.input_data['CD_flag'] == 1:
                        random_state += 1
                        gpu_rand.gpu_tauCD_gauss_rand(seed=random_state, discrete=discrete, nchains=self.input_data['Nchains'], count=d_tau_CD_used_CD, 
                                                      SDtoggle=False,CDflag=self.input_data['CD_flag'], gauss_rand=d_tau_CD_gauss_rand_CD, pcd_array=pcd_array,
                                                      pcd_table_eq=pcd_table_eq,pcd_table_cr=pcd_table_cr, pcd_table_tau=pcd_table_tau, refill=True)
                    random_state += 1
                    gpu_rand.gpu_uniform_rand(seed=random_state, nchains=self.input_data['Nchains'], count=d_rand_used, uniform_rand=d_uniform_rand,refill=True)
                    
                    step_count = 0
            
            
            #array handling when num_time_syncs are > 1 (this is to remove the t=0 value after the first time sync)
            stress_host = d_stress.copy_to_host()
            if x_sync == num_time_syncs and x_sync!=1:
                last = int((simulation_time - (x_sync-1)*max_sync_time)/time_resolution)+1
                stress_host = stress_host[:,:last,:]
            if x_sync>1:
                stress_host = stress_host[:,1:,:]

            #calulcate ensemble average and write to file
            self.calc_avg_stress(x_sync,next_sync_time,stress_host)
            
            #update progress bar
            self.progbar(x_sync,num_time_syncs,20)
            
        #SIMULATION ENDS---------------------------------------------------------------------------------------------------------------------------
            
        t1 = time.time()
        print("Total simulation time: %.2f seconds"%(t1-t0))
        
        #copy final chain conformations and entanglement numbers from device to host
        QN_final = d_QN.copy_to_host()
        Z_final = d_Z.copy_to_host()
        
        #calculate entanglement lifetime distribution
        enttime_run_sum = 0
        filename = os.path.join(self.output_path,'./f_dt_%d.txt'%self.sim_ID)
        with open(filename, 'w') as f:
            for k in range(0,len(enttime_bins)):
                if enttime_bins[k]!=0:
                    enttime_run_sum += enttime_bins[k]
                    f.write('%d  %d\n'%(k,enttime_bins[k]))
                
        #save final distributions to file
        np.savetxt(os.path.join(self.output_path,'Z_final_%d.txt'%self.sim_ID),Z_final,fmt='%d')
        self.save_distributions(os.path.join(self.output_path,'distr_Q_final_%d.txt'%self.sim_ID),QN_final,Z_final)
        
        
        #read in stress files for stress autocorrelation function
        print("Loading stress data for G(t) calculation...",end="",flush=True)
        
        stress_array = np.loadtxt(self.stress_output,delimiter=',',skiprows=1) #load stress file
        time_array = stress_array[0:,0] # tau_K values
        sampf = 1/(time_array[1]-time_array[0]) #normalize time by smallest resolution
        stress = np.array(np.reshape(stress_array[0:,1:],(len(time_array),self.input_data['Nchains']))) #reshape stress array
        print("Done.")
        
        #calculate stress correlation of each chain and average
        print("Calculating G(t), this may take some time...",end="",flush=True)
        
        #parameters for block transformation
        p = 8
        m = 2
        uplim=int(math.floor(np.log(len(stress[0:,0])/p)/np.log(m)))
        
        #counter for initializing final array size
        count = 0
        for k in range(1,p*m):
            count+=1
        for l in range(1,int(uplim)):
            for j in range(p*m**l,p*m**(l+1),m**l):
                count+=1
        
        #initialize arrays for output
        time_corr = np.zeros(count,dtype=float) #array to hold correlated times (log scale) 
        stress_corr = np.zeros(shape=(self.input_data['Nchains'],count,2),dtype=float) #hold average chain stress correlations 
        corr_array =np.zeros(len(time_array),dtype=float) #array to store correlation values for averaging (single chain)
        
        #transfer to device
        d_time = cuda.to_device(time_corr)
        d_stress_corr = cuda.to_device(stress_corr)
        d_stress = cuda.to_device(stress)
        d_corr_array = cuda.to_device(corr_array)
        
        #run the block transformation and calculate stress autocorrelation with error
        stress_acf.stress_sample[blockspergrid,threadsperblock](d_stress,sampf,uplim,d_time,d_stress_corr,d_corr_array)
        
        #copy results to host and calculate average over all chains
        time_array = d_time.copy_to_host().astype(int) 
        stress_corr_final = d_stress_corr.copy_to_host()
        average_corr = np.mean(stress_corr_final[:,:,0],axis=0) #mean stress correlation
        average_err = np.sum(stress_corr_final[:,:,1],axis=0)/(self.input_data['Nchains']*np.sqrt(self.input_data['Nchains'])) #error propagation

        #make combined result array and write to file
        combined = np.hstack((time_array, average_corr, average_err))
        with open('./DSM_results/Gt_result_%d.txt'%self.sim_ID, "w") as f:
            f.write('time, G(t), Error\n')
            for m in range(0,len(time_array)):
                    f.write("%d, %.4f, %.4f \n"%(time_array[m],average_corr[m],average_err[m]))
            
        print('Done.')
        
if __name__ == "__main__":
    run_dsm = FSM_LINEAR(1)
    run_dsm.main()


