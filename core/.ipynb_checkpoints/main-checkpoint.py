import os
import sys
import time
import numpy as np
from numba import cuda
import yaml
import math
from core.chain import ensemble_chains
from core.pcd_tau import p_cd
import core.random_gen as rng
import core.ensemble_kernel as ensemble_kernel
import core.gpu_random as gpu_rand 


class FSM_LINEAR(object):

    def __init__(self,sim_ID,device_ID):

        self.sim_ID = sim_ID
        
        if os.path.isfile('input.yaml'):
            with open('input.yaml') as f:
                self.input_data = yaml.load(f, Loader=yaml.FullLoader)
            
            #get output stress file ready
            with open('stress_%d.txt'%self.sim_ID,'w') as f:
                f.write('time tau_xy tau_yz tau_xz sigma_xy sigma_yz sigma_xz\n')
        else:
            sys.exit("No input file found.")

        if sim_ID != 0:
            SEED = sim_ID*self.input_data['Nchains']
            print("Simulation ID: %d"%(sim_ID))
            print("Using %d as a seed for the random number generator"%(sim_ID))
        else:
            SEED = self.input_data['Nchains']

        rng.initialize_generator(SEED)
        self.rs = np.random.RandomState(SEED)
        self.SEED=SEED
        num_devices = len(cuda.gpus)
        if num_devices == 0:
            sys.exit("No GPU found.")

        cuda.select_device(device_ID)
        
        return
        
        

    def save3Darray(self,filename,data,fmt):
        with open(filename, 'w') as outfile:
            # I'm writing a header here just for the sake of readability
            # Any line starting with "#" will be ignored by numpy.loadtxt
            outfile.write('# Array shape: {0}\n'.format(data.shape))

            for data_slice in data:
                np.savetxt(outfile, data_slice, fmt=fmt)#, fmt='%-7.2f')
                outfile.write('#New chain\n')
                
        return


    def save_distributions(self,filename,QN,Z):

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
        frac = curr/total
        filled_progbar = round(frac*full_progbar)
        print('\r', '#'*filled_progbar + '-'*(full_progbar-filled_progbar), '[{:>7.2%}]'.format(frac), end='')
        
        return

    
    def calc_avg_stress(self,num_sync,time,stress_array):
        
        avg_stress = np.mean(stress_array,axis=2)
        stdev_stress = np.std(stress_array,axis=2)
        
        if num_sync == 1:
            self.old_sync_time = 0
            time_index = 0
        else:
            time_index = 1
            
        time_resolution = self.input_data['tau_K']
        time_array = np.arange(self.old_sync_time,time+0.5,time_resolution)
        time_array = np.reshape(time_array[time_index:],(1,len(time_array[time_index:])))
        
        combined = np.hstack((time_array.T,avg_stress, stdev_stress))
        
        with open("stress_%d.txt"%self.sim_ID, "a") as f:
            np.savetxt(f, combined,fmt='%.4f')

        self.old_sync_time = time
        
        return 

            
    def main(self):

        #if multi-mode maxwell fit param file exists, read and set modes
        if os.path.isfile("pcd_MMM.txt"):
            with open("pcd_MMM.txt",'r') as pcd_file:
                nmodes = int(pcd_file.readline())

                lines = pcd_file.readlines()
                tauArr = [float(line.split()[0]) for line in lines]
                gArr = [float(line.split()[1]) for line in lines]
           
            

        #if CD_flag is set (constraint dynamics is on), set probability of CD parameters
        if self.input_data['CD_flag']==1:
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
        
        #else, set probability factors to 0
        else:
            pcd=None
            pcd_table_tau = np.zeros(1)
            pcd_table_cr = np.zeros(1)
            pcd_table_eq = np.zeros(1)
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
        self.save_distributions('distr_Q_initial_%d.dat'%self.sim_ID,chain.QN,chain.Z)

        #self.save3Darray('QN_initial.txt',chain.QN)
        np.savetxt('Z_initial_%d.txt'%self.sim_ID,chain.Z,fmt='%d')

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
        random_state = 1
        gpu_rand.gpu_tauCD_gauss_rand(seed=random_state, nchains=self.input_data['Nchains'], count=250, SDtoggle=True, CDflag=self.input_data['CD_flag'],
                                      gauss_rand=tau_CD_gauss_rand_SD, pcd_table_eq=pcd_table_eq, pcd_table_cr=pcd_table_cr,
                                      pcd_table_tau=pcd_table_tau,refill=False)
        
        if self.input_data['CD_flag'] == 1:
            random_state += 1
            gpu_rand.gpu_tauCD_gauss_rand(seed=random_state, nchains=self.input_data['Nchains'], count=250, SDtoggle=False, CDflag=self.input_data['CD_flag'],
                                          gauss_rand=tau_CD_gauss_rand_CD, pcd_table_eq=pcd_table_eq, pcd_table_cr=pcd_table_cr,
                                          pcd_table_tau=pcd_table_tau,refill=False)
        
        random_state += 1
        gpu_rand.gpu_uniform_rand(seed=random_state, nchains=self.input_data['Nchains'], count=250, uniform_rand=uniform_rand,refill=False)
        
        
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
        d_uniform_rand=cuda.to_device(uniform_rand)
        d_tau_CD_gauss_rand_SD=cuda.to_device(tau_CD_gauss_rand_SD)
        d_tau_CD_gauss_rand_CD=cuda.to_device(tau_CD_gauss_rand_CD)
        d_tau_CD_used_SD=cuda.to_device(tau_CD_used_SD)
        d_tau_CD_used_CD=cuda.to_device(tau_CD_used_CD)
        

        #set timesteps and begin simulation
        simulation_time = self.input_data['sim_time']
        enttime_bins = np.zeros(shape=(20000),dtype=int)
        
        #initialize some time constants used for simulation
        step_count = 0
        max_sync_time = 100 #arbitrary, just setting every chain to sync at t = 100
        num_time_syncs = int(math.floor(self.input_data['sim_time'] / max_sync_time) + 1)
        
        #set grid dimensions
        dimBlock = (8,32)
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
        stress = np.zeros(shape=(int(max_sync_time/time_resolution)+1,3,chain.QN.shape[0]),dtype=float)
        d_stress = cuda.to_device(stress)
        
        #timer start
        t0 = time.time()
        
        #initialize cuda streams
        stream1 = cuda.stream()
        stream2 = cuda.stream()
        stream3 = cuda.stream()
        
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
            
            
            #SIMULATION STARTS ----------------------------------------------------------------------------------------------------------------
            while reach_flag_all != True:
                
                #calculate shuffle probabilities
                ensemble_kernel.calc_probs_shuffle[dimGrid, dimBlock, stream1](d_Z,d_QN,d_tau_CD,d_shift_probs,self.input_data['CD_flag'],d_CD_create_prefact)

                #calculate probabilities at chain ends
                ensemble_kernel.calc_probs_chainends[blockspergrid, threadsperblock, stream2](d_Z,d_QN,d_shift_probs,self.input_data['CD_flag'],
                                                                                             d_CD_create_prefact,self.input_data['beta'])
                stream2.synchronize()

                #control chain time 
                ensemble_kernel.chain_control_kernel[blockspergrid, threadsperblock, stream3](d_Z,d_QN,d_chain_time,d_stress,d_reach_flag,next_sync_time,
                                                                                              max_sync_time,d_write_time,time_resolution)

                #find jump type and location
                ensemble_kernel.scan_kernel[blockspergrid, threadsperblock,stream1](d_Z, d_shift_probs, d_sum_W_sorted, d_uniform_rand, d_rand_used, 
                                                                                     d_found_index, d_found_shift,d_add_rand, self.input_data['CD_flag'])
                
                reach_flag_host = d_reach_flag.copy_to_host(stream=stream3)
                
                stream1.synchronize()
                stream3.synchronize()

                #find which chains create slip link (from SD or CD) so that shifted arrays can be stored
                found_shift_SDCD = d_found_shift.copy_to_host(stream=stream1)
                shared_size = np.argwhere((found_shift_SDCD==6) | (found_shift_SDCD==4))
                create_SDCD_chains = np.ones(shape=(chain.QN.shape[0]),dtype=int)*-1
                for j in range(0,len(shared_size)):
                    create_SDCD_chains[shared_size[j][0]] = j
                
################################DEBUGGING#######################################               
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
#################################################################################
                
                #initialize array for which chains create a slip link from sliding dynamics at beginning of chain
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
                
                
##################DEBUGGING######################################
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


                #if random numbers are used, refill array with new random numbers and advance the random seed number
                if step_count % 250 == 0:
                    
                    random_state += 1
                    
                    gpu_rand.gpu_tauCD_gauss_rand(seed=random_state, nchains=self.input_data['Nchains'], count=d_tau_CD_used_SD, SDtoggle=True,
                                                  CDflag=self.input_data['CD_flag'], gauss_rand=d_tau_CD_gauss_rand_SD, pcd_table_eq=pcd_table_eq, 
                                                  pcd_table_cr=pcd_table_cr, pcd_table_tau=pcd_table_tau, refill=True)
                    if self.input_data['CD_flag'] == 1:
                        random_state += 1
                        gpu_rand.gpu_tauCD_gauss_rand(seed=random_state, nchains=self.input_data['Nchains'], count=d_tau_CD_used_CD, SDtoggle=False,
                                                      CDflag=self.input_data['CD_flag'], gauss_rand=d_tau_CD_gauss_rand_CD, pcd_table_eq=pcd_table_eq, 
                                                      pcd_table_cr=pcd_table_cr, pcd_table_tau=pcd_table_tau, refill=True)
                    random_state += 1
                    gpu_rand.gpu_uniform_rand(seed=random_state, nchains=self.input_data['Nchains'], count=d_rand_used, uniform_rand=d_uniform_rand, refill=True)
                    
                    
                    step_count = 0
            
            
            #array handling when num_time_syncs are > 1
            stress_host = d_stress.copy_to_host()
            if x_sync == num_time_syncs and x_sync!=1:
                last = int((simulation_time - (x_sync-1)*max_sync_time)/time_resolution)+1
                stress_host = stress_host[:last]
            if x_sync>1:
                stress_host = stress_host[1:]

            #calulcate ensemble average and write to file
            self.calc_avg_stress(x_sync,next_sync_time,stress_host)
            
            #update progress bar
            self.progbar(x_sync,num_time_syncs,20)
            
        #SIMULATION ENDS------------------------------------------------------------------------------------------------------------
            
        t1 = time.time()
        print("Total simulation time: %.2f seconds"%(t1-t0))
        
        #copy final chain conformations and entanglement numbers from device to host
        QN_final = d_QN.copy_to_host()
        Z_final = d_Z.copy_to_host()
        
        #calculate entanglement lifetime distribution
        enttime_run_sum = 0
        filename = './f_dt_%d.txt'%self.sim_ID
        f = open(filename,'w')
        for k in range(0,len(enttime_bins)):
            if enttime_bins[k]!=0:
                enttime_run_sum += enttime_bins[k]
                f.write('%d  %d\n'%(k,enttime_bins[k]))
        f.close()
                
        
        #save distributions to file
        np.savetxt('Z_final_%d.txt'%self.sim_ID,Z_final,fmt='%d')
        self.save_distributions('distr_Q_final_%d.dat'%self.sim_ID,QN_final,Z_final)


if __name__ == "__main__":
    run_dsm = FSM_LINEAR(1)
    run_dsm.main()


