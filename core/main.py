import os
import sys
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

		if os.path.isfile('input.yaml'):
			with open('input.yaml') as f:
				self.input_data = yaml.load(f, Loader=yaml.FullLoader)
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
                


	def save3Darray(self,filename,data,fmt):
		with open(filename, 'w') as outfile:
	    	# I'm writing a header here just for the sake of readability
	    	# Any line starting with "#" will be ignored by numpy.loadtxt
			outfile.write('# Array shape: {0}\n'.format(data.shape))

			for data_slice in data:
	                        np.savetxt(outfile, data_slice, fmt=fmt)#, fmt='%-7.2f')
	                        outfile.write('#New chain\n')


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



	def main(self):

		if os.path.isfile("pcd_MMM.dat"):
		    pcd_file = open("pcd_MMM.dat",'r')
		    nmodes = int(pcd_file.readline())
		   
		    lines = pcd_file.readlines()
		    tauArr = [float(line.split()[0]) for line in lines]
		    gArr = [float(line.split()[1]) for line in lines]

		    pcd_file.close()

		if self.input_data['CD_flag']==1:
			pcd = p_cd(tauArr, gArr, nmodes)
			d_CD_create_prefact = cuda.to_device([pcd.W_CD_destroy_aver()])

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

		else:
			pcd=None
			pcd_table_tau = np.zeros(1)
			pcd_table_cr = np.zeros(1)
			pcd_table_eq = np.zeros(1)
			d_CD_create_prefact = cuda.to_device([0.0])

	    #generate initial chain conformations on host CPU
		print('Generating initial chain conformations on host...',end="",flush=True)
		chain = ensemble_chains(self.input_data)
		for m in range(0,self.input_data['Nchains']):
			
			chain.chain_init(self.input_data['Nc'],z_max=self.input_data['Nc'],pcd=pcd)

		print('Done.')

	    #reshape arrays for CUDA kernels
		chain.QN = chain.QN.reshape(self.input_data['Nchains'],self.input_data['Nc'],4)
		chain.tau_CD = chain.tau_CD.reshape(self.input_data['Nchains'],self.input_data['Nc'])
		chain.Z = chain.Z.reshape(self.input_data['Nchains'])

		self.save_distributions('distr_Q_initial.dat',chain.QN,chain.Z)

		#self.save3Darray('QN_initial.txt',chain.QN)
		np.savetxt('Z_initial.txt',chain.Z,fmt='%d')

		d_QN = cuda.to_device(chain.QN)
		d_tau_CD = cuda.to_device(chain.tau_CD)
		d_Z = cuda.to_device(chain.Z)


		#initialize arrays for finding jump index
		rand_used = np.zeros(shape=chain.QN.shape[0])
		tau_CD_used_SD = np.zeros(shape=chain.QN.shape[0])
		tau_CD_used_CD = np.zeros(shape=chain.QN.shape[0])
		tau_CD_gauss_rand_SD = np.zeros(shape=(chain.QN.shape[0],250,4),dtype=float)
		tau_CD_gauss_rand_CD = np.zeros(shape=(chain.QN.shape[0],250,4),dtype=float)


		uniform_rand = np.zeros(shape=(chain.QN.shape[0],256),dtype=float)
		gpu_rand.gpu_tauCD_gauss_rand(seed=1, nchains=self.input_data['Nchains'], count=250, SDtoggle=True, CDflag=self.input_data['CD_flag'], gauss_rand=tau_CD_gauss_rand_SD, pcd_table_eq=pcd_table_eq, pcd_table_cr=pcd_table_cr, pcd_table_tau=pcd_table_tau,refill=False)
		gpu_rand.gpu_uniform_rand(seed=1, nchains=self.input_data['Nchains'], count=250, uniform_rand=uniform_rand,refill=False)
		
		d_offset = np.zeros(shape=(chain.QN.shape[0]))
		add_rand = np.zeros(shape=(chain.QN.shape[0]))
		t_cr = np.zeros(shape=(chain.QN.shape[0],chain.QN.shape[1]))
		f_t = np.zeros(shape=chain.QN.shape[0])

		#initialize arrays for chain time
		time = np.zeros(shape=chain.Z.shape[0])
		time_compensation = np.zeros(shape=chain.Z.shape[0])
		stall_flag = np.zeros(shape=chain.Z.shape[0])
		tdt = np.zeros(shape=chain.Z.shape[0],dtype=float)


		found_index = np.zeros(shape=(chain.QN.shape[0]))
		found_shift = np.zeros(shape=(chain.QN.shape[0]))


		d_found_index = cuda.to_device(found_index)
		d_found_shift = cuda.to_device(found_shift)
		d_add_rand = cuda.to_device(add_rand)

		d_t_cr = cuda.to_device(t_cr)
		d_f_t = cuda.to_device(f_t)
		d_time = cuda.to_device(time)
		d_time_compensation = cuda.to_device(time_compensation)
		d_stall_flag = cuda.to_device(stall_flag)
		d_tdt = cuda.to_device(tdt)
		d_rand_used = cuda.to_device(rand_used)
		d_uniform_rand=cuda.to_device(uniform_rand)
		d_tau_CD_gauss_rand_SD=cuda.to_device(tau_CD_gauss_rand_SD)
		d_tau_CD_used_SD=cuda.to_device(tau_CD_used_SD)

		num_steps = self.input_data['sim_time']
		for j in range(1,num_steps+1):

			stream1 = cuda.stream()
			stream2 = cuda.stream()

			shift_probs = np.zeros(shape=chain.QN.shape,dtype=int)
			sum_W_sorted = np.zeros(shape=(chain.QN.shape[0],chain.QN.shape[1]),dtype=int)

			found_index = np.zeros(shape=(chain.QN.shape[0]))
			found_shift = np.zeros(shape=(chain.QN.shape[0]))

			dimBlock = (32,8)
			dimGrid_x = (self.input_data['Nchains']+dimBlock[0]-1)//dimBlock[0]
			dimGrid_y = (self.input_data['Nc']+dimBlock[1]-1)//dimBlock[1]
			dimGrid = (dimGrid_x,dimGrid_y)

			ensemble_kernel.calc_probs_shuffle[dimGrid, dimBlock,stream1](d_Z,d_QN,d_tau_CD,shift_probs,self.input_data['CD_flag'],d_CD_create_prefact)


			threadsperblock = 256
			blockspergrid = (chain.QN.shape[0] + threadsperblock - 1)//threadsperblock
			ensemble_kernel.calc_probs_chainends[blockspergrid, threadsperblock,stream2](d_Z,d_QN,d_tau_CD,shift_probs,self.input_data['CD_flag'],d_CD_create_prefact,self.input_data['beta'])
			

			threadsperblock = 1024
			#blockspergrid = (self.input_data['Nchains'], (threadsperblock[1]*2 - 1)//threadsperblock[1])
			blockspergrid = self.input_data['Nchains'] + threadsperblock // threadsperblock
			ensemble_kernel.scan_kernel[blockspergrid, threadsperblock,stream1](d_Z, shift_probs, sum_W_sorted, d_uniform_rand, d_rand_used, found_index, found_shift, d_add_rand, self.input_data['CD_flag'])


			Z = d_Z.copy_to_host()
			stream1.synchronize()
			#if np.any(Z>self.input_data['Nc']):
			#	print('Chain exceeds number of strands')

			for k in range(0,len(Z)):
				if Z[k] > 10 or Z[k] < 1:
					print(shift_probs[k])
					#print(d_QN.copy_to_host()[k])

			ensemble_kernel.chain_kernel[blockspergrid, threadsperblock,stream1](d_Z, d_QN, d_time, d_time_compensation, d_stall_flag, found_shift, found_index, d_offset, d_tdt, sum_W_sorted,\
																			d_t_cr, d_f_t, d_tau_CD, d_rand_used, d_add_rand, d_tau_CD_used_SD, tau_CD_used_CD, d_tau_CD_gauss_rand_SD)


			#print(uniform_rand[0])
			#rand_used = d_rand_used.copy_to_host()
			#uniform_rand = d_uniform_rand.copy_to_host()
			#tau_CD_used_SD = d_tau_CD_used_SD.copy_to_host()
			#tau_CD_gauss_rand_SD = d_tau_CD_gauss_rand_SD.copy_to_host()


			stream1.synchronize()

			#print(tau_CD_used_SD[0])
			#print(tau_CD_gauss_rand_SD[0,0])

			if j % 250 == 0:

				gpu_rand.gpu_tauCD_gauss_rand(seed=j, nchains=self.input_data['Nchains'], count=d_tau_CD_used_SD, SDtoggle=True, CDflag=self.input_data['CD_flag'], gauss_rand=d_tau_CD_gauss_rand_SD, pcd_table_eq=pcd_table_eq, pcd_table_cr=pcd_table_cr, pcd_table_tau=pcd_table_tau, refill=True)
				gpu_rand.gpu_uniform_rand(seed=j, nchains=self.input_data['Nchains'], count=d_rand_used, uniform_rand=d_uniform_rand, refill=True)
				#rand_used = np.zeros(shape=chain.QN.shape[0])
				#d_rand_used = cuda.to_device(rand_used,stream=stream1)
				#tau_CD_used_SD = np.zeros(shape=chain.QN.shape[0])

			self.progbar(j,num_steps-1,20)

		check_array = d_QN.copy_to_host()
		Z_final = d_Z.copy_to_host()
                
		np.savetxt('Z_final.txt',Z_final,fmt='%d')
		self.save_distributions('distr_Q_final.dat',check_array,Z_final)


if __name__ == "__main__":
	run_dsm = FSM_LINEAR(1)
	run_dsm.main()


