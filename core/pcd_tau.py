import numpy as np
import core.random_gen as rng

class p_cd(object):

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
		p = rng.genrand_real3()
		total = 0
		for i in range(0,self.nmodes):
			if total>p:
				break
			else:
				total += self.g[i]*self.tau[i]/self.ptau_sum

		return self.tau[i-1]


	def W_CD_destroy_aver(self):

		return 1.0/self.ptau_sum

