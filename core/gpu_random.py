import numpy as np
from numba import cuda, float32

from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_normal_float32


def gpu_uniform_rand(seed, nchains, count, uniform_rand, refill=False):

	threadsperblock = 256
	blockspergrid = (nchains + threadsperblock - 1) // threadsperblock

	rng_states = create_xoroshiro128p_states(threadsperblock*blockspergrid, seed=seed)

	if refill:
		refill_uniform_rand[blockspergrid,threadsperblock](rng_states, nchains, count, uniform_rand)
	else:
		fill_uniform_rand[blockspergrid,threadsperblock](rng_states, nchains, count, uniform_rand)

	return


def gpu_tauCD_gauss_rand(seed, nchains, count, CDflag, SDtoggle, gauss_rand, pcd_table_eq, pcd_table_cr, pcd_table_tau, refill=False):

	threadsperblock = 256
	blockspergrid = (nchains + threadsperblock - 1) // threadsperblock

	rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid, seed=seed)
	
	if refill:
		refill_gauss_rand_tauCD[blockspergrid,threadsperblock](rng_states, nchains, count, SDtoggle, CDflag, gauss_rand, pcd_table_eq, pcd_table_cr, pcd_table_tau)

	else:
		fill_gauss_rand_tauCD[blockspergrid,threadsperblock](rng_states, nchains, count, SDtoggle, CDflag, gauss_rand, pcd_table_eq, pcd_table_cr, pcd_table_tau)

	return


@cuda.jit
def fill_uniform_rand(rng_states, nchains, count, uniform_rand):

	i = cuda.grid(1)

	if i >= nchains:
		return

	for j in range(0, count):
		x = 0.0 #xoroshiro128p_uniform_float32(rng_states, i)
		while x <= 0.0:
			x = xoroshiro128p_uniform_float32(rng_states, i)

		uniform_rand[i,j] = x

	return

@cuda.jit
def refill_uniform_rand(rng_states, nchains, count, uniform_rand):

	i = cuda.grid(1)

	if i >= nchains:
		return

	for j in range(0, count[i]):
		x = xoroshiro128p_uniform_float32(rng_states, i)
		while x <= 0.0:
			x = xoroshiro128p_uniform_float32(rng_states, i)

		uniform_rand[i,j] = x

	count[i] = 0.0

	return


@cuda.jit
def fill_gauss_rand_tauCD(rng_states, nchains, count, SDtoggle, CD_flag, gauss_rand, pcd_table_eq, pcd_table_cr, pcd_table_tau):

	i = cuda.grid(1)

	if i>=nchains:
		return

	for j in range(0,count):
		x = 0.0
		while x <= 0.0:
			x = xoroshiro128p_uniform_float32(rng_states, i)

		if CD_flag == 1:

			if SDtoggle==True:
				gauss_rand[i,j,3] = tau_CD_eq(x, pcd_table_eq, pcd_table_tau)
			else:
				gauss_rand[i,j,3] = tau_CD_cr(x, pcd_table_cr, pcd_table_tau)

		else:
			gauss_rand[i,j,3] = 0.0
			gauss_rand[i,j,3] = 0.0

		gauss_rand[i,j,0] = xoroshiro128p_normal_float32(rng_states, i)
		gauss_rand[i,j,1] = xoroshiro128p_normal_float32(rng_states, i)
		gauss_rand[i,j,2] = xoroshiro128p_normal_float32(rng_states, i)


	return

@cuda.jit
def refill_gauss_rand_tauCD(rng_states, nchains, count, SDtoggle, CD_flag, gauss_rand, pcd_table_eq, pcd_table_cr, pcd_table_tau):

	i = cuda.grid(1)

	if i>=nchains:
		return

	for j in range(0,int(count[i])):
		x = xoroshiro128p_uniform_float32(rng_states, i)

		if CD_flag == 1:

			if SDtoggle==True:
				gauss_rand[i,j,3] = tau_CD_eq(x, pcd_table_eq, pcd_table_tau)
			else:
				gauss_rand[i,j,3] = tau_CD_cr(x, pcd_table_cr, pcd_table_tau)

		else:
			gauss_rand[i,j,3] = 0.0
			gauss_rand[i,j,3] = 0.0

		gauss_rand[i,j,0] = xoroshiro128p_normal_float32(rng_states, i)
		gauss_rand[i,j,1] = xoroshiro128p_normal_float32(rng_states, i)
		gauss_rand[i,j,2] = xoroshiro128p_normal_float32(rng_states, i)

	count[i] = 0.0


	return


@cuda.jit(device=True)
def tau_CD_cr(p, pcd_table_cr, pcd_table_tau):

	for i in range(0,len(pcd_table_cr)):

		if pcd_table_cr[i] >= p:

			return 1.0/pcd_table_tau[i]


@cuda.jit(device=True)
def tau_CD_eq(p, pcd_table_eq, pcd_table_tau):

	for i in range(0,len(pcd_table_eq)):

		if pcd_table_eq[i] >= p:

			return 1.0/pcd_table_tau[i]


