from numba import cuda
import math

from numba.cuda.random import xoroshiro128p_uniform_float64, xoroshiro128p_normal_float64


@cuda.jit
def fill_uniform_rand(rng_states, nchains, count, uniform_rand):
    '''
    GPU kernel to fill array with random uniform values in uniform_rand for each chain
    '''

    i = cuda.grid(1)

    if i >= nchains:
        return

    for j in range(0, count):
        x = 0.0
        while x == 0.0: #do not include 0 in uniform random numbers
            x = xoroshiro128p_uniform_float64(rng_states, i)

        uniform_rand[i,j] = x

    return

@cuda.jit
def refill_uniform_rand(rng_states, nchains, count, uniform_rand):
    '''
    GPU kernel to refill all used random values in uniform_rand for each chain
    '''

    i = cuda.grid(1)

    if i >= nchains:
        return

    for j in range(0, int(count[i])):
        x = 0.0
        while x == 0.0: #do not include 0 in uniform random numbers
            x = xoroshiro128p_uniform_float64(rng_states, i)

        uniform_rand[i,j] = x

    count[i] = 0.0

    return


@cuda.jit
def fill_gauss_rand_tauCD(rng_states, analytic, nchains, count, SDtoggle, CD_flag, pdi_array, gauss_rand, pcd_array, pcd_table_eq, pcd_table_cr, pcd_table_tau):
    '''
    GPU kernel to fill random values from a guassian distribution in gauss_rand array
    '''
    i = cuda.grid(1)

    if i>=nchains:
        return

    for j in range(0,count):
        x = 0.0
        while x == 0.0: #do not include 0 in uniform random numbers
            x = xoroshiro128p_uniform_float64(rng_states, i)

        if CD_flag == 1 and not analytic:

            if SDtoggle==True:
                gauss_rand[i,j,3] = tau_CD_eq(x, pcd_table_eq, pcd_table_tau)
            else:
                gauss_rand[i,j,3] = tau_CD_cr(x, pcd_table_cr, pcd_table_tau)

        elif CD_flag == 1 and analytic:

            if pdi_array[0]:
                p = xoroshiro128p_normal_float64(rng_states, i)
                Mlognorm = (math.exp(p*pdi_array[2] + pdi_array[1]))
                NK_PD = Mlognorm/pdi_array[4]
                while (Mlognorm > pdi_array[3]) or (NK_PD < 1):
                    p = xoroshiro128p_normal_float64(rng_states, i)
                    Mlognorm = (math.exp(p*pdi_array[2] + pdi_array[1]))
                    NK_PD = Mlognorm/pdi_array[4]
                
                g, tau_D_inverse, At, Dt, Ct, Adt, Bdt, Cdt, Ddt = p_cd_linear(NK_PD,pdi_array[5])    

            else:
                g, tau_D_inverse, At, Dt, Ct, Adt, Bdt, Cdt, Ddt = pcd_array[0], pcd_array[5], pcd_array[6], pcd_array[7], pcd_array[8], pcd_array[9], pcd_array[10], pcd_array[11], pcd_array[12]
                    
            if SDtoggle==True:
                gauss_rand[i,j,3] = tau_CD_f_t(x,At,Ct,Dt,tau_D_inverse,g)
            else:
                gauss_rand[i,j,3] = tau_CD_f_d_t(x,Adt,Bdt,Cdt,Ddt,tau_D_inverse)

        else:
            gauss_rand[i,j,3] = 0.0
            gauss_rand[i,j,3] = 0.0

        for k in range(0,3):
            gauss_rand[i,j,k] = xoroshiro128p_normal_float64(rng_states, i)

    return

@cuda.jit
def refill_gauss_rand_tauCD(rng_states, analytic, nchains, count, SDtoggle, CD_flag, pdi_array, gauss_rand, pcd_array, pcd_table_eq, pcd_table_cr, pcd_table_tau):
    '''
    GPU kernel to refill all used random values from the gauss_rand array
    '''
    i = cuda.grid(1)

    if i>=nchains:
        return

    for j in range(0,int(count[i])):
        x = 0.0
        while x == 0.0: #do not include 0 in uniform random numbers
            x = xoroshiro128p_uniform_float64(rng_states, i)

        if CD_flag == 1 and not analytic:

            if SDtoggle==True:
                gauss_rand[i,j,3] = tau_CD_eq(x, pcd_table_eq, pcd_table_tau)
            else:
                gauss_rand[i,j,3] = tau_CD_cr(x, pcd_table_cr, pcd_table_tau)

        elif CD_flag == 1 and analytic:
            if pdi_array[0]:
                p = xoroshiro128p_normal_float64(rng_states, i)
                Mlognorm = (math.exp(p*pdi_array[2] + pdi_array[1]))
                NK_PD = Mlognorm/pdi_array[4]
                while (Mlognorm > pdi_array[3]) or (NK_PD < 1):
                    p = xoroshiro128p_normal_float64(rng_states, i)
                    Mlognorm = (math.exp(p*pdi_array[2] + pdi_array[1]))
                    NK_PD = Mlognorm/pdi_array[4]
                
                g, tau_D_inverse, At, Dt, Ct, Adt, Bdt, Cdt, Ddt = p_cd_linear(NK_PD,pdi_array[5])    
                
            else:
                g, tau_D_inverse, At, Dt, Ct, Adt, Bdt, Cdt, Ddt = pcd_array[0], pcd_array[5], pcd_array[6], pcd_array[7], pcd_array[8], pcd_array[9], pcd_array[10], pcd_array[11], pcd_array[12]
                    
            if SDtoggle==True:
                gauss_rand[i,j,3] = tau_CD_f_t(x,At,Ct,Dt,tau_D_inverse,g)
            else:
                gauss_rand[i,j,3] = tau_CD_f_d_t(x,Adt,Bdt,Cdt,Ddt,tau_D_inverse)

        else:
            gauss_rand[i,j,3] = 0.0
            gauss_rand[i,j,3] = 0.0

        for k in range(0,3):
            gauss_rand[i,j,k] = xoroshiro128p_normal_float64(rng_states, i)

    count[i] = 0.0

    return


@cuda.jit(device=True)
def tau_CD_cr(p, pcd_table_cr, pcd_table_tau):
    '''
    Device function to calculate probability of creation due to CD using discrete pCD modes
    '''

    for i in range(0,len(pcd_table_cr)):

        if pcd_table_cr[i] >= p:

            return 1.0/pcd_table_tau[i]


@cuda.jit(device=True)
def tau_CD_eq(p, pcd_table_eq, pcd_table_tau):
    '''
    Device function to calculate probability of creation due to SD using discrete pCD modes
    '''
    for i in range(0,len(pcd_table_eq)):

        if pcd_table_eq[i] >= p:

            return 1.0/pcd_table_tau[i]
        

@cuda.jit(device=True)
def tau_CD_f_d_t(prob,d_Adt,d_Bdt,d_Cdt,d_Ddt,d_tau_D_inverse):

    if prob < d_Bdt:
        return math.pow(prob*d_Adt + d_Ddt,d_Cdt) 
    else:
        return d_tau_D_inverse 


@cuda.jit(device=True)
def tau_CD_f_t(prob,d_At,d_Ct,d_Dt,d_tau_D_inverse,d_g):

    if prob < 1.0 - d_g:
        return math.pow(prob * d_At + d_Dt,d_Ct) 
    else:
        return d_tau_D_inverse 


@cuda.jit(device=True)
def p_cd_linear(NK,beta):

    g = 0.667
    z = (NK + beta) / (beta + 1.0)

    if beta != 1.0:

        alpha = (0.053 * math.log(beta) + 0.31) * math.pow(z,(-0.012 * math.log(beta) - 0.024))
        tau_0 = 0.285 * math.pow((beta + 2.0),0.515)
        if NK < 2:
            tau_max = tau_0
            tau_D = tau_0 
        else: 
            tau_max = 0.025 * math.pow((beta+2.0),2.6) * math.pow(z,2.83)
            tau_D = 0.036 * math.pow((beta+2.0),3.07) * math.pow((z - 1),3.02)

    else:

        alpha = 0.267096 - 0.375571 * math.exp(-0.0838237 * NK)
        tau_0 = 0.460277 + 0.298913 * math.exp(-0.0705314 * NK)
        if NK < 4:
            tau_max = tau_0
            tau_D = tau_0
        else:
            tau_max = 0.0156137 * math.pow(NK, 3.18849)
            tau_D = 0.0740131 * math.pow(NK, 3.18363)


    tau_alpha = math.pow(tau_max,alpha) - math.pow(tau_0,alpha)
    tau_alpha_m1 = math.pow(tau_max,(alpha-1.0)) - math.pow(tau_0,(alpha - 1.0))
    if tau_alpha == 0.0:
        ratio_tau_alpha = (alpha - 1.0)/alpha/tau_0
    else:
        ratio_tau_alpha = tau_alpha_m1 / tau_alpha 

    At = (1.0 - g)
    Adt = At * alpha / (alpha - 1.0)
    Bdt = Adt * ratio_tau_alpha
    normdt = Bdt + g / tau_D
    
    # pcd_array[0] = g
    # # pcd_array[1] = alpha
    # # pcd_array[2] = tau_0
    # # pcd_array[3] = tau_max
    # # pcd_array[4] = tau_D
    # pcd_array[5] = 1.0/tau_D
    # pcd_array[6] = 1.0*tau_alpha/At
    # pcd_array[7] = math.pow(tau_0,alpha)
    # pcd_array[8] = -1.0/alpha
    # pcd_array[9] = normdt*tau_alpha/Adt
    # pcd_array[10] = Bdt/normdt
    # pcd_array[11] = -1.0/(alpha - 1.0)
    # pcd_array[12] = tau_0**(alpha - 1.0)

    return g, 1.0/tau_D, 1.0*tau_alpha/At, math.pow(tau_0,alpha), -1.0/alpha, normdt*tau_alpha/Adt, Bdt/normdt, -1.0/(alpha - 1.0), tau_0**(alpha - 1.0)