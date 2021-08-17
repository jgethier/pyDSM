import numpy as np
import math, cmath
from numba import cuda,float32,int32


# @cuda.jit(device=True)
# def offset_code(offset_index, offset_dir):
#     return ((offset_dir + 1) | (offset_index << 8))


# @cuda.jit(device=True)
# def make_offset(i, offset):
#     if i >= ((offset & 0xffff00) >> 8):
#         return i + ((offset & 0xff) - 1)
#     else:
#         return i


# @cuda.jit(device=True)
# def offset_index(offset):
#     return ((offset & 0xffff00) >> 8)


# @cuda.jit(device=True)
# def offset_dir(offset):
#     return (offset & 0xff) - 1   


# @cuda.jit(device=True)
# def fetch_new_strent(i, offset):
#     return (i == offset_index(offset)) & (offset_dir(offset) == -1) 

# @cuda.jit(device=True)
# def read_strent(i, j, offset, new_strent, d_QN):
#     if (fetch_new_strent(j, offset)):
#         new_strent
#     else: 
#         d_QN[make_offset(j, offset), i]  
    

@cuda.jit
def calc_probs_shuffle(Z,QN,tau_CD,shift_probs,CD_flag,d_CD_create_prefact):

        i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
        j = cuda.blockIdx.y*cuda.blockDim.y + cuda.threadIdx.y

        tz = int(Z[i])

        if j >= tz:
            return

        if i >= QN.shape[0]:
            return

        QN_i = QN[i, j, :]
        if j<tz-1:
            QN_ip1 = QN[i, j+1, :]
            Q_i = QN_i[0]*QN_i[0] + QN_i[1]*QN_i[1] + QN_i[2]*QN_i[2]
            Q_ip1 = QN_ip1[0]*QN_ip1[0] + QN_ip1[1]*QN_ip1[1] + QN_ip1[2]*QN_ip1[2]

            if QN_ip1[3] > 1.0:
                    sig1 = 0.75 / (QN_i[3]*(QN_i[3]+1))
                    sig2 = 0.75 / (QN_ip1[3]*(QN_ip1[3]-1))
                    if Q_i==0.0:
                            prefactor1 = 1.0
                    else:
                            prefactor1 = QN_i[3] / (QN_i[3] + 1)
                    if Q_ip1 == 0.0:
                            prefactor2 = 1.0
                    else:
                            prefactor2 = QN_ip1[3] / (QN_ip1[3] - 1)
                    if j==0:
                            f1 = 2.0*(QN_i[3]+0.5)
                    else:
                            f1 = QN_i[3]
                    if j==tz-2:
                            f2 = 2.0*(QN_ip1[3]-0.5)
                    else:
                            f2 = QN_ip1[3]

                    friction = 2.0 / (f1 + f2)
                    shift_probs[i, j, 0] = int(1000000.0*friction*math.pow(prefactor1*prefactor2,0.75)*math.exp(Q_i*sig1-Q_ip1*sig2))

            if QN_i[3] > 1.0:
                    sig1 = 0.75 / (QN_i[3]*(QN_i[3]-1))
                    sig2 = 0.75 / (QN_ip1[3]*(QN_ip1[3]+1))
                    if Q_i == 0.0:
                            prefactor1 = 1.0
                    else:
                            prefactor1 = QN_i[3] / (QN_i[3] - 1)
                    if Q_ip1 == 0.0:
                            prefactor2 = 1.0
                    else:
                            prefactor2 = QN_ip1[3] / (QN_ip1[3] + 1)
                    if j==0:
                            f1 = 2.0*(QN_i[3]-0.5)
                    else:
                            f1 = QN_i[3]
                    if j==tz-2:
                            f2 = 2.0*(QN_ip1[3]+0.5)
                    else:
                            f2 = QN_ip1[3]

                    friction = 2.0 / (f1 + f2)
                    shift_probs[i, j, 1] = int(1000000.0*friction*math.pow(prefactor1*prefactor2,0.75)*math.exp(-Q_i*sig1+Q_ip1*sig2))

            if CD_flag==1:
                    shift_probs[i, j, 2] = int(tau_CD[i,j]*1000000.0)

                    shift_probs[i, j, 3] = int(d_CD_create_prefact[0]*1000000.0*(QN_i[3]-1.0))
        
        return


@cuda.jit
def calc_probs_chainends(Z, QN, tau_CD, shift_probs, CD_flag, d_CD_create_prefact, beta):

	i = cuda.grid(1)

	if i >= QN.shape[0]:
		return

	tz = int(Z[i])


	QNfirst = QN[i,0]
	QNlast = QN[i,tz-1]

	if tz == 1:
	 	shift_probs[i,tz-1,1] = int(1000000.0*(1.0 / (beta*QN[i,0,3])))
	 	shift_probs[i,tz,1] = int(1000000.0*(1.0 / (beta*QN[i,0,3])))

	else:
		if QNfirst[3] == 1.0: #destruction by SD at the beginning
			QNfirst_n = QN[i,1]
			if tz == 2:
				c = QNfirst_n[3] + 0.25
			else:
				c = QNfirst_n[3] * 0.5

			shift_probs[i,tz,0] = int(1000000.0*(1.0 / (c+0.75)))

		else: #creation by SD at the beginning
			shift_probs[i,tz,1] = int(1000000.0*(2.0 / (beta * (QNfirst[3]+0.5) )))

		if QNlast[3] == 1.0: #destruction by SD at the end

			QNlast_p = QN[i,tz-2]
			if tz == 2:
				c = QNlast_p[3] + 0.25
			else:
				c = QNlast_p[3] * 0.5
			shift_probs[i,tz-1,0] = int(1000000.0*(1.0 / (c+0.75)))
		else: #creation by SD at the end
			shift_probs[i,tz-1,1] = int(1000000.0*(2.0 / (beta * (QNlast[3]+0.5) )))

	if CD_flag==1:
	 	shift_probs[i,tz-1,2] = int(1000000.0*d_CD_create_prefact[0]*(QNlast[3]-1.0))

	return


@cuda.jit
def scan_kernel(Z,shift_probs,sum_W_sorted,uniform_rand,rand_used,found_index,found_shift,add_rand,CD_flag):


    prefix_sum = cuda.shared.array(1024,dtype=int32)
    
    bid = cuda.blockIdx.x*cuda.blockDim.x
    tid = cuda.threadIdx.x
    #bid = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
    #tid = cuda.blockIdx.y*cuda.blockDim.y + cuda.threadIdx.y

    if bid >= shift_probs.shape[0] or tid >= shift_probs.shape[1]:
        return

    for m in range(0,int(Z[bid])+1):
        temp = shift_probs[bid,m,:]
        if m==0:
            sum_W_sorted[bid,m] = (temp[0] + temp[1] + temp[2] + temp[3])
        else:
            sum_W_sorted[bid,m] = sum_W_sorted[bid,m-1] + (temp[0] + temp[1] + temp[2] + temp[3])

    
    x = math.ceil(sum_W_sorted[bid,int(Z[bid])]*uniform_rand[bid,int(rand_used[bid])])

    xFound = yFound = zFound = wFound = False

    temp = shift_probs[bid,tid,:]

    if tid == 0:
        left = 0.0
    else:
        left = sum_W_sorted[bid,tid-1]
    xFound = (left < x) & (x <= left + temp[0])
    yFound = (left + temp[0] < x) & (x <= left + temp[0] + temp[1])

    if CD_flag==1:
        zFound = (left + temp[0] + temp[1] < x) & (x <= left + temp[0] + temp[1] + temp[2])
        wFound = (left + temp[0] + temp[1] + temp[2] < x) & (x <= left + temp[0] + temp[1] + temp[2] + temp[3])


    if xFound or yFound or zFound or wFound:
        found_index[bid] = tid
        if xFound==True:
               	found_shift[bid] = 0.0
               	if (tid == int(Z[bid]) - 1):
                       	found_index[bid] = tid-1
                       	found_shift[bid] = 5.0 #destroy at end by SD
               	if (tid == int(Z[bid])):
                       	found_index[bid] = 0
                       	found_shift[bid] = 5.0 #destroy at beginning by SD
       	elif yFound==True:
               	found_shift[bid] = 1.0
               	if (tid == int(Z[bid]) - 1):
                       	found_shift[bid] = 3.0 #create at end by SD
               	if (tid == int(Z[bid])):
                       	found_index[bid] = 0
                       	found_shift[bid] = 6.0 #create at beginning by SD
       	elif zFound:
            found_shift[bid] = 2 #destroy by CD
            print('Destroy by CD found')
       	elif wFound:
            found_shift[bid] = 4 #create by CD
            print('Create by CD found')
            add_rand[bid] = (x - left - temp[0] - temp[1] - temp[2]) / temp[3]                                           

    return


@cuda.jit
def chain_kernel(Z, QN, time, time_compensation, stall_flag, found_shift, found_index, d_offset, tdt, sum_W_sorted, \
                        t_cr, f_t, tau_CD, rand_used, add_rand, tau_CD_used_SD, tau_CD_used_CD, tau_CD_gauss_rand_SD):

    shared_QN = cuda.shared.array((1024,4),dtype=float32)
    shared_tcr = cuda.shared.array(1024,dtype=float32)
    shared_tauCD = cuda.shared.array(1024,dtype=float32)

    i = cuda.blockIdx.x*cuda.blockDim.x
    j = cuda.threadIdx.x
    #i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x #chain index
    #j = cuda.blockIdx.y*cuda.blockDim.y + cuda.threadIdx.y #str + ent index

    tz = int(Z[i])

    if j >= QN.shape[1] or i >= QN.shape[0]:
        return

    jumpIdx = int(found_index[i])
    jumpType = int(found_shift[i])

    olddt = tdt[i]
    tdt[i] = 1.0 / (sum_W_sorted[i,tz] / 1000000.0)

    #if j == 0 and i == 0:
    #    print(sum_W_sorted[i,tz])
    
    if sum_W_sorted[i,tz] == 0.0:
        if j == tz:
            print('Timestep size is infinity')

    if tdt[i] == 0.0:
        stall_flag[i] = 1
    elif cmath.isnan(tdt[i]):
        stall_flag[i] = 2
    elif cmath.isinf(tdt[i]):
        stall_flag[i] = 3

    #Use Kahan summation to update time of chain
    y = tdt[i] - time_compensation[i]
    t = time[i] + y
    time_compensation[i] = (t - time[i] - y)
    time[i] = t 

    rand_used[i]+=1

    if jumpType == 0 or jumpType == 1:
        if j == jumpIdx:
            apply_shuffle(i, jumpIdx, jumpType, QN)


    elif jumpType == 2 or jumpType == 5:
        apply_destroy(i, j, jumpIdx, QN, shared_QN, Z, t_cr, shared_tcr, tau_CD, shared_tauCD, f_t, time)

        if j == jumpIdx:
            Z[i]=Z[i]-1

        for m in range(0,4):
            QN[i,j,m] = shared_QN[j,m]
        t_cr[i,j] = shared_tcr[j]
        tau_CD[i,j] = shared_tauCD[j]


    elif jumpType == 3 or jumpType == 6:
        apply_create_SD(i, j, jumpIdx, jumpType, QN, shared_QN, Z, t_cr, shared_tcr, tau_CD, shared_tauCD, time, tau_CD_used_SD, tau_CD_gauss_rand_SD, add_rand)
        
        if j == jumpIdx:
            Z[i] = Z[i] + 1
            #print(Z[i])
        
        for m in range(0,4):
            QN[i,j,m] = shared_QN[j,m]
        t_cr[i,j] = shared_tcr[j]
        tau_CD[i,j] = shared_tauCD[j]


    #elif jumpType == 4:
    #    apply_create_CD(i, j, jumpIdx, jumpType, d_QN, shared_QN, Z, t_cr, shared_tcr, tau_CD, shared_tauCD, time, tau_CD_used_CD, tau_CD_gauss_rand_CD, add_rand)
    

    return


@cuda.jit(device=True)
def apply_shuffle(chainIdx, jumpIdx, jumpType, QN):
    QN1=QN[chainIdx,jumpIdx,:]
    QN2=QN[chainIdx,jumpIdx+1,:]
    if jumpType == 0: #shuffling left
        QN1[3] = QN1[3] + 1
        QN2[3] = QN2[3] - 1
    else: #shuffling right
        QN1[3] = QN1[3] - 1
        QN2[3] = QN2[3] + 1

    return


@cuda.jit(device=True)
def apply_destroy(chainIdx, threadIdx, jumpIdx, QN, shared_QN, Z, t_cr, shared_tcr, tau_CD, shared_tauCD, f_t, time):
    tz = int(Z[chainIdx])
    
    QN1 = QN[chainIdx,jumpIdx,:]
    QN2 = QN[chainIdx,jumpIdx+1,:]

    cr_time = t_cr[chainIdx,jumpIdx]

    if cr_time != 0:
        f_t[chainIdx] = math.log10(time[chainIdx] - cr_time) + 10

    if jumpIdx == 0:

        if threadIdx == jumpIdx:

            shared_QN[threadIdx,3] = QN1[3]+QN2[3]
            shared_QN[threadIdx,0] = shared_QN[threadIdx,1] = shared_QN[threadIdx,2] = 0.0
            #cuda.syncthreads()

            shared_tcr[threadIdx] = t_cr[chainIdx,threadIdx+1]
            shared_tauCD[threadIdx] = tau_CD[chainIdx,threadIdx+1]
        
        elif threadIdx > 0:
            for m in range(0,4):
                shared_QN[threadIdx,m] = QN[chainIdx,threadIdx+1,m]
            shared_tcr[threadIdx] = t_cr[chainIdx,threadIdx+1]
            shared_tauCD[threadIdx] = tau_CD[chainIdx,threadIdx+1]

    elif jumpIdx == tz-2:

        if threadIdx == jumpIdx:
            shared_QN[threadIdx,3] = QN1[3]+QN2[3]
            shared_QN[threadIdx,0] = shared_QN[threadIdx,1] = shared_QN[threadIdx,2] = 0.0

            shared_tcr[threadIdx] = 0.0
            shared_tauCD[threadIdx] = 0.0
        
        elif threadIdx < jumpIdx:
        
            for m in range(0,4):
                shared_QN[threadIdx,m] = QN[chainIdx,threadIdx,m]

            shared_tcr[threadIdx] = t_cr[chainIdx,threadIdx]
            shared_tauCD[threadIdx] = tau_CD[chainIdx,threadIdx]
        
        elif threadIdx > jumpIdx:
            for m in range(0,4):
                shared_QN[threadIdx,m] = 0.0

            shared_tcr[threadIdx] = 0.0
            shared_tauCD[threadIdx] = 0.0

    else:

        if threadIdx == jumpIdx:
            for m in range(0,4):
                shared_QN[threadIdx,m] = QN1[m]+QN2[m]
        
            shared_tcr[threadIdx] = t_cr[chainIdx,threadIdx]
            shared_tauCD[threadIdx] = tau_CD[chainIdx,threadIdx]
            
        
        elif threadIdx < jumpIdx:
            for m in range(0,4):
                shared_QN[threadIdx,m] = QN[chainIdx,threadIdx,m]
            shared_tcr[threadIdx] = t_cr[chainIdx,threadIdx]
            shared_tauCD[threadIdx] = tau_CD[chainIdx,threadIdx]
        
        elif threadIdx > jumpIdx:
            for m in range(0,4):
                shared_QN[threadIdx,m] = QN[chainIdx,threadIdx+1,m]
            shared_tcr[threadIdx] = t_cr[chainIdx,threadIdx+1]
            shared_tauCD[threadIdx] = tau_CD[chainIdx,threadIdx+1]

    cuda.syncthreads()
    
    return


@cuda.jit(device=True)
def apply_create_SD(chainIdx, threadIdx, jumpIdx, jumpType, QN, shared_QN, Z, t_cr, shared_tcr, tau_CD, shared_tauCD, time, tau_CD_used_SD, tau_CD_gauss_rand_SD, add_rand):

    tz = int(Z[chainIdx])

    QN1 = QN[chainIdx,jumpIdx,:]

    temp = tau_CD_gauss_rand_SD[chainIdx,int(tau_CD_used_SD[chainIdx]),:]
    
    if threadIdx == jumpIdx:
        tau_CD_used_SD[chainIdx]+=1

    shared_tcr[jumpIdx] = time[chainIdx]
    shared_tauCD[jumpIdx] = temp[3]

    temp[3] = QN1[3] - 1.0

    if tz==1:
        sigma = 0.0
    else:
        sigma = math.sqrt(temp[3] / 3.0)

    temp[0]*=sigma
    temp[1]*=sigma
    temp[2]*=sigma

    if jumpIdx == tz-1:

        if threadIdx == jumpIdx: 

            for m in range(0,4):
                shared_QN[threadIdx,m] = temp[m]

        elif threadIdx == jumpIdx+1:

            shared_QN[threadIdx,0] = shared_QN[threadIdx,1] = shared_QN[threadIdx,2] = 0.0
            shared_QN[threadIdx,3] = 1.0

            shared_tcr[threadIdx] = 0.0
            shared_tauCD[threadIdx] = 0.0

        elif threadIdx < jumpIdx:

            for m in range(0,4):
                shared_QN[threadIdx,m] = QN[chainIdx,threadIdx,m]
            shared_tcr[threadIdx] = t_cr[chainIdx,threadIdx]
            shared_tauCD[threadIdx] = tau_CD[chainIdx,threadIdx]

        elif threadIdx > jumpIdx+1:

            shared_QN[threadIdx,0] = shared_QN[threadIdx,1] = shared_QN[threadIdx,2] = shared_QN[threadIdx,3] = 0.0

            shared_tcr[threadIdx] = 0.0
            shared_tauCD[threadIdx] = 0.0


    elif jumpIdx == 0:
        
        if threadIdx == jumpIdx:
            shared_QN[threadIdx,3] = 1.0
            shared_QN[threadIdx,0] = shared_QN[threadIdx,1] = shared_QN[threadIdx,2] = 0.0


        elif threadIdx == jumpIdx+1:
            for m in range(0,4):
                shared_QN[threadIdx,m] = temp[m]

            shared_tcr[threadIdx] = t_cr[chainIdx,threadIdx-1]
            shared_tauCD[threadIdx] = tau_CD[chainIdx,threadIdx-1]
        
        elif threadIdx > jumpIdx+1:
            for m in range(0,4):
                shared_QN[threadIdx,m] = QN[chainIdx,threadIdx-1,m]

            shared_tcr[threadIdx] = t_cr[chainIdx,threadIdx-1]
            shared_tauCD[threadIdx] = tau_CD[chainIdx,threadIdx-1]

    cuda.syncthreads()

    return


@cuda.jit(device=True)
def apply_create_CD(chainIdx, threadIdx, jumpIdx, jumpType, QN, shared_QN, Z, t_cr, shared_tcr, tau_CD, shared_tauCD, time, tau_CD_used_CD, tau_CD_gauss_rand_CD, add_rand):

    tz = int(Z[chainIdx])

    QN1 = QN[chainIdx,jumpIdx,:]







