import numpy as np
import math, cmath
from numba import cuda,float32,int32
    

@cuda.jit
def calc_probs_shuffle(Z,QN,tau_CD,shift_probs,CD_flag,CD_create_prefact):

    i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x #chain index
    j = cuda.blockIdx.y*cuda.blockDim.y + cuda.threadIdx.y #strand index

    if i >= QN.shape[0]:
        return

    tz = int(Z[i])

    if j >= tz:
        return
    
    shift_probs[i,j,0] = shift_probs[i,j,1] = shift_probs[i,j,2] = shift_probs[i,j,3] = 0

    tcd = tau_CD[i,j]
    
    QN_i = QN[i, j, :]
    if j<tz-1:
        QN_ip1 = QN[i, j+1, :]
        Q_i = QN_i[0]*QN_i[0] + QN_i[1]*QN_i[1] + QN_i[2]*QN_i[2]
        Q_ip1 = QN_ip1[0]*QN_ip1[0] + QN_ip1[1]*QN_ip1[1] + QN_ip1[2]*QN_ip1[2]

        if int(QN_ip1[3]) > 1.0:
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
                shift_probs[i, j, 0] = int(1e6*friction*math.pow(prefactor1*prefactor2,0.75)*math.exp(Q_i*sig1-Q_ip1*sig2))
        else:
            pass


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
                shift_probs[i, j, 1] = int(1e6*friction*math.pow(prefactor1*prefactor2,0.75)*math.exp(-Q_i*sig1+Q_ip1*sig2))
        else:
            pass


        if CD_flag==1:
                shift_probs[i, j, 2] = int(tcd*1e6)
                shift_probs[i, j, 3] = int(CD_create_prefact[0]*1e6*(QN_i[3]-1.0))
    
    return


@cuda.jit
def calc_probs_chainends(Z, QN, shift_probs, CD_flag, CD_create_prefact, beta):

    i = cuda.grid(1)

    if i >= QN.shape[0]:
        return

    tz = int(Z[i])


    QNfirst = QN[i,0]
    QNlast = QN[i,tz-1]
    
    
    shift_probs[i,tz,0] = shift_probs[i,tz,1] = shift_probs[i,tz,2] = shift_probs[i,tz,3] = 0
    shift_probs[i,tz-1,0] = shift_probs[i,tz-1,1] = shift_probs[i,tz-1,2] = shift_probs[i,tz-1,3] = 0

    if tz == 1:
        shift_probs[i,tz-1,1] = int(1e6*(1.0 / (beta*QN[i,0,3])))
        shift_probs[i,tz,1] = int(1e6*(1.0 / (beta*QN[i,0,3])))

    else:
        if QNfirst[3] == 1.0: #destruction by SD at the beginning
            QNfirst_n = QN[i,1]
            if tz == 2:
                c = QNfirst_n[3] + 0.25
            else:
                c = QNfirst_n[3] * 0.5

            shift_probs[i,tz,0] = int(1e6*(1.0 / (c+0.75)))

        else: #creation by SD at the beginning
            shift_probs[i,tz,1] = int(1e6*(2.0 / (beta * (QNfirst[3]+0.5) )))

        if QNlast[3] == 1.0: #destruction by SD at the end

            QNlast_p = QN[i,tz-2]
            if tz == 2:
                c = QNlast_p[3] + 0.25
            else:
                c = QNlast_p[3] * 0.5
            shift_probs[i,tz-1,0] = int(1e6*(1.0 / (c+0.75)))
        else: #creation by SD at the end
            shift_probs[i,tz-1,1] = int(1e6*(2.0 / (beta * (QNlast[3]+0.5) )))

    if CD_flag==1:
        shift_probs[i,tz-1,3] = int(1e6*CD_create_prefact[0]*(QNlast[3]-1.0))

    return


@cuda.jit
def scan_kernel(Z,shift_probs,sum_W_sorted,uniform_rand,rand_used,found_index,found_shift,add_rand,CD_flag):


    i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
    
    if i >= shift_probs.shape[0]:
        return
    
    tz = int(Z[i])

    sum1 = 0
    for j in range(0,tz+1):
        temp = shift_probs[i,j,:]

        if CD_flag:
            sum1 += (temp[0] + temp[1] + temp[2] + temp[3])
        else:
            sum1 += (temp[0] + temp[1])
                
    sum_W_sorted[i] = sum1
    x = math.ceil(sum1*uniform_rand[i,int(rand_used[i])])
    
    xFound = yFound = zFound = wFound = False
    sum2 = 0
    
    for j in range(0,tz+1):
        
        temp = shift_probs[i,j,:]
        
        if sum2 < x:
            
            xFound = (sum2 < x) & (x <= sum2 + temp[0])
            sum2+=temp[0]
            yFound = (sum2 < x) & (x <= sum2 + temp[1])
            sum2+=temp[1]

            if CD_flag==1:
                zFound = (sum2 < x) & (x <= sum2 + temp[2])
                sum2+=temp[2]
                wFound = (sum2 < x) & (x <= sum2 + temp[3])
                sum2+=temp[3]
                
            if xFound or yFound or zFound or wFound:
                break
                

    ii=j
    
    
    if xFound or yFound or zFound or wFound:
        found_index[i] = j
        if xFound:
               	found_shift[i] = 0.0
               	if (ii == int(Z[i]) - 1):
                       	found_index[i] = j-1
                       	found_shift[i] = 5.0 #destroy at end by SD
               	if (ii == int(Z[i])):
                       	found_index[i] = 0
                       	found_shift[i] = 5.0 #destroy at beginning by SD
       	elif yFound:
               	found_shift[i] = 1.0
               	if (ii == int(Z[i]) - 1):
                       	found_shift[i] = 3.0 #create at end by SD
               	if (ii == int(Z[i])):
                       	found_index[i] = 0
                       	found_shift[i] = 6.0 #create at beginning by SD
       	elif zFound:
            found_shift[i] = 2 #destroy by CD
       	elif wFound:
            found_shift[i] = 4 #create by CD
            add_rand[i] = (x - (sum2 - temp[3])) / temp[3]
    else:
        print("Error: no jump found for chain",i)

    return


@cuda.jit
def chain_control_kernel(Z,QN,chain_time,stress,reach_flag,next_sync_time,max_sync_time,write_time,time_resolution):
    
    sum_stress = cuda.local.array(shape=3,dtype=float32)
    
    i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x #chain index
    
    
    if i >= QN.shape[0]:
        return
            
    if reach_flag[i] != 0:
        return
    
    if (chain_time[i] >= next_sync_time) and chain_time[i] <= (write_time[i]*time_resolution):
        
        reach_flag[i] = 1
        
        return
        
    if (chain_time[i] > write_time[i]*time_resolution):
        
        sum_stress[0] = sum_stress[1] = sum_stress[2] = 0.0

        if int((chain_time[i]%max_sync_time)/time_resolution)==0 and write_time[i] != 0:
            arr_index = int(max_sync_time/time_resolution)
        else:
            arr_index = int((chain_time[i]%max_sync_time)/time_resolution)            
        
        tz = int(Z[i])
        
        for j in range(0,tz):

            sum_stress[0] -= (3.0*QN[i,j,0]*QN[i,j,1] / QN[i,j,3])
            sum_stress[1] -= (3.0*QN[i,j,1]*QN[i,j,2] / QN[i,j,3])
            sum_stress[2] -= (3.0*QN[i,j,0]*QN[i,j,2] / QN[i,j,3])

        stress[arr_index,0,i] = sum_stress[0]
        stress[arr_index,1,i] = sum_stress[1]
        stress[arr_index,2,i] = sum_stress[2]
        
        write_time[i]+=1
    
        return
        
        
    
    
@cuda.jit
def chain_kernel(Z, QN, create_SDCD_chains, QN_create_SDCD, chain_time, time_compensation, reach_flag, found_shift, found_index, tdt, sum_W_sorted, \
                        t_cr, new_t_cr, f_t, tau_CD, new_tau_CD, rand_used, add_rand, tau_CD_used_SD, tau_CD_used_CD, tau_CD_gauss_rand_SD, tau_CD_gauss_rand_CD):
   
    
    i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x #chain index

       
    if i >= QN.shape[0]:
        return
    
    if reach_flag[i]!=0:
        return
    
    tz = int(Z[i])
    
    #store shifted arrays for chains that create an entanglement by SD or CD
    createIdx = int(create_SDCD_chains[i])
    if createIdx >= 0:
        for j in range(1,tz+1):
            QN_create_SDCD[createIdx,j,0] = QN[i,j-1,0]
            QN_create_SDCD[createIdx,j,1] = QN[i,j-1,1]
            QN_create_SDCD[createIdx,j,2] = QN[i,j-1,2]
            QN_create_SDCD[createIdx,j,3] = QN[i,j-1,3]
        for j in range(1,tz+1):
            new_t_cr[createIdx,j] = t_cr[i,j-1]
            new_tau_CD[createIdx,j] = tau_CD[i,j-1]
    else:
        pass
    
    jumpIdx = int(found_index[i])
    jumpType = int(found_shift[i])

    if sum_W_sorted[i] == 0.0:
        print('Error: timestep size is infinity for chain',i)

    olddt = tdt[i]
    tdt[i] = 1.0 / (sum_W_sorted[i] / 1e6)   

    #Use Kahan summation to update time of chain
    y = tdt[i] - time_compensation[i]
    t = chain_time[i] + y
    time_compensation[i] = (t - chain_time[i] - y)
    chain_time[i] = t 
        
    rand_used[i]+=1

    if jumpType == 0 or jumpType == 1:
        apply_shuffle(i, jumpIdx, jumpType, QN)

    elif jumpType == 2 or jumpType == 5:
        apply_destroy(i, jumpIdx, jumpType, QN, Z, t_cr, tau_CD, f_t, chain_time)
        
    elif jumpType == 3 or jumpType == 6:
        apply_create_SD(i, jumpIdx, createIdx, jumpType, QN, QN_create_SDCD, Z, t_cr, new_t_cr, tau_CD, new_tau_CD, chain_time, tau_CD_used_SD, tau_CD_gauss_rand_SD)

    elif jumpType == 4:
        apply_create_CD(i, jumpIdx, createIdx, QN, QN_create_SDCD, Z, t_cr, new_t_cr, tau_CD, new_tau_CD, tau_CD_used_CD, tau_CD_gauss_rand_CD, add_rand)
    
    else:
        return
    


@cuda.jit(device=True)
def apply_shuffle(chainIdx, jumpIdx, jumpType, QN):
    if jumpType == 0: #shuffling left
        QN[chainIdx,jumpIdx,3] += 1
        QN[chainIdx,jumpIdx+1,3] -= 1
    elif jumpType == 1: #shuffling right
        QN[chainIdx,jumpIdx,3] -= 1
        QN[chainIdx,jumpIdx+1,3] += 1
    return


@cuda.jit(device=True)
def apply_destroy(chainIdx, jumpIdx, jumpType, QN, Z, t_cr, tau_CD, f_t, chain_time):
    
    tz = int(Z[chainIdx])

    cr_time = t_cr[chainIdx,jumpIdx]
    
    Z[chainIdx]-=1

    if cr_time != 0:
        f_t[chainIdx] = math.log10(chain_time[chainIdx]- cr_time) + 10
        
        
    if jumpIdx == 0:
        #destroy entanglement at beginning of chain
        
        #destroy first strand and set N
        QN[chainIdx,jumpIdx,3] = QN[chainIdx,jumpIdx,3] + QN[chainIdx,jumpIdx+1,3]
        QN[chainIdx,jumpIdx,0] = QN[chainIdx,jumpIdx,1] = QN[chainIdx,jumpIdx,2] = 0.0

        t_cr[chainIdx,jumpIdx] = t_cr[chainIdx,jumpIdx+1]
        tau_CD[chainIdx,jumpIdx] = tau_CD[chainIdx,jumpIdx+1]

        #shift all strands -1 in array for deleted strand
        for threadIdx in range(jumpIdx+1,tz-1):
            for m in range(0,4):
                QN[chainIdx,threadIdx,m] = QN[chainIdx,threadIdx+1,m]

            t_cr[chainIdx,threadIdx] = t_cr[chainIdx,threadIdx+1]
            tau_CD[chainIdx,threadIdx] = tau_CD[chainIdx,threadIdx+1]
        
        #set previous free strand at end of chain to 0s
        QN[chainIdx,tz-1,0] = QN[chainIdx,tz-1,1] = QN[chainIdx,tz-1,2] = QN[chainIdx,tz-1,3] = 0.0

        return


    elif jumpIdx == tz-2:
        #destroy entanglement at end of chain
        
        QN[chainIdx,jumpIdx,3] = QN[chainIdx,jumpIdx,3] + QN[chainIdx,jumpIdx+1,3]
        QN[chainIdx,jumpIdx,0] = QN[chainIdx,jumpIdx,1] = QN[chainIdx,jumpIdx,2] = 0.0

        t_cr[chainIdx,jumpIdx] = 0.0
        tau_CD[chainIdx,jumpIdx] = 0.0

        QN[chainIdx,jumpIdx+1,3] = 0.0
        QN[chainIdx,jumpIdx+1,0] = QN[chainIdx,jumpIdx+1,1] = QN[chainIdx,jumpIdx+1,2] = 0.0

        t_cr[chainIdx,jumpIdx+1] = 0.0
        tau_CD[chainIdx,jumpIdx+1] = 0.0

        return
            
    else:
        
        #destroy entanglement at jumpIdx
        for m in range(0,4):
            QN[chainIdx,jumpIdx,m] = QN[chainIdx,jumpIdx,m] + QN[chainIdx,jumpIdx+1,m]
        
        t_cr[chainIdx,jumpIdx] = t_cr[chainIdx,jumpIdx+1]
        tau_CD[chainIdx,jumpIdx] = tau_CD[chainIdx,jumpIdx+1]

        #shift all other strands to the strand+1 value in array (shifting -1 in array)
        for threadIdx in range(jumpIdx+1,tz-1):
            for m in range(0,4):
                QN[chainIdx,threadIdx,m] = QN[chainIdx,threadIdx+1,m]

            t_cr[chainIdx,threadIdx] = t_cr[chainIdx,threadIdx+1]
            tau_CD[chainIdx,threadIdx] = tau_CD[chainIdx,threadIdx+1]
        
        #set last strand in old array to 0s
        QN[chainIdx,tz-1,0] = QN[chainIdx,tz-1,1] = QN[chainIdx,tz-1,2] = QN[chainIdx,tz-1,3] = 0.0
        
        return 


@cuda.jit(device=True)
def apply_create_SD(chainIdx, jumpIdx, createIdx, jumpType, QN, QN_create_SD, Z, t_cr, new_t_cr, tau_CD, new_tau_CD, chain_time, tau_CD_used_SD, tau_CD_gauss_rand_SD):

    tz = int(Z[chainIdx])
    
    Z[chainIdx]+=1
    
    QN1 = QN[chainIdx,jumpIdx,:]
    
    #pull random gaussian and tau_CD for sliding dynamics
    temp = tau_CD_gauss_rand_SD[chainIdx,int(tau_CD_used_SD[chainIdx]),:]
    
    #if random numbers are used, add 1 to counter to shift random number array
    tau_CD_used_SD[chainIdx]+=1
    
    #set tau_CD and new N for new strand
    tCD = temp[3]
    temp[3] = QN1[3] - 1.0

    if tz==1:
        sigma = 0.0
    else:
        sigma = math.sqrt(temp[3] / 3.0)

    #calculate Q for new strand
    temp[0]*=sigma
    temp[1]*=sigma
    temp[2]*=sigma
    
    
    if jumpType == 3.0:
        #create new strand at end of chain from sliding dynamics
        
        #set strand at end
        QN[chainIdx,jumpIdx+1,0] = QN[chainIdx,jumpIdx+1,1] = QN[chainIdx,jumpIdx+1,2] = 0.0
        QN[chainIdx,jumpIdx+1,3] = 1.0

        t_cr[chainIdx,jumpIdx+1] = 0.0
        tau_CD[chainIdx,jumpIdx+1] = 0.0

        #set new strand at tz-1 
        for m in range(0,4):
            QN[chainIdx,jumpIdx,m] = temp[m]
        
        tau_CD[chainIdx,jumpIdx] = tCD
        t_cr[chainIdx,jumpIdx] = chain_time[chainIdx]


    elif jumpType == 6.0:
        #create new strand at beginning of chain from sliding dynamics
        
        #shift all indices of other strands in array +1 to create new strand
        for entIdx in range(jumpIdx+1,tz):
            for m in range(0,4):
                QN[chainIdx,entIdx+1,m] = QN_create_SD[createIdx,entIdx+1,m]
        
            t_cr[chainIdx,entIdx] = new_t_cr[createIdx,entIdx]
            tau_CD[chainIdx,entIdx] = new_tau_CD[createIdx,entIdx]
        
        #create new strand Q and N
        for m in range(0,4):
            QN[chainIdx,jumpIdx+1,m] = temp[m]
        
        #update free end at beginning
        QN[chainIdx,jumpIdx,3] = 1.0
        QN[chainIdx,jumpIdx,0] = QN[chainIdx,jumpIdx,1] = QN[chainIdx,jumpIdx,2] = 0.0
        
        t_cr[chainIdx,jumpIdx] = chain_time[chainIdx]
        tau_CD[chainIdx,jumpIdx] = tCD
        
            
    else:
        return


@cuda.jit(device=True)
def apply_create_CD(chainIdx, jumpIdx, createIdx, QN, QN_create_SD, Z, t_cr, new_t_cr, tau_CD, new_tau_CD, tau_CD_used_CD, tau_CD_gauss_rand_CD, add_rand):

    tz = int(Z[chainIdx])
    
    Z[chainIdx]+=1
    
    QN1 = QN[chainIdx,jumpIdx,:]

    temp = tau_CD_gauss_rand_CD[chainIdx,int(tau_CD_used_CD[chainIdx]),:]

    tau_CD_used_CD[chainIdx]+=1

    tCD = temp[3]
    
    new_N = math.floor(0.5 + add_rand[chainIdx] * (QN1[3] - 2.0)) + 1.0
    
    ratio_N = new_N / QN1[3]
    
    
    if jumpIdx == 0:
        
        #shift other entanglements
        for entIdx in range(jumpIdx+1,tz):
            for m in range(0,4):
                QN[chainIdx,entIdx+1,m] = QN_create_SD[createIdx,entIdx+1,m]
        
            t_cr[chainIdx,entIdx] = new_t_cr[createIdx,entIdx]
            tau_CD[chainIdx,entIdx] = new_tau_CD[createIdx,entIdx]
            
        #set tau_CD t_cr (creation time of entanglement is 0 for constraint dynamics)
        tau_CD[chainIdx,jumpIdx] = tCD
        t_cr[chainIdx,jumpIdx] = 0.0
        
        #calculate Q and N for new and previous strand 
        sigma = math.sqrt( (QN1[3] - new_N) / 3.0)
        temp[0] *= sigma
        temp[1] *= sigma
        temp[2] *= sigma
        temp[3] = QN1[3] - new_N
        
        #previous strand is updated
        for m in range(0,4):
            QN[chainIdx,jumpIdx+1,m] = temp[m]
        
        #at jump index, create new strand Q and N
        QN[chainIdx,jumpIdx,3] = new_N
        QN[chainIdx,jumpIdx,0] = QN[chainIdx,jumpIdx,1] = QN[chainIdx,jumpIdx,2] = 0.0
        
        return
        

    elif jumpIdx == tz-1:
        
        #set tau_CD t_cr (creation time of entanglement is 0 for constraint dynamics)
        tau_CD[chainIdx,jumpIdx] = tCD
        t_cr[chainIdx,jumpIdx] = 0.0
        
        #calculate Q and N for new and previous strand
        sigma = math.sqrt(new_N / 3.0)
        temp[0] *= sigma
        temp[1] *= sigma
        temp[2] *= sigma
        temp[0] += (QN1[0] * ratio_N)
        temp[1] += (QN1[1] * ratio_N)
        temp[2] += (QN1[2] * ratio_N)
        temp[3] = new_N
        
        #set new last strand
        QN[chainIdx,jumpIdx+1,0] = QN[chainIdx,jumpIdx+1,1] = QN[chainIdx,jumpIdx+1,2] = 0.0
        QN[chainIdx,jumpIdx+1,3] = QN1[3] - new_N
        
        #set new strand Q and  N
        for m in range(0,4):
            QN[chainIdx,jumpIdx,m] = temp[m]
        
        return
        
    elif jumpIdx !=0 and jumpIdx != tz-1:
        
        #shift all strands in front of new entanglement
        for entIdx in range(jumpIdx+1,tz):
            for m in range(0,4):
                QN[chainIdx,entIdx+1,m] = QN_create_SD[createIdx,entIdx+1,m]
        
            t_cr[chainIdx,entIdx] = new_t_cr[createIdx,entIdx]
            tau_CD[chainIdx,entIdx] = new_tau_CD[createIdx,entIdx]
        
        #calculate Q and N for new and previous strand
        sigma = math.sqrt((new_N * (QN1[3] - new_N)) / (3.0 * QN1[3]))
        temp[0] *= sigma
        temp[1] *= sigma
        temp[2] *= sigma
        temp[0] += (QN1[0] * ratio_N)
        temp[1] += (QN1[1] * ratio_N)
        temp[2] += (QN1[2] * ratio_N) 
        temp[3] = new_N
        
        #create new strands Q and N at jumpIdx and jumpIdx+1
        for m in range(0,4):
            QN[chainIdx,jumpIdx+1,m] = QN1[m] - temp[m]   
            QN[chainIdx,jumpIdx,m] = temp[m]
        
        #set tau_CD and creation time of new entanglement
        tau_CD[chainIdx,jumpIdx] = tCD
        t_cr[chainIdx,jumpIdx] = 0
        
        return
        
        
    
    else:
        return
        
            
        
        
        
    
    

        
    

    







