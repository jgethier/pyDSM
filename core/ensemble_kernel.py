import math
from numba import cuda, float32, int32

@cuda.jit(device=True)
def apply_flow(Q,dt,kappa):
    '''
    Device function (run on GPU) to apply shear flow to ensemble of chains
    Inputs: Q - strand conformation
            dt - chain time step
            kappa - strain tensor
    Returns: deformed strand orientation
    '''
    return Q[0] + dt*kappa[0]*Q[0] + dt*kappa[1]*Q[1] + dt*kappa[2]*Q[2], Q[1] + dt*kappa[3]*Q[0] + dt*kappa[4]*Q[1] + dt*kappa[5]*Q[2], Q[2] + dt*kappa[6]*Q[0] + dt*kappa[7]*Q[1] + dt*kappa[8]*Q[2], Q[3]

@cuda.jit
def reset_chain_flag(reach_flag):
    '''
    GPU function to reset the chain flag after the chain time has reached the sync time
    Inputs: reach_flag - array of binary values (0 or 1) indicating whether a chain has reached the sync time
    Returns: None (updates reach_flag device array)
    '''
    i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x #chain index
    
    if i >= reach_flag.shape[0]:
        return 

    reach_flag[i] = 0

    return 

@cuda.jit
def reset_chain_time(chain_time,write_time,flow_time):

    i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x #chain index
    
    if i >= chain_time.shape[0]:
        return 

    chain_time[i] -= flow_time
    write_time[i] = 0

    return

@cuda.jit
def calc_new_Q_fraction(Z,new_Z,temp_Z,found_shift,found_index,result,chain_time,max_sync_time,time_resolution,write_time):

    i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x #chain index
    
    if i >= Z.shape[0]:
        return
    
    jumpIdx = int(found_index[i])
    jumpType = int(found_shift[i])
    tz = int(Z[i])

    total_Z = tz

    # if (jumpType == 3) or (jumpType == 6) or (jumpType == 4):
    #     new_QN[i] += 1
    
    if jumpType == 4 or jumpType == 6:
        for j in range(1,tz+1):
            temp_Z[i,j] = int(new_Z[i,j-1])

    if jumpType == 4:
        total_Z+=1
        if jumpIdx == 0:
            #shift other entanglements
            for entIdx in range(jumpIdx+1,tz+1):
                new_Z[i,entIdx] = int(temp_Z[i,entIdx])
            new_Z[i,jumpIdx] = 1

        for entIdx in range(jumpIdx+1,tz+1):
            new_Z[i,entIdx] = temp_Z[i,entIdx]
        new_Z[i,jumpIdx] = 1

    if jumpType == 6:
        total_Z+=1
        for entIdx in range(jumpIdx+1,tz+1):
            new_Z[i,entIdx] = temp_Z[i,entIdx]
        new_Z[i,jumpIdx] = 1
    
    if jumpType == 3:
        total_Z += 1
        new_Z[i,jumpIdx+1] = 0
        new_Z[i,jumpIdx] = 1

    if jumpType == 2 or jumpType == 5:
        total_Z -= 1
        if jumpIdx == 0:
            new_Z[i,jumpIdx] = new_Z[i,jumpIdx+1]
            #shift all strands -1 in array for deleted strand
            for threadIdx in range(jumpIdx+1,tz-1):
                new_Z[i,threadIdx] = new_Z[i,threadIdx+1]
        elif jumpIdx == tz-2:
            new_Z[i,jumpIdx] = 0
            new_Z[i,jumpIdx+1] = 0
        else:
            new_Z[i,jumpIdx] = new_Z[i,jumpIdx+1]
            #shift all other strands to the strand+1 value in array (shifting -1 in array)
            for threadIdx in range(jumpIdx+1,tz-1):
                new_Z[i,threadIdx] = new_Z[i,threadIdx+1]

    count_new_Z = 0
    for j in range(0,total_Z-1):
        if new_Z[i,j] == 1:
            count_new_Z+=1

    if int((chain_time[i]%max_sync_time)/time_resolution[0])==0 and write_time[i] != 0:
        arr_index = int(max_sync_time/time_resolution[0])
    else:
        arr_index = int((chain_time[i]%max_sync_time)/time_resolution[0])

    result[i,arr_index,7] = count_new_Z/(total_Z-1) # normalize by number of entanglements (Z-2)

    return   

@cuda.jit
def calc_flow_stress(Z,QN,stress):
    '''
    GPU function that calculates the flow stress tensor (only used when EQ_calc is set to 'msd')
    Inputs: Z - number of entangled strands (including dangling ends)
            QN - chain conformations and number of Kuhn steps in each strand
            stress - array to hold stress values
    Returns: None (updates stress device array)
    '''
    i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x #chain index
    
    if i >= QN.shape[0]:
        return 
    
    stress_xx = stress_yy = stress_zz = stress_xy = stress_yz = stress_xz = 0.0
    for j in range(0,int(Z[i])):
        stress_xx -= (3.0*QN[i,j,0]*QN[i,j,0] / QN[i,j,3]) #tau_xx
        stress_yy -= (3.0*QN[i,j,1]*QN[i,j,1] / QN[i,j,3]) #tau_yy
        stress_zz -= (3.0*QN[i,j,2]*QN[i,j,2] / QN[i,j,3]) #tau_zz
        stress_xy -= (3.0*QN[i,j,0]*QN[i,j,1] / QN[i,j,3]) #tau_xy
        stress_yz -= (3.0*QN[i,j,1]*QN[i,j,2] / QN[i,j,3]) #tau_yz
        stress_xz -= (3.0*QN[i,j,0]*QN[i,j,2] / QN[i,j,3]) #tau_xz
    
    stress[i,0,0] = stress_xx
    stress[i,0,1] = stress_yy
    stress[i,0,2] = stress_zz
    stress[i,0,3] = stress_xy
    stress[i,0,4] = stress_yz
    stress[i,0,5] = stress_xz
    stress[i,0,6] = Z[i]
    
    return

        
@cuda.jit
def calc_strand_prob(Z,QN,flow,tdt,kappa,tau_CD,shift_probs,CD_flag,CD_create_prefact,beta,NK):
    '''
    GPU function to calculate probabilities for Kuhn step shuffling, entanglement creation or destruction
    Inputs: Z - number of entangled strands for each chain
            QN - chain conformations and number of Kuhn steps in each strand
            flow - boolean variable to determine whether to apply deformation
            tdt - time steps of each chain
            kappa - strain tensor (if flow = True, kappa contains non-zero values)
            tau_CD - entanglement lifetime for probability of destruction due to constraint dynamics
            shift_probs - array to store probabilities for chain entanglement process (shuffle, creation, destroy, etc)
            CD_flag - binary flag for determining whether constraint dynamics are implemented (0 - off, 1 - on)
            CD_create_prefact - variable used to calculate probability to create entanglement
    Returns: None
    '''
    i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x #chain index
    j = cuda.blockIdx.y*cuda.blockDim.y + cuda.threadIdx.y #strand index

    if i >= QN.shape[0]:
        return

    tz = int(Z[i])

    if j >= tz:
        return
    
    shift_probs[i,j,0] = shift_probs[i,j,1] = shift_probs[i,j,2] = shift_probs[i,j,3] = 0.0
    shift_probs[i,tz,0] = shift_probs[i,tz,1] = shift_probs[i,tz,2] = shift_probs[i,tz,3] = 0.0
    shift_probs[i,tz-1,0] = shift_probs[i,tz-1,1] = shift_probs[i,tz-1,2] = shift_probs[i,tz-1,3] = 0.0

    tcd = tau_CD[i,j]
    
    QN_i = QN[i, j, :]
    
    if bool(flow[0]):
        dt = tdt[i]
        QN[i,j,:] = apply_flow(QN_i,dt,kappa)
        cuda.syncthreads()
    
    if j<tz-1:
        
        QN_ip1 = QN[i, j+1, :]
            
        Q_i = QN_i[0]**2 + QN_i[1]**2 + QN_i[2]**2
        Q_ip1 = QN_ip1[0]**2 + QN_ip1[1]**2 + QN_ip1[2]**2

        if QN_ip1[3] > 1.0:
                sig1 = 0.75 / (QN_i[3]*(QN_i[3]+1))
                sig2 = 0.75 / (QN_ip1[3]*(QN_ip1[3]-1))
                if Q_i==0.0:
                        prefactor1 = 1.0
                        f1 = 2.0*QN_i[3]+0.5
                else:
                        prefactor1 = QN_i[3] / (QN_i[3] + 1)
                        f1 = QN_i[3]
                if Q_ip1 == 0.0:
                        prefactor2 = 1.0
                        f2 = 2.0*QN_ip1[3]-0.5
                else:
                        prefactor2 = QN_ip1[3] / (QN_ip1[3] - 1)
                        f2 = QN_ip1[3]
                        
                friction = 2.0 / (f1 + f2)
                shift_probs[i, j, 0] = friction*math.pow(prefactor1*prefactor2,0.75)*math.exp(Q_i*sig1-Q_ip1*sig2)

        if QN_i[3] > 1.0:
                sig1 = 0.75 / (QN_i[3]*(QN_i[3]-1))
                sig2 = 0.75 / (QN_ip1[3]*(QN_ip1[3]+1))
                
                if Q_i == 0.0:
                        prefactor1 = 1.0
                        f1 = 2.0*QN_i[3]-0.5
                else:
                        prefactor1 = QN_i[3] / (QN_i[3] - 1)
                        f1 = QN_i[3]
                if Q_ip1 == 0.0:
                        prefactor2 = 1.0
                        f2 = 2.0*QN_ip1[3]+0.5
                else:
                        prefactor2 = QN_ip1[3] / (QN_ip1[3] + 1)
                        f2 = QN_ip1[3]

                friction = 2.0 / (f1 + f2)
                shift_probs[i, j, 1] = friction*math.pow(prefactor1*prefactor2,0.75)*math.exp(-Q_i*sig1+Q_ip1*sig2)

        if CD_flag[0]==1:
                shift_probs[i, j, 2] = tcd
                shift_probs[i, j, 3] = CD_create_prefact[0]*(QN_i[3]-1.0)

    return


@cuda.jit
def calc_chainends_prob(Z, QN, shift_probs, CD_flag, CD_create_prefact, beta, Nk):
    '''
    GPU function to calculate probabilities for Kuhn step shuffling, entanglement creation or destruction at chain ends
    Inputs: Z - number of entangled strands for each chain
            QN - chain conformations and number of Kuhn steps in each strand
            shift_probs - array to store probabilities for chain entanglement process (shuffle, creation, destroy, etc)
            CD_flag - binary flag for determining whether constraint dynamics are implemented (0 - off, 1 - on)
            CD_create_prefact - variable used to calculate probability to create entanglement
            beta - entanglement activity parameter (e.g. beta = 1 for CFSM)
            Nk - maximum number of Kuhn steps in chain (from input.yaml file)
    Returns: None
    '''
    i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x #chain index

    if i >= QN.shape[0]:
        return

    tz = int(Z[i])
    
    QNfirst = QN[i,0]
    QNlast = QN[i,tz-1]
    
    shift_probs[i,tz,0] = shift_probs[i,tz,1] = shift_probs[i,tz,2] = shift_probs[i,tz,3] = 0.0
    shift_probs[i,tz-1,0] = shift_probs[i,tz-1,1] = shift_probs[i,tz-1,2] = shift_probs[i,tz-1,3] = 0.0

    if tz == 1:
        shift_probs[i,tz-1,1] = (1.0 / (beta[0]*Nk[0]))           
        shift_probs[i,tz,1] = (1.0 / (beta[0]*Nk[0]))

    else:
        if QNfirst[3] == 1.0: #destruction by SD at the beginning
            QNfirst_n = QN[i,1]
            if tz == 2:
                c = QNfirst_n[3] + 0.25
            else:
                c = QNfirst_n[3] * 0.5

            shift_probs[i,tz,0] = (1.0 / (c+0.75))

        else: #creation by SD at the beginning
            shift_probs[i,tz,1] = (2.0 / (beta[0] * (QNfirst[3]+0.5)))

        if QNlast[3] == 1.0: #destruction by SD at the end

            QNlast_p = QN[i,tz-2]
            if tz == 2:
                c = QNlast_p[3] + 0.25
            else:
                c = QNlast_p[3] * 0.5
            shift_probs[i,tz-1,0] = (1.0 / (c+0.75))
            
        else: #creation by SD at the end
            shift_probs[i,tz-1,1] = (2.0 / (beta[0] * (QNlast[3]+0.5) ))

    if CD_flag[0]==1:
        shift_probs[i,tz-1,3] = CD_create_prefact[0]*(QNlast[3]-1.0)

    return


# @cuda.jit
# def choose_kernel(Z,shift_probs,sum_W_sorted,uniform_rand,rand_used,found_index,found_shift,add_rand,CD_flag,NK):

#     s = cuda.shared.array(1024,float32)

#     i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x #chain
#     j = cuda.blockIdx.y*cuda.blockDim.y + cuda.threadIdx.y #strand index
    
#     if NK[0]>1024 and i == 0 and j == 0:
#         print("NK is larger than 1024. Reduce chain size.")
 
#     tz = int(Z[i])

#     temp = shift_probs[i,i,:]
#     s[i] = 0.0
#     cuda.syncthreads()

#     if CD_flag[0] == 1:
#         var = temp[0] + temp[1] + temp[2] + temp[3]
#     else:
#         var = temp[0] + temp[1]

#     d = 1
#     while d < 32:
#         var2 = cuda.shfl_up_sync(-1,var,d)
#         if (i % 32 >= d):
#             var += var2
#         d <<= 1

#     if (i % 32 == 31): 
#         s[int(i / 32)] = var

#     cuda.syncthreads()

#     if (i < 32):
#         var2 = 0.0
#         if (i < cuda.blockDim.x / 32):
#             var2 = s[i]
#         d=1
#         while d<32:
#             var3 = cuda.shfl_up_sync(-1,var2,d)
#             if (i % 32 >= d): 
#                 var2 += var3
#             d <<= 1
        
#         if (i < cuda.blockDim.y / 32):
#             s[i] = var2
    
#     cuda.syncthreads()

#     if (i >= 32):
#         var += s[int(i / 32 - 1)]
    
#     cuda.syncthreads()

#     s[i] = var

#     cuda.syncthreads()

#     sum_W_sorted[i] = s[NK[0]-1]
#     if i == 0 and i == 0:
#         print(s[NK[0]-1])
#     x = float(s[NK[0]-1])*uniform_rand[i,int(rand_used[i])]
    
#     if i == 0:
#         left = 0
#     elif i>0:
#         left = s[i-1]

#     if i == 0:
#         print(left)

#     xFound = bool((left < x) & (x <= left + temp[0]))
#     yFound = bool((left + temp[0] < x) & (x <= left + temp[0] + temp[1]))
#     zFound = bool((left + temp[0] + temp[1] < x) & (x <= left + temp[0] + temp[1] + temp[2]))
#     wFound = bool((left + temp[0] + temp[1] + temp[2] < x) & (x <= left + temp[0] + temp[1] + temp[2] + temp[3]))

#     ii = i
#     if xFound or yFound or zFound or wFound:
#         found_index[i] = i
#         if xFound:
#             found_shift[i] = 0
#             if (ii == tz - 1):
#                 found_index[i] = ii-1
#                 found_shift[i] = 5 #destroy at end by SD
#             if (ii == tz):
#                 found_index[i] = 0
#                 found_shift[i] = 5 #destroy at beginning by SD
#         elif yFound:
#             found_shift[i] = 1
#             if (ii == tz - 1):
#                 found_shift[i] = 3 #create at end by SD
#             if (ii == tz):
#                 found_index[i] = 0
#                 found_shift[i] = 6 #create at beginning by SD
#         elif zFound:
#             found_shift[i] = 2 #destroy by CD
#         elif wFound:
#             found_shift[i] = 4 #create by CD
#             add_rand[i] = float((x - left - temp[0] - temp[1] - temp[2])) / float(temp[3])
#     # else:
#     #     print("Error: no jump found for chain",i)

#     return


@cuda.jit
def choose_step_kernel(Z,shift_probs,sum_W_sorted,uniform_rand,rand_used,found_index,found_shift,add_rand,CD_flag):
    '''
    GPU function to calculate which entanglement process will be applied to each chain (Kuhn step shuffle, entanglement creation/destruction)
    Inputs: Z - number of entangled strands for each chain
            shift_probs - probabilities for entanglement process
            sum_W_sorted - sum of all probabilities for each chain
            uniform_rand - uniform random number array for determining entanglement process
            rand_used - array to keep track of which random numbers were used in uniform_rand
            found_index - strand index at which entanglement process will occur in chain
            found_shift - value assigned to each chain for which entanglement process that will occur
            add_rand - fraction of remaining probability for determining number of Kuhn steps during creation of a strand (see apply_create_CD in apply_step_kernel)
            CD_flag - binary flag for determining whether constraint dynamics are implemented (0 - off, 1 - on)
    '''
    i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x #chain index
    
    if i >= shift_probs.shape[0]:
        return
    
    tz = int(Z[i])

    sum1 = 0
    for j in range(0,tz+1):
        temp = shift_probs[i,j,:]

        if CD_flag[0]==1:
            sum1 += (temp[0] + temp[1] + temp[2] + temp[3])
        else:
            sum1 += (temp[0] + temp[1])
                
    sum_W_sorted[i] = sum1
    x = sum1*uniform_rand[i,int(rand_used[i])]
    
    xFound = yFound = zFound = wFound = False
    sum2 = 0
    
    for j in range(0,tz+1):
        
        temp = shift_probs[i,j,:]
        
        if sum2 < x:
            
            xFound = bool((sum2 < x) & (x <= sum2 + temp[0]))
            sum2+=temp[0]

            yFound = bool((sum2 < x) & (x <= sum2 + temp[1]))
            sum2+=temp[1]

            if CD_flag[0]==1:
                zFound = bool((sum2 < x) & (x <= sum2 + temp[2]))
                sum2+=temp[2]
                wFound = bool((sum2 < x) & (x <= sum2 + temp[3]))
                sum2+=temp[3]

        if xFound or yFound or zFound or wFound:
            break
    
    ii=j
    
    if xFound or yFound or zFound or wFound:
        found_index[i] = j
        if xFound:
            found_shift[i] = 0
            if (ii == tz - 1):
                found_index[i] = ii-1
                found_shift[i] = 5 #destroy at end by SD
            if (ii == tz):
                found_index[i] = 0
                found_shift[i] = 5 #destroy at beginning by SD
        elif yFound:
            found_shift[i] = 1
            if (ii == tz - 1):
                found_shift[i] = 3 #create at end by SD
            if (ii == tz):
                found_index[i] = 0
                found_shift[i] = 6 #create at beginning by SD
        elif zFound:
            found_shift[i] = 2 #destroy by CD
        elif wFound:
            found_shift[i] = 4 #create by CD
            add_rand[i] = float(x - (sum2 - temp[3])) / float(temp[3])
    else:
        print("Error: no jump found for chain",i)

    return


@cuda.jit
def time_control_kernel(Z,QN,QN_first,NK,chain_time,tdt,result,calc_type,flow,flow_off,reach_flag,next_sync_time,max_sync_time,write_time,time_resolution,result_index,postprocess):
    
    
    i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x #chain index
    
    if i >= QN.shape[0]:
        return

    if not postprocess:
        result[i,result_index,0] = result[i,result_index,1] = result[i,result_index,2] = result[i,result_index,3] = 0.0
            
    if reach_flag[i] != 0:
        return

    if (chain_time[i] >= next_sync_time) and chain_time[i] <= (write_time[i]*time_resolution[0]):
        
        #if sync time is reached and stress was recorded, set reach flag to 1
        reach_flag[i] = 1
        tdt[i] = 0.0
        
        return
        
    if (chain_time[i] > write_time[i]*time_resolution[0]): #if chain time reaches next time to record stress/CoM (every time_resolution)
        
        if not bool(flow[0]) and not bool(flow_off[0]):
            
            tz = int(Z[i])
            
            if postprocess:
                if int((chain_time[i]%max_sync_time)/time_resolution[0])==0 and write_time[i] != 0:
                    arr_index = int(max_sync_time/time_resolution[0])
                else:
                    arr_index = int((chain_time[i]%max_sync_time)/time_resolution[0]) 
            else:
                arr_index = result_index 

            if calc_type[0] == 1:
                stress_xy = stress_yz = stress_xz = 0.0 

                for j in range(0,tz):
                    stress_xy -= (3.0*QN[i,j,0]*QN[i,j,1] / QN[i,j,3]) #tau_xy
                    stress_yz -= (3.0*QN[i,j,1]*QN[i,j,2] / QN[i,j,3]) #tau_yz
                    stress_xz -= (3.0*QN[i,j,0]*QN[i,j,2] / QN[i,j,3]) #tau_xz
                
                result[i,arr_index,0] = stress_xy
                if not postprocess:
                    result[i,arr_index,1] = stress_yz
                    result[i,arr_index,2] = stress_xz
                    result[i,arr_index,3] = 1.0
            
            elif calc_type[0] == 2:
                QN_1 = QN_first[i,:] #need fixed frame of reference, choosing first entanglement which is tracked during simulation
                chain_com = cuda.local.array(3,float32)
                temp = cuda.local.array(3,float32)
                prev_QN = cuda.local.array(3,float32)
                
                chain_com[0] = chain_com[1] = chain_com[2] = 0.0
                temp[0] = temp[1] = temp[2] = 0.0
                prev_QN[0] = prev_QN[1] = prev_QN[2] = 0.0
                
                for j in range(0,tz):
                    QN_i = QN[i,j,:]
                    term = cuda.local.array(3,float32)
                    term[0] = term[1] = term[2] = 0.0
                    for k in range(0,3):
                        temp[k] += prev_QN[k]
                        term[k] += temp[k]
                        term[k] += QN_i[k]/2.0
                        chain_com[k] += term[k] * QN_i[3] / NK[0]
                        prev_QN[k] = QN_i[k]
                    
                result[i,arr_index,0] = chain_com[0] + QN_1[0]
                result[i,arr_index,1] = chain_com[1] + QN_1[1]
                result[i,arr_index,2] = chain_com[2] + QN_1[2]
                if not postprocess:
                    result[i,arr_index,3] = 1.0
        
        if not bool(flow[0]) and bool(flow_off[0]): #track equilibrium variables after cessation of flow
            
            tz = int(Z[i])

            if int((chain_time[i]%max_sync_time)/time_resolution[0])==0 and write_time[i] != 0:
                arr_index = int(max_sync_time/time_resolution[0])
            else:
                arr_index = int((chain_time[i]%max_sync_time)/time_resolution[0])

            
            stress_xx = stress_yy = stress_zz = stress_xy = stress_yz = stress_xz = 0.0
            for j in range(0,int(Z[i])):
                stress_xx -= (3.0*QN[i,j,0]*QN[i,j,0] / QN[i,j,3]) #tau_xx
                stress_yy -= (3.0*QN[i,j,1]*QN[i,j,1] / QN[i,j,3]) #tau_yy
                stress_zz -= (3.0*QN[i,j,2]*QN[i,j,2] / QN[i,j,3]) #tau_zz
                stress_xy -= (3.0*QN[i,j,0]*QN[i,j,1] / QN[i,j,3]) #tau_xy
                stress_yz -= (3.0*QN[i,j,1]*QN[i,j,2] / QN[i,j,3]) #tau_yz
                stress_xz -= (3.0*QN[i,j,0]*QN[i,j,2] / QN[i,j,3]) #tau_xz
            
            result[i,arr_index,0] = stress_xx
            result[i,arr_index,1] = stress_yy
            result[i,arr_index,2] = stress_zz
            result[i,arr_index,3] = stress_xy
            result[i,arr_index,4] = stress_yz
            result[i,arr_index,5] = stress_xz
            result[i,arr_index,6] = tz

        write_time[i]+=1
    
    return
        
        
    
@cuda.jit
def apply_step_kernel(Z, QN, QN_first, QN_create_SDCD, chain_time, time_compensation, sum_W_sorted,
                 found_shift, found_index, reach_flag, tdt,
                 t_cr, new_t_cr, f_t, tau_CD, new_tau_CD, rand_used, add_rand, tau_CD_used_SD, tau_CD_used_CD, tau_CD_gauss_rand_SD, tau_CD_gauss_rand_CD):
   
    
    i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x #chain index
    
    if i >= QN.shape[0]:
        return
    
    if reach_flag[i]!=0:
        return
    
    tz = int(Z[i])

    #chosen process and location along chain
    jumpIdx = int(found_index[i])
    jumpType = int(found_shift[i])
    
    if jumpType == 4 or jumpType == 6:

        for j in range(1,tz+1):
            for m in range(0,4):
                QN_create_SDCD[i,j,m] = QN[i,j-1,m]

            new_t_cr[i,j] = t_cr[i,j-1]
            new_tau_CD[i,j] = tau_CD[i,j-1]

    # if sum_W_sorted[i] == 0:
    #     print('Error: timestep size is infinity for chain',i)

    #set time step to be length of time to make single jump
    tdt[i] = 1.0 / sum_W_sorted[i]

    #Use Kahan summation to update time of chain
    y = tdt[i] - time_compensation[i]
    t = chain_time[i] + y
    time_compensation[i] = (t - chain_time[i] - y)
    chain_time[i] = t 
        
    rand_used[i]+=1

    #apply jump processes to each chain
    if jumpType == 0 or jumpType == 1:
        apply_shuffle(i, jumpIdx, jumpType, QN)

    elif jumpType == 2 or jumpType == 5:
        apply_destroy(i, jumpIdx, jumpType, QN, QN_first, Z, t_cr, tau_CD, f_t, chain_time)
        
    elif jumpType == 3 or jumpType == 6:
        apply_create_SD(i, jumpIdx, jumpType, QN, QN_first, QN_create_SDCD[i], Z, t_cr, new_t_cr[i], tau_CD, new_tau_CD[i], chain_time, tau_CD_used_SD, tau_CD_gauss_rand_SD)

    elif jumpType == 4:
        apply_create_CD(i, jumpIdx, QN, QN_first, QN_create_SDCD[i], Z, t_cr, new_t_cr[i], tau_CD, new_tau_CD[i], tau_CD_used_CD, tau_CD_gauss_rand_CD, add_rand[i])
        
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
def apply_destroy(chainIdx, jumpIdx, jumpType, QN, QN_first, Z, t_cr, tau_CD, f_t, chain_time):
    
    tz = int(Z[chainIdx])

    cr_time = t_cr[chainIdx,jumpIdx]
    
    Z[chainIdx]-=1

    if cr_time != 0:
        f_t[chainIdx] = math.log10(chain_time[chainIdx]- cr_time) + 10
        
        
    if jumpIdx == 0:
        #destroy entanglement at beginning of chain
        
        #update change to first entanglement location
        for k in range(0,3):
            QN_first[chainIdx,k] += QN[chainIdx,jumpIdx+1,k]
            
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
def apply_create_SD(chainIdx, jumpIdx, jumpType, QN, QN_first, QN_create_SDCD, Z, t_cr, new_t_cr, tau_CD, new_tau_CD, chain_time, tau_CD_used_SD, tau_CD_gauss_rand_SD):

    tz = int(Z[chainIdx])
    
    Z[chainIdx]+=1
    
    QN1 = QN[chainIdx,jumpIdx,:]
    
    #pull random gaussian and tau_CD for sliding dynamics
    temp = tau_CD_gauss_rand_SD[chainIdx,int(tau_CD_used_SD[chainIdx]),:]
    
    #if random numbers are used, add 1 to counter to shift random number array
    tau_CD_used_SD[chainIdx]+=1
    
    #set tau_CD and new N for new strand
    tCD = temp[3]
    new_N = QN1[3] - 1.0

    if tz==1:
        sigma = 0.0
    else:
        sigma = math.sqrt(new_N / 3.0)

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
        for m in range(0,3):
            QN[chainIdx,jumpIdx,m] = temp[m]
        QN[chainIdx,jumpIdx,3] = new_N
        
        tau_CD[chainIdx,jumpIdx] = tCD
        t_cr[chainIdx,jumpIdx] = chain_time[chainIdx]


    elif jumpType == 6.0:
        #create new strand at beginning of chain from sliding dynamics
        
        #shift all indices of other strands in array +1 to create new strand
        for entIdx in range(jumpIdx+1,tz+1):
            for m in range(0,4):
                QN[chainIdx,entIdx,m] = QN_create_SDCD[entIdx,m]
        
            t_cr[chainIdx,entIdx] = new_t_cr[entIdx]
            tau_CD[chainIdx,entIdx] = new_tau_CD[entIdx]
        
        #create new strand Q and N
        for m in range(0,3):
            QN[chainIdx,jumpIdx+1,m] = temp[m]
        QN[chainIdx,jumpIdx+1,3] = new_N
        
        #update free end at beginning
        QN[chainIdx,jumpIdx,3] = 1.0
        QN[chainIdx,jumpIdx,0] = QN[chainIdx,jumpIdx,1] = QN[chainIdx,jumpIdx,2] = 0.0
        
        t_cr[chainIdx,jumpIdx] = chain_time[chainIdx]
        tau_CD[chainIdx,jumpIdx] = tCD
        
        for k in range(0,3):
            QN_first[chainIdx,k] -= temp[k]
        
            
    else:
        return


@cuda.jit(device=True)
def apply_create_CD(chainIdx, jumpIdx, QN, QN_first, QN_create_SDCD, Z, t_cr, new_t_cr, tau_CD, new_tau_CD, tau_CD_used_CD, tau_CD_gauss_rand_CD, add_rand):

    tz = int(Z[chainIdx])
    
    Z[chainIdx]+=1
    
    QN1 = QN[chainIdx,jumpIdx,:]

    temp = tau_CD_gauss_rand_CD[chainIdx,int(tau_CD_used_CD[chainIdx]),:]

    tau_CD_used_CD[chainIdx]+=1

    tCD = temp[3]
    
    new_N = math.floor(0.5 + add_rand * (QN1[3] - 2.0)) + 1.0
    
    temp[3] = new_N
    
    sigma = math.sqrt(float(new_N * (QN1[3] - new_N)) / float(3.0 * QN1[3]))
    
    if jumpIdx == tz-1:
        sigma = math.sqrt(new_N / 3.0) 
    
    ratio_N = new_N / QN1[3]
        
    if jumpIdx == 0:
        
        #calculate Q and N for new and previous strand 
        temp[3] = QN1[3] - new_N
        sigma = math.sqrt(temp[3] / 3.0)
        temp[0] *= sigma
        temp[1] *= sigma
        temp[2] *= sigma
        
        #shift other entanglements
        for entIdx in range(jumpIdx+1,tz+1):
            for m in range(0,4):
                QN[chainIdx,entIdx,m] = QN_create_SDCD[entIdx,m]
        
            t_cr[chainIdx,entIdx] = new_t_cr[entIdx]
            tau_CD[chainIdx,entIdx] = new_tau_CD[entIdx]
            
        #set tau_CD t_cr (creation time of entanglement is 0 for constraint dynamics)
        tau_CD[chainIdx,jumpIdx] = tCD
        t_cr[chainIdx,jumpIdx] = 0.0
    
        #previous strand is updated
        for m in range(0,4):
            QN[chainIdx,jumpIdx+1,m] = temp[m]

        #at jump index, create new strand Q and N
        QN[chainIdx,jumpIdx,3] = new_N
        QN[chainIdx,jumpIdx,0] = QN[chainIdx,jumpIdx,1] = QN[chainIdx,jumpIdx,2] = 0.0
        
        for k in range(0,3):
            QN_first[chainIdx,k] -= temp[k]
            
        return 
    
    temp[0] *= sigma
    temp[1] *= sigma
    temp[2] *= sigma
    temp[0] += (QN1[0] * ratio_N)
    temp[1] += (QN1[1] * ratio_N)
    temp[2] += (QN1[2] * ratio_N) 
        
    #shift all strands in front of new entanglement
    for entIdx in range(jumpIdx+1,tz+1):
        for m in range(0,4):
            QN[chainIdx,entIdx,m] = QN_create_SDCD[entIdx,m]

        t_cr[chainIdx,entIdx] = new_t_cr[entIdx]
        tau_CD[chainIdx,entIdx] = new_tau_CD[entIdx]
        
    #create new strands Q and N at jumpIdx and jumpIdx+1
    for m in range(0,4):
        QN[chainIdx,jumpIdx+1,m] = QN1[m] - temp[m]   
        QN[chainIdx,jumpIdx,m] = temp[m]
    
    #if create by CD at end of chain, set jumpIdx+1 to free end
    if jumpIdx == tz-1:
        QN[chainIdx,jumpIdx+1,0] = QN[chainIdx,jumpIdx+1,1] = QN[chainIdx,jumpIdx+1,2] = 0.0
        
    #set tau_CD and creation time of new entanglement
    tau_CD[chainIdx,jumpIdx] = tCD
    t_cr[chainIdx,jumpIdx] = 0
        
        
    return