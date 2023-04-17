from numba import cuda, float64
import math

#correlator parameters
p = 16 
m = 2

@cuda.jit(device=True)
def add_to_correlator(data,corrLevel,D,temp_D,C,N,A,M,corrtype):

    if corrLevel >= D.shape[0]:
        return 

    for j in range(1,p):
        for k in range(0,3):
            temp_D[corrLevel,j,k] = D[corrLevel,j-1,k] #temporarily store shifted data values
    
    for j in range(1,p):
        for k in range(0,3):
            D[corrLevel,j,k] = temp_D[corrLevel,j,k] #set new data values to shifted values

    for k in range(0,3):
        D[corrLevel,0,k] = data[k] #put new data value in 0 index
    
    if corrLevel == 0: #if corrLevel is 0, run calculation from 0 to p-1
        for j in range(0,p):
            N[corrLevel,j] += 1 #correlation counter incremented
            if corrtype == 1: 
                stress_corr = D[corrLevel,0,0]*D[corrLevel,j,0] #new correlation value
                C[corrLevel,j] += stress_corr                  #update running sum
            if corrtype == 2:
                msd = (D[corrLevel,0,0]-D[corrLevel,j,0])**2 + (D[corrLevel,0,1]-D[corrLevel,j,1])**2 + (D[corrLevel,0,2]-D[corrLevel,j,2])**2
                C[corrLevel,j] += msd

    else: #if corrLevel > 0, run calculation from p/m to p-1
        for j in range(int(p/m),p):
            N[corrLevel,j] += 1 #correlation counter incremented
            if corrtype == 1:
                stress_corr = D[corrLevel,0,0]*D[corrLevel,j,0] #new correlation value
                C[corrLevel,j] += stress_corr                  #update running sum
            if corrtype == 2:
                msd = (D[corrLevel,0,0]-D[corrLevel,j,0])**2 + (D[corrLevel,0,1]-D[corrLevel,j,1])**2 + (D[corrLevel,0,2]-D[corrLevel,j,2])**2
                C[corrLevel,j] += msd
    
    if (corrtype == 1) or (corrtype == 2 and M[corrLevel]==0): #add data to accumulator
        A[corrLevel,0] += data[0] #only updating accumulator if counter is 0 (non-averaging method) for MSD calc
        A[corrLevel,1] += data[1] 
        A[corrLevel,2] += data[2]
    
    #update counter
    M[corrLevel] += 1

    return


@cuda.jit
def update_correlator(n,result_array,D,D_shift,C,N,A,M,corrtype):

    i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x #chain index

    if i >= result_array.shape[0]:
        return

    S_corr = D.shape[1]
    chain_result = result_array[i]

    for j in range(0,n): #search through result array and find stress values that need to be added
        result = chain_result[j]
        if result[3] == 1.0: #if last value in results array is 1, add to correlator (1 means stress was recorded for chain i at index j)
            add_to_correlator(result,0,D[i],D_shift[i],C[i],N[i],A[i],M[i],corrtype[0])

        for corrLevel in range(0,S_corr): #after updating correlator level 0 with result value (above), check if accumulator needs to be sent to next level
            if M[i,corrLevel] == m:
                if corrtype[0] == 1:
                    for k in range(0,3):
                        A[i,corrLevel,k] /= m
                    add_to_correlator(A[i,corrLevel],corrLevel+1,D[i],D_shift[i],C[i],N[i],A[i],M[i],corrtype[0])
                if corrtype[0] == 2: 
                    add_to_correlator(A[i,corrLevel],corrLevel+1,D[i],D_shift[i],C[i],N[i],A[i],M[i],corrtype[0])
                A[i,corrLevel,0] = A[i,corrLevel,1] = A[i,corrLevel,2] = 0.0
                M[i,corrLevel] = 0
    return 

@cuda.jit
def coarse_result_array(data,g,calc_type):

    i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x #chain index
    
    if i >= data.shape[0]:
        return

    for j in range(1,p*g+1):
        if calc_type[0] == 1:
            data[i,j,0]=data[i,j*m,0]
        if calc_type[0] == 2:
            data[i,j,0]=data[i,j*m,0]
            data[i,j,1]=data[i,j*m,1]
            data[i,j,2]=data[i,j*m,2]
    return

@cuda.jit
def calc_corr(rawdata, calc_type, num_time_syncs, corrLevel, data_corr, corr_array, array_index, last_index, time_res, sim_time):
    
    i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x #chain index
    
    if i >= data_corr.shape[0]:
        return
    
    data = rawdata[i,0:last_index,:] #raw data for chain i
    corr = corr_array[i,:] #store correlation values for time t and t+lag for chain i

    if corrLevel == 0: #if first correlator level
        for j in range(0,p*m):
            array_index[i] += 1
            time_lag = j
            corr_block(i, data, time_lag, data_corr, array_index[i], corr, calc_type[0]) #get the average correlation and error for time lag
    else:
        for j in range(p*m**corrLevel,p*m**(corrLevel+1),m**corrLevel):
            if j*time_res[0] <= sim_time:
                array_index[i] += 1
                if corrLevel >= num_time_syncs: #if correlator is above last time sync
                    time_lag = int(j/m**(num_time_syncs-1))
                else:
                    time_lag = int(j/m**corrLevel)
                corr_block(i, data, time_lag, data_corr, array_index[i], corr, calc_type[0]) #get the average correlation and error for time lag
    return


@cuda.jit(device=True)
def corr_block(chainIdx, chainData, tj, corr, arr_index, xV, calc_type):
    
    #number of correlations
    n = int(len(chainData[:,0])-tj)
    
    #begin correlation averaging for timelag tj
    xav = 0
    for r in range(0,n):
        if calc_type==1:
            xV[r] = chainData[r,0]*chainData[int(r+tj),0] #correlation between time and time + lag
        elif calc_type == 2:
            xV[r] = (chainData[r,0]-chainData[int(r+tj),0])**2+(chainData[r,1]-chainData[int(r+tj),1])**2+(chainData[r,2]-chainData[int(r+tj),2])**2
        xav+=xV[r]/n  #calculate average
    c0=(xV[0]-xav)**2   
    for r in range(1,n):
        c0+=(xV[r]-xav)**2/n
    sa=math.sqrt(c0/(n-1))
    sb=sa/math.sqrt(2*(n-1))
    n=int(math.floor(n/2))
    for r in range(0,n):
        xV[r]=(xV[2*r+1]+xV[2*r])/2
    c0=(xV[0]-xav)**2
    for r in range(1,n):
        c0=c0+(xV[r]-xav)**2
    c0=c0/n
    sap=math.sqrt(c0/(n-1))
    sbp=sap/math.sqrt(2*(n-1))
    while (math.fabs(sa-sap) > sbp+sb) and (n > 4):
        sa=sap
        sb=sbp
        n=int(math.floor(n/2))
        for r in range(0,n):
            xV[r]=(xV[2*r+1]+xV[2*r])/2
        c0=(xV[0]-xav)**2
        for r in range(1,n):
            c0=c0+(xV[r]-xav)**2
        c0=c0/n
        sap=math.sqrt(c0/(n-1))
        sbp=sap/math.sqrt(2*(n-1))
    
    corr[chainIdx,arr_index,0] = xav #set average correlation value for chain i 
    corr[chainIdx,arr_index,1] = sap #set error of average correlation value for chain i