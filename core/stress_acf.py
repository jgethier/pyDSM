import numpy as np
from numba import cuda, float32
import math


@cuda.jit
def stress_sample(stress, sampf, uplim, time, stress_corr, corr_array):
    
    i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x #chain index
    
    if i >= stress_corr.shape[0]:
        return
   
    data = stress[0:,i] #read stress data for chain i 
    corr = corr_array[0:,i]
        
    p = 8 #block transformation parameters
    m = 2 #block transformation parameters
    array_index = -1 #initialize array indexing for final results
    for k in range(1,p*m):
        array_index+=1
        if i == 0: #only record time for one chain, do not need to repeat for each chain
            time[array_index] = (k/sampf)
        stress_block(i, data, k, stress_corr, array_index, corr) #get the average correlation and error for time lag k
    for l in range(1,int(uplim)):
        for j in range(p*m**l,p*m**(l+1),m**l):
            array_index += 1
            if i == 0:
                time[array_index] = (j/sampf) #only record time for one chain, do not need to repeat for each chain
            stress_block(i, data, j, stress_corr, array_index, corr) #get the average correlation and error for time lag j
    return


@cuda.jit(device=True)
def stress_block(chainIdx, chainData, tj, corr, arr_index, xV):
    
    #number of correlations
    n=int(len(chainData)-tj)
    
    #begin correlation averaging for timelag tj
    xav = 0
    for r in range(0,n):
        xV[r] = chainData[r]*chainData[int(r+tj)] #correlation between time and time + lag
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