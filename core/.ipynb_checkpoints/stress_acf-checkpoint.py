import numpy as np
from numba import cuda, float32
import math


@cuda.jit
def stress_sample(stress, sampf, uplim, time, stress_corr, corr_array):
    
    i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x #chain index
    
    if i >= stress.shape[1]:
        return
    
    data = stress[0:,i] #read stress data for chain i 

    p = 8 #block transformation parameters
    m = 2 #block transformation parameters
    array_index = -1 #initialize array indexing for final results
    for k in range(1,p*m):
        array_index+=1
        if i == 0: #only record time for one chain, do not need to repeat for each chain
            time[array_index] = (k/sampf)
        stress_block(i, data, k, stress_corr, array_index, corr_array) #get the average correlation and error for time lag k
    for l in range(1,int(uplim)):
        for j in range(p*m**l,p*m**(l+1),m**l):
            array_index += 1
            if i == 0:
                time[array_index] = (j/sampf) #only record time for one chain, do not need to repeat for each chain
            stress_block(i, data, j, stress_corr, array_index, corr_array) #get the average correlation and error for time lag j
    return


@cuda.jit(device=True)
def stress_block(chainIdx, data, tj, stress_corr, array_index, xV):
    
    
    tj = int(tj)
    n=int(len(data)-tj)
    
    #reset correlation array to 0s after each block averaging
    for m in range(0,len(xV)):
        xV[m] = 0.0 
    
    #begin correlation averaging for timelag tj
    xav = 0
    for m in range(0,n):
        xV[m] = data[m]*data[int(m+tj)] #correlation between time and time + lag
        xav+=xV[m]/n  #calculate average
    c0=(xV[0]-xav)**2   
    for m in range(1,n):
        c0+=(xV[m]-xav)**2
    c0=c0/n
    sa=math.sqrt(c0/(n-1))
    sb=sa/math.sqrt(2*(n-1))
    n=int(math.floor(n/2))
    for m in range(0,n):
        xV[m]=(xV[2*m+1]+xV[2*m])/2
    c0=(xV[0]-xav)**2
    for m in range(1,n):
        c0=c0+(xV[m]-xav)**2
    c0=c0/n
    sap=math.sqrt(c0/(n-1))
    sbp=sap/math.sqrt(2*(n-1))
    while math.fabs(sa-sap) > sbp+sb and n > 4:
        sa=sap
        sb=sbp
        n=int(math.floor(n/2))
        for m in range(0,n):
            xV[m]=(xV[2*m+1]+xV[2*m])/2
        c0=(xV[0]-xav)**2
        for m in range(1,n):
            c0=c0+(xV[m]-xav)**2
        c0=c0/n
        sap=math.sqrt(c0/(n-1))
        sbp=sap/math.sqrt(2*(n-1))
    
    stress_corr[chainIdx,array_index,0] = xav #set average correlation value for chain i 
    stress_corr[chainIdx,array_index,1] = sap #set error of average correlation value for chain i
    
    return