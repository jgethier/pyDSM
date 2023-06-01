import sys
import math
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
matplotlib.rc('text', usetex = True)
from scipy.optimize import curve_fit
from scipy import optimize


#This script calculates the maximum fraction of new entangled strands after cessation of shear flow.
#As input it needs:
# 1. The fraction of new entangled strands after cessation as a function of dimensionless time as obtained from a DSM simulation
# 2. It also takes as runtime argument the dimensionless shear rate at which the simulation was run.
# Using a simple cooling model the script calculates the time at which re-entanglement effectively stops and the 
#fraction new entangled strands that is achieved at that time. The script does that calculation for several values
#of the cooling constant. Running this script for simulations performed at different shear rates one can then 
#construct a phase diagram for the maximum fraction of new entangled strands as a function of cooling rate and shear rate.
#Andres Cordoba (andcorduri@gmail.com)

data1=[]
with open('./stress_1.txt','r') as csvfile:
    for line in csvfile:
        plots=csv.reader(csvfile, delimiter=',')
        for row in plots:
            if float(row[8])>0:
                data1.append([float(row[0]),float(row[8])])


#Cluster-shuffling characteristic time at the reference temperature for polycarbonate:
tcref=6.6*10**-7
#temperature at the nozzle through which the printing material is extruded:
Tn=250
#Reference temperature:
Tref=260
#Williams-Landel-Ferry (WLF) parameter:
c1=3
c2=160
a=math.exp((-c1*(Tn-Tref))/(Tn+c2-Tref))

#Nonlinear relation between the dimensionless time of the simulation (after cessation) and the dimensional time, accounting for cooling.
#totc is the dimensionless time from the simulation, t is the dimensional time.
#This relation is solved numerically (below) for the dimensional time.
def froot(t,l,totc):
    sol = -47/5-96/(5*(-3+math.exp(t/l)))
    return math.exp(sol)*tcref*totc/t-1
    

#Here the dimensionless time from the simulation (after cessation) is transformed to dimensional time using the relation above
#To make this more efficient we do not transform all the data points outputted from the simulation. Instead we 
#chose points evenly spaced in a log scale.
def makedim(data,l,sr):
    p=8
    m=2
    uplim=int(math.floor(np.log(len(data)/p)/np.log(m)))
    res=[]
    for i in range(1,p*m):
        totc=data[i][0]-data[0][0]
        sol=optimize.root_scalar(froot, args=(l,totc), x0=0.9*tcref*totc, x1=1.1*tcref*totc, method='secant')
        if sol.root >0:
            res.append([sr/(a*tcref),1/l,sol.root,data[i][1]])
    slope=(math.log(res[-1][3])-math.log(res[-2][3]))/(math.log(res[-1][2])-math.log(res[-2][2]))
    if slope>0 and sol.root >0:
        for i in range(1,uplim):
            for j in range(p*m**i,p*m**(i+1),m**i):
                if slope >0 and sol.root >0:
                    totc=data[j][0]-data[0][0]
                    sol=optimize.root_scalar(froot, args=(l,totc),x0=0.9*tcref*totc, x1=1.1*tcref*totc, method='secant')
                    if sol.root >0:
                        res.append([sr/(a*tcref),1/l,sol.root,data[j][1]])
                    slope=(math.log(res[-1][3])-math.log(res[-2][3]))/(math.log(res[-1][2])-math.log(res[-2][2]))
    return res


#Here we read as an argument the shear rate at which the simulation was run.
#Then perform the calculations for different values of the cooling rate.
sr = float(sys.argv[1])
lrang=[0.001,2]
deltal=(math.log(lrang[1])-math.log(lrang[0]))/10
maxfrac=[]
count=math.log(lrang[0])
while count<math.log(lrang[1]):
    reseval=makedim(data1,round(math.exp(count),3),sr)
    maxfrac.append(reseval[-1])
    count=count+deltal

#Here the results are written to a file.
with open("maxentfrac.txt", "w") as f:
    f.write('shear rate (1/s), cooling rate (1/s), max time (s), max fraction\n')
    for m in range(0,len(maxfrac)):
        f.write("%.2f, %.2f, %s, %s \n"%(maxfrac[m][0],maxfrac[m][1],maxfrac[m][2],maxfrac[m][3]))


