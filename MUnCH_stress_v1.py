#This script is part of the Supplementary Material for the paper:
#"MUnCH: a calculator for propagating statistical and other sources of error in passive microrheology" 
#Submitted to Rheologica Acta (August, 2021)
#authors: Andres Cordoba and Jay D. Schieber
#Copyright (2021) Andres Cordoba and Jay D. Schieber. 
#This script is distributed under the MIT License.
#email: andcorduri@gmail.com

#This script was tested on Python 2.7.17 and on Python 3.7.6

#The next few lines import libraries that are required for the script to work. If you do not have them 
#installed on your system you will have to install them.

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



#The next few lines import the data file, in this case the first column is 
#time made dimensionless by the smallest relaxation time of the medium and 
#the  second column is voltage in units of mV. For this line to work save 
#the data file in the same folder as the script

t1=[]
data1=[]
with open('./stress_0.txt','r') as f:
    f.readline()
    for line in f.readlines():
        t1.append(float(line.split()[0]))
        data1.append(float(line.split()[2]))
            
            
#This next function, "msdblt" implements the blocking
#transformations method (H. Flyvbjerg and H. G. Petersen, The Journal
#of Chemical Physics, vol. 91, no. 1, pp. 461-466, 1989) for
#calculating the uncertainty in the MSD. The first argument "data" is
#the data file where the first column is the time made dimensionless
#by the smallest relaxation time of the medium, and the second column
#is voltage in units of mV. The second argument "tj" is the lag
#time also made dimensionless by the smallest relaxation time of the
#medium. The third argument "drb" is a vector that contains the
#measurement errors      
 
def msdblt(data, tj):
    n=len(data)-tj
    xV=[]
    for i in range(0,n):
        xV.append((data[i]*data[i+tj]))
    xav=sum(xV)/n
    c0=(xV[0]-xav)**2
    for i in range(1,n):
        c0=c0+(xV[i]-xav)**2
    c0=c0/n
    sa=math.sqrt(c0/(n-1))
    sb=sa/math.sqrt(2*(n-1))
    n=int(math.floor(n/2))
    for i in range(0,n):
        xV[i]=(xV[2*i+1]+xV[2*i])/2
    c0=(xV[0]-xav)**2
    for i in range(1,n):
        c0=c0+(xV[i]-xav)**2
    c0=c0/n
    sap=math.sqrt(c0/(n-1))
    sbp=sap/math.sqrt(2*(n-1))
    while math.fabs(sa-sap) > sbp+sb and n > 4:
        sa=sap
        sb=sbp
        n=int(math.floor(n/2))
        for i in range(0,n):
            xV[i]=(xV[2*i+1]+xV[2*i])/2
        c0=(xV[0]-xav)**2
        for i in range(1,n):
            c0=c0+(xV[i]-xav)**2
        c0=c0/n
        sap=math.sqrt(c0/(n-1))
        sbp=sap/math.sqrt(2*(n-1))
    return [xav, sap]


#This next function calls the previous functions at evenly spaced points in a log scale, i.e. 
#calculates the MSD and its uncertainty at evenly spaced lag times in a log scale, 
#i.e. it calculates the MSD and its uncertainty at evenly spaced lag times in a log scale. The first
#argument "data" is the data file, The second argument "sens" is the
#sensitivity in units of nm/mV. The fourth argument "dsens" is the
#uncertainty in the sensitivity in units of nm/mV. The fourth argument
#"dV" is the offset due to electric noise or optical system
#misalignment in units of mV


def msdsamp(data, sampf):
    rest=[]
    res=[]
    p=8
    m=2
    for i in range(1,p*m):
        rest.append(i/sampf)
        res.append(msdblt(data, i))
    uplim=int(math.floor(np.log(len(data)/p)/np.log(m)))
    for i in range(1,uplim):
        for j in range(p*m**i,p*m**(i+1),m**i):
            rest.append(j/sampf)
            res.append(msdblt(data, j))
    return [rest, res]


#This next functions exports the MSD and its uncertainty to a file. 
#The first column in the written file is time made dimensionless by the smallest relaxation time of the medium. 
#The second column is the MSD in nm^2. The third column is the MSD error also in nm^2

def write_msd_file(outFile,msd):                                                
    file = open(outFile,"w")
    for i in range(0,len(msd[0])):
        file.write("%s,%s,%s\n" % (msdt[0][i],msdt[1][i][0],msdt[1][i][1]))                                 
    file.close() 
    

#This next two lines call the functions that calculate the MSD and its uncertainty.

print('Calculating the ACF and its uncertainty, please wait, this will take some time ...')

msdt=msdsamp(data1,1/(t1[1]-t1[0]))

write_msd_file('./acf_wth_err_py.csv',msdt)


tmsd=msdt[0]
msde=[]
msdtt=[]
for i in range(0,len(msdt[1])):
    msde.append(msdt[1][i][1])
    msdtt.append(msdt[1][i][0])
            

print('ACF saved to file acf_wth_err_py.csv.')

#The function "msdmodel" is the analytic expression used to fit the MSD
#of the probe bead.

#The next few lines plot the MSD with uncertainty and its fit.
#the dynamic modulus with error bars is also plotted. The figures are saved in an eps file          


# MSD plot 

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.errorbar(tmsd, msdtt,yerr=msde,fmt="o",color="r",capsize=8,elinewidth=2,capthick=2)
plt.xlabel(r'$t/\tau_{\rm K}$',fontsize=40)
plt.ylabel(r'${\rm ACF}$',fontsize=40)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.9*tmsd[0],1.1*tmsd[len(tmsd)-1])
plt.ylim(0.9*msdtt[0],1.1*msdtt[len(msdtt)-1])

ax.tick_params(which='major', direction='in', length=20, axis='both', colors='k', pad=35)
ax.tick_params(which='minor', direction='in', length=10, axis='both', colors='k', pad=35, labelsize=55)
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
plt.legend()
chartBox = ax.get_position()
ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.8, chartBox.height])
ax.legend(loc='upper left', shadow=True, ncol=1, prop={'size': 40})

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(26, 35)
fig.savefig('acf_py.eps', dpi=300, bbox_inches='tight',pad_inches=0.5)


print('Saved figure to the file acf_py.eps. Finished.')
