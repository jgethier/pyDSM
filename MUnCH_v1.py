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
with open('./Vdata_lmax_15.98_traj_1.csv','r') as csvfile:
    for line in csvfile:
        plots=csv.reader(csvfile, delimiter=',')
        for row in plots:
            t1.append(float(row[0]))
            data1.append(float(row[1]))
            
#The next few lines define the global parameters:
            
#Sensitivity in nm/mV:
sens = 0.25
#Uncertainty in the sensitivity in nm/mV:
dsens = 0.025
#Offset due to electric noise or optical system misalignment in mV:
dV = 50
#Probe bead diameter in micro meters:
DExp = 1.53
#Uncertainty in the probe bead diameter in micro meters:
dDExp = 0.05*DExp
#Temperature measured during the experiment in Celsius:
TExp = 20
#Trap stiffness in pN/nm:
HeExp = 0.08
#Uncertainty in the trap stiffness in pN/nm:
dHeExp = 0.1*HeExp
#Parameters for the discrete relaxation spectrum of the fluid used here as input in the BD simulations to generate the data (dimensionless).
#see A. Cordoba, T. Indei, and J. D. Schieber, Journal of Rheology, vol. 56, no. 1, pp. 185-212, 2012 for details.
Hin = [4, 3, 2, 1]
lin = [1, 5, 10, 20]
z0in = 0.1
#Given the relaxation times and moduli, above, the
#parameters for an analytic expression of the MSD of the probe bead
#can be obtained, see the reference cited above for more details on how to do the calculation.
#The expression is useful to obtain an initial guess for the fitting of the MSD done here:
Lp=[0.009,1.46,6.84,15.4,71.3]
cpV0=[0.009018,0.0034,0.0022,0.0036,0.082]


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
 
def msdblt(data, tj, drb):
    n=len(data)-tj
    xV=[]
    dxV=[]
    for i in range(0,n):
        xV.append((data[i]-data[i+tj])**2)
    xav=sum(xV)/n
    for i in range(0,n):
        dxV.append(2*math.sqrt(xV[i]*(drb[i]**2+drb[i+tj]**2)))
    dxAv=dxV[0]**2
    for i in range(1,n):
        dxAv=dxAv+dxV[i]**2
    dxAv=math.sqrt(dxAv)/n
    c0=(xV[0]-xav)**2
    for i in range(1,n):
        c0=c0+(xV[i]-xav)**2
    c0=c0/n
    sa=math.sqrt(c0/(n-1))+dxAv
    sb=sa/math.sqrt(2*(n-1))
    n=int(math.floor(n/2))
    for i in range(0,n):
        xV[i]=(xV[2*i+1]+xV[2*i])/2
    for i in range(0,n):
        dxV[i]=(math.sqrt(dxV[2*i+1]**2+dxV[2*i]**2))/2
    dxAv=dxV[0]**2
    for i in range(1,n):
        dxAv=dxAv+dxV[i]**2
    dxAv=math.sqrt(dxAv)/n
    c0=(xV[0]-xav)**2
    for i in range(1,n):
        c0=c0+(xV[i]-xav)**2
    c0=c0/n
    sap=math.sqrt(c0/(n-1))+dxAv
    sbp=sap/math.sqrt(2*(n-1))
    while math.fabs(sa-sap) > sbp+sb and n > 4:
        sa=sap
        sb=sbp
        n=int(math.floor(n/2))
        for i in range(0,n):
            xV[i]=(xV[2*i+1]+xV[2*i])/2
        for i in range(0,n):
            dxV[i]=(math.sqrt(dxV[2*i+1]**2+dxV[2*i]**2))/2
        dxAv=dxV[0]**2
        for i in range(1,n):
            dxAv=dxAv+dxV[i]**2
        dxAv=math.sqrt(dxAv)/n
        c0=(xV[0]-xav)**2
        for i in range(1,n):
            c0=c0+(xV[i]-xav)**2
        c0=c0/n
        sap=math.sqrt(c0/(n-1))+dxAv
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


def msdsamp(data, sens, dsens, dV, sampf):
    rest=[]
    res=[]
    drbV=[]
    datax=[]
    p=8
    m=2
    for i in range(0,len(data)):
        drbV.append(math.sqrt((data[i]*dsens)**2+(sens*dV)**2))
    for i in range(0,len(data)):
        datax.append(data[i]*sens)
    for i in range(1,p*m):
        rest.append(i/sampf)
        res.append(msdblt(datax, i, drbV))
    uplim=int(math.floor(np.log(len(data)/p)/np.log(m)))
    for i in range(1,uplim):
        for j in range(p*m**i,p*m**(i+1),m**i):
            rest.append(j/sampf)
            res.append(msdblt(datax, j, drbV))
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

print('Calculating the MSD and its uncertainty, please wait, this will take some time ...')

msdt=msdsamp(data1,sens,dsens,dV,1/(t1[1]-t1[0]))

write_msd_file('./msd_wth_err_py.csv',msdt)


tmsd=msdt[0]
msde=[]
msdtt=[]
for i in range(0,len(msdt[1])):
    msde.append(msdt[1][i][1])
    msdtt.append(msdt[1][i][0])
            

print('MSD saved to file msd_wth_err_py.csv. Fitting the MSD with an analytic expression ...')

#The function "msdmodel" is the analytic expression used to fit the MSD
#of the probe bead.

def msdmodel(t,cp0,cp1,cp2,cp3,cp4):
    kB=1.38*10**(-23)
    T=TExp+273.15
    He=HeExp*10**(-12)/10**(-9)
    prefac=(10**9)**2*(2*kB*T)/He
    msdfun=prefac*(1-cp0*np.exp(-t/Lp[0])-cp1*np.exp(-t/Lp[1])-cp2*np.exp(-t/Lp[2])-cp3*np.exp(-t/Lp[3])-cp4*np.exp(-t/Lp[4]))
    return msdfun


#The next line fits the analytic expression above to the MSD data

for i in range(0,len(cpV0)):
    cpV0[i]=cpV0[i]/sum(cpV0)
pars, cov = curve_fit(f=msdmodel, xdata=tmsd, ydata=msdtt, p0=cpV0, sigma=msde, absolute_sigma=True, bounds=(0, np.inf))

#The next few line exports the parameters of the fitted MSD function to a file 

def write_fit_file(outFile,fit):                                                
    file = open(outFile,"w")
    for i in range(0,len(fit)):
        file.write("%s,%s\n" % (Lp[i],fit[i]/sum(fit)))                                 
    file.close() 
    
write_fit_file('./msd_fit_py.csv',pars)


print('MSD fit saved to file msd_fit_py.csv. Propagating the error to the dynamic modulus ...')

#The functions "Gs" and "GsErr" propagate the error from the MSD fit to the
#dynamic modulus. It uses the generalized Stokes-Einstein relation
#(GSER). It also includes the uncertainties in the bead radius and the
#trap stiffness in the calculation, see main text for details


def Gs(Cp, s):
    kB=1.38*10**(-23)
    T=TExp+273.15
    He=HeExp*10**(-12)/10**(-9)
    R=DExp/2*10**(-6)
    prefac=(2*kB*T)/He
    prod1=1
    sum1=[]
    for i in range(0,len(Cp)):
        prod1=Cp[i]/sum(Cp)
        for j in range (0,len(Lp)):
            if j != i: 
                prod1=prod1*(s+1/Lp[j])  
        sum1.append(prod1)
    num1=sum(sum1)
    deno1=s+1/Lp[0]
    for i in range(1,len(Lp)):
        deno1=deno1*(s+1/Lp[i])
    msdfunb=prefac*(1/s-num1/deno1)
    Gsexp=kB*T/(3*np.pi*R*s*msdfunb)-He/(6*np.pi*R)
    return Gsexp


def GsErr(Cp, dCp, s):
    kB=1.38*10**(-23)
    T=TExp+273.15
    He=HeExp*10**(-12)/10**(-9)
    dHe=0.1*He
    R=DExp/2*10**(-6)
    dR=0.05*R
    prefac=(2*kB*T)/He
    prod1=1
    sum1=[]
    for i in range(0,len(Cp)):
        prod1=Cp[i]/sum(Cp)
        for j in range (0,len(Lp)):
            if j != i: 
                prod1=prod1*(s+1/Lp[j])  
        sum1.append(prod1)
    num1=sum(sum1)
    deno1=s+1/Lp[0]
    for i in range(1,len(Lp)):
        deno1=deno1*(s+1/Lp[i])
    msdfunb=prefac*(1/s-num1/deno1)
    Gsexp=kB*T/(3*np.pi*R*s*msdfunb)-He/(6*np.pi*R)
    errCp=[]
    for i in range (0, len(Cp)):
        for j in range (0, len(Cp)):
            errCp.append(dCp[i][j]*He**2/((6*np.pi*R*s)**2*(s+1/Lp[i])*(s+1/Lp[j])*(1/s-num1/deno1)**4))
    errHe=(dHe*(1/(6*np.pi*R)*(1/(s*(1/s-num1/deno1))-1)))**2
    errR=(dR*(He/(6*np.pi*R**2)*(1-1/(s*(1/s-num1/deno1)))))**2
    errGs=np.sqrt(sum(errCp)+errHe+errR)
    return errGs/Gsexp

#The next function "GsA" is the expression for the known analytic dynamic modulus that 
#is used as input in the GBD simulation that is used here to generate the synthetic data.

def GsA(h, l, mu, s):
    kB=1.38*10**(-23)
    T=TExp+273.15
    He=HeExp*10**(-12)/10**(-9)
    R=DExp/2*10**(-6)
    res=[]
    for i in range (0,len(h)):
        res.append(h[i]*l[i]*s/(1+l[i]*s))
    return He/(6*np.pi*R)*(mu*s+sum(res))



#The next few lines plot the MSD with uncertainty and its fit.
#the dynamic modulus with error bars is also plotted. The figures are saved in an eps file          


# MSD plot 

msdfitp=[]
for i in range(0,len(tmsd)):
    msdfitp.append(msdmodel(tmsd[i],pars[0],pars[1],pars[2],pars[3],pars[4]))
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
ax.errorbar(tmsd, msdtt,yerr=msde,fmt="o",color="r",capsize=8,elinewidth=2,capthick=2)
ax.plot(tmsd, msdfitp,color="k",label=r'Fit')
plt.xlabel(r'$t/\lambda_1$',fontsize=40)
plt.ylabel(r'$\langle \Delta r_\gamma^2(t/\lambda_1) \rangle, {\rm nm}^2$',fontsize=40)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.6*tmsd[0],1.5*tmsd[len(tmsd)-1])
plt.ylim(0.6*msdtt[0],1.5*msdtt[len(msdtt)-1])

ax.tick_params(which='major', direction='in', length=20, axis='both', colors='k', pad=35)
ax.tick_params(which='minor', direction='in', length=10, axis='both', colors='k', pad=35, labelsize=55)
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
plt.legend()
chartBox = ax.get_position()
ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.8, chartBox.height])
ax.legend(loc='upper left', shadow=True, ncol=1, prop={'size': 40})

print('Finished G* error propagation. Generating plot ...')

# Dynamic modulus plot. The shaded regions represent the uncertainty in the calculated MSD 
# The black line is the known analytic modulus used here as input in the BD simulations.
wGs=[]
for i in range (0, len(tmsd)):
    wGs.append(2*np.pi/tmsd[i])
Gp=[]
for i in range (0, len(wGs)):
    Gp.append(Gs(pars, wGs[i]*1.j).real)
Gpp=[]
for i in range (0, len(wGs)):
    Gpp.append(Gs(pars, wGs[i]*1.j).imag)
GpUp=[]
for i in range (0, len(wGs)):
    GpUp.append(Gs(pars, wGs[i]*1.j).real*(1+GsErr(pars, cov, wGs[i])))
GpLo=[]
for i in range (0, len(wGs)):
    GpLo.append(Gs(pars, wGs[i]*1.j).real*(1-GsErr(pars, cov, wGs[i])))
GppUp=[]
for i in range (0, len(wGs)):
    GppUp.append(Gs(pars, wGs[i]*1.j).imag*(1+GsErr(pars, cov, wGs[i])))
GppLo=[]
for i in range (0, len(wGs)):
    GppLo.append(Gs(pars, wGs[i]*1.j).imag*(1-GsErr(pars, cov, wGs[i])))

    
GpA=[]
for i in range (0, len(wGs)):
    GpA.append(GsA(Hin,lin,z0in,wGs[i]*1.j).real)
GppA=[]
for i in range (0, len(wGs)):
    GppA.append(GsA(Hin,lin,z0in,wGs[i]*1.j).imag)


#The next few lines export the dynamic modulus and its uncertanty to a file.
#The first column is the dimensionless frequency, the second column is the storage modulus (Pa),
#the third column is the uncertainty in the storage modulus (Pa),
#the fourth column is the loss modulus (Pa) and the fifth colum is the uncertainty in the loss modulus (Pa).
    
def write_Gs_file(outFile, wGs, Gp, GpUp, Gpp, GppUp):                                                
    file = open(outFile,"w")
    endf=len(wGs)-1
    for i in range(0,len(wGs)):
        file.write("%s,%s,%s,%s,%s\n" % (wGs[endf-i],Gp[endf-i],GpUp[endf-i]-Gp[endf-i],Gpp[endf-i],GppUp[endf-i]-Gpp[endf-i]))                                 
    file.close()
    
write_Gs_file('Gs_wth_err_py.csv',wGs,Gp,GpUp,Gpp,GppUp)

print('Dynamic modulus and its uncertainty saved to file Gs_wth_err_py.csv. Generating plot ...')
    
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB Tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


ax = fig.add_subplot(2,1,2)
ax.plot(wGs, GpA,color='k')
ax.plot(wGs, GppA,color='k')
plt.fill_between(wGs, GpUp, GpLo,alpha=0.001, edgecolor='salmon', facecolor=lighten_color('salmon',0.4),label=r'$G^\prime$')
plt.fill_between(wGs, GppUp, GppLo,alpha=0.001, edgecolor='violet', facecolor=lighten_color('violet',0.4),label=r'$G^{\prime\prime}$')
plt.xlabel(r'$\omega\lambda_1$',fontsize=40)
plt.ylabel(r'$G^*(\omega\lambda_1), {\rm Pa}$',fontsize=40)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.005,5000)
plt.ylim(0.1,200)

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
fig.savefig('msd_Gs_py.eps', dpi=300, bbox_inches='tight',pad_inches=0.5)


print('Saved figure to the file msd_Gs_py.eps. Finished.')
