import numpy as np
from scipy.optimize import least_squares, minimize
import matplotlib.pyplot as plt 
from scipy.special import logsumexp

import matplotlib as mpl

mpl.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 2

x = []
y = []

#Plot gt_fit results
def plot_gt_results(gt_result_x, gt_result_y):

    with open('./gt_MMM_fit.dat') as f:
        lines = f.readlines()
        gt_lambdaArr = np.array([float(line.split()[0]) for line in lines[1:]])
        gt_gArr = np.array([float(line.split()[1]) for line in lines[1:]])

    omegaPoints = 1000
    omegaMin = -8
    omegaMax = 5
    omegaArr=10**(omegaMin+(np.array(range(omegaPoints), float) + 1.0)/omegaPoints*(omegaMax-omegaMin))

    fig3 = plt.figure(figsize=(16, 6))

    # ax0 = fig3.add_subplot(121)

    # ax0.xaxis.set_tick_params(which='major',size=7, width=2, direction='in', top='on')
    # ax0.xaxis.set_tick_params(which='minor',size=4, width=2, direction='in', top='on')
    # ax0.yaxis.set_tick_params(which='major',size=7, width=2, direction='in', right='on')
    # ax0.yaxis.set_tick_params(which='minor',size=4, width=2, direction='in', right='on')

    # ax0.tick_params(bottom=True, top=True, left = True, right=True)
    # ax0.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    # ax0.tick_params(direction='in')

    # ax0.set_xlabel(r'$\lambda$')
    # ax0.set_ylabel(r'$g$')

    # ax0.scatter(gt_lambdaArr, gt_gArr, c='r', label=r'MMM fit')
    # leg = ax0.legend()
    # ax0.set_xscale('log')
    # ax0.set_yscale('log')

    ax1 = fig3.add_subplot(121)

    ax1.xaxis.set_tick_params(which='major',size=7, width=2, direction='in', top='on')
    ax1.xaxis.set_tick_params(which='minor',size=4, width=2, direction='in', top='on')
    ax1.yaxis.set_tick_params(which='major',size=7, width=2, direction='in', right='on')
    ax1.yaxis.set_tick_params(which='minor',size=4, width=2, direction='in', right='on')

    ax1.tick_params(bottom=True, top=True, left = True, right=True)
    ax1.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    ax1.tick_params(direction='in')

    ax1.set_title("Relaxation modulus G(t)")
    ax1.set_xlabel(r'$t/\tau_c$')
    ax1.set_ylabel(r'G(t/$\tau_c$)/($\rho RT/M_w$)')

    ax1.scatter(gt_result_x, gt_result_y, c='r', label=r'Simulation',edgecolor='black')
    ax1.plot(gt_result_x, gt_result_y[0]*Gt_MMM_vec(time=gt_result_x, params=np.append(gt_lambdaArr, gt_gArr)), c='black', label=r'Fit')
    leg = ax1.legend()
    plt.legend(frameon=False)
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    ax2 = fig3.add_subplot(122)

    ax2.xaxis.set_tick_params(which='major',size=7, width=2, direction='in', top='on')
    ax2.xaxis.set_tick_params(which='minor',size=4, width=2, direction='in', top='on')
    ax2.yaxis.set_tick_params(which='major',size=7, width=2, direction='in', right='on')
    ax2.yaxis.set_tick_params(which='minor',size=4, width=2, direction='in', right='on')

    ax2.tick_params(bottom=True, top=True, left = True, right=True)
    ax2.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    ax2.tick_params(direction='in')

    ax2.set_title(r'Dynamic modulus $G*(\omega)$')
    ax2.set_xlabel(r'$\omega$')
    ax2.set_ylabel(r'$G*$')

    ax2.plot(omegaArr,gt_result_y[0]*Gp_MMM_vec(omega=omegaArr,params=np.append(gt_lambdaArr, gt_gArr)), c='k', label=r'$G^\prime$')
    ax2.plot(omegaArr,gt_result_y[0]*Gdp_MMM_vec(omega=omegaArr,params=np.append(gt_lambdaArr, gt_gArr)), c='k',linestyle='--', label=r'$G^{\prime\prime}$')

    leg = ax2.legend()
    plt.legend(frameon=False)
    ax2.set_yscale('log')
    ax2.set_xscale('log')

    plt.show()

    return 

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def Gt_MMM(time, params):
    #Variable frequencies
    lambdaArr = np.split(params,2)[0]
    gArr = np.split(params,2)[1]/np.sum(np.split(params,2)[1])
    #Fixed frequencies
    #lambdaArr=10.0**((np.array(range(nmodes), float) + 1.0)/nmodes*np.log10(tfinal))
    #gArr = params/np.sum(params)
    return np.dot(np.exp(-time/lambdaArr), gArr)

def log_Gt_MMM(time,params):
    lambdaArr = np.split(params,2)[0]
    gArr = np.split(params,2)[1]/np.sum(np.split(params,2)[1])
    #lambdaArr=10.0**((np.array(range(nmodes), float) + 1.0)/nmodes*np.log10(tfinal))
    #gArr = params/np.sum(params)
    return logsumexp(-time/lambdaArr, b=gArr)

#Vectorize function fdt and log_fdt
Gt_MMM_vec=np.vectorize(Gt_MMM, excluded=['params'])
log_Gt_MMM_vec=np.vectorize(log_Gt_MMM, excluded=['params'])

def Gp_MMM(omega, params):
    lambdaArr = np.split(params,2)[0]
    gArr = np.split(params,2)[1]/np.sum(np.split(params,2)[1])

    return np.sum((gArr * lambdaArr**2 * omega**2)/(1 + lambdaArr**2 * omega**2))

def Gdp_MMM(omega, params):
    lambdaArr = np.split(params,2)[0]
    gArr = np.split(params,2)[1]/np.sum(np.split(params,2)[1])

    return np.sum((gArr * lambdaArr * omega)/(1 + lambdaArr**2 * omega**2))

#Vectorize function Gp and Gdp
Gp_MMM_vec=np.vectorize(Gp_MMM, excluded=['params'])
Gdp_MMM_vec=np.vectorize(Gdp_MMM, excluded=['params'])

def gt_fit():
    global x
    global y

    with open('DSM_results/Gt_result_1.txt') as f:
        f.readline()
        lines = f.readlines()
        x = np.array([float(line.split(",")[0]) for line in lines])
        y = np.array([float(line.split(",")[1]) for line in lines])
    GN0=y[0]
    cutoff_list = np.array([np.argmax(y[1:]-y[:-1]>0), np.argmax(y<0), find_nearest(y,0.01*GN0), np.size(x)])

    cutoff = min(cutoff_list[cutoff_list>0])
    x=x[:cutoff]
    y=y[:cutoff]

    tfinal=x[-1]
    tstart=x[1]

    #Save calculated G(t) to file
    gt_result=zip(x, y)
    file = open("gt_result.dat","w")
    for i in gt_result:
        file.write(str(i[0])+'\t'+str(i[1])+'\n')
    file.close()

    #Define residuals
    def residuals_Gt_MMM(param):
        return Gt_MMM_vec(time=x, params=param)*GN0-y

    #Define log-residuals
    def residuals_log_Gt_MMM(param):
        if np.any(Gt_MMM_vec(time=x[:-1], params=param) < 0):
            return np.full(x[:-1].shape,1e8) #Penalty for negative f_d(t)
        else:
            return log_Gt_MMM_vec(time=x, params=param)+np.log(GN0)-np.log(y)

    #Define Mean-Squared Error
    def MSE_MMM(param):
        return np.dot(residuals_Gt_MMM(param),residuals_Gt_MMM(param))/np.size(x)

    def log_MSE_MMM(param):
        return np.dot(residuals_log_Gt_MMM(param),residuals_log_Gt_MMM(param))/np.size(x)

    fits_1 = [] #output of fitting function for all tested numbers of modes
    successful_fits_1 = [] #number of modes for successful fits

    lambdaArrInit=np.e**(np.log(tstart)+(np.array(range(2), float))/(np.log(tfinal)-np.log(tstart)))
    fit = np.linalg.lstsq(np.exp(-np.outer(x,1.0/lambdaArrInit)), y)[0]
    fits_1.append(fit)
    min_log_SME = log_MSE_MMM(np.append(lambdaArrInit, fit))
    best_fit = 2
    print(2, fit, MSE_MMM(np.append(lambdaArrInit, fit)))

    for nmodes in range(3, 15):
        lambdaArrInit=np.e**(np.log(tstart)+(np.array(range(nmodes), float))/(nmodes-1)*(np.log(tfinal)-np.log(tstart)))
        fit = np.linalg.lstsq(np.exp(-np.outer(x,1.0/lambdaArrInit)), y)[0]
        fits_1.append(fit)
        print(nmodes, fit, MSE_MMM(np.append(lambdaArrInit, fit)), log_MSE_MMM(np.append(lambdaArrInit, fit)))

        #if not np.any(Gt_MMM_vec(time=x, params=np.append(lambdaArrInit, fit)) < 0):
        if not np.any(fit < 0) and log_MSE_MMM(np.append(lambdaArrInit, fit))<min_log_SME:
            min_log_SME = log_MSE_MMM(np.append(lambdaArrInit, fit))
            best_fit = nmodes


    fit = fits_1[best_fit-2]
    li=np.e**(np.log(tstart)+(np.array(range(best_fit), float))/(best_fit-1)*(np.log(tfinal)-np.log(tstart)))
    gi=fit
    result=zip(li, gi)
    f = open('gt_MMM_fit.dat','w')
    f.write(str(best_fit))
    for i in result:
        f.write('\n'+str(i[0])+'\t'+str(i[1]))
    f.close()

    plot_gt_results(x,y)

    return 

if __name__== "__main__":
    gt_fit()