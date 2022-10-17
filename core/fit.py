import numpy as np
import os 
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 
from scipy.special import logsumexp
import matplotlib as mpl

mpl.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 2



class CURVE_FIT(object):

    def __init__(self,file_path,output_path):
        '''
        Initialize class. Set vectorized functions for G(t), log G(t), G'(omega), and G''(omega)
        '''
        self.filepath = file_path
        self.output_path = output_path

        #get results path ready
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        self.output_dir = output_path

        #Vectorize function fdt and log_fdt
        self.Gt_MMM_vec=np.vectorize(self.Gt_MMM, excluded=['params'])
        self.log_Gt_MMM_vec=np.vectorize(self.log_Gt_MMM, excluded=['params'])
        self.Gp_MMM_vec = np.vectorize(self.Gp_MMM,excluded=['params'])
        self.Gdp_MMM_vec = np.vectorize(self.Gdp_MMM,excluded=['params'])
        self.Gp_MMM_stdev_vec=np.vectorize(self.Gp_MMM_stdev,excluded=['cov'])
        self.Gdp_MMM_stdev_vec=np.vectorize(self.Gdp_MMM_stdev,excluded=['cov'])
        self.Gt_MMM_stdev_vec=np.vectorize(self.Gt_MMM_stdev,excluded=['cov'])

        return 

    #Plot gt_fit results
    def plot_gt_results(self,gt_result_x, gt_result_y,error,cov):
        '''
        Plots G(t), G(t)_fit, G'(omega), and G''(omega) with 95% confidence intervals. Standard error is reported for simulation data. 
        '''

        with open(os.path.join(self.output_path,'./gt_MMM_fit.dat')) as f:
            lines = f.readlines()
            g = np.array([float(line.split()[1]) for line in lines[1:]])

        #only fitting {g_i} values
        params = g
        self.nmodes = len(g) #modes = number of g values
        self.tstart = gt_result_x[1] #first t value to determine set of lambda
        self.tfinal = gt_result_x[-1] #last t value to determine set of lambda

        #frequency range
        omegaPoints = 100
        omegaMin = -4
        omegaMax = 1
        omegaArr=10**(omegaMin+(np.array(range(omegaPoints), float) + 1.0)/omegaPoints*(omegaMax-omegaMin))

        #start figure
        fig = plt.figure(figsize=(16, 6))

        #fig 1
        ax1 = fig.add_subplot(121)
        
        #axis settings
        ax1.xaxis.set_tick_params(which='major',size=7, width=2, direction='in', top='on')
        ax1.xaxis.set_tick_params(which='minor',size=4, width=2, direction='in', top='on')
        ax1.yaxis.set_tick_params(which='major',size=7, width=2, direction='in', right='on')
        ax1.yaxis.set_tick_params(which='minor',size=4, width=2, direction='in', right='on')

        #tick marks
        ax1.tick_params(bottom=True, top=True, left = True, right=True)
        ax1.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
        ax1.tick_params(direction='in')
        
        #title and axis labels
        ax1.set_title("Relaxation modulus G(t)")
        ax1.set_xlabel(r'$t/\tau_c$')
        ax1.set_ylabel(r'G(t/$\tau_c$)/($\rho RT/M_w$)')

        #get standard deviation of G(t) and calculate 95% confidence interval
        if self.return_error:
            Gt_error = self.Gt_MMM_stdev_vec(time=gt_result_x,cov=cov)
            Gp_error = self.Gp_MMM_stdev_vec(omega=omegaArr,cov=cov)
            Gdp_error = self.Gdp_MMM_stdev_vec(omega=omegaArr,cov=cov)

        #plotting results
        Gt = self.Gt_MMM_vec(time=gt_result_x, params=params)
        ax1.plot(gt_result_x, Gt, c='black', label=r'Fit',zorder=3)
        ax1.scatter(gt_result_x, gt_result_y,edgecolor='black',c='r', label=r'Simulation',zorder=2)
        if self.return_error:
            ax1.fill_between(gt_result_x,Gt+Gt_error*1.96,Gt-Gt_error*1.96,color='gray',alpha=0.5,label=r'95% confidence interval')
            ax1.errorbar(gt_result_x, gt_result_y,yerr=error, fmt='',linewidth=0,elinewidth=2,c='r',capsize=4,zorder=1,markersize=0,label=r'Standard Error')

        #legend
        plt.legend(frameon=False)
        
        #set log-log scale
        ax1.set_xscale('log')
        ax1.set_yscale('log')

        #fig 2
        ax2 = fig.add_subplot(122)

        #axis settings
        ax2.xaxis.set_tick_params(which='major',size=7, width=2, direction='in', top='on')
        ax2.xaxis.set_tick_params(which='minor',size=4, width=2, direction='in', top='on')
        ax2.yaxis.set_tick_params(which='major',size=7, width=2, direction='in', right='on')
        ax2.yaxis.set_tick_params(which='minor',size=4, width=2, direction='in', right='on')

        #tick parameters
        ax2.tick_params(bottom=True, top=True, left = True, right=True)
        ax2.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
        ax2.tick_params(direction='in')

        #set title and axis labels
        ax2.set_title(r'Dynamic modulus $G*(\omega)$')
        ax2.set_xlabel(r'$\omega$')
        ax2.set_ylabel(r'$G*$')

        #plot G' and G'' results
        Gp = self.Gp_MMM_vec(omega=omegaArr,params=params)
        Gdp = self.Gdp_MMM_vec(omega=omegaArr,params=params)
        ax2.plot(omegaArr,self.Gp_MMM_vec(omega=omegaArr,params=params), c='k', label=r'$G^\prime$')
        ax2.plot(omegaArr,self.Gdp_MMM_vec(omega=omegaArr,params=params), c='k',linestyle='--', label=r'$G^{\prime\prime}$')
        if self.return_error:
            ax2.fill_between(omegaArr,Gp+Gp_error*1.96,Gp-Gp_error*1.96,color='gray',alpha=0.5,label=r'95% confidence interval')
            ax2.fill_between(omegaArr,Gdp+Gdp_error*1.96,Gdp-Gdp_error*1.96,color='gray',alpha=0.5)

        #legend and log-log scale
        plt.legend(frameon=False)
        ax2.set_yscale('log')
        ax2.set_xscale('log')
        
        if self.return_error:
            #write G(t) predictions and standard deviation to file
            with open(os.path.join(self.output_path,"predictions_Gt.txt"),"w") as f:
                f.write("t" + '\t' + "G(t)" + '\t' + "sigma[G(t)]"+'\n')
                for i in range(0,len(Gt)):
                    f.write('%d'%gt_result_x[i] + '\t' + '%.5g'%Gt[i]+'\t'+'%.5g'%Gt_error[i]+'\n')

            #write G' and G'' predictions and standard deviation to file
            with open(os.path.join(self.output_path,"predictions_Gp_Gpp.txt"),"w") as f:
                f.write("omega" + '\t' + "G'(omega)" + '\t' + "sigma[G'(omega)]" + '\t' +"G''(omega)" + '\t' + "sigma[G''(omega)]"+'\n')
                for i in range(0,len(Gp)):
                    f.write('%.5g'%omegaArr[i]+'\t'+'%.5g'%Gp[i]+'\t'+'%.5g'%Gp_error[i]+'\t'+'%.5g'%Gdp[i]+'\t'+'%.5g'%Gdp_error[i]+'\n')

        else:
            with open(os.path.join(self.output_path,"predictions_Gt.txt"),"w") as f:
                f.write("t" + '\t' + "G(t)" + '\n')
                for i in range(0,len(Gt)):
                    f.write('%d'%gt_result_x[i] + '\t' + '%.5g'%Gt[i]+'\n')

            with open(os.path.join(self.output_path,"predictions_Gp_Gpp.txt"),"w") as f:
                f.write("omega" + '\t' + "G'(omega)" + '\t' +"G''(omega)" + '\n')
                for i in range(0,len(Gp)):
                    f.write('%.5g'%omegaArr[i]+'\t'+'%.5g'%Gp[i]+'\t'+'%.5g'%Gdp[i]+'\n')

        plt.show()

        return 

    def find_nearest(self,array,value):
        '''
        Find nearest value in array
        '''
        idx = (np.abs(array-value)).argmin()
        return idx

    def Gt_MMM(self,time,params):
        '''
        Calculate G(t) using multi-mode Maxwell model
        '''
        #Variable frequencies
        #lambdaArr = np.split(params,2)[0]
        #gArr = np.split(params,2)[1]#/np.sum(np.split(params,2)[1])

        #Fixed frequencies
        lambdaArr=np.geomspace(self.tstart,self.tfinal,self.nmodes)
        gArr = params#/np.sum(params)
        return np.dot(np.exp(-time/lambdaArr), gArr)

    def Gt_MMM_stdev(self,time,cov):
        '''
        Calculate sigma[G(t)], standard deviation of G(t) using multi-mode Maxwell model
        '''
        #Variable frequencies
        #lambdaArr = np.split(params,2)[0]
        #gArr = np.split(params,2)[1]#/np.sum(np.split(params,2)[1])

        #Fixed frequencies
        lambdaArr=np.geomspace(self.tstart,self.tfinal,self.nmodes)
        
        error = 0
        for i in range(0,len(lambdaArr)):
            for j in range(0,len(lambdaArr)):
                error += (np.exp(-time/lambdaArr[i]))*(np.exp(-time/lambdaArr[j]))*cov[i,j]

        return np.sqrt(error)

    def log_Gt_MMM(self,time,params):
        '''
        Calculate log G(t) using multi-mode Maxwell model
        '''
        #Variable frequencies
        #lambdaArr = np.split(params,2)[0]
        #gArr = np.split(params,2)[1]#/np.sum(np.split(params,2)[1])

        #Fixed frequencies
        lambdaArr=np.geomspace(self.tstart,self.tfinal,self.nmodes)
        gArr = params#/np.sum(params)
        return logsumexp(-time/lambdaArr, b=gArr)


    def Gp_MMM(self,omega, params):
        '''
        Calculate G'(omega) from multi-mode Maxwell model
        '''
        lambdaArr=np.geomspace(self.tstart,self.tfinal,self.nmodes)
        gArr = params#/np.sum(params)

        return np.sum((gArr * lambdaArr**2 * omega**2)/(1 + lambdaArr**2 * omega**2))

    def Gdp_MMM(self,omega, params):
        '''
        Calculate G''(omega) from multi-mode Maxwell model
        '''
        lambdaArr=np.geomspace(self.tstart,self.tfinal,self.nmodes)
        gArr = params#/np.sum(params)

        return np.sum((gArr * lambdaArr * omega)/(1 + lambdaArr**2 * omega**2))

    def Gp_MMM_stdev(self,omega,cov):
        '''
        Calculate sigma[G'(omega)], standard deviation for G'(omega) from multi-mode Maxwell model
        '''
        lambdaArr=np.geomspace(self.tstart,self.tfinal,self.nmodes)
        
        error = 0
        for i in range(0,len(lambdaArr)):
            for j in range(0,len(lambdaArr)):
                error += ((lambdaArr[i]**2 * omega**2)/(1 + lambdaArr[i]**2 * omega**2))*((lambdaArr[j]**2 * omega**2)/(1 + lambdaArr[j]**2 * omega**2))*cov[i,j]

        return np.sqrt(error) 

    def Gdp_MMM_stdev(self,omega,cov):
        '''
        Calculate sigma[G''(omega)], standard deviation for G'(omega) from multi-mode Maxwell model
        '''
        lambdaArr=np.geomspace(self.tstart,self.tfinal,self.nmodes)
        
        error = 0
        for i in range(0,len(lambdaArr)):
            for j in range(0,len(lambdaArr)):
                error += ((lambdaArr[i] * omega)/(1 + lambdaArr[i]**2 * omega**2))*((lambdaArr[j] * omega)/(1 + lambdaArr[j]**2 * omega**2))*cov[i,j]

        return np.sqrt(error) 

    def multivariate_distribution(self,omegaArr, gt_result_x, params,cov):
        '''
        Estimate uncertainty of params, G(t), G'(omega), and G''(omega) by drawing {g_i} values from a multivariate normal distribution.
        Note: this is an approximation due to sampling and is only used to compare to exact error from propagation equation. 
        Exact error is found in Gt_MMM_stdev, Gp_MMM_stdev, and Gdp_MMM_stdev functions. 
        '''

        #get multivariate normal distribution with mean = parameter values and covariance = pcov
        dist = np.random.multivariate_normal(mean=params,cov=cov,size=(1000))
        print("Average fit parameters from distribution: \n",dist.mean(axis=0))
        print("Standard deviation from distribution: \n",dist.std(axis=0))

        #calculate G(t), G'(omega), and G''(omega) for all sampled parameters from multivariate normal distribution
        gt_result_total = []
        Gp_result_total = []
        Gdp_result_total = []
        print("")
        print(r'Calculating predictions from all {g_i} sampled from multivariate normal distribution...',end="",flush=True)
        for j in dist:
            param = j
            gt_result = np.array(self.Gt_MMM_vec(time=gt_result_x, params=param))
            gt_result_total.append(gt_result)

            Gp_result = np.array(self.Gp_MMM_vec(omega=omegaArr, params=param))
            Gdp_result = np.array(self.Gdp_MMM_vec(omega=omegaArr, params=param))
            Gp_result_total.append(Gp_result)
            Gdp_result_total.append(Gdp_result)
        print("Done.")

        #get average and standard devation of all G(t) predictions
        gt_result_total = np.array(gt_result_total) #mean 
        gt_result_aver = np.mean(gt_result_total,axis=0) #sigma
        gt_result_std = np.std(gt_result_total,axis=0)*1.96  #1.96*sigma = 95% confidence interval

        #get average and standard devation of all G'(omega) predictions
        Gp_result_total = np.array(Gp_result_total)
        Gp_result_aver = np.mean(Gp_result_total,axis=0)
        Gp_result_std = np.std(Gp_result_total,axis=0)*1.96

        #get average and standard devation of all G''(omega) predictions
        Gdp_result_total = np.array(Gdp_result_total)
        Gdp_result_aver = np.mean(Gdp_result_total,axis=0)
        Gdp_result_std = np.std(Gdp_result_total,axis=0)*1.96

        return gt_result_aver, gt_result_std, Gp_result_aver, Gp_result_std, Gdp_result_aver, Gdp_result_std

    def fit(self):

        #fit function (G(t))
        def func(x, *params):
            fit_params = np.array(params)
            return self.Gt_MMM_vec(time=x,params=fit_params)

        #Define chi_sq = sum((r/sigma)**2) where r = (ydata - func(x,*popt))
        def chi_sq(popt,xdata,ydata,yerr):
            return np.sum((func(xdata,*popt)-ydata)**2 / yerr**2)

        #curve_fit func
        def fit_curvefit(p0, xdata, ydata, f, yerr):
            """
            Function to apply curve_fit and get optimal parameters, error estimates, and covariance matrix
            
            Note: As per the current documentation (Scipy V1.1.0), sigma (yerr) must be:
                None or M-length sequence or MxM array, optional
            """
            #get optimal params and covariance matrix, set absolute_sigma=False if you want cov = hess_inv*chi_sq/(len(ydata)-len(params))
            pfit, pcov = curve_fit(f,xdata,ydata,p0=p0,sigma=yerr,bounds=(0,np.inf),absolute_sigma=True)

            error = [] 
            for i in range(len(pfit)):
                try:
                    error.append(np.absolute(pcov[i][i])**0.5) #parameter errors are sqrt of the diagonal of the covariance matrix
                except:
                    error.append( 0.00 )

            pfit_curvefit = pfit
            perr_curvefit = np.array(error)

            return pfit_curvefit, perr_curvefit, pcov

        #read in file and set x, y, and error arrays
        with open(self.filepath) as f:
            f.readline()
            lines = f.readlines()
            x = np.array([float(line.split(",")[0]) for line in lines]) #x values (time)
            y = np.array([float(line.split(",")[1]) for line in lines]) #y values (G(t))
            try:
                s = np.array([float(line.split(",")[2]) for line in lines]) #standard error
                self.return_error = True 
            except:
                print("G(t) error not found. Confidence intervals will not be reported.")
                s = np.ones(len(x))
                self.return_error = False 

        GN0=y[0] #G at t=0
        cutoff_list = np.array([np.argmax(y[1:]-y[:-1]>0), np.argmax(y<0), self.find_nearest(y,0.01*GN0), np.size(x)]) #list to determine where to cut G(t) off at long times

        cutoff = min(cutoff_list[cutoff_list>0]) #determine best place to cut G(t) at end
        x=x[:cutoff]
        y=y[:cutoff]
        s=s[:cutoff]

        self.tfinal=x[-1] #end time for lambda
        self.tstart=x[1] #start at t=1

        #Save reduced G(t) to file
        gt_result=zip(x, y)
        file = open(os.path.join(self.output_path,"gt_data.dat"),"w")
        for i in gt_result:
            file.write(str(i[0])+'\t'+str(i[1])+'\n')
        file.close()

        min_chi_sq = 1e10 #initial minimum chi^2 value
        print("")
        print("# modes, Chi^2")
        for nmodes in range(2, 15): #loop over number of modes
            self.nmodes = nmodes #set modes to self.nmodes
            lambdaArrInit=np.geomspace(self.tstart,self.tfinal,nmodes)
            gInit = 1.0/lambdaArrInit #initial g values
            params = tuple(gInit) #set g values to params
            pfit, perr, pcov = fit_curvefit(params, x, y, func,yerr=s) #send to curve fit function

            chi_sq_val = chi_sq(pfit,x,y,s) #get chi^2
            print(nmodes, chi_sq_val) #print current number of modes and chi^2
            
            #check if current chi_sq value is less than the minimum and if so, set new minimum and popt (note: also checks if any g values are ~ 0.00005 or less)
            if chi_sq_val< min_chi_sq:
                min_chi_sq = chi_sq_val
                best_modes = nmodes
                best_error = perr
                best_params = pfit
                best_cov = pcov

        #print best parameters
        print("")
        print("Best fit parameters: \n", np.round(best_params,5))
        if self.return_error:
            print("Standard deviation in parameters: \n",best_error)

        #save lambda and g values to file
        li = lambdaArrInit=np.geomspace(self.tstart,self.tfinal,best_modes)
        gi = best_params
        result=zip(li, gi)
        output = os.path.join(self.output_path,'gt_MMM_fit.dat')
        f = open(output,'w')
        f.write(str(best_modes))
        for i in result:
            f.write('\n'+str(i[0])+'\t'+str(i[1]))
        f.close()

        #plot results
        self.plot_gt_results(x,y,s,best_cov)

        return 

if __name__== "__main__":
    filepath='./Gt_result_1.txt'
    output_path = './'
    gt_fit = CURVE_FIT(filepath,output_path)
    gt_fit.fit()
