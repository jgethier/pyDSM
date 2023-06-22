import numpy as np
import os
import pickle


def save_distributions(input_data,distr,QN,Z,output_dir,sim_ID):
    '''
    Function to save Q distributions to file
    
    Args: 
        input_data - input parameters from yaml file
        distr - name of distribution (initial or final) 
        QN - distribution of strand orientations and number of Kuhn steps in cluster
        Z - number of entanglements in each chain
        output_dir - path to output directory
        sim_ID - simulation ID number 
    '''
    Q = []
    L = []
    for i in range(0,input_data['Nchains']):
        L_value = 0.0
        for j in range(1,int(Z[i])-1):
            x2 = QN[i,j,0]**2
            y2 = QN[i,j,1]**2
            z2 = QN[i,j,2]**2
            Q_value = np.sqrt(x2+y2+z2)
            L_value += Q_value
            Q.append(Q_value)
            
        L.append(L_value)

    Q_sorted = np.sort(Q)
    L_sorted = np.sort(L)
    
    Q_file = os.path.join(output_dir,'distr_Q_%s_%d.txt'%(distr,sim_ID))
    L_file = os.path.join(output_dir,'distr_L_%s_%d.txt'%(distr,sim_ID))
    
    np.savetxt(Q_file, Q_sorted)
    np.savetxt(L_file, L_sorted)

    return

def write_stress(input_data,flow,turn_flow_off,num_sync,time,stress_array,output_dir,sim_ID):
    '''
    Write the stress of all chains in ensemble
    
    Args: 
        input_data - input parameters from yaml file
        flow - boolean to indicate whether flow is on or off
        turn_flow_off - boolean to indicate whether flow will be turned off
        num_sync - sync number for simulation
        time - simulation time
        stress_array - array of stress values to write
        output_dir - path to output directory
        sim_ID - simulation ID number
    
    Returns: 
        stress over time for each chain in stress.txt file
    '''
    global old_sync_time

    stress_output = os.path.join(output_dir,'stress_%d.txt'%sim_ID)
    
    #if first time sync, need to include 0 array index
    if num_sync == 1:
        old_sync_time = 0
        if flow:
            time_index = 1
        else:
            time_index = 0
    else: #do not include the t=0 spot of array (only used for first time sync)
        if flow:
            time_index = 1
        else:
            time_index = 0
    
    #set time array depending on num_sync
    time_resolution = input_data['tau_K']
    time_array = np.arange(old_sync_time,time+time_resolution/2.0,time_resolution)
    if not flow and turn_flow_off:
        time_array = np.arange(old_sync_time,(time+input_data['flow']['flow_time']),time_resolution)
    time_array = np.reshape(time_array[time_index:],(1,len(time_array[time_index:])))
    len_array = len(time_array[0])+1
    
    #if flow, take average stress tensor over all chains, otherwise write out only tau_xy stress of all chains
    if flow or turn_flow_off:
        if flow:
            stress = np.array([np.mean(stress_array[:,0,i],axis=0) for i in range(0,8)])
            error = np.array([np.std(stress_array[:,0,i],axis=0)/np.sqrt(input_data['Nchains']) for i in range(0,8)])
            stress = np.reshape(stress,(8,1))
            error = np.reshape(error,(8,1))
        else:
            stress = np.array([np.mean(stress_array[:,:,i],axis=0) for i in range(0,8)])
            error = np.array([np.std(stress_array[:,:,i],axis=0)/np.sqrt(input_data['Nchains']) for i in range(0,8)])
            stress = np.reshape(stress[:,time_index:len_array],(8,len(stress_array[0,time_index:len_array,0])))
            error = np.reshape(error[:,time_index:len_array],(8,len(stress_array[0,time_index:len_array,0])))
        combined = np.hstack((time_array.T, stress.T, error.T))
    else:
        stress = np.reshape(stress_array[:,time_index:len_array,0],(input_data['Nchains'],len(stress_array[0,time_index:len_array,0])))    
        combined = np.hstack((time_array.T, stress.T))
    
    #write stress to file
    if num_sync == 1:
        with open(stress_output,'w') as f:
            if flow or turn_flow_off:
                f.write('time, tau_xx, tau_yy, tau_zz, tau_xy, tau_yz, tau_xz, Z, f_newQ, stderr_xx, stderr_yy, stderr_zz, stderr_xy, stderr_yz, stderr_xz, stderr_Z, stderr_f_newQ\n')
            else:
                f.write('time, tau_xy_chain0, tau_xy_chain1, ... , tau_xy_Nchains\n')
            np.savetxt(f, combined, delimiter=',', fmt='%.8f')
    else:
        with open(stress_output,'a') as f:
            np.savetxt(f, combined, delimiter=',', fmt='%.8f')
    
    #keeping track of the last simulation time for beginning of next array
    if not flow and turn_flow_off:
        old_sync_time = time + input_data['flow']['flow_time']
    else:
        old_sync_time = time
    
    return 


def write_com(input_data,num_sync,time,com_array,output_dir,sim_ID):
    '''
    Write the CoM of all chains in ensemble
    
    Inputs: 
        input_data - input parameters from yaml file
        num_sync - sync number for simulation
        time - simulation time
        com_array - array of center of mass (CoM) values to write
        output_dir - path to output directory
        sim_ID - simulation ID number 
    
    Returns: 
        Center-of-mass over time for each chain and dimension (x,y,z) in CoM.txt file
    '''
    
    if num_sync == 1: #if first time sync, need to include 0 and initialize file path
        com_output_x = os.path.join(output_dir, 'CoM_%d_x.dat'%sim_ID)
        com_output_y = os.path.join(output_dir, 'CoM_%d_y.dat'%sim_ID)
        com_output_z = os.path.join(output_dir, 'CoM_%d_z.dat'%sim_ID)
        old_sync_time = 0
        time_index = 0
    
    else:
        time_index = 1
    
    #set time array depending on num_sync
    time_resolution = input_data['tau_K']
    time_array = np.arange(old_sync_time,time+time_resolution/2.0,time_resolution)
    time_array = np.reshape(time_array[time_index:],(1,len(time_array[time_index:])))
    len_array = len(time_array[0])+1
    
    #reshape arrays
    com_x = np.reshape(com_array[:,time_index:len_array,0],(input_data['Nchains'],len(com_array[0,time_index:len_array,0])))    
    com_y = np.reshape(com_array[:,time_index:len_array,1],(input_data['Nchains'],len(com_array[0,time_index:len_array,1])))    
    com_z = np.reshape(com_array[:,time_index:len_array,2],(input_data['Nchains'],len(com_array[0,time_index:len_array,2])))    
    
    #combine time and CoM for each dimension
    combined_x = np.hstack((time_array.T, com_x.T))
    combined_y = np.hstack((time_array.T, com_y.T))
    combined_z = np.hstack((time_array.T, com_z.T))
    
    if num_sync == 1: #write data to files and overwrite if file exists
        with open(com_output_x,"wb") as f:
            pickle.dump(combined_x,f)
        with open(com_output_y,"wb") as f:
            pickle.dump(combined_y,f)
        with open(com_output_z,"wb") as f:
            pickle.dump(combined_z,f)
    else: #append file for num_sync > 1
        with open(com_output_x,"ab") as f:
            pickle.dump(combined_x,f)
        with open(com_output_y,"ab") as f:
            pickle.dump(combined_y,f)
        with open(com_output_z,"ab") as f:
            pickle.dump(combined_z,f)
    
    #keeping track of the last simulation time for beginning of next array
    old_sync_time = time


def load_results(filename,block_num,block_size,num_chains):
    '''
    Load in part of the binary .dat file into the result array
    
    Args: 
        filename - filename of the data file
        block_num - chain block number (total chains split into n blocks of size block_size)
        num_chains - block_num*block_size
    
    Returns: 
        an array of data read from filename
    '''
    result_array = []
    with open(filename,'rb') as f:
        try:
            while True:
                data = pickle.load(f)
                for j in data:
                    if block_num == 0:
                        first_idx = 1
                    else:
                        first_idx = block_num*block_size + 1
                    last_idx = num_chains+1
                    result_array.append(j[first_idx:last_idx])
        except EOFError:
            pass

    return result_array