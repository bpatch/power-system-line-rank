# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 18:38:38 2019

@author: Brendan Patch
"""

from IPython import get_ipython
get_ipython().magic('reset -sf') 
import numpy as np
import scipy
import scipy.optimize
import csv
import matplotlib.pyplot as plt
import numpy.matlib
# import multiprocessing as mp


np.random.seed(2020)
number_MC_samples = 1000

noise_multiplier = 5 ## This changes the scale of variance at each generator
correlation_multiplier = 5 ## This changes the covariance between generation at different locations

number_overloads = 1
number_overloads_safety = 1
beta_tolerance = 0.00000001
threshold_tolerance = 20



with open('lines.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    lines = []
    for row in readCSV:
        lines.append([float(row[0]), float(row[1])])
        
with open('line_susceptances.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    line_susceptances = []
    for row in readCSV:
        line_susceptances.append(float(row[0]))
line_susceptances = np.array(line_susceptances)
   
# with open('line_limits.csv') as csvfile:
#     readCSV = csv.reader(csvfile, delimiter=',')
#     line_limits = []
#     for row in readCSV:
#         line_limits.append(float(row[0]))
line_limits = np.array([600,1000,500,500,900,500,500,600,500,1200,900,900,480,1800,900,900,900,600,600,900,500,500,600,600,600,600,600,600,600,600,600,900,900,900,900,600,900,600,900,600,900,600,600,600,600,1200])



with open('nominal_injections.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    nominal_injections = []
    original_generator_indices = []
    for row in readCSV:
        nominal_injections.append(float(row[1]))    
        original_generator_indices.append(float(row[0]))
        
nominal_injections = np.array(nominal_injections)
          
generator_indices = np.array(original_generator_indices) -1
number_generators = generator_indices.size

  
with open('nominal_demand.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    nominal_demand = []
    counter = 0
    for row in readCSV:
        nominal_demand.append(float(row[0]))   
        
nominal_demand = -np.array(nominal_demand)  

number_buses = np.amax(lines)
number_lines = len(lines)

load_indices = np.setdiff1d(np.arange(start=0,stop=number_buses, step=1), generator_indices)

incidence_matrix_temp = np.zeros((number_lines,int(number_buses)))

for line in range(number_lines):
    incidence_matrix_temp[line, int(lines[line][0]-1)] = 1
    incidence_matrix_temp[line, int(lines[line][1]-1)] = -1

## Rearrange nodes so that the generators are indexed first
incidence_matrix = np.concatenate((incidence_matrix_temp[:, generator_indices.astype(int)], incidence_matrix_temp[:, load_indices.astype(int)]),axis=1)
nominal_demand = np.concatenate((nominal_demand[generator_indices.astype(int)], nominal_demand[load_indices.astype(int)]),axis=0)


weighted_graph_laplacian = np.linalg.multi_dot([ np.transpose(incidence_matrix), np.diag(line_susceptances), incidence_matrix])
weighted_graph_laplacian_pinv = np.linalg.pinv(weighted_graph_laplacian)
vee_matrix = np.linalg.multi_dot([np.diag(line_susceptances), incidence_matrix, weighted_graph_laplacian_pinv])

nominal_line_flows = np.abs(np.dot(vee_matrix, (np.concatenate((nominal_injections, np.zeros((1,int(number_buses-number_generators)))[0]),axis=0)+nominal_demand).reshape(-1,1)))

test_beta_values = np.ones((1,number_generators))
for i in range(number_generators):
    test_beta_values[0][i] =  (1/np.sqrt(1/(noise_multiplier*nominal_injections[i])))

test_beta_values = test_beta_values[0]
np.random.seed(2020)

A = correlation_multiplier*np.random.randn(number_generators,number_generators)
off_diagonal_cov_entries = numpy.dot(A,A.transpose())
off_diagonal_cov_entries[np.diag_indices(number_generators)] = np.zeros((1,number_generators))
test_cov_matrix = noise_multiplier*np.diag(nominal_injections) + off_diagonal_cov_entries

def gauss_true_rank(cov_matrix):
    cov_matrix_inv = np.linalg.inv(cov_matrix)
    
    def gauss_convex_con(x):
        return 0.5*np.linalg.multi_dot([x-nominal_injections, cov_matrix_inv, (x-nominal_injections).reshape(-1,1)])[0]
   
    def gauss_convex_con_jac(x):
         return np.dot(x-nominal_injections, cov_matrix_inv)
     
    rate_estimates = []
    for ell in range(number_lines):
       
        def constraint(x):
            line_flows = np.abs(np.dot(vee_matrix, (np.concatenate((x, np.zeros((1,int(number_buses-number_generators)))[0]),axis=0)+nominal_demand).reshape(-1,1)))
            return (line_flows[ell]-line_limits[ell])[0]
        
        con = {'type': 'ineq', 'fun': constraint}
       
        x_initial = nominal_injections
   
        sol = scipy.optimize.minimize(gauss_convex_con, x_initial, constraints=con, method='SLSQP',options={'disp': False,'maxiter': 1000})
   
        rate_estimates.append(sol.fun)
        
        
        
    return np.argsort(rate_estimates)

gauss_true = gauss_true_rank(test_cov_matrix)

print('True rank Gauss:', gauss_true)

P = 10


rangeSamples = np.hstack((np.linspace(10,10**2,P), np.linspace(10**2,10**3,P),np.linspace(10**3,10**4,P),np.linspace(10**4,10**5,P)))

total_simulation_effort = number_MC_samples*P*4


alg2_data_k1 = np.empty((number_MC_samples, P*4))
alg2_data_k2 = np.empty((number_MC_samples,P*4))
alg2_data_k3 = np.empty((number_MC_samples, P*4))
alg2_data_k4 = np.empty((number_MC_samples, P*4))
alg2_data_k5 = np.empty((number_MC_samples, P*4))


counter = 1
counter2 = 0

np.random.seed(2020)

x_initial_alg2 = np.repeat([nominal_injections],repeats = number_lines, axis=0)


for number_test_samples in rangeSamples:
    for sample in range(number_MC_samples):

        data = np.transpose(np.random.multivariate_normal(nominal_injections, test_cov_matrix, size = number_test_samples.astype(int)))
        
        number_samples = data.shape[1]
        new_data = data - np.array(nominal_injections).reshape(-1,1)
        estimated_cov = np.dot(new_data,np.transpose(new_data))/number_samples
        estimated_cov_inv = np.linalg.inv(estimated_cov)
        def gauss_convex_con(x):
            return 0.5*np.linalg.multi_dot([x-nominal_injections, estimated_cov_inv, (x-nominal_injections).reshape(-1,1)])[0]
    
        def gauss_convex_con_jac(x):
            return np.dot(x-nominal_injections, estimated_cov_inv)
        
        rate_estimates = []
        for ell in range(number_lines):
           
            def constraint(x):
                line_flows = np.abs(np.dot(vee_matrix, (np.concatenate((x, np.zeros((1,int(number_buses-number_generators)))[0]),axis=0)+nominal_demand).reshape(-1,1)))
                return (line_flows[ell]-line_limits[ell])[0]
           
            con = {'type': 'ineq', 'fun': constraint}
                        
            initial_condition = np.copy(x_initial_alg2)
            
            sol = scipy.optimize.minimize(gauss_convex_con, initial_condition[ell,:], constraints=con, method='SLSQP', jac = gauss_convex_con_jac, options={'disp': False,'maxiter': 1000})
    
            x_initial_alg2[ell,:] = sol.x
           
            rate_estimates.append(sol.fun)

        sorted_lines = np.argsort(rate_estimates)
        
        indices_of_true_ranks_1 = np.empty(1)
        for checks in range(1):
            indices_of_true_ranks_1[checks] = np.where(sorted_lines==gauss_true[:1][checks])[0][0]
        alg2_data_k1[sample, counter2] = max(indices_of_true_ranks_1)+1
        
        indices_of_true_ranks_2 = np.empty(2)
        for checks in range(2):
            indices_of_true_ranks_2[checks] = np.where(sorted_lines==gauss_true[:2][checks])[0][0]
        alg2_data_k2[sample, counter2] = max(indices_of_true_ranks_2)+1       
        
        indices_of_true_ranks_3 = np.empty(3)
        for checks in range(3):
            indices_of_true_ranks_3[checks] = np.where(sorted_lines==gauss_true[:3][checks])[0][0]
        alg2_data_k3[sample, counter2] = max(indices_of_true_ranks_3)+1  

        indices_of_true_ranks_4 = np.empty(4)
        for checks in range(4):
            indices_of_true_ranks_4[checks] = np.where(sorted_lines==gauss_true[:4][checks])[0][0]
        alg2_data_k4[sample, counter2] = max(indices_of_true_ranks_4)+1   


        indices_of_true_ranks_5 = np.empty(5)
        for checks in range(5):
            indices_of_true_ranks_5[checks] = np.where(sorted_lines==gauss_true[:5][checks])[0][0]
        alg2_data_k5[sample, counter2] = max(indices_of_true_ranks_5)+1  
        

        print(100*counter/total_simulation_effort, '%')
        counter += 1
    counter2 += 1

#plt.plot(range(len(np.mean(alg2_data_k1 <= 1, axis=0))),1-np.mean(alg2_data_k2 <= 3, axis=0), 'r-')

np.savetxt('alg2data_k1_G_MC=1000_P=10.csv',alg2_data_k1.astype(int))
np.savetxt('alg2data_k2_G_MC=1000_P=10.csv',alg2_data_k2.astype(int))
#np.savetxt('alg2data_k3_G_MC=1000_P=10.csv',alg2_data_k3.astype(int))
#np.savetxt('alg2data_k4_G_MC=1000_P=10.csv',alg2_data_k4.astype(int))
#np.savetxt('alg2data_k5_G_MC=1000_P=10.csv',alg2_data_k5.astype(int))
