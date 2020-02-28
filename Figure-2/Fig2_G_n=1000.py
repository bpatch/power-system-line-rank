# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:22:40 2020

@author: brend
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
from scipy.stats import norm
import tikzplotlib

np.random.seed(2020)

noise_multiplier = 5 ## This changes the scale of variance at each generator
correlation_multiplier = 5 ## This changes the covariance between generation at different locations

number_overloads = 2
number_overloads_safety = 2
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

net_gen_input = nominal_injections+nominal_demand[:number_generators]
net_load = nominal_demand[number_generators:]

weighted_graph_laplacian = np.linalg.multi_dot([ np.transpose(incidence_matrix), np.diag(line_susceptances), incidence_matrix])
weighted_graph_laplacian_pinv = np.linalg.pinv(weighted_graph_laplacian)
vee_matrix = np.linalg.multi_dot([np.diag(line_susceptances), incidence_matrix, weighted_graph_laplacian_pinv])

veeS_matrix = vee_matrix[:, 0:number_generators]
veeD_matrix = vee_matrix[:, number_generators:]

nominal_line_flows = np.dot(vee_matrix, (np.concatenate((nominal_injections, np.zeros((1,int(number_buses-number_generators)))[0]),axis=0)+nominal_demand).reshape(-1,1))

nominal_line_flows_dir = np.dot(vee_matrix, (np.concatenate((nominal_injections, np.zeros((1,int(number_buses-number_generators)))[0]),axis=0)+nominal_demand).reshape(-1,1))

test_beta_values = np.ones((1,number_generators))
for i in range(number_generators):
    test_beta_values[0][i] =  np.sqrt(0.5*noise_multiplier*nominal_injections[i])

test_beta_values = test_beta_values[0]

A = correlation_multiplier*np.random.randn(number_generators,number_generators)
off_diagonal_cov_entries = numpy.dot(A,A.transpose())
off_diagonal_cov_entries[np.diag_indices(number_generators)] = np.zeros((1,number_generators))
test_cov_matrix = noise_multiplier*np.diag(nominal_injections) + off_diagonal_cov_entries

test_cov_matrix_lines = np.linalg.multi_dot([veeS_matrix,test_cov_matrix, np.transpose(veeS_matrix)])
         
overflow_prob_estimates_numerical = []

for ell in range(number_lines):
    overflow_prob_estimates_numerical.append(scipy.stats.norm.cdf(-line_limits[ell], nominal_line_flows_dir[ell], np.sqrt(test_cov_matrix_lines[ell,ell]))[0]+scipy.stats.norm.sf(line_limits[ell], nominal_line_flows_dir[ell], np.sqrt(test_cov_matrix_lines[ell,ell]))[0])

gauss_true= np.argsort(overflow_prob_estimates_numerical)[::-1]

print('True rank Gauss:', gauss_true[:10])


b3vals_1 = -line_limits + np.dot(veeD_matrix, net_load)

b3vals_2 = -line_limits - np.dot(veeD_matrix, net_load)

A3vals_1 = np.hstack((-veeS_matrix, np.zeros((number_lines, number_generators))))

A3vals_2 = np.hstack((veeS_matrix, np.zeros((number_lines, number_generators))))

# A1 = np.hstack((np.diag(1/test_beta_values), -np.identity(number_generators)))

# A2 = np.hstack((-np.diag(1/test_beta_values), -np.identity(number_generators)))
        
# b1 = np.multiply(1/test_beta_values, net_gen_input)

# b2 = np.multiply(-1/test_beta_values, net_gen_input)

# c = np.hstack((np.zeros((1,number_generators))[0], 1/test_beta_values))

# rate_estimates = []
# for ell in range(number_lines):
    
#     if nominal_line_flows[ell][0] > 0:            
#         A = np.vstack((A1, A2, A3vals_1[ell,:]))

#         b = np.hstack((b1, b2, b3vals_1[ell]))
        
#         soln = scipy.optimize.linprog(c, A, b, A_eq=None, b_eq= None, bounds= None, method='interior-point')
#         soln2 = scipy.optimize.linprog(c, A, b, A_eq=None, b_eq= None, bounds= None, method='revised simplex')
        
#     else:
#         A = np.vstack((A1, A2, A3vals_2[ell,:]))

#         b = np.hstack((b1, b2, b3vals_2[ell]))
        
#         soln1 = scipy.optimize.linprog(c, A, b, A_eq=None, b_eq= None, bounds= None, method='interior-point')
#         soln2 = scipy.optimize.linprog(c, A, b, A_eq=None, b_eq= None, bounds= None, method='revised simplex')
        
#     rate_estimates.append(min(soln1.fun,soln2.fun))
    
# sorted_lines = np.argsort(rate_estimates)
# laplace_true = np.argsort(rate_estimates)


# print('True rank Laplace:', laplace_true[:10])

sample_size = 10**3

k1 = 46
j1 = 1

number_MC_samples = 10**3


indices_of_true_ranks_1 = np.empty((number_MC_samples, k1))
indices_of_true_ranks_2 = np.empty((number_MC_samples, k1))
indices_of_true_ranks_3 = np.empty((number_MC_samples, k1))
indices_of_true_ranks_4 = np.empty((number_MC_samples, k1))




for sample in range(number_MC_samples):
    
    # V_samples = np.empty((number_generators, sample_size))
    # for i in range(number_generators):
    #     V_samples[i,:] = np.random.gamma(1, test_beta_values[i], size=sample_size)   
    # R_samples = np.random.binomial(1,0.5,(number_generators, sample_size))
    # R_samples[R_samples==0] = -1
    # data = np.multiply(V_samples, R_samples)
            
    # data = data + np.array(net_gen_input).reshape(-1,1)
    
    data = np.transpose(np.random.multivariate_normal(net_gen_input, test_cov_matrix, size = sample_size))



    counter2 = 0 
    
    line_flows = np.abs(np.dot(vee_matrix,np.concatenate((data, np.tile(net_load.reshape(-1,1), sample_size)),axis=0)))

#### ALGORITHM 2
    overflow_probability_estimates = []
    for ell in range(number_lines): ## This can be made faster with an online estimate
        overflow_probability_estimates.append(np.mean(line_flows[ell,:] > line_limits[ell]))
    
    jitter =np.random.random(number_lines)
    
    sorted_lines2 = np.lexsort((jitter, overflow_probability_estimates))[::-1]
    
    for checks in range(k1):
        indices_of_true_ranks_2[sample,checks] = np.where(sorted_lines2==gauss_true[:k1][checks])[0][0]+1
        

#### ALGORITHM 3
    new_data = data - np.array(net_gen_input).reshape(-1,1)
    estimated_cov = np.dot(new_data, np.transpose(new_data))/sample_size

    estimated_cov_lines = np.linalg.multi_dot([veeS_matrix,estimated_cov, np.transpose(veeS_matrix)])

    overflow_prob_estimates_numerical = []
    for ell in range(number_lines):
        overflow_prob_estimates_numerical.append(scipy.stats.norm.cdf(-line_limits[ell], nominal_line_flows_dir[ell], np.sqrt(estimated_cov_lines[ell,ell]))[0]+scipy.stats.norm.sf(line_limits[ell], nominal_line_flows_dir[ell], np.sqrt(estimated_cov_lines[ell,ell]))[0])
    
    sorted_lines3 = np.argsort(overflow_prob_estimates_numerical)[::-1]

    for checks in range(k1):
        indices_of_true_ranks_3[sample,checks] = np.where(sorted_lines3==gauss_true[:k1][checks])[0][0]+1
        


#### ALGORITHM 4
    beta_estimates = np.sum(np.abs(data-net_gen_input.reshape(-1,1)),axis=1)/sample_size       
            
    A1 = np.hstack((np.diag(1/beta_estimates), -np.identity(number_generators)))
    
    A2 = np.hstack((-np.diag(1/beta_estimates), -np.identity(number_generators)))
            
    b1 = np.multiply(1/beta_estimates, net_gen_input)
    
    b2 = np.multiply(-1/beta_estimates, net_gen_input)

    c = np.hstack((np.zeros((1,number_generators))[0], 1/beta_estimates))
    
    rate_estimates4 = []
    for ell in range(number_lines):
        
        if nominal_line_flows[ell][0] > 0:            
            A = np.vstack((A1, A2, A3vals_1[ell,:]))

            b = np.hstack((b1, b2, b3vals_1[ell]))
            
            soln1 = scipy.optimize.linprog(c, A, b, A_eq=None, b_eq= None, bounds= None, method='revised simplex')
            soln2 = scipy.optimize.linprog(c, A, b, A_eq=None, b_eq= None, bounds= None, method='interior-point')
        else:
            A = np.vstack((A1, A2, A3vals_2[ell,:]))

            b = np.hstack((b1, b2, b3vals_2[ell]))
            
            soln1 = scipy.optimize.linprog(c, A, b, A_eq=None, b_eq= None, bounds= None, method='revised simplex')
            soln2 = scipy.optimize.linprog(c, A, b, A_eq=None, b_eq= None, bounds= None, method='interior-point')
                        
        rate_estimates4.append(min(soln1.fun, soln2.fun))
        
    sorted_lines4 = np.argsort(rate_estimates4)
    
    for checks in range(k1):
        indices_of_true_ranks_4[sample,checks] = np.where(sorted_lines4==gauss_true[:k1][checks])[0][0]+1
        
#### ALGORITHM 1

    rate_estimates = []
    for ell in range(number_lines):
        def y_convex_con_subproblem(lambda_variable):
            return -lambda_variable*line_limits[ell] + scipy.special.logsumexp(np.multiply(lambda_variable,line_flows[ell]), b= 1/sample_size)

        
        soln = scipy.optimize.minimize(y_convex_con_subproblem, line_limits[ell], method='SLSQP',options={'disp': False, 'maxiter': 1000})
                                
        rate_estimates.append(-soln.fun)
        
    sorted_lines1 = np.argsort(rate_estimates)
    
    for checks in range(k1):
        indices_of_true_ranks_1[sample,checks] = np.where(sorted_lines1==gauss_true[:k1][checks])[0][0]+1
        

    print(100*(sample+1)/number_MC_samples, '%')
    counter2 += 1

lower_bracket_pi_1 = np.empty((1, k1))
upper_bracket_pi_1 = np.empty((1, k1))
mean_1 = np.empty((1, k1))

lower_bracket_pi_2 = np.empty((1, k1))
upper_bracket_pi_2 = np.empty((1, k1))
mean_2 = np.empty((1, k1))


lower_bracket_pi_3 = np.empty((1, k1))
upper_bracket_pi_3 = np.empty((1, k1))
mean_3 = np.empty((1, k1))


lower_bracket_pi_4 = np.empty((1, k1))
upper_bracket_pi_4 = np.empty((1, k1))
mean_4 = np.empty((1, k1))


for checks in range(k1):
    lower_bracket_pi_1[0,checks] = np.percentile(indices_of_true_ranks_1[:,checks], 5)
    upper_bracket_pi_1[0,checks] = np.percentile(indices_of_true_ranks_1[:,checks], 95)
    mean_1[0,checks] = np.mean(indices_of_true_ranks_1[:,checks])

    
    lower_bracket_pi_2[0,checks] = np.percentile(indices_of_true_ranks_2[:,checks], 5)
    upper_bracket_pi_2[0,checks] = np.percentile(indices_of_true_ranks_2[:,checks], 95)
    mean_2[0,checks] = np.mean(indices_of_true_ranks_2[:,checks])

    
    lower_bracket_pi_3[0,checks] = np.percentile(indices_of_true_ranks_3[:,checks], 5)
    upper_bracket_pi_3[0,checks] = np.percentile(indices_of_true_ranks_3[:,checks], 95)
    mean_3[0,checks] = np.mean(indices_of_true_ranks_3[:,checks])

    
    lower_bracket_pi_4[0,checks] = np.percentile(indices_of_true_ranks_4[:,checks], 5)
    upper_bracket_pi_4[0,checks] = np.percentile(indices_of_true_ranks_4[:,checks], 95)
    mean_4[0,checks] = np.mean(indices_of_true_ranks_4[:,checks])



plt.figure(1)

for checks in range(k1):
    plt.plot([checks+1, checks+1], [lower_bracket_pi_1[0,checks],upper_bracket_pi_1[0,checks]],   linestyle = 'solid', color = 'black')
    plt.plot(checks+1, mean_1[0,checks],   marker = 'o', color = 'red')
    plt.plot(np.linspace(0,46.5,10), np.linspace(0,46.5,10), linestyle = 'dashed', color = 'gray')
    plt.axis([0.5, 46.5, 0.5, 46.5])
    plt.title('Alg 1 ECGF')

#tikzplotlib.save("G1_ss=1000_mc=1000_Jan30.tex")


plt.figure(2)

for checks in range(k1):
    plt.plot([checks+1, checks+1], [lower_bracket_pi_2[0,checks],upper_bracket_pi_2[0,checks]] , linestyle = 'solid', color = 'black')
    plt.plot(checks+1, mean_2[0,checks],   marker = 'o', color = 'red')
    plt.plot(np.linspace(0,46.5,10), np.linspace(0,46.5,10), linestyle = 'dashed', color = 'gray')
    plt.axis([0.5, 46.5, 0.5, 46.5])
    plt.title('Alg 2 Indicator')

#tikzplotlib.save("G2_ss=1000_mc=1000_Jan30.tex")


plt.figure(3)

for checks in range(k1):
    plt.plot([checks+1, checks+1], [lower_bracket_pi_3[0,checks],upper_bracket_pi_3[0,checks]] , linestyle = 'solid', color = 'black')
    plt.plot(checks+1, mean_3[0,checks],   marker = 'o', color = 'red')
    plt.plot(np.linspace(0,46.5,10), np.linspace(0,46.5,10), linestyle = 'dashed', color = 'gray')
    plt.axis([0.5, 46.5, 0.5, 46.5])
    plt.title('Alg 3 Gauss')

#tikzplotlib.save("G3_ss=1000_mc=1000_Jan30.tex")


plt.figure(4)

for checks in range(k1):
    plt.plot([checks+1, checks+1], [lower_bracket_pi_4[0,checks],upper_bracket_pi_4[0,checks]] , linestyle = 'solid', color = 'black')
    plt.plot(checks+1, mean_4[0,checks],   marker = 'o', color = 'red')
    plt.plot(np.linspace(0,46.5,10), np.linspace(0,46.5,10), linestyle = 'dashed', color = 'gray')
    plt.axis([0.5, 46.5, 0.5, 46.5])
    plt.title('Alg 4 Laplace')
    
#tikzplotlib.save("G4_ss=1000_mc=1000_Jan30.tex")
