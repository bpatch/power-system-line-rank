"""
Author: Brendan Patch

Date: 14 Feb 2020

Purpose: Experiment 3 in the paper 'Analyzing large frequency disruptions in power
systems using large deviations theory' by Brendan Patch and Bert Zwart

Notes: 
- Depends on several csv files which are contained within the same GitHub repository. 
- As written it should run in Spyder. 
"""

from IPython import get_ipython
get_ipython().magic('reset -sf') 
import numpy as np
import scipy
import scipy.optimize
import csv
import matplotlib.pyplot as plt
import numpy.matlib
import scipy.stats
#import tikzplotlib

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

gauss_true_DC = np.argsort(overflow_prob_estimates_numerical)[::-1]

b3vals_1 = -line_limits + np.dot(veeD_matrix, net_load)

b3vals_2 = -line_limits - np.dot(veeD_matrix, net_load)

A3vals_1 = np.hstack((-veeS_matrix, np.zeros((number_lines, number_generators))))

A3vals_2 = np.hstack((veeS_matrix, np.zeros((number_lines, number_generators))))

A1 = np.hstack((np.diag(1/test_beta_values), -np.identity(number_generators)))

A2 = np.hstack((-np.diag(1/test_beta_values), -np.identity(number_generators)))
        
b1 = np.multiply(1/test_beta_values, net_gen_input)

b2 = np.multiply(-1/test_beta_values, net_gen_input)

c = np.hstack((np.zeros((1,number_generators))[0], 1/test_beta_values))

rate_estimates = []
for ell in range(number_lines):
    
    if nominal_line_flows[ell][0] > 0:            
        A = np.vstack((A1, A2, A3vals_1[ell,:]))

        b = np.hstack((b1, b2, b3vals_1[ell]))
        
        soln = scipy.optimize.linprog(c, A, b, A_eq=None, b_eq= None, bounds= None, method='interior-point')
        soln2 = scipy.optimize.linprog(c, A, b, A_eq=None, b_eq= None, bounds= None, method='revised simplex')
        
    else:
        A = np.vstack((A1, A2, A3vals_2[ell,:]))
        
        b = np.hstack((b1, b2, b3vals_2[ell]))
        
        soln1 = scipy.optimize.linprog(c, A, b, A_eq=None, b_eq= None, bounds= None, method='interior-point')
        soln2 = scipy.optimize.linprog(c, A, b, A_eq=None, b_eq= None, bounds= None, method='revised simplex')
        
    rate_estimates.append(min(soln1.fun,soln2.fun))
    
sorted_lines = np.argsort(rate_estimates)
laplace_true_DC = np.argsort(rate_estimates)

indices_of_true_ranks_1 = np.loadtxt('alg1rank_AC_G_MC=100_ss=1000.csv')

overflow_count_G = np.loadtxt('overflow_probability_count_G.csv')

number_samples = 1093000 #np.loadtxt('number_samples_G.csv')

a = scipy.stats.norm.sf(1.96,0,1)

overflow_probability_estimates_G = overflow_count_G/number_samples

overflow_probability_lower_bound_G = (a**2+2*overflow_count_G-a*np.sqrt(a**2-4*overflow_count_G*(overflow_probability_estimates_G-1)))/(2*(a**2+number_samples))

overflow_probability_upper_bound_G = (a**2+2*overflow_count_G+a*np.sqrt(a**2-4*overflow_count_G*(overflow_probability_estimates_G-1)))/(2*(a**2+number_samples))

interval_sizes_G = overflow_probability_upper_bound_G-overflow_probability_lower_bound_G

positive_estimate_indices = np.argwhere(overflow_probability_estimates_G>0)

jitter = np.random.random(number_lines)

sorted_lines_AC_G = np.lexsort((jitter, overflow_probability_estimates_G))[::-1]

k = number_lines

indices_of_AC_ranks_G = np.empty((1,number_lines))

for checks in range(k):
   indices_of_AC_ranks_G[0,checks] = np.where(sorted_lines_AC_G==gauss_true_DC[:k][checks])[0][0]
        

plt.plot(np.linspace(1,46,46),indices_of_AC_ranks_G[0]+1,'.b' )
plt.plot(np.linspace(0,46.5,10), np.linspace(0,46.5,10), linestyle = 'dashed', color = 'gray')
plt.axis([0.5, 46.5, 0.5, 46.5])
#tikzplotlib.save("ACDC_G.tex")


indices_of_true_ranks_1 = np.loadtxt('alg1rank_AC_L_MC=100_ss=1000.csv')

overflow_count_L = np.loadtxt('overflow_probability_count_L.csv')

number_samples = 1078000 #np.loadtxt('number_samples_L.csv')

a = scipy.stats.norm.sf(1.96,0,1)

overflow_probability_estimates_L = overflow_count_L/number_samples

overflow_probability_lower_bound_L = (a**2+2*overflow_count_L-a*np.sqrt(a**2-4*overflow_count_L*(overflow_probability_estimates_L-1)))/(2*(a**2+number_samples))

overflow_probability_upper_bound_L = (a**2+2*overflow_count_L+a*np.sqrt(a**2-4*overflow_count_L*(overflow_probability_estimates_L-1)))/(2*(a**2+number_samples))

interval_sizes_L = overflow_probability_upper_bound_L-overflow_probability_lower_bound_L

jitter = np.random.random(number_lines)

sorted_lines_AC_L = np.lexsort((jitter, overflow_probability_estimates_L))[::-1]

positive_estimate_indices = np.argwhere(overflow_probability_estimates_L>0)


indices_of_AC_ranks_L = np.empty((1,number_lines))

for checks in range(k):
   indices_of_AC_ranks_L[0,checks] = np.where(sorted_lines_AC_L==laplace_true_DC[:k][checks])[0][0]
        


plt.figure()

plt.plot(np.linspace(1,46,46),indices_of_AC_ranks_L[0]+1,'.b' )
plt.plot(np.linspace(0,46.5,10), np.linspace(0,46.5,10), linestyle = 'dashed', color = 'gray')
plt.axis([0.5, 46.5, 0.5, 46.5])
#tikzplotlib.save("ACDC_L.tex")
