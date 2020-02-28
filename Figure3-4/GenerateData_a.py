
from IPython import get_ipython
get_ipython().magic('reset -sf') 
import numpy as np
import scipy
import scipy.optimize
import csv
# import matplotlib.pyplot as plt
import numpy.matlib
# import multiprocessing as mp
from scipy.stats import norm
# import tikzplotlib
import pypsa
import pandas as pd
import random

np.random.seed(2020)
number_MC_samples = 1000

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
# line_susceptances = np.array(line_susceptances)
   
    
with open('maximum_power_generation.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    max_power = []
    for row in readCSV:
        max_power.append(float(row[0]))

line_limits = [600,1000,500,500,900,500,500,600,500,1200,900,900,480,1800,900,900,900,600,600,900,500,500,600,600,600,600,600,600,600,600,600,900,900,900,900,600,900,600,900,600,900,600,600,600,600,1200]
        
# nominal_injections = [831.339591715101, 646.0, 606.996266940159, 652.0, 508.0, 687.0, 580.0, 564.0, 865.0, 313.894141344742]

with open('nominal_injections.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    nominal_injections = []
    original_generator_indices = []
    for row in readCSV:
        nominal_injections.append(float(row[1]))    
        original_generator_indices.append(float(row[0]))
         
with open('line_resistances.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    line_resistances = []
    for row in readCSV:
        if float(row[0]) > 0:
            line_resistances.append(float(row[0]))
        else:            
            line_resistances.append(0.0001)

        
# line_resistances = np.array(line_resistances)

# line_resistances[np.argwhere(line_resistances==0)] = 0.00001

with open('line_reactances.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    line_reactances = []
    for row in readCSV:
        line_reactances.append(float(row[0]))

# line_reactances = np.array(line_reactances)

with open('nominal_voltages.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    nominal_voltages = []
    for row in readCSV:
        nominal_voltages.append(float(row[0]))

# nominal_voltages = np.array(nominal_voltages)

# generator_indices = np.array(original_generator_indices) -1
# number_generators = generator_indices.size

with open('nominal_demand.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    nominal_demand = []
    counter = 0
    for row in readCSV:
        nominal_demand.append(float(row[0]))   
        
# nominal_demand = -np.array(nominal_demand)  
        
number_lines = len(lines)

number_generators = len(nominal_injections)

A = correlation_multiplier*np.random.randn(number_generators,number_generators)
off_diagonal_cov_entries = numpy.dot(A,A.transpose())
off_diagonal_cov_entries[np.diag_indices(number_generators)] = np.zeros((1,number_generators))
test_cov_matrix = noise_multiplier*np.diag(nominal_injections) + off_diagonal_cov_entries




network = pypsa.Network()

#add three buses
n_buses = 39

for i in range(n_buses):
    network.add("Bus","My bus {}".format(i),
               v_nom= 20*nominal_voltages[i])

# print(network.buses)

for i in range(number_lines):
    network.add("Line","My line {}".format(i),
                bus0="My bus {}".format(int(lines[i][0]-1)),
                bus1="My bus {}".format(int(lines[i][1]-1)),
                x=line_reactances[i],
                r=line_resistances[i])
    
# print(network.lines)

for i in range(number_generators):
        network.add("Generator","My gen {}".format(i),
            bus="My bus {}".format(i+29),
            p_set=nominal_injections[i])
    
# print(network.generators)

# print(network.generators.p_set)

load_indices = np.argwhere(nominal_demand)

for i in range(load_indices.size):
    network.add("Load","My load {}".format(i),
                bus="My bus {}".format(int(load_indices[i])),
                p_set=nominal_demand[load_indices[i][0]])#,
                # q_set=100.0)


print(network.generators_t.p.loc['now'].tolist())

# print(network.loads)

# print(network.loads.p_set)


k1 = 46
j1 = 1

number_MC_samples = 10**2

number_samples = 10**3

overflow_probability_estimates = np.zeros((1,number_lines))[0]

network.generators_t.p.loc['now'].tolist()

indices_of_true_ranks_1 = np.empty((number_MC_samples, 46))
indices_of_true_ranks_2 = np.empty((number_MC_samples, 46))


for sample in range(number_MC_samples):

     data = np.empty((number_samples,number_lines))
     for observation in range(number_samples):
         network.generators.p_set = np.random.multivariate_normal(nominal_injections,test_cov_matrix)
         network.lpf()
         network.pf()
         line_flows= np.abs(network.lines_t.p0.loc['now'].tolist())
         data[observation,:] = line_flows
        
     rate_estimates = []
     for ell in range(number_lines):
         def y_convex_con_subproblem(lambda_variable):
             return -lambda_variable*line_limits[ell] + scipy.special.logsumexp(np.multiply(lambda_variable,data[:,ell]), b= 1/number_samples)
    
         soln = scipy.optimize.minimize(y_convex_con_subproblem, line_limits[ell], method='SLSQP',options={'disp': False, 'maxiter': 1000})
                                    
         rate_estimates.append(-soln.fun)
        
     sorted_lines1 = np.argsort(rate_estimates)
    
     overflow_probability_estimates = []
     for ell in range(number_lines): ## This can be made faster with an online estimate
         overflow_probability_estimates.append(np.mean(data[:,ell] > line_limits[ell]))
    
     jitter =np.random.random(number_lines)
    
     sorted_lines2 = np.lexsort((jitter, overflow_probability_estimates))[::-1]
 
     for checks in range(46):
         indices_of_true_ranks_1[sample,checks] = np.where(sorted_lines1==checks)[0][0]+1
         indices_of_true_ranks_2[sample,checks] = np.where(sorted_lines2==checks)[0][0]+1
        
     print()
     print()
     print(100*sample/number_MC_samples, ' %')
     print(100*sample/number_MC_samples, ' %')
     print(100*sample/number_MC_samples, ' %')
     print(100*sample/number_MC_samples, ' %')
     print()
     print()

np.savetxt('alg1rank_AC_G_MC=1000_ss=1000.csv', indices_of_true_ranks_1.astype(int))
np.savetxt('alg2rank_AC_G_MC=1000_ss=1000.csv', indices_of_true_ranks_2.astype(int))


#indices_of_true_ranks_1_loaded = np.loadtxt('alg1rank_AC_G_MC=1000_ss=1000.csv')
