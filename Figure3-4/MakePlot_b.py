
from IPython import get_ipython
get_ipython().magic('reset -sf') 
import numpy as np
import scipy
# import matplotlib.pyplot as plt
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.stats import norm
import tikzplotlib


number_lines = 46

indices_of_true_ranks_1 = np.loadtxt('alg1rank_AC_L_MC=1000_ss=1000.csv')
indices_of_true_ranks_2 = np.loadtxt('alg2rank_AC_L_MC=1000_ss=1000.csv')

overflow_count = np.loadtxt('overflow_probability_count_L.csv')

number_samples = 1078000 #np.loadtxt('number_samples_L.csv')

a = scipy.stats.norm.sf(1.96,0,1)

overflow_probability_estimates = overflow_count/number_samples

overflow_probability_lower_bound = (a**2+2*overflow_count-a*np.sqrt(a**2-4*overflow_count*(overflow_probability_estimates-1)))/(2*(a**2+number_samples))

overflow_probability_upper_bound = (a**2+2*overflow_count+a*np.sqrt(a**2-4*overflow_count*(overflow_probability_estimates-1)))/(2*(a**2+number_samples))

interval_sizes = overflow_probability_upper_bound-overflow_probability_lower_bound

jitter = np.random.random(number_lines)

sorted_lines = np.lexsort((jitter, overflow_probability_estimates))[::-1]

positive_estimate_indices = np.argwhere(overflow_probability_estimates>0)

k = number_lines



counter = 1
for line in sorted_lines[:k]:
    plt.plot(counter, np.mean(indices_of_true_ranks_1[:,line]),   marker = 'o', color = 'red')
    plt.plot([counter, counter],[np.percentile(indices_of_true_ranks_1[:,line], 5), np.percentile(indices_of_true_ranks_1[:,line], 95)],   linestyle = 'solid', color = 'black')
    counter += 1
plt.plot(np.linspace(0,46.5,10), np.linspace(0,46.5,10), linestyle = 'dashed', color = 'gray')
plt.axis([0.5, 46.5, 0.5, 46.5])

#tikzplotlib.save("L1_AC_ss=1000_mc=1000_Feb18.tex")

plt.figure()
counter = 1
for line in sorted_lines[:k]:
    plt.plot(counter, np.mean(indices_of_true_ranks_2[:,line]),   marker = 'o', color = 'red')
    plt.plot([counter, counter],[np.percentile(indices_of_true_ranks_2[:,line], 5), np.percentile(indices_of_true_ranks_2[:,line], 95)],   linestyle = 'solid', color = 'black')
    counter += 1
plt.plot(np.linspace(0,46.5,10), np.linspace(0,46.5,10), linestyle = 'dashed', color = 'gray')
plt.axis([0.5, 46.5, 0.5, 46.5])

#tikzplotlib.save("L2_AC_ss=1000_mc=1000_Feb18.tex")
