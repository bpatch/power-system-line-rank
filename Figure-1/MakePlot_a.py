# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 16:16:31 2020

@author: brend
"""
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib


alg1_k1 = np.loadtxt('alg1data_k1_G_MC=10000.csv')
alg1_k2 = np.loadtxt('alg1data_k2_G_MC=10000.csv')
#alg1_k3 = np.loadtxt('alg1data_k3_G_MC=10000.csv')
#alg1_k4 = np.loadtxt('alg1data_k4_G_MC=10000.csv')
#alg1_k5 = np.loadtxt('alg1data_k5_G_MC=10000.csv')


alg2_k1 = np.loadtxt('alg2data_k1_G_MC=1000_P=10.csv')
alg2_k2 = np.loadtxt('alg2data_k2_G_MC=1000_P=10.csv')
#alg2_k3 = np.loadtxt('alg2data_k3_G_MC=1000_P=10.csv')
#alg2_k4 = np.loadtxt('alg2data_k4_G_MC=1000_P=10.csv')
#alg2_k5 = np.loadtxt('alg2data_k5_G_MC=1000_P=10.csv')

alg3_k1 = np.loadtxt('alg3data_k1_G_MC=1000_P=10.csv')
alg3_k2 = np.loadtxt('alg3data_k2_G_MC=1000_P=10.csv')
#alg3_k3 = np.loadtxt('alg3data_k3_G_MC=1000_P=10.csv')
#alg3_k4 = np.loadtxt('alg3data_k4_G_MC=1000_P=10.csv')
#alg3_k5 = np.loadtxt('alg3data_k5_G_MC=1000_P=10.csv')

alg4_k1 = np.loadtxt('alg4data_k1_G_MC=1000_P=10.csv')
alg4_k2 = np.loadtxt('alg4data_k2_G_MC=1000_P=10.csv')
#alg4_k3 = np.loadtxt('alg4data_k3_G_MC=1000_P=10.csv')
#alg4_k4 = np.loadtxt('alg4data_k4_G_MC=1000_P=10.csv')
#alg4_k5 = np.loadtxt('alg4data_k5_G_MC=1000_P=10.csv')


P1 = 10
P2 = 10
P3 = 10
P4 = 10

### k = 1, j = 1

plt.figure(0)

plt.plot(range(P1), 1-np.mean(alg1_k1[:,:P1] <= 1, axis=0), linestyle = 'solid', color = 'orangered')
plt.plot(range(P2), 1-np.mean(alg2_k1[:,:P2] <= 1, axis=0), linestyle = 'dashdot', color = 'darkgreen')
plt.plot(range(P3), 1-np.mean(alg3_k1[:,:P3] <= 1, axis=0), linestyle = 'dotted', color = 'limegreen')
plt.plot(range(P4), 1-np.mean(alg4_k1[:,:P4] <= 1, axis=0), linestyle='dashed', color = 'darkblue')

#tikzplotlib.save("G11_Part1_Jan26.tex")


plt.figure(1)

plt.plot(range(P1), 1-np.mean(alg1_k1[:,P1:2*P1] <= 1, axis=0), linestyle = 'solid', color = 'orangered')
plt.plot(range(P2), 1-np.mean(alg2_k1[:,P2:2*P2] <= 1, axis=0), linestyle = 'dashdot', color = 'darkgreen')
plt.plot(range(P3), 1-np.mean(alg3_k1[:,P3:2*P3] <= 1, axis=0), linestyle = 'dotted', color = 'limegreen')
plt.plot(range(P4), 1-np.mean(alg4_k1[:,P4:2*P4] <= 1, axis=0), linestyle='dashed', color = 'darkblue')

#tikzplotlib.save("G11_Part2_Jan26.tex")


plt.figure(2)

plt.plot(range(P1), 1-np.mean(alg1_k1[:,2*P1:3*P1] <= 1, axis=0), linestyle = 'solid', color = 'orangered')
plt.plot(range(P2), 1-np.mean(alg2_k1[:,2*P2:3*P2] <= 1, axis=0), linestyle = 'dashdot', color = 'darkgreen')
plt.plot(range(P3), 1-np.mean(alg3_k1[:,2*P3:3*P3] <= 1, axis=0), linestyle = 'dotted', color = 'limegreen')
plt.plot(range(P4), 1-np.mean(alg4_k1[:,2*P4:3*P4] <= 1, axis=0), linestyle='dashed', color = 'darkblue')

#tikzplotlib.save("G11_Part3_Jan26.tex")


plt.figure(3)

plt.plot(range(P1), 1-np.mean(alg1_k1[:,3*P1:] <= 1, axis=0), linestyle = 'solid', color = 'orangered')
plt.plot(range(P2), 1-np.mean(alg2_k1[:,3*P2:] <= 1, axis=0), linestyle = 'dashdot', color = 'darkgreen')
plt.plot(range(P3), 1-np.mean(alg3_k1[:,3*P3:] <= 1, axis=0), linestyle = 'dotted', color = 'limegreen')
plt.plot(range(P4), 1-np.mean(alg4_k1[:,3*P4:] <= 1, axis=0), linestyle='dashed', color = 'darkblue')

#tikzplotlib.save("G11_Part4_Jan26.tex")


### k = 2, j = 3
plt.figure(4)

plt.plot(range(P1), 1-np.mean(alg1_k2[:,:P1] <= 3, axis=0), linestyle = 'solid', color = 'orangered')
plt.plot(range(P2), 1-np.mean(alg2_k2[:,:P2] <= 3, axis=0), linestyle = 'dashdot', color = 'darkgreen')
plt.plot(range(P3), 1-np.mean(alg3_k2[:,:P3] <= 3, axis=0), linestyle = 'dotted', color = 'limegreen')
plt.plot(range(P4), 1-np.mean(alg4_k2[:,:P4] <= 3, axis=0), linestyle='dashed', color = 'darkblue')

#tikzplotlib.save("G21_Part1_Jan26.tex")


plt.figure(5)

plt.plot(range(P1), 1-np.mean(alg1_k2[:,P1:2*P1] <= 3, axis=0), linestyle = 'solid', color = 'orangered')
plt.plot(range(P2), 1-np.mean(alg2_k2[:,P2:2*P2] <= 3, axis=0), linestyle = 'dashdot', color = 'darkgreen')
plt.plot(range(P3), 1-np.mean(alg3_k2[:,P3:2*P3] <= 3, axis=0), linestyle = 'dotted', color = 'limegreen')
plt.plot(range(P4), 1-np.mean(alg4_k2[:,P4:2*P4] <= 3, axis=0), linestyle='dashed', color = 'darkblue')

#tikzplotlib.save("G21_Part2_Jan26.tex")


plt.figure(6)

plt.plot(range(P1), 1-np.mean(alg1_k2[:,2*P1:3*P1] <= 3, axis=0), linestyle = 'solid', color = 'orangered')
plt.plot(range(P2), 1-np.mean(alg2_k2[:,2*P2:3*P2] <= 3, axis=0), linestyle = 'dashdot', color = 'darkgreen')
plt.plot(range(P3), 1-np.mean(alg3_k2[:,2*P3:3*P3] <= 3, axis=0), linestyle = 'dotted', color = 'limegreen')
plt.plot(range(P4), 1-np.mean(alg4_k2[:,2*P4:3*P4] <= 3, axis=0), linestyle='dashed', color = 'darkblue')

#tikzplotlib.save("G21_Part3_Jan26.tex")


plt.figure(7)

plt.plot(range(P1), 1-np.mean(alg1_k2[:,3*P1:] <= 3, axis=0), linestyle = 'solid', color = 'orangered')
plt.plot(range(P2), 1-np.mean(alg2_k2[:,3*P2:] <= 3, axis=0), linestyle = 'dashdot', color = 'darkgreen')
plt.plot(range(P3), 1-np.mean(alg3_k2[:,3*P3:] <= 3, axis=0), linestyle = 'dotted', color = 'limegreen')
plt.plot(range(P4), 1-np.mean(alg4_k2[:,3*P4:] <= 3, axis=0), linestyle='dashed', color = 'darkblue')

#tikzplotlib.save("G21_Part4_Jan26.tex")
