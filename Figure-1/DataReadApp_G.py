# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 16:16:31 2020

@author: brend
"""
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib


alg1_k1 = np.loadtxt('alg1data_k1_L_MC=1000.csv')
alg1_k2 = np.loadtxt('alg1data_k2_L_MC=1000.csv')
alg1_k3 = np.loadtxt('alg1data_k3_L_MC=1000.csv')
alg1_k4 = np.loadtxt('alg1data_k4_L_MC=1000.csv')
alg1_k5 = np.loadtxt('alg1data_k5_L_MC=1000.csv')


alg2_k1 = np.loadtxt('alg2data_k1_L_MC=100.csv')
alg2_k2 = np.loadtxt('alg2data_k2_L_MC=100.csv')
alg2_k3 = np.loadtxt('alg2data_k3_L_MC=100.csv')
alg2_k4 = np.loadtxt('alg2data_k4_L_MC=100.csv')
alg2_k5 = np.loadtxt('alg2data_k5_L_MC=100.csv')

alg3_k1 = np.loadtxt('alg3data_k1_L_MC=100.csv')
alg3_k2 = np.loadtxt('alg3data_k2_L_MC=100.csv')
alg3_k3 = np.loadtxt('alg3data_k3_L_MC=100.csv')
alg3_k4 = np.loadtxt('alg3data_k4_L_MC=100.csv')
alg3_k5 = np.loadtxt('alg3data_k5_L_MC=100.csv')

alg4_k1 = np.loadtxt('alg4data_k1_L_MC=100_P=10.csv')
alg4_k2 = np.loadtxt('alg4data_k2_L_MC=100_P=10.csv')
alg4_k3 = np.loadtxt('alg4data_k3_L_MC=100_P=10.csv')
alg4_k4 = np.loadtxt('alg4data_k4_L_MC=100_P=10.csv')
alg4_k5 = np.loadtxt('alg4data_k5_L_MC=100_P=10.csv')

alg4plus_k1 = np.loadtxt('alg4plusdata_k1_L_MC=100_P=10.csv')
alg4plus_k2 = np.loadtxt('alg4plusdata_k2_L_MC=100_P=10.csv')
alg4plus_k3 = np.loadtxt('alg4plusdata_k3_L_MC=100_P=10.csv')
alg4plus_k4 = np.loadtxt('alg4plusdata_k4_L_MC=100_P=10.csv')
alg4plus_k5 = np.loadtxt('alg4plusdata_k5_L_MC=100_P=10.csv')


P = 10


### k = 1, j = 1

plt.figure(0)

plt.plot(range(P), 1-np.mean(alg1_k1[:,:P] <= 1, axis=0), linestyle = 'solid', color = 'orangered')
plt.plot(range(P), 1-np.mean(alg2_k1[:,:P] <= 1, axis=0), linestyle = 'dashdot', color = 'darkgreen')
plt.plot(range(P), 1-np.mean(alg3_k1[:,:P] <= 1, axis=0), linestyle = 'dotted', color = 'limegreen')
plt.plot(range(P), 1-np.mean(alg4_k1[:,:P] <= 1, axis=0), linestyle='dashed', color = 'darkblue')
plt.plot(range(P), 1-np.mean(alg4plus_k1[:,:P] <= 1, axis=0), linestyle=(0,(5,10)), color = 'royalblue')

tikzplotlib.save("L11_Part1_Jan20.tex")


plt.figure(1)

plt.plot(range(P), 1-np.mean(alg1_k1[:,P:2*P] <= 1, axis=0), linestyle = 'solid', color = 'orangered')
plt.plot(range(P), 1-np.mean(alg2_k1[:,P:2*P] <= 1, axis=0), linestyle = 'dashdot', color = 'darkgreen')
plt.plot(range(P), 1-np.mean(alg3_k1[:,P:2*P] <= 1, axis=0), linestyle = 'dotted', color = 'limegreen')
plt.plot(range(P), 1-np.mean(alg4_k1[:,P:2*P] <= 1, axis=0), linestyle='dashed', color = 'darkblue')
plt.plot(range(P), 1-np.mean(alg4plus_k1[:,P:2*P] <= 1, axis=0), linestyle=(0,(5,10)), color = 'royalblue')

tikzplotlib.save("L11_Part2_Jan20.tex")


plt.figure(2)

plt.plot(range(P), 1-np.mean(alg1_k1[:,2*P:3*P] <= 1, axis=0), linestyle = 'solid', color = 'orangered')
plt.plot(range(P), 1-np.mean(alg2_k1[:,2*P:3*P] <= 1, axis=0), linestyle = 'dashdot', color = 'darkgreen')
plt.plot(range(P), 1-np.mean(alg3_k1[:,2*P:3*P] <= 1, axis=0), linestyle = 'dotted', color = 'limegreen')
plt.plot(range(P), 1-np.mean(alg4_k1[:,2*P:3*P] <= 1, axis=0), linestyle='dashed', color = 'darkblue')
plt.plot(range(P), 1-np.mean(alg4plus_k1[:,2*P:3*P] <= 1, axis=0), linestyle=(0,(5,10)), color = 'royalblue')

tikzplotlib.save("L11_Part3_Jan20.tex")


plt.figure(3)

plt.plot(range(P), 1-np.mean(alg1_k1[:,3*P:] <= 1, axis=0), linestyle = 'solid', color = 'orangered')
plt.plot(range(P), 1-np.mean(alg2_k1[:,3*P:] <= 1, axis=0), linestyle = 'dashdot', color = 'darkgreen')
plt.plot(range(P), 1-np.mean(alg3_k1[:,3*P:] <= 1, axis=0), linestyle = 'dotted', color = 'limegreen')
plt.plot(range(P), 1-np.mean(alg4_k1[:,3*P:] <= 1, axis=0), linestyle='dashed', color = 'darkblue')
plt.plot(range(P), 1-np.mean(alg4plus_k1[:,3*P:] <= 1, axis=0), linestyle=(0,(5,10)), color = 'royalblue')

tikzplotlib.save("L11_Part4_Jan20.tex")


### k = 2, j = 3
plt.figure(4)

plt.plot(range(P), 1-np.mean(alg1_k2[:,:P] <= 3, axis=0), linestyle = 'solid', color = 'orangered')
plt.plot(range(P), 1-np.mean(alg2_k2[:,:P] <= 3, axis=0), linestyle = 'dashdot', color = 'darkgreen')
plt.plot(range(P), 1-np.mean(alg3_k2[:,:P] <= 3, axis=0), linestyle = 'dotted', color = 'limegreen')
plt.plot(range(P), 1-np.mean(alg4_k2[:,:P] <= 3, axis=0), linestyle='dashed', color = 'darkblue')
plt.plot(range(P), 1-np.mean(alg4plus_k2[:,:P] <= 3, axis=0), linestyle=(0,(5,10)), color = 'royalblue')

tikzplotlib.save("L21_Part1_Jan20.tex")


plt.figure(5)

plt.plot(range(P), 1-np.mean(alg1_k2[:,P:2*P] <= 3, axis=0), linestyle = 'solid', color = 'orangered')
plt.plot(range(P), 1-np.mean(alg2_k2[:,P:2*P] <= 3, axis=0), linestyle = 'dashdot', color = 'darkgreen')
plt.plot(range(P), 1-np.mean(alg3_k2[:,P:2*P] <= 3, axis=0), linestyle = 'dotted', color = 'limegreen')
plt.plot(range(P), 1-np.mean(alg4_k2[:,P:2*P] <= 3, axis=0), linestyle='dashed', color = 'darkblue')
plt.plot(range(P), 1-np.mean(alg4plus_k2[:,P:2*P] <= 3, axis=0), linestyle=(0,(5,10)), color = 'royalblue')

tikzplotlib.save("L21_Part2_Jan20.tex")


plt.figure(6)

plt.plot(range(P), 1-np.mean(alg1_k2[:,2*P:3*P] <= 3, axis=0), linestyle = 'solid', color = 'orangered')
plt.plot(range(P), 1-np.mean(alg2_k2[:,2*P:3*P] <= 3, axis=0), linestyle = 'dashdot', color = 'darkgreen')
plt.plot(range(P), 1-np.mean(alg3_k2[:,2*P:3*P] <= 3, axis=0), linestyle = 'dotted', color = 'limegreen')
plt.plot(range(P), 1-np.mean(alg4_k2[:,2*P:3*P] <= 3, axis=0), linestyle='dashed', color = 'darkblue')
plt.plot(range(P), 1-np.mean(alg4plus_k2[:,2*P:3*P] <= 3, axis=0), linestyle=(0,(5,10)), color = 'royalblue')

tikzplotlib.save("L21_Part3_Jan20.tex")


plt.figure(7)

plt.plot(range(P), 1-np.mean(alg1_k2[:,3*P:] <= 3, axis=0), linestyle = 'solid', color = 'orangered')
plt.plot(range(P), 1-np.mean(alg2_k2[:,3*P:] <= 3, axis=0), linestyle = 'dashdot', color = 'darkgreen')
plt.plot(range(P), 1-np.mean(alg3_k2[:,3*P:] <= 3, axis=0), linestyle = 'dotted', color = 'limegreen')
plt.plot(range(P), 1-np.mean(alg4_k2[:,3*P:] <= 3, axis=0), linestyle='dashed', color = 'darkblue')
plt.plot(range(P), 1-np.mean(alg4plus_k2[:,3*P:] <= 3, axis=0), linestyle=(0,(5,10)), color = 'royalblue')

tikzplotlib.save("L21_Part4_Jan20.tex")
