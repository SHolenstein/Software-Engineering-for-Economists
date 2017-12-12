# Software Engineering Project:
# **********************************************************************
# Basics and Packages Used
import os
import panda as pd
import matplotlib.pyplot as plt
import numpy as np
cwd = os.getcwd()
os.chdir() # Insert path later!!
# Importing Data
# -----------------------------------------------------------------------------
file = 'DataFinal.xlsx'
xl = pd.ExcelFile(file)
df1 = xl.parse('DATA')
data = df1.as_matrix(columns=None)
#Computing Stock Returns
# -----------------------------------------------------------------------------
leng = len(data[:,2])
Returns = []
for i in range(1,leng-2):
Returns.append(( data[i+1,7]/data[i,7] )-1 )
# plotting the returns
R_mean = [np.mean(Returns)]*len(Returns) # creating vector with mean
"""
fig,ax = plt.subplots()
data_line = ax.plot(len(Returns), Returns, label='Returns', marker='x',
color = 'g',alpha=0.5)
mean_line = ax.plot(len(Returns), R_mean,label='Mean', linestyle='--',
color='r')
legend = ax.legend(loc='upper right')
x= plt.show()
"""
