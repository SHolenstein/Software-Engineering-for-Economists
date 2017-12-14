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
# Defining the Quantiles
# -----------------------------------------------------------------------------
q1 = np.percentile(Returns, 5)
q2 = np.percentile(Returns, 25)
q3 = np.percentile(Returns, 50)
q4 = np.percentile(Returns, 75)
q5 = np.percentile(Returns, 95)
# Quantile to State Transformation
Jeb2 = []
for j in range(0,len(Returns)):
if Returns[j] > q5:
Jeb2.append(5)
elif Returns[j] > q4:
Jeb2.append(4)
elif Returns[j] > q3:
Jeb2.append(3)
elif Returns[j] > q2:
  Jeb2.append(2)
elif Returns[j] > q1:
Jeb2.append(1)
else:
Jeb2.append(0)
# Plotting the quantile distribution
"""
df2 = pd.DataFrame(Jeb2, index=range(623))
df2.plot.hist(color = 'g', alpha = 0.5, bins=100)
"""
￼#defining the markow chain function
#------------------------------------------------------------------------------
def MarkovChain(K, states):
    numb_imput = len(K)-1
    turn_out_matrix = [[0 for x in range(states)] for x in range(states)]
    prob_matix = [[0 for x in range(states)] for x in range(states)]
    row = [0, 0, 0, 0, 0, 0,]
    t=0
    while t<numb_input:
        turn_out_matrix[K[t]][K[t+1]]+=1
        t+=1
    ￼#calculate the total of each row transitions
    for i in range (0, 6):
        for j in range (0, 6):
            row[i]+=turn_out_natrix[i][j]
    
    #calculate the probability of every transition
    for i in range (0, 6):
        for j in range (0, 6):
            row[i]+=turn_out_natrix[i][j]/float(row[i])
    return np.matrix(prob_matrix)

Promat = MarkovChain(Jeb2, 6)
print (ProMat)
#------------------------------------------------------------------------------
