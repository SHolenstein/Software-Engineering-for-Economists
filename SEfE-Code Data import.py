# Software Engineering Project:
# **********************************************************************
# In the beginning of the code we should do an interactive part: For instance,
# (1) Explain the procedure of Data import
# (2) Explain What the code does
# (3) ...
# Basics and Packages Used
# -----------------------------------------------------------------------------
import os
import panda as pd
import matplotlib.pyplot as plt
import numpy as np
cwd = os.getcwd()
os.chdir(" ") # Insert path later!! # How are we going to do that? Is there an alternative approach?
# Importing Data
# -----------------------------------------------------------------------------
file = 'DataFinal.xlsx'
xl = pd.ExcelFile(file)
df1 = xl.parse('DATA')
data = df1.as_matrix(columns=None)

#------------------------------------------------------------------------------
# Collection of Functions Used:
#------------------------------------------------------------------------------
def Return_of_Asset(x):             # Return of Asset
    Returns = []
    for i in range(0,len(x)-1):
        Returns.append((x[i+1]/x[i])-1)
    return Returns

def Return_of_Matrix(x):            # Data Matrix Return
    Return_Matrix = []
    Return = []
    for i in range(0, len(x[1,:])):
        Return_Matrix = np.c_[Return_of_Asset(x[:,i])]
        Return.append(Return_Matrix)
    return Return


def MarkovChain(K, states):         # Markov Chain
    numb_inputs = len(K)-1
    turn_out_matrix = [[0 for x in range(states)] for x in range(states)]
    prob_matrix = [[0 for x in range(states)] for x in range(states)]
    row = [0, 0, 0, 0, 0, 0]
    t=0
    while t<numb_inputs:
        turn_out_matrix[K[t]][K[t+1]]+=1
        t+=1

    for i in range(0, 6):           # calculate the total of each row transistions
        for j in range(0, 6):
            row[i]+=turn_out_matrix[i][j]

    for i in range(0, 6):           # calculate the probability of every transition
        for j in range(0, 6):
            prob_matrix[i][j] = turn_out_matrix[i][j]/float(row[i])
    return np.matrix(prob_matrix)

def State_Transformation(x, quant1):    # This function does transform the financial data into the state vector.
    q1= np.percentile(x,quant1)         # this is later used for the markov chain
    q2= np.percentile(x,25)
    q3= np.percentile(x,50)
    q4= np.percentile(x,75)
    q5= np.percentile(x,100-quant1)
    Jeb = []                          
    for j in range(0,len(x)):
        if x[j] > q5:
            Jeb.append(5)
        elif x[j] > q4:
            Jeb.append(4)
        elif x[j] > q3:
            Jeb.append(3)
        elif x[j] > q2:
            Jeb.append(2)
        elif x[j] > q1:
            Jeb.append(1)
        else:
            Jeb.append(0)
    return Jeb

def Prob_High_S_Tplus1(D, inp):     # Prob[S_{t+1} in {6,5,4} | S_{t} = 1]
    K = Return_of_Matrix(D)
    x = K[inp]
    TRX = []
    for i in range(1,11):
        Jeb = State_Transformation(x,i)
        F = MarkovChain(Jeb,6)
        l = 0
        for k in range(3,6):
            l = l + F[0,k]
        TRX.append(l)
    return TRX

def Prob_Low_S_Tplus1(D, inp):      # Prob[S_{t+1} in {1,2,3} | S_{t} = 6]
    K = Return_of_Matrix(D)
    x = K[inp]
    TRX = []
    for i in range(1,11):
        Jeb = State_Transformation(x,i)
        F = MarkovChain(Jeb,6)
        n = 0
        for r in range(0,3):
            n = n + F[5,r]
        TRX.append(n)
    return TRX

def Crit_Title_Low(x):                          # Critical State for "LOW"?
    K = Return_of_Matrix(x)
    Low = []
    for i in range(0,len(data[1,:])):
        if K[i][-1] < np.nanpercentile(K, 5):   # quantile has to be defined here
            Low.append(1)
        else:
            Low.append(0)
    return Low

def Crit_Title_High(x):              # Is there a critical state "High" 
    K = Return_of_Matrix(x)
    High = []
    for i in range(0,len(data[1,:])):
        if K[i][-1] > np.nanpercentile(K, 90):  # quantile has to be definded
            High.append(1)
        else:
            High.append(0)
return High


#------------------------------------------------------------------------------
def MarkovChain(K, states):     # function which couts the differnt states
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
# Functions that show the probabilities that the return changes from one extreme to the other extreme! 
# We have to do functions, in order that we can call them later!
# (1) FOR Prob[S_{t+1}= {5,4,3} | S_{t} = 1] AND Prob[S_{t+1}= {1,2,3} | S_{t} = 5]
sum_s1 = 0
for i in range(3,6):
    sum_s1 = sum_s1 + ProMat[0, i]
print(sum_s1)       # computes Prob[S_{t+1}= {5,4,3} | S_{t} = 1]
sum_s5 = 0
for i in range(0,3):
    sum_s5 = sum_s5 + ProMat[5, i]
print(sum_s5)       # computes Prob[S_{t+1}= {1,2,3} | S_{t} = 5]
# (2) FOR Prob[S_{t+1}= {5,4,3} | S_{t} = 2] AND Prob[S_{t+1}= {1,2,3} | S_{t} = 4]
sum_s2 = 0
for i in range(3,6):
    sum_s2 = sum_s2 + ProMat[1, i]
print(sum_s2)   # Prob[S_{t+1}= {5,4,3} | S_{t} = 2]
sum_s4 = 0
for i in range(0,3):
    sum_s4 = sum_s4 + ProMat[4, i]
print(sum_s4)       # Prob[S_{t+1}= {1,2,3} | S_{t} = 4]
# do a function for that procedure!!! So that you can call it later on!
TRX = []
TRX2 = []
for i in range(1,11):
    q1= np.percentile(Returns,i)
    q2= np.percentile(Returns,25)
    q3= np.percentile(Returns,50)
    q4= np.percentile(Returns,75)
    q5= np.percentile(Returns,(100-i))
    Jeb = []                          # "Jeb" is going to be the state vector
    for j in range(0,len(Returns)):
        if Returns[j] > q5:
            Jeb.append(5)
        elif Returns[j] > q4:
            Jeb.append(4)
        elif Returns[j] > q3:
            Jeb.append(3)
        elif Returns[j] > q2:
            Jeb.append(2)
        elif Returns[j] > q1:
            Jeb.append(1)
        else:
            Jeb.append(0)
    F = MarkovChain(Jeb,6)
    l = 0
    for k in range(3,6):
        l = l + F[0,k]
    n = 0
    for r in range(0,3):
        n = n + F[5,r]
    TRX2.append(n)
    TRX.append(l)

print('------- TRX -------')
print(TRX)              # Prob[S_{t+1} = {5,4,3} | S_{t} = 1] for different
                        # quantiles
print('------ TRX2 -------')
print(TRX2)             # Prob[S_{t+1} = {1,2,3} | S_{t} = 6] for different
                        # quantiles
zerofive = []
for i in range(1,11):
    zerofive.append(0.5)
fig,ax = plt.subplots()
quantiless = ax.plot(range(1,11), TRX, label='Prob[S_{t+1} in {6,5,4} | S_{t} = 1]',
                     marker='o', color = 'g', alpha=0.8)
quantiles = ax.plot(range(1,11), TRX2, label='Prob[S_{t+1} in {1,2,3} | S_{t} = 6]',
                    marker='o', color = 'r', alpha=0.8)
pointfive = ax.plot(range(1,11), zerofive, label='50% line', color='grey',alpha = 0.5)
legend = ax.legend(loc='upper right')
plt.xlabel('Quantiles in %')
plt.ylabel('Probabilty')
fig.savefig('Quantiles.jpg')
fff = plt.show()  # indicates the probability that the returns in {t+1} are going to
            # be in the oposite quantile, when t was in an extreme quantile


