# Software Engineering Project:
# **********************************************************************
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
df2 = xl.parse('King')
df3 = xl.parse('Weggli')
df4 = xl.parse('Number3')
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



# the Graph function visualizes the Tradings strategy. It  calls almost all of the above functions. That is:
    # Prob_High... (inlk. State_Transformation, Return_of_Matrix, Markov_Chain)
    # Prob_Low... (inkl. State_Transformation, Return_of_Matrix, Markov_Chain)
def Graph(D, inp):
    zerofive = []
    for i in range(1,11):
        zerofive.append(0.5)
    fig,ax = plt.subplots()
    ax.plot(range(1,11), Prob_High_S_Tplus1(D, inp), label='Prob[S_{t+1} in {6,5,4} | S_{t} = 1]',
                         marker='o', color = 'g', alpha=0.8)
    ax.plot(range(1,11), Prob_Low_S_Tplus1(D, inp), label='Prob[S_{t+1} in {1,2,3} | S_{t} = 6]',
                        marker='o', color = 'r', alpha=0.8)
    ax.plot(range(1,11), zerofive, label='50% Probability', color='grey',alpha = 0.5)
    ax.legend(loc='upper right')
    plt.xlabel('Quantiles in %')
    plt.ylabel('Probabilty')
    fff = plt.show()
    return fff

def count_critical_statesH(x):      # How many critical States for "Low"?
    sum = 0
    CRIT = Crit_Title_High(x);
    for i in range(0,len(data[1,:])):
        if CRIT[i] == 1:
            sum = sum + 1
        else:
            sum = sum
    return sum

def count_critical_statesL(x):      # How many critical States for "High"?
    sum = 0
    CRIT = Crit_Title_Low(x);
    for i in range(0,len(data[1,:])):
        if CRIT[i] == 1:
            sum = sum + 1
        else:
            sum = sum
    return sum

# A new interactive part has to be coded!
#END
