# Import the functions and packages that are used
from dwave.system import EmbeddingComposite, DWaveSampler
from dimod import BinaryQuadraticModel #ConstrainedQuadraticModel
from dimod.reference.samplers import ExactSolver
import dimod
import neal
import math
import pandas as pd
import numpy as np
from numpy import sum,isrealobj,sqrt
from numpy.random import standard_normal
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)
pd.set_option('display.max_rows',None)

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)
def Hamming(temp, ec):
    c=0
    for i in range(len(temp)):
        if temp[i]!=ec[i]:
            c+=1
    return c
# Trying example from https://machinelearningmastery.com/method-of-lagrange-multipliers-the-theory-behind-support-vector-machines-part-1-the-separable-case/
X_train, X_test, y_train, y_test = np.load('./data.npy', allow_pickle=True)
# Loading data for SVM Classification
x = X_train
#print("Shape of x is {}".format(x.shape))
n = len(x)
y = y_train
#print("Shape of y is {}".format(y.shape))
#x = np.load()
#y = np.load()
ll = -2
ul = 1
# Define Precision Vector P
p = [math.pow(2,i) for i in range(ll,ul,1)] 
p = np.array(p).reshape(ul-ll,1)
#print("Shape of p is {}".format(p))
I = np.eye(n)
P = np.kron(I,np.transpose(p))
#print("Shape of P is {}".format(P.shape))

# Constructing quadratic part of QUBO matrix
X = x@np.transpose(x)
#print("Shape of X is {}".format(X))
Y = y@np.transpose(y)
#print("Shape of Y is {}".format(Y))
A = 0.5*(np.transpose(P)@np.multiply(X,Y)@P)
#print("Shape of A is {}".format(A.shape))
# Constructing linear part of QUBO matrix
B = -np.transpose(P)@np.ones((n,1))
#print("Shape of B is {}".format(B.shape))
# Final QUBO matrix
Q = A
for i in range(len(Q)):
    Q[i][i]=B[i]
#print("Shape of Q is {}".format(Q.shape))

var = check_symmetric(Q)
if var:
    print("Array symmetric")

# Constructing BQM 
bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(Q)
#print("Linear terms are {}".format(bqm.linear))
#print("Quadratic terms are {}".format(bqm.quadratic))   

# Define the sampler that will be used to run the problem
#sampler = ExactSolver()
sampler = neal.SimulatedAnnealingSampler()
#sampler = EmbeddingComposite(DWaveSampler())

# Run the problem on the sampler and print the results
#sampleset = sampler.sample(bqm)
num_reads = 20
sampleset = sampler.sample(bqm,num_reads=num_reads)

frame = sampleset.to_pandas_dataframe()
s_frame = frame.sort_values(by='energy')
   
#print(s_frame)
   
# Analyzing lowest energy solution
temp = list(s_frame.iloc[0])
temp = temp[0:n*(ul-ll)]        #Lowest energy array
temp = np.array(temp)
temp = temp.reshape(n,ul-ll)
# Reconstructing the lambdas
lagrange = []
for i in range(n):
    lagrange.append(np.sum(np.multiply(temp[i].reshape(ul-ll,1),p)))
#print(lagrange)  

def compute_sigma(test_point):
    alpha = 0
    for i in range(n):
        alpha += lagrange[i]*y[i]*np.transpose(test_point)@x[i]
    return alpha

# The lagrange values obtained after annealing don't match the actual values given in the example website given above
# After above is fixed, I will implement the code to find the weights and bias corresponding to the hyperplane 

# Computing bias w0
w_all = []
for i in range(len(lagrange)):
    if(lagrange[i]>0):
        t = compute_sigma(x[i])
        if(y[i]>0):
            w_all.append(1-t)
        else:
            w_all.append(-1-t)    
w0 = np.sum(w_all)/len(w_all)
#print(w0)

# Testing for arbitrary points pt
pt = X_test
actual = []
predictions = []
for i in pt:
    t = compute_sigma(i)
    predictions.append(1 if(t+w0)>0 else 0)

# for i in range(len(predictions)):
#     if predictions[i]>0:
#         print("{} belongs to class {}".format(pt[i],1))
#     else:
#         print("{} belongs to class {}".format(pt[i],0))

error = Hamming(predictions,y_test)
print("Total no of errors is : {}".format(error))
       