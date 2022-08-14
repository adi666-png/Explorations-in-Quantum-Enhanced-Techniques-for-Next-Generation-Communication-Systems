# Import the functions and packages that are used
from dwave.system import EmbeddingComposite, DWaveSampler
from dimod import BinaryQuadraticModel #ConstrainedQuadraticModel
from dimod.reference.samplers import ExactSolver
import neal
import math
import pandas as pd
import numpy as np
import scipy.integrate
from numpy import sum,isrealobj,sqrt
from numpy.random import standard_normal, normal
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)
pd.set_option('display.max_rows',None)

# Setup the problem
n = 32
k = 16
R = (n-k)/n
W1 = 5.5
W2 = 1
#SNR = [i for i in range(4,11,1)]
SNR=[8]
BER_batch = 10**1
#beta=0.5

def awgn(s,SNRdB,L=1):
    gamma = 10**(SNRdB/10) #SNR to linear scale
    if s.ndim==1:# if s is single dimensional vector
        P=L*sum(abs(s)**2)/len(s) #Actual power in the vector
    else: # multi-dimensional signals like MFSK
        P=L*sum(sum(abs(s)**2))/len(s) # if s is a matrix [MxN]
    N0=P/gamma # Find the noise spectral density
    if isrealobj(s):# check if input is real/complex object type
        n = sqrt(N0/2)*standard_normal(s.shape) # computed noise
    else:
        n = sqrt(N0/2)*(standard_normal(s.shape)+1j*standard_normal(s.shape))
    r = s + n # received signal
    return math.sqrt(N0/2),r

def generate_codeword(generator, message):
    final_codeword = np.dot(message,generator)
    final_codeword = [i%2 for i in final_codeword]
    return final_codeword

def prob(z, sigma):
    return 1/(1+math.exp(-2*z/(sigma**2)))

def rayleigh(z,sigma):
    sigma, w, y = sigma,1,z
    hlim = np.inf
    llim = 0
    p1 = scipy.integrate.quad(integrand, llim, hlim, args=(sigma, w, y))
    w = -1
    p0 = scipy.integrate.quad(integrand, llim, hlim, args=(sigma, w, y))
    return 1/((p0[0]/p1[0])+1) 

def rayleigh_channel(s,SNRdB, L=1):
    gamma = 10**(SNRdB/10)

    if s.ndim==1:# if s is single dimensional vector
        P=L*sum(abs(s)**2)/len(s) #Actual power in the vector

    else: # multi-dimensional signals like MFSK
        P=L*sum(sum(abs(s)**2))/len(s) # if s is a matrix [MxN]

    N0=P/(2*gamma) # Find the noise spectral density
    w = normal(0, N0/2, s.shape)+1j*normal(0,N0/2, s.shape)
    h = normal(0, 1/2, s.shape)+1j*normal(0,1/2, s.shape)
    y,r= [],[]
    for i in range(len(s)):
        y.append(np.multiply(h[i],s[i])+w[i])    
    # r = np.multiply(h,s) + w # received signal
    for i in range(len(y)):
        r.append((np.conj(h[i]/math.sqrt(h[i].real**2+h[i].imag**2))*y[i]).real)
        # r0.append(math.sqrt(h[i].real**2+h[i].imag**2))
    r=np.array(r)
    # print("r: "+str(r))
    # print("w: "+str(w))
    # print("h: "+str(h))
    # print("y: "+str(y))
    return math.sqrt(N0/2),r

def integrand(x, sigma, w, y):
    return (1/math.sqrt(2*math.pi*sigma**2))*math.e**(-((y-w*x)**2)/(2*sigma**2))*2*x*math.e**(-x**2)

def rayleigh_llr(z,sigma):    
    sigma_tilde = sigma*sigma*(1+2*sigma*sigma)
    a0 = 0
    a1 = math.sqrt(2*math.pi/sigma_tilde)
    a2 = 0
    a3 = -math.sqrt(math.pi/2)*((15-30*math.pi+8*math.pi*math.pi)/(30*(-3+math.pi)*sigma_tilde*math.sqrt(sigma_tilde)))
    b0 = 1
    b1 = 0
    b2 = ((-35+30*math.pi-6*math.pi*math.pi)/(20*(-3+math.pi)*sigma_tilde))

    llr = (a0+a1*z+a2*z*z+a3*z*z*z)/(b0+b1*z+b2*z*z)

    #return (1/(1/math.pow(math.e,llr))+1)
    return llr


def dist_noise(temp, noise):
    d=0
    temp, noise = np.array(temp).reshape(n,1), np.transpose(np.array(noise))
    #print(temp.shape, noise.shape)
    for i in range(len(temp)):
        value1=temp[i]-noise[i]
        d+=math.pow(value1,2)
    return math.sqrt(d)

def Hamming(temp, ec):
    c=0
    for i in range(len(temp)):
        if temp[i]!=ec[i]:
            c+=1
    return c

BER_list = []
FER_list=[]
frac_list=[]
for ind_1 in range(len(SNR)):
    
    # variance = 10**(-SNR[ind_1]/10)
    # sigma = math.sqrt(variance)
    with open('parity.txt', 'r') as f:
        parity_check = [[int(num) for num in line.split(',')] for line in f]

    with open('generator.txt', 'r') as f:
        generator = [[int(num) for num in line.split(',')] for line in f]
    BER=0
    FER=0
    frac=0
    for ind_2 in range(BER_batch):
        x = [i+1000 for i in range(n)]
        message = [np.random.randint(0,2) for i in range(n-k)]
        #scale = np.random.exponential(scale = beta, size=(1,n)).flatten()
        #scale = [math.sqrt(i) for i in scale]
        #message = np.array(message).reshape(n-k,1)
        encoded_message = generate_codeword(generator, message)
        #print("Encoded Message is {}".format(encoded_message))
        def verify(z):
            H = np.array(parity_check)
            z = np.array(z)
            prod=np.dot(H,np.transpose(z))
            prod=np.array([i%2 for i in prod])
            #print(prod)
            return np.array_equal(prod,np.zeros(k))
    
        #print("Encoded message is {}".format(encoded_message))
        bpsk_encoded = np.array([2*t-1 for t in encoded_message])
        #print("BPSK message is {}".format(bpsk_encoded))
        #noise = math.sqrt(variance)*np.random.randn(1,n)
        #print("Noise Vector is {}".format(noise))
        #received = bpsk_encoded + noise
        sigma, received = rayleigh_channel(bpsk_encoded, SNR[ind_1])
        #print(bpsk_encoded, received)
        received = received.flatten()
        #print("Received signal is {}".format(received))

        intrinsic = [rayleigh(i, sigma) for i in received] # to be discussed
        # intrinsic = [rayleigh_llr(i, sigma) for i in received] # to be discussed
        
        print(np.array(intrinsic))
        exit()
        #print("Intrinsic signal is {}".format(intrinsic))
        # Define QUBO variables
        codeword = [f'codeword_{i}' for i in x]
        # Initialize BQM
        bqm = BinaryQuadraticModel('BINARY')
        # Objective
        for i in (x):
            bqm.add_variable(codeword[i-1000],W1*(1-2*intrinsic[i-1000]))
        #qubo = bqm.to_qubo()
        #print("After adding objective QUBO looks like: {}".format(qubo))
        anc_count = 0    
        # Parity Check Constraint
        for i in range(k):
            cnt = 0
            eligible_codewords = []
            for j in range(n):
                if(parity_check[i][j]==1):
                    bqm.add_linear(codeword[j],1)
                    eligible_codewords.append(codeword[j])
                    cnt = cnt + 1
            #print("Eligible codewords look like: {}".format(encoded_message))
            #qubo = bqm.to_qubo()
            #print("After adding first constraint QUBO looks like: {}".format(qubo))
            num_anc = math.floor(math.log2(math.floor(cnt/2)))+1   
            for _ in range(num_anc+anc_count,anc_count,-1):
                bqm.add_variable('ancillary_{}'.format(_),W2*math.pow((math.pow(2,_-anc_count)),2))
            # Add quadratic terms
            for a in eligible_codewords:
                for b in range(num_anc+anc_count,anc_count,-1):
                    bqm.add_quadratic(a,'ancillary_{}'.format(b),-2*W2*(math.pow(2,b-anc_count)))
            for a in range(len(eligible_codewords)-1):
                for b in range(a+1,len(eligible_codewords),1):
                    bqm.add_quadratic(eligible_codewords[a],eligible_codewords[b],2*W2)
            for a in range(num_anc+anc_count,anc_count+1,-1):
                for b in range(a-1,anc_count,-1):
                    bqm.add_quadratic('ancillary_{}'.format(a),'ancillary_{}'.format(b),2*W2*(math.pow(2,a-anc_count))*(math.pow(2,b-anc_count)))                       
            anc_count = anc_count + num_anc
        #qubo = bqm.to_qubo()
        #print(qubo)    
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
        for t in range(anc_count):
            s_frame.drop(['ancillary_{}'.format(t+1)],axis=1,inplace=True)
        #s_frame.columns = s_frame.columns.astype(int)    
        #print(s_frame)
                    
        #Analyzing lowest energy solution
        temp = list(s_frame.iloc[0])
        temp = temp[0:n]        #Lowest energy array
        #print("Lowest energy codeword is {}".format(verify(temp)))
        BER1 = Hamming(temp, encoded_message)
        #print(temp) 
        accepted_list = []

        #Minimum dist
        for i in range(len(s_frame)):
            temp1 = list(s_frame.iloc[i])
            temp1 = temp1[0:n]
            if(temp1 not in accepted_list) and (verify(temp1)):
                tup = (temp1, dist_noise(temp1, received))
                accepted_list.append(tup)
        
        
        try:
            mini = accepted_list[0][1]
            shortest_codeword_index=0

            for j in range(len(accepted_list)):
                if accepted_list[j][1]<mini:
                    shortest_codeword_index=j
            shortest_codeword=accepted_list[shortest_codeword_index][0]
            frac=frac+int(verify(shortest_codeword))
            BER2 = Hamming(shortest_codeword, encoded_message)
            
            temp, encoded_message, shortest_codeword = np.array(temp), np.array(encoded_message), np.array(shortest_codeword)
            #print(temp.shape, encoded_message.shape, shortest_codeword.shape)

            BER+=(min(BER1, BER2))/n

            if BER1>BER2 and not np.array_equal(shortest_codeword,encoded_message):
                FER+=1
            elif BER1<BER2 and not np.array_equal(temp,encoded_message):
                FER+=1
            elif BER1==BER2 and  not np.array_equal(temp,encoded_message) and not np.array_equal(shortest_codeword,encoded_message):
                FER+=1
            #print(shortest_codeword)

        except:
            BER+=BER1/n
            temp, encoded_message = np.array(temp), np.array(encoded_message)
            frac=frac+int(verify(temp))
            #print(temp.shape, encoded_message.shape)
            #print(temp)
            if not np.array_equal(np.array(temp), np.array(encoded_message)):
                FER+=1
        if ind_2%410==0:   
            print(ind_2)
    
    BER_list.append(BER/BER_batch)
    FER_list.append(FER/BER_batch)
    frac_list.append(frac/BER_batch)
    print("SNR:{} done".format(SNR[ind_1]))

    #BER_list=np.array(BER_list)
    #FER_list=np.array(FER_list)
    
    print(BER_list, FER_list, frac_list)
    #np.savez_compressed("rayleigh_0.5_{}.npz".format(SNR[ind_1]), x=BER_list, y=FER_list)