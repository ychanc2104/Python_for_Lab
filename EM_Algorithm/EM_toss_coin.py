
import numpy as np
import math
import matplotlib.pyplot as plt
## E-M Coin Toss Example as given in the EM tutorial paper by Do and Batzoglou* ##
def get_binomial_log_likelihood(obs,probs):
    """ Return the (log)likelihood of obs, given the probs"""
    # Binomial Distribution Log PDF
    # ln (pdf)      = Binomial Coeff * product of probabilities
    # ln[f(x|n, p)] =   comb(N,k)    * num_heads*ln(pH) + (N-num_heads) * ln(1-pH)
    N = sum(obs) #number of trials  
    k = obs[0] # number of heads
    binomial_coeff = math.factorial(N) / (math.factorial(N-k) * math.factorial(k))
    prod_probs = obs[0]*math.log(probs[0]) + obs[1]*math.log(1-probs[0])
    log_lik = binomial_coeff + prod_probs
    return log_lik
# 1st:  Coin B, {HTTTHHTHTH}, 5H,5T
# 2nd:  Coin A, {HHHHTHHHHH}, 9H,1T
# 3rd:  Coin A, {HTHHHHHTHH}, 8H,2T
# 4th:  Coin B, {HTHTTTHHTT}, 4H,6T
# 5th:  Coin A, {THHHTHHHTH}, 7H,3T
# so, from MLE: pA(heads) = 0.80 and pB(heads)=0.45
# represent the experiments
head_counts = np.array([5,9,8,4,7])
tail_counts = 10-head_counts
# initialise the pA(heads) and pB(heads)
# pA_heads = np.zeros(100); pA_heads[0] = 0.10
# pB_heads = np.zeros(100); pB_heads[0] = 0.50

pA_heads = [0.3]
pB_heads = [0.48]
# E-M begins!
delta = 10e-15
j = 0 # iteration counter
improvement = float('inf')
while (improvement>delta):
    expectation_A = np.zeros((len(head_counts),2), dtype=float) 
    expectation_B = np.zeros((len(head_counts),2), dtype=float)
    i = 0
    for h, t in zip(head_counts,tail_counts):
        e = [h, t] # i'th experiment
          # loglikelihood of e given coin A:
        ll_A = get_binomial_log_likelihood(e,np.array([pA_heads[j],1-pA_heads[j]])) 
          # loglikelihood of e given coin B
        ll_B = get_binomial_log_likelihood(e,np.array([pB_heads[j],1-pB_heads[j]])) 
# corresponding weight of A proportional to likelihood of A 
        weightA = math.exp(ll_A) / ( math.exp(ll_A) + math.exp(ll_B) ) 
# corresponding weight of B proportional to likelihood of B
        weightB = math.exp(ll_B) / ( math.exp(ll_A) + math.exp(ll_B) ) 
        expectation_A[i,:] = np.dot(weightA, e) 
        expectation_B[i,:] = np.dot(weightB, e)
        i += 1
    # pA_heads[j+1] = sum(expectation_A)[0] / sum(sum(expectation_A)) 
    # pB_heads[j+1] = sum(expectation_B)[0] / sum(sum(expectation_B))
    pA_heads += [sum(expectation_A)[0] / sum(sum(expectation_A))]
    pB_heads += [sum(expectation_B)[0] / sum(sum(expectation_B))]
    improvement = ( max( abs(np.array([pA_heads[j+1],pB_heads[j+1]]) - np.array([pA_heads[j],pB_heads[j]]) )) )
    j = j+1
plt.figure();
plt.plot(np.arange(0,j+1),pA_heads, 'ro--')
plt.plot(np.arange(0,j+1),pB_heads, 'bo-')
plt.show()

print(f'pA is {pA_heads[-1]}')
print(f'pB is {pB_heads[-1]}')


