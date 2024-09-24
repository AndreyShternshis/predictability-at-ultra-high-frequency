import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
def KLstatistics(Data,k,A):
    N=np.size(Data)
    n=N-k+1
    Count=np.zeros([A**(k),1])
    K=A**(np.arange(k-1,-1,-1))
    for i in range(n):
        Count[int(np.dot(Data[i:(i+k)],K))]+=1
    Freq=np.reshape(Count,(A**(k-1),A))
    fi=np.sum(Freq,axis=1)
    fj=np.sum(Freq,axis=0)
    KL=0
    for i in range(A**(k-1)):
        for j in range(A):
            if Freq[i,j]!=0:
                KL+=Freq[i,j]*np.log(Freq[i,j]*n/fi[i]/fj[j])
    return 2*KL
beta=0.42
gamma=0.24
C0=0.20
l0=20
Gamma0=2.8*10**(-3)
def G0_function(l,Gamma0,l0,beta):
    return Gamma0/(l+l0)**beta
sigma1=0.01
sigma2=0.01
p0=0
n=10**5
G0=G0_function(np.arange(n),Gamma0,l0,beta)
A=2
AGG=50
Ndays=80
fracpred_signs=np.zeros(AGG)
fracpred_prices=np.zeros(AGG)
for day in range(Ndays):
    eps=np.random.normal(0,sigma1,n)
    eta=np.random.normal(0,sigma2,n)
    lnV=np.random.normal(5.5,1.8,n)
    signs=np.zeros(n)
    p=np.zeros(n)
    p[0]=p0+eps[0]
    if p[0]>p0:
        signs[0]=1
    else:
        signs[0]=-1
    for t in range(1,n):
        p[t]=np.dot(np.multiply(G0[0:t],signs[0:t]),lnV[0:t])+sum(eta[0:t])+eps[t]
        if p[t] > p[t-1]:
            signs[t] = 1
        else:
            signs[t] = -1
    signs[np.where(signs == -1)[0]] = 0
    for agg in range(1,AGG+1):
        Data_signs=signs[range(0,np.size(signs),agg)]
        k=int(np.round(np.log(np.size(Data_signs))/np.log(A)/2))
        nu=(A**(k-1)-1)*(A-1)
        H=KLstatistics(Data_signs,k,A)
        empquant=stats.chi2.cdf(H,df=nu)
        if empquant>0.99:
                fracpred_signs[agg-1]+=1
        Prices = p[range(0, np.size(p), agg)]
        returns = Prices[1:] - Prices[:-1]
        returns = np.delete(returns, (np.where(returns == 0))[0])
        Data_prices = np.zeros_like(returns)
        Data_prices[np.where(returns > 0)[0]] = 1
        Data_prices[np.where(returns < 0)[0]] = 0
        k = int(np.round(np.log(np.size(Data_prices)) / np.log(A) / 2))
        nu = (A ** (k - 1) - 1) * (A - 1)
        H = KLstatistics(Data_prices, k, A)
        empquant = stats.chi2.cdf(H, df=nu)
        if empquant > 0.99:
            fracpred_prices[agg - 1] += 1
fracpred_signs=fracpred_signs/Ndays
fracpred_prices=fracpred_prices/Ndays
np.save('simulation_TS_signs',fracpred_signs)
np.save('simulation_TS_prices',fracpred_prices)
plt.figure(0)
plt.plot(range(1,AGG+1),fracpred_signs)
plt.title('Fraction of predictable simulated intervals', fontsize=14)
plt.ylabel('Fraction of predictable intervals', fontsize=14)
plt.xlabel('time lag', fontsize=14)
plt.savefig('simulation_TS_signs.eps', format='eps',bbox_inches='tight')
plt.figure(1)
plt.plot(range(1,AGG+1),fracpred_prices)
plt.title('Fraction of predictable simulated intervals', fontsize=14)
plt.ylabel('Fraction of predictable intervals', fontsize=14)
plt.xlabel('aggregation level', fontsize=14)
plt.savefig('simulation_TS_prices.eps', format='eps',bbox_inches='tight')