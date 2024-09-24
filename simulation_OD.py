import numpy as np
import random
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
    return 2*KL,Freq
def roundbyticksize(x,delta):
    return np.round(x/delta)*delta
A=2
AGG=50
Traders=1000
n=10**5
Ndays=1
T=100
tau=2*T
Lmax=T
pf=1000
sigma1=1
sigma2=1.4
n0=0.5
lambda1=0.5
kmax=0.5
delta=0.1
fracpred=np.zeros(AGG)
day=0
while day < Ndays:
    while True:
        try:
            day=day+1
            print(day)
            g1=abs(np.random.normal(0,sigma1,Traders))
            g2=np.random.normal(0,sigma2,Traders)
            noise=np.random.normal(0,n0,Traders)
            ki=np.random.uniform(0,kmax,Traders)
            L = []
            for i in range(Traders):
                L.append(random.randint(1, Lmax+1))
            bid=[]
            ask=[]
            timeofbid=[]
            timeofask=[]
            p=pf*np.ones(2)
            Output_prices=[]
            Output_signs=[]
            t=1
            while np.size(Output_prices)<n:
                t+=1
                if np.size(timeofbid)>0:
                    timeofbid=timeofbid+1
                    if timeofbid[0]>tau:
                        bid = np.delete(bid, 0)
                        timeofbid = np.delete(timeofbid, 0)
                if np.size(timeofask) > 0:
                    timeofask = timeofask + 1
                    if timeofask[0] > tau:
                        ask = np.delete(ask, 0)
                        timeofask = np.delete(timeofask, 0)
                liquiditycheck=np.random.binomial(size=1, n=1, p=lambda1)
                if liquiditycheck==0:
                    if np.size(bid)*np.size(ask)>0:
                        p = np.append(p, (np.max(bid) + np.min(ask)) / 2)
                    else:
                        p = np.append(p, p[t-1])
                else:
                    Tradernumber=random.randrange(Traders)
                    Llocal=np.min([L[Tradernumber],t-1])
                    rL=sum(np.divide(p[(t-Llocal):t]-p[(t-Llocal-1):(t-1)],p[(t-Llocal-1):(t-1)]))/Llocal
                    eps=np.random.normal(0,1)
                    rhat=g1[Tradernumber]*(pf-p[t-1])/p[t-1]+g2[Tradernumber]*rL+noise[Tradernumber]*eps
                    phat = p[t - 1] * np.exp(rhat)
                    if phat>p[t-1]:
                        bidt = roundbyticksize(phat * (1 - ki[Tradernumber]), delta)
                        if np.size(ask)>0 and bidt>=np.min(ask):
                            p = np.append(p, np.min(ask))
                            Output_signs=np.append(Output_signs,1)
                            Output_prices = np.append(Output_prices, np.min(ask))
                            timeofask = np.delete(timeofask, np.argmin(ask))
                            ask=np.delete(ask,np.argmin(ask))
                        else:
                            bid = np.append(bid, bidt)
                            timeofbid = np.append(timeofbid, 0)
                            if np.size(bid)*np.size(ask)>0:
                                p = np.append(p, (np.max(bid) + np.min(ask)) / 2)
                            else:
                                p = np.append(p, p[t-1])
                    else:
                        askt = roundbyticksize(phat * (1 + ki[Tradernumber]), delta)
                        if np.size(bid)>0 and askt<=np.max(bid):
                            p = np.append(p, np.max(bid))
                            Output_signs=np.append(Output_signs,0)
                            Output_prices = np.append(Output_prices, np.max(bid))
                            timeofbid = np.delete(timeofbid, np.argmax(bid))
                            bid=np.delete(bid,np.argmax(bid))
                        else:
                            ask = np.append(ask, askt)
                            timeofask = np.append(timeofask, 0)
                            if np.size(bid)*np.size(ask)>0:
                                p = np.append(p, (np.max(bid) + np.min(ask)) / 2)
                            else:
                                p = np.append(p, p[t - 1])
                    if np.isnan(rhat) or t>10*n:#or p[t-1]>10**(4) or p[t-1]<10**(2):
                        print(t,np.size(Output_prices))
                        print(p)
                        print(phat,rhat,(pf-p[t-1])/p[t-1],rL)
                        print(bidt,askt,ki[Tradernumber])
                        print(bid)
                        print(ask)
                        raise Exception("error in price dynamics")
            for agg in range(1,AGG+1):
                Prices = Output_prices[range(0, np.size(Output_prices), agg)]
                returns = Prices[1:] - Prices[:-1]
                returns = np.delete(returns, (np.where(returns == 0))[0])
                Data = np.zeros_like(returns)
                Data[np.where(returns > 0)[0]] = 1
                Data[np.where(returns < 0)[0]] = 0
                k=int(np.round(np.log(np.size(Data))/np.log(A)/2))
                nu=(A**(k-1)-1)*(A-1)
                H,Freq=KLstatistics(Data,k,A)
                empquant=stats.chi2.cdf(H,df=nu)
                if empquant>0.99:
                        fracpred[agg-1]+=1
            break
        except:
            print('error')
            day = day - 1
fracpred=fracpred/Ndays
np.save('simulation_OD',fracpred)
plt.figure(0)
plt.plot(range(1,AGG+1),fracpred)
plt.title('Fraction of predictable simulated intervals', fontsize=14)
plt.ylabel('Fraction of predictable intervals', fontsize=14)
plt.xlabel('aggregation level', fontsize=14)
plt.savefig('simulation_OD.eps', format='eps',bbox_inches='tight')