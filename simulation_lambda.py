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
    return 2*KL
A=2
AGG=50
n=10**5
N0=21
alpha=1.63
prob=0.38
Ndays=80
fracpred=np.zeros(AGG)
for day in range(Ndays):
    N=N0
    Signs=np.random.binomial(size=N, n=1, p= 0.5)
    Volumes=np.ceil(np.random.pareto(a=alpha,size=N))
    Output=np.zeros(n)
    for t in range(n):
        newV=np.random.binomial(size=1, n=1, p=prob)
        if N==0 or newV==1:
            Volumes=np.append(Volumes,np.ceil(np.random.pareto(a=alpha,size=1)))
            Signs=np.append(Signs,np.random.binomial(size=1, n=1, p= 0.5))
            N+=1
        randomorder=random.randrange(N)
        Output[t]=Signs[randomorder]
        Volumes[randomorder]-=1
        if int(Volumes[randomorder])==0:
            Volumes=np.delete(Volumes,int(randomorder))
            Signs=np.delete(Signs, randomorder)
            N-=1
    for agg in range(1,AGG+1):
        Data=Output[range(0,np.size(Output),agg)]
        k=int(np.round(np.log(np.size(Data))/np.log(A)/2))
        nu=(A**(k-1)-1)*(A-1)
        H=KLstatistics(Data,k,A)
        empquant=stats.chi2.cdf(H,df=nu)
        if empquant>0.99:
                fracpred[agg-1]+=1
fracpred=fracpred/Ndays
np.save('simulation_lambda',fracpred)
plt.plot()
plt.plot(range(1,AGG+1),fracpred)
plt.title('Fraction of predictable simulated intervals', fontsize=14)
plt.ylabel('Fraction of predictable intervals', fontsize=14)
plt.xlabel('aggregation level', fontsize=14)
plt.savefig('simulation_lambda.eps', format='eps',bbox_inches='tight')
plt.show()