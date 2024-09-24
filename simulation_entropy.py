import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
def Entropystatistics(Data,k,A):
    N=np.size(Data)
    n=int(N/k)
    Count=np.zeros([A**(k),1])
    K=A**(np.arange(k-1,-1,-1))
    for i in range(n):
        Count[int(np.dot(Data[i*k:(i+1)*k],K))]+=1
    H=0
    for i in range(A**k):
        if Count[i]!=0:
            pi=Count[i]/n
            H=H-pi*np.log(pi)
    return 2*n*(k*np.log(A)-H)
N=10**5
fig = plt.figure()
###
A=2
n=10**2
k=int(np.round(np.log(n)/np.log(A)/2))
print(k)
Entropies=[]
np.random.seed(0)
for j in range(N):
    Data =np.random.randint(A, size=(n)) #bernoulli.rvs(0.5, size=n)
    KL=Entropystatistics(Data,k,A)
    Entropies=np.append(Entropies, KL)
ax = fig.add_subplot(3, 3, 1)
sm.graphics.qqplot(Entropies, stats.chi2, distargs=((A**k-1),),fit=True,line="45",ax=ax)
plt.title("n=100,s=2")
plt.ylabel("")
plt.xlabel("")
###
A=3
n=10**2
k=int(np.round(np.log(n)/np.log(A)/2))
print(k)
Entropies=[]
np.random.seed(0)
for j in range(N):
    Data =np.random.randint(A, size=(n)) #bernoulli.rvs(0.5, size=n)
    KL=Entropystatistics(Data,k,A)
    Entropies=np.append(Entropies, KL)
ax = fig.add_subplot(3, 3, 2)
sm.graphics.qqplot(Entropies,stats.chi2,distargs=((A**k-1),),fit=True,line="45",ax=ax)
plt.title("n=100,s=3")
plt.ylabel("")
plt.xlabel("")
###
A=4
n=10**2
k=int(np.round(np.log(n)/np.log(A)/2))
print(k)
Entropies=[]
np.random.seed(0)
for j in range(N):
    Data =np.random.randint(A, size=(n)) #bernoulli.rvs(0.5, size=n)
    KL=Entropystatistics(Data,k,A)
    Entropies=np.append(Entropies, KL)
ax = fig.add_subplot(3, 3, 3)
sm.graphics.qqplot(Entropies,stats.chi2,distargs=((A**k-1),),fit=True,line="45",ax=ax)
plt.title("n=100,s=4")
plt.ylabel("")
plt.xlabel("")
###
A=2
n=10**3
k=int(np.round(np.log(n)/np.log(A)/2))
print(k)
Entropies=[]
np.random.seed(0)
for j in range(N):
    Data =np.random.randint(A, size=(n)) #bernoulli.rvs(0.5, size=n)
    KL=Entropystatistics(Data,k,A)
    Entropies=np.append(Entropies, KL)
ax = fig.add_subplot(3, 3, 4)
sm.graphics.qqplot(Entropies, stats.chi2, distargs=((A**k-1),),fit=True,line="45",ax=ax)
plt.title("n=1000,s=2")
plt.ylabel("")
plt.xlabel("")
###
A=3
n=10**3
k=int(np.round(np.log(n)/np.log(A)/2))
print(k)
Entropies=[]
np.random.seed(0)
for j in range(N):
    Data =np.random.randint(A, size=(n)) #bernoulli.rvs(0.5, size=n)
    KL=Entropystatistics(Data,k,A)
    Entropies=np.append(Entropies, KL)
ax = fig.add_subplot(3, 3, 5)
sm.graphics.qqplot(Entropies,stats.chi2,distargs=((A**k-1),),fit=True,line="45",ax=ax)
plt.title("n=1000,s=3")
plt.ylabel("")
plt.xlabel("")
###
A=4
n=10**3
k=int(np.round(np.log(n)/np.log(A)/2))
print(k)
Entropies=[]
np.random.seed(0)
for j in range(N):
    Data =np.random.randint(A, size=(n)) #bernoulli.rvs(0.5, size=n)
    KL=Entropystatistics(Data,k,A)
    Entropies=np.append(Entropies, KL)
ax = fig.add_subplot(3, 3, 6)
sm.graphics.qqplot(Entropies,stats.chi2,distargs=((A**k-1),),fit=True,line="45",ax=ax)
plt.title("n=1000,s=4")
plt.ylabel("")
plt.xlabel("")
###
A=2
n=10**4
k=int(np.round(np.log(n)/np.log(A)/2))#-1
print(k)
Entropies=[]
np.random.seed(0)
for j in range(N):
    Data =np.random.randint(A, size=(n)) #bernoulli.rvs(0.5, size=n)
    KL=Entropystatistics(Data,k,A)
    Entropies=np.append(Entropies, KL)
ax = fig.add_subplot(3, 3, 7)
sm.graphics.qqplot(Entropies, stats.chi2, distargs=((A**k-1),),fit=True,line="45",ax=ax)
plt.title("n=10000,s=2")
plt.ylabel("")
plt.xlabel("")
###
A=3
n=10**4
k=int(np.round(np.log(n)/np.log(A)/2))
print(k)
Entropies=[]
np.random.seed(0)
for j in range(N):
    Data =np.random.randint(A, size=(n)) #bernoulli.rvs(0.5, size=n)
    KL=Entropystatistics(Data,k,A)
    Entropies=np.append(Entropies, KL)
ax = fig.add_subplot(3, 3, 8)
sm.graphics.qqplot(Entropies,stats.chi2,distargs=((A**k-1),),fit=True,line="45",ax=ax)
plt.title("n=10000,s=3")
plt.ylabel("")
plt.xlabel("")
plt.show
###
A=4
n=10**4
k=int(np.round(np.log(n)/np.log(A)/2))
print(k)
Entropies=[]
np.random.seed(0)
for j in range(N):
    Data =np.random.randint(A, size=(n)) #bernoulli.rvs(0.5, size=n)
    KL=Entropystatistics(Data,k,A)
    Entropies=np.append(Entropies, KL)
ax = fig.add_subplot(3, 3, 9)
sm.graphics.qqplot(Entropies,stats.chi2,distargs=((A**k-1),),fit=True,line="45",ax=ax)
plt.title("n=10000,s=4")
plt.ylabel("")
plt.xlabel("")
fig.suptitle('QQ plots', fontsize=16)
fig.text(0.5, 0, 'Theoretical Quantiles', ha='center', fontsize=16)
fig.text(0, 0.2, 'Sample Quantiles', ha='center', fontsize=16, rotation='vertical')
fig.tight_layout()
plt.show
plt.savefig('QQplots_entropy05.eps', format='eps',bbox_inches='tight')
plt.savefig('QQplots_entropy05.png', format='png',bbox_inches='tight')