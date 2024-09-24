import numpy as np
from scipy import stats
import statsmodels.api as sm
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
N=100000
fig = plt.figure()
###
A=2
p1=2/(A+1)
p2=1/(A+1)
n=10**2
k=int(np.round(np.log(n)/np.log(A)/2))
Entropies=[]
np.random.seed(0)
for j in range(N):
    Data =np.random.choice(np.arange(0, A), p=[p1, p2],size=n)
    KL=KLstatistics(Data,k,A)
    Entropies=np.append(Entropies, KL)
ax = fig.add_subplot(3, 3, 1)
sm.graphics.qqplot(Entropies, stats.chi2, distargs=((A**(k-1)-1)*(A-1),),fit=True,line="45",ax=ax)
plt.title("n=100,s=2")
plt.ylabel("")
plt.xlabel("")
###
A=3
p1=2/(A+1)
p2=1/(A+1)
n=10**2
k=int(np.round(np.log(n)/np.log(A)/2))
Entropies=[]
np.random.seed(0)
for j in range(N):
    Data =np.random.choice(np.arange(0, A), p=[p1, p2,p2],size=n)
    KL=KLstatistics(Data,k,A)
    Entropies=np.append(Entropies, KL)
ax = fig.add_subplot(3, 3, 2)
sm.graphics.qqplot(Entropies,stats.chi2,distargs=((A**(k-1)-1)*(A-1)),fit=True,line="45",ax=ax)
plt.title("n=100,s=3")
plt.ylabel("")
plt.xlabel("")
###
A=4
p1=2/(A+1)
p2=1/(A+1)
n=10**2
k=int(np.round(np.log(n)/np.log(A)/2))
Entropies=[]
np.random.seed(0)
for j in range(N):
    Data =np.random.choice(np.arange(0, A), p=[p1, p2,p2,p2],size=n)
    KL=KLstatistics(Data,k,A)
    Entropies=np.append(Entropies, KL)
ax = fig.add_subplot(3, 3, 3)
sm.graphics.qqplot(Entropies,stats.chi2,distargs=((A**(k-1)-1)*(A-1)),fit=True,line="45",ax=ax)
plt.title("n=100,s=4")
plt.ylabel("")
plt.xlabel("")
###
A=2
p1=2/(A+1)
p2=1/(A+1)
n=10**3
k=int(np.round(np.log(n)/np.log(A)/2))
Entropies=[]
np.random.seed(0)
for j in range(N):
    Data =np.random.choice(np.arange(0, A), p=[p1, p2],size=n)
    KL=KLstatistics(Data,k,A)
    Entropies=np.append(Entropies, KL)
ax = fig.add_subplot(3, 3, 4)
sm.graphics.qqplot(Entropies, stats.chi2, distargs=((A**(k-1)-1)*(A-1),),fit=True,line="45",ax=ax)
plt.title("n=1000,s=2")
plt.ylabel("")
plt.xlabel("")
###
A=3
p1=2/(A+1)
p2=1/(A+1)
n=10**3
k=int(np.round(np.log(n)/np.log(A)/2))
Entropies=[]
np.random.seed(0)
for j in range(N):
    Data =np.random.choice(np.arange(0, A), p=[p1, p2,p2],size=n)
    KL=KLstatistics(Data,k,A)
    Entropies=np.append(Entropies, KL)
ax = fig.add_subplot(3, 3, 5)
sm.graphics.qqplot(Entropies,stats.chi2,distargs=((A**(k-1)-1)*(A-1)),fit=True,line="45",ax=ax)
plt.title("n=1000,s=3")
plt.ylabel("")
plt.xlabel("")
###
A=4
p1=2/(A+1)
p2=1/(A+1)
n=10**3
k=int(np.round(np.log(n)/np.log(A)/2))
Entropies=[]
np.random.seed(0)
for j in range(N):
    Data =Data =np.random.choice(np.arange(0, A), p=[p1, p2,p2,p2],size=n)
    KL=KLstatistics(Data,k,A)
    Entropies=np.append(Entropies, KL)
ax = fig.add_subplot(3, 3, 6)
sm.graphics.qqplot(Entropies,stats.chi2,distargs=((A**(k-1)-1)*(A-1)),fit=True,line="45",ax=ax)
plt.title("n=1000,s=4")
plt.ylabel("")
plt.xlabel("")
###
A=2
p1=2/(A+1)
p2=1/(A+1)
n=10**4
k=int(np.round(np.log(n)/np.log(A)/2))
Entropies=[]
np.random.seed(0)
for j in range(N):
    Data =np.random.choice(np.arange(0, A), p=[p1, p2],size=n)
    KL=KLstatistics(Data,k,A)
    Entropies=np.append(Entropies, KL)
ax = fig.add_subplot(3, 3, 7)
sm.graphics.qqplot(Entropies, stats.chi2, distargs=((A**(k-1)-1)*(A-1),),fit=True,line="45",ax=ax)
plt.title("n=10000,s=2")
plt.ylabel("")
plt.xlabel("")
###
A=3
p1=2/(A+1)
p2=1/(A+1)
n=10**4
k=int(np.round(np.log(n)/np.log(A)/2))
Entropies=[]
np.random.seed(0)
for j in range(N):
    Data =np.random.choice(np.arange(0, A), p=[p1, p2,p2],size=n)
    KL=KLstatistics(Data,k,A)
    Entropies=np.append(Entropies, KL)
ax = fig.add_subplot(3, 3, 8)
sm.graphics.qqplot(Entropies,stats.chi2,distargs=((A**(k-1)-1)*(A-1)),fit=True,line="45",ax=ax)
plt.title("n=10000,s=3")
plt.ylabel("")
plt.xlabel("")
plt.show
###
A=4
p1=2/(A+1)
p2=1/(A+1)
n=10**4
k=int(np.round(np.log(n)/np.log(A)/2))
Entropies=[]
np.random.seed(0)
for j in range(N):
    Data =Data =np.random.choice(np.arange(0, A), p=[p1, p2,p2,p2],size=n)
    KL=KLstatistics(Data,k,A)
    Entropies=np.append(Entropies, KL)
ax = fig.add_subplot(3, 3, 9)
sm.graphics.qqplot(Entropies,stats.chi2,distargs=((A**(k-1)-1)*(A-1)),fit=True,line="45",ax=ax)
plt.title("n=10000,s=4")
plt.ylabel("")
plt.xlabel("")
fig.suptitle('QQ plots', fontsize=16)
fig.text(0.5, 0, 'Theoretical Quantiles', ha='center', fontsize=16)
fig.text(0, 0.2, 'Sample Quantiles', ha='center', fontsize=16, rotation='vertical')
fig.tight_layout()
plt.show
plt.savefig('QQplots_KL06.eps', format='eps',bbox_inches='tight')
plt.savefig('QQplots2_KL06.png', format='png',bbox_inches='tight')