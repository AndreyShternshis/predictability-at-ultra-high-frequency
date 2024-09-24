import pandas
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
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
    return 2*KL, Count/n
location='/home'
file='AAPL_2022-08-02_34200000_57600000_message_1.csv'
agg=1
Datafile=pandas.read_csv(location+'/'+file, low_memory=False,header=None)
Datafile = Datafile.iloc[:, [0, 1, 3, 4]]
Datafile.columns = ['Time', 'Type', 'Volume', 'Price']
Datafile['Price']=Datafile['Price']/10000
Datafile['logprice']=np.log(Datafile['Price'])
Datafile = Datafile[(Datafile.Type >= 4) & (Datafile.Type <= 5)]
Delta=np.copy(Datafile.Time - Datafile.Time.shift(1))
Datafile['key']=np.floor(np.arange(np.shape(Datafile)[0])/agg)
Df=Datafile.groupby('key').aggregate({'Time': 'last','Volume': 'sum', 'logprice': 'last'})
Df['Ret']=Df.logprice - Df.logprice.shift(1)
Df = Df.iloc[1: , :]
Data=np.zeros_like(Df.Ret)
medianm=0
Data[Df.Ret<medianm]=1
Data[Df.Ret>medianm]=2
Data = Data[Data!=0]
N=np.size(Data)
print(N)
A=2
Kmax=int(np.round(np.log(N)/np.log(A)/2))
Data=Data-1
statistics=[]
quantiles=[]
med=[]
for k in range(2,Kmax+1,1):
    print(k)
    KL, prob=KLstatistics(Data,k,A)
    print(np.where(prob==np.max(prob))[0])
    nu=(A**(k-1)-1)*(A-1)
    q99=chi2.ppf(0.99,nu)
    statistics=np.append(statistics, KL)
    quantiles=np.append(quantiles, q99)
    med=np.append(med, nu)
fig, ax = plt.subplots()
plt.plot(range(2,Kmax+1),statistics, label='statistics D')
plt.plot(range(2,Kmax+1),med, label='mean')
plt.plot(range(2,Kmax+1),quantiles, label='99% CI')
plt.title('NP-statistics, 02.08.2022', fontsize=14)
plt.ylabel('D', fontsize=14)
plt.xlabel('k', fontsize=14)
plt.xticks(range(2,Kmax+1),range(2,Kmax+1))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.legend(fontsize=14)
plt.savefig('AAPL0208D.eps', format='eps',bbox_inches='tight')
fig, ax = plt.subplots()
plt.plot(Delta)
plt.title('Intraday pattern of durations for AAPL, 02.08.2022', fontsize=14)
plt.ylabel('seconds between transactions', fontsize=14)
plt.xlabel('number of transactions', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.savefig('AAPL0208time.eps', format='eps',bbox_inches='tight')