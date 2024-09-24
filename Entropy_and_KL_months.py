import pandas
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import os
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
A=2
Nagg=50
fig = plt.figure()
ASSETS=['AAPL','MSFT','TSLA','INTC','LLY','SNAP','F','CCL','SPY']
# folder path
for ticker in range(1,10):
    dir_path = ASSETS[ticker-1]+'_August2022'
    res = []
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            if path.endswith('_message_1.csv'):
                res.append(path)
    ax = fig.add_subplot(3, 3, ticker)
    AUGUST=np.zeros(Nagg)
    countA=0
    SEPTEMBER=np.zeros(Nagg)
    countS=0
    OCTOBER=np.zeros(Nagg)
    countO=0
    NOVEMBER=np.zeros(Nagg)
    countN=0
    for i in range(np.size(res)):
        file=res[i]
        if '2022-08' in file:
            countA+=1
        elif '2022-09' in file:
            countS+=1
        elif '2022-10' in file:
            countO+=1
        else:
            countN+=1
        Datafile=pandas.read_csv(dir_path+'/'+file, low_memory=False,header=None)
        Datafile = Datafile.iloc[:, [0, 1, 3, 4]]
        Datafile.columns = ['Time', 'Type', 'Volume', 'Price']
        Datafile = Datafile[(Datafile.Type >= 4) & (Datafile.Type <= 5)]
        Datafile['delta']=Datafile.Time - Datafile.Time.shift(1)
        for agg in range(1,Nagg+1,1):
            Datafile['key']=np.floor(np.arange(np.shape(Datafile)[0])/agg)
            Df=Datafile.groupby('key').aggregate({'Time': 'last','Volume': 'sum', 'Price': 'last'})
            Df['Price']=Df['Price']/10000
            Df['Ret']=np.log(Df.Price) - np.log(Df.Price.shift(1))
            Df.drop(index=Df.index[0], axis=0, inplace=True)
            Data=np.zeros_like(Df.Ret)
            Data[Df.Ret<0]=1
            Data[Df.Ret>0]=2
            Data = Data[Data!=0]
            Data=Data-1
            N=np.size(Data)
            k=int(np.round(np.log(N)/np.log(A)/2))
            if k<2:
                print('warning', Nagg)
            nu=A**k-1 #(A**(k-1)-1)*(A-1) if NP statistics is used
            H=Entropystatistics(Data,k,A)
            empquant=chi2.cdf(H,df=nu)
            if '2022-08' in file:
                if empquant>0.99:
                    AUGUST[agg-1]+=1
            elif '2022-09' in file:
                if empquant>0.99:
                    SEPTEMBER[agg-1]+=1
            elif '2022-10' in file:
                if empquant>0.99:
                    OCTOBER[agg-1]+=1
            else:
                if empquant>0.99:
                    NOVEMBER[agg-1]+=1
    plt.plot(range(1,Nagg+1),AUGUST/countA)
    plt.plot(range(1,Nagg+1),SEPTEMBER/countS)
    plt.plot(range(1,Nagg+1),OCTOBER/countO)
    plt.plot(range(1,Nagg+1),NOVEMBER/countN)
    plt.ylabel("")
    plt.xlabel("")
    plt.title(ASSETS[ticker-1])
fig.suptitle('Fraction of predictable daily intervals', fontsize=12)
fig.text(0.5, 0, 'aggregation level', ha='center', fontsize=12)
fig.text(0, 0.1, 'Fraction of predictable days', ha='center', fontsize=12, rotation='vertical')
handles, labels = ax.get_legend_handles_labels()
fig.tight_layout()
fig.legend(['August','September','October','November'],loc='upper center',ncol=2, bbox_to_anchor=(0.5, -0.05),fontsize=14)
plt.savefig('months.eps', format='eps',bbox_inches='tight')