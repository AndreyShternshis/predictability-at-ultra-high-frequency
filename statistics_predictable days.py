import pandas
import numpy as np
import scipy.stats as stats
import math
import os
dir_path = r'AAPL_August2022'
res = []
for path in os.listdir(dir_path):
    if os.path.isfile(os.path.join(dir_path, path)):
        if path.endswith('_message_1.csv'):
            res.append(path)
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
def jumps_Lee(logprices):
    returns=logprices[1:]-logprices[:-1]
    n=np.size(returns)
    KJ=int(np.ceil(np.sqrt(n)))
    BV=np.zeros(n-KJ+1)
    for i in range(KJ-1,n):
        BV[i-KJ+1]=1/(KJ-2)*np.dot(abs(returns[(i-KJ+2):(i)]),abs(returns[(i-KJ+1):(i-1)]))
    returns=returns[KJ-1:]
    returns=returns[BV>0]
    BV=BV[BV>0]
    LJ=np.divide(returns,np.sqrt(BV))
    c=np.sqrt(2/math.pi)
    n=np.size(LJ)
    Cn=np.sqrt(2*np.log(n))/c-(np.log(math.pi)+np.log(np.log(n)))/(2*c*np.sqrt(2*np.log(n)))
    Sn=1/c/np.sqrt(2*np.log(n))
    beta=-np.log(-np.log(0.99))
    test=(abs(LJ)-Cn)/Sn
    return sum(test>beta)/np.size(test)
zeroseff, zerosineff, autocorreff, autocorrineff, voleff, volineff, repeateff,repeatineff,probdiffeff,probdiffineff,pricechangeeff,pricechangeineff,nueff,nuineff,scaleeff,scaleineff,loceff,locineff,loc2eff,loc2ineff,sigeff,sigineff,sizeineff,sizeeff,jumpineff,jumpeff,volumeineff,volumeeff,meanineff,meaneff,keff,kineff,durationineff,durationeff,STATprice,STATstd,STATvolume,STATtransactions,STATduration = ([] for i in range(39))
A=2
agg=1
print('agg is', agg)
for i in range(np.size(res)):
    file=res[i]
    Datafile=pandas.read_csv(dir_path+'/'+file, low_memory=False,header=None)
    Datafile = Datafile.iloc[:, [0, 1, 3, 4]]
    Datafile.columns = ['Time', 'Type', 'Volume', 'Price']
    Datafile = Datafile[(Datafile.Type >= 4) & (Datafile.Type <= 5)]
    Datafile['Price']=Datafile['Price']/10000
    STATprice=np.append(STATprice,np.mean(Datafile.Price))
    STATstd=np.append(STATstd,np.std(Datafile.Price))
    STATvolume=np.append(STATvolume,sum(Datafile.Volume))
    STATtransactions=np.append(STATtransactions,np.size(Datafile.Price))
    Datafile['Duration']=Datafile.Time - Datafile.Time.shift(1)
    STATduration=np.append(STATduration,np.mean(Datafile.Duration))
    Datafile['logprice']=np.log(Datafile['Price'])
    Datafile=Datafile.groupby('Time').aggregate({'Time': 'last','Volume': 'sum', 'logprice': 'last', 'Duration':'sum'})
    Datafile['key']=np.floor(np.arange(np.shape(Datafile)[0])/agg)
    Datafile['Mean']=Datafile['logprice']
    Df=Datafile.groupby('key').aggregate({'Time': 'last','Volume': 'sum', 'logprice': 'last', 'Mean': 'mean', 'Duration':'sum'})
    Jumpsfrac=jumps_Lee(np.copy(Df['Mean']))
    #log returns
    Df['Ret']=Df.logprice - Df.logprice.shift(1)
    Df = Df.iloc[1: , :]
    Non0returns=Df.Ret[Df.Ret!=0]
    Squaredret=Non0returns**2
    Data=np.zeros_like(Df.Ret)
    Data[Df.Ret<0]=1
    Data[Df.Ret>0]=2
    zs=sum(Df.Ret==0)
    Data = Data[Data!=0]
    Data=Data-1
    Data=Data[0:]
    N=np.size(Data)
    zeros=zs/(N+zs)
    priceinc=abs(Df['logprice'].iat[-1]-Df['logprice'].iat[0])
    totalvolume=sum(Df.Volume)
    corr=abs(Non0returns.autocorr(lag=1))
    vol=abs(Squaredret.autocorr(lag=1))
    nu1, loc1, scale1 = stats.t.fit(Df.Ret)
    meanreturn=np.mean(Df.Ret)
    loc2=loc1
    loc1=abs(loc1)
    zerotimes=np.sum(Df.Duration==0)/np.size(Df.Duration)
    k=int(np.round(np.log(N)/np.log(A)/2))
    nu=(A**(k-1)-1)*(A-1)
    H, probs=KLstatistics(Data,k,A)
    sumofprob=(probs[0]+probs[-1])
    difofprob=abs((np.sum(Data==0)-np.sum(Data==1))/N)
    empquant=stats.chi2.cdf(H,df=nu)
    if empquant>0.99:
        #print(i)
        #print(res[i])
        zerosineff=np.append(zerosineff,zeros)
        autocorrineff=np.append(autocorrineff,corr)
        repeatineff=np.append(repeatineff,sumofprob)
        probdiffineff=np.append(probdiffineff,difofprob)
        volineff=np.append(volineff,vol)
        nuineff=np.append(nuineff,nu1)
        scaleineff=np.append(scaleineff,scale1)
        locineff=np.append(locineff,loc1)
        loc2ineff=np.append(loc2ineff,loc2)
        pricechangeineff=np.append(pricechangeineff,priceinc)
        sizeineff=np.append(sizeineff,N)
        jumpineff=np.append(jumpineff,Jumpsfrac)
        volumeineff=np.append(volumeineff,totalvolume)
        meanineff=np.append(meanineff,meanreturn)
        kineff=np.append(kineff,k)
        durationineff=np.append(durationineff,zerotimes)
    else:
        zeroseff=np.append(zeroseff,zeros)
        autocorreff=np.append(autocorreff,corr)
        repeateff=np.append(repeateff,sumofprob)
        probdiffeff=np.append(probdiffeff,difofprob)
        voleff=np.append(voleff,vol)
        nueff=np.append(nueff,nu1)
        scaleeff=np.append(scaleeff,scale1)
        loceff=np.append(loceff,loc1)
        loc2eff=np.append(loc2eff,loc2)
        pricechangeeff=np.append(pricechangeeff,priceinc)
        sizeeff=np.append(sizeeff,N)
        jumpeff=np.append(jumpeff,Jumpsfrac)
        volumeeff=np.append(volumeeff,totalvolume)
        meaneff=np.append(meaneff,meanreturn)
        keff=np.append(keff,k)
        durationeff=np.append(durationeff,zerotimes)
print(np.mean(STATprice),np.mean(STATstd),np.mean(STATvolume),np.mean(STATtransactions),np.mean(STATduration))
print('number of ineff days is', np.size(repeatineff))
print('pvalue<0.05 stands for different means')
print('size:')
print(np.mean(sizeineff),np.mean(sizeeff))
print(stats.ttest_ind(sizeineff, sizeeff, equal_var=False))
print('frac of 0s:')
print(np.mean(zerosineff),np.mean(zeroseff))
print(stats.ttest_ind(zerosineff, zeroseff, equal_var=False))
print('k:')
print(np.mean(kineff),np.mean(keff))
print(stats.ttest_ind(kineff, keff, equal_var=False))
print('sum of probs:')
print(np.mean(repeatineff),np.mean(repeateff))
print(stats.ttest_ind(repeatineff, repeateff, equal_var=False))
print('diff of probs:')
print(np.mean(probdiffineff),np.mean(probdiffeff))
print(stats.ttest_ind(probdiffineff, probdiffeff, equal_var=False))
print('price changes:')
print(np.mean(pricechangeineff),np.mean(pricechangeeff))
print(stats.ttest_ind(pricechangeineff, pricechangeeff, equal_var=False))
print('mean return:')
print(np.mean(meanineff),np.mean(meaneff))
print(stats.ttest_ind(meanineff, meaneff, equal_var=False))
print('autocorrelation:')
print(np.mean(autocorrineff),np.mean(autocorreff))
print(stats.ttest_ind(autocorrineff, autocorreff, equal_var=False))
print('autocorr for abs:')
print(np.mean(volineff),np.mean(voleff))
print(stats.ttest_ind(volineff, voleff, equal_var=False))
print('nu of t-distribution:')
print(np.mean(nuineff),np.mean(nueff))
print(stats.ttest_ind(nuineff, nueff, equal_var=False))
print('scale of t-distribution:')
print(np.mean(scaleineff),np.mean(scaleeff))
print(stats.ttest_ind(scaleineff, scaleeff, equal_var=False))
print('abs shift of t-distribution:')
print(np.mean(locineff),np.mean(loceff))
print(stats.ttest_ind(locineff, loceff, equal_var=False))
print('volume:')
print(np.mean(volumeineff),np.mean(volumeeff))
print(stats.ttest_ind(volumeineff, volumeeff, equal_var=False))
print('jumps:')
print(np.mean(jumpineff),np.mean(jumpeff))
print(stats.ttest_ind(jumpineff, jumpeff, equal_var=False))