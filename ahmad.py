import numpy as np
import math
import matplotlib.pyplot as plt

def ahmadStats(data1, data2):
    size1 = np.size(data1, 0)
    size2 = np.size(data2, 0)
    covar1 = np.matrix(np.cov(np.transpose(data1)))
    covar2 = np.matrix(np.cov(np.transpose(data2)))
    e1 = eNum(covar1, data1)
    e2 = eNum(covar2, data2)
    e12 = np.trace(covar1*covar2)
    q = e1 + e2 -2*e12
    covar = ((size1-1)*covar1 + (size2-1)*covar2)/float(size1+size2-2)
    qvar = 4*(np.trace(covar**2))**2 * (1.0/size1 + 1.0/size2)**2
    return q/math.sqrt(qvar)

def eNum(s, data):
    size = np.size(data, 0)
    coeff = float(size-1)/(size*(size-2)*(size-3))
    result = coeff*((size-1)*(size-2)*np.trace(s**2)+(np.trace(s))**2-partE(data))
    return result

def partE(data):
    size = np.size(data, 0)
    mean = np.mean(data, axis=0)
    deviation = data - mean
    result = 0 
    for i in range(size):
        x = np.matrix(deviation[i, :])
        result = result + x*x.T*x*x.T
    result = float(result * size)/(size-1)
    return result
       
if __name__ == '__main__':
    reduceDimSet = [10, 25, 50, 75, 90, 100] 
    dataDim = 100
    sampleSize = 12
    rep = 100
    sigLevel = 0.05
    reducedPowerSet = [] 
    prePowerSet = []
    meanVec = np.zeros(dataDim)
    sig1 = 0.99*np.eye(dataDim) + 0.01*np.ones((dataDim, dataDim))
    sig2 = 0.95*np.eye(dataDim) + 0.05*np.ones((dataDim, dataDim))
   
    for reducedDims in reducedDimSet:
        reducedTestStats = []
        preTestStas = []
        for i in range(rep):
            sample1 = np.random.multivariate_normal(mean=meanVec, cov=sig1, size=sampleSize)
            sample2 = np.random.multivariate_normal(mean=meanVec, cov=sig1, size=sampleSize)
            sampleCov1 = np.cov(np.transpose(sample1)) 
            sampleCov2 = np.cov(np.transpose(sample2)) 
            M = sampleCov1 - sampleCov2
            Fq, z, g = np.linalg.svd(M)
            Fq = Fq[:,0:reduceDims]
            reduced1 = (np.matmul(Fq.T,sample1.T)).T
            reduced2 = (np.matmul(Fq.T, sample2.T)).T
            reducedTestStats.append(ahmadStats(reduced1, reduced2))
            preTestStats.append(ahmadStats(sample1, sample2))
        reducedTestStats.sort()
        preTestStats.sort()
        reducedLower, reducedUpper = np.percentile(reducedTestStats, [2.5, 97.5])
        preLower, preUpper = np.percentile(preTestStats, [2.5, 97.5])
        reducedPower, prePower = 0, 0
        for i in range(rep):
            sample1 = np.random.multivariate_normal(mean=meanVec, cov=sig1, size=sampleSize)
            sample2 = np.random.multivariate_normal(mean=meanVec, cov=sig2, size=sampleSize)
            sampleCov1 = np.cov(np.transpose(sample1)) 
            sampleCov2 = np.cov(np.transpose(sample2)) 
            M = sampleCov1 - sampleCov2
            Fq, z, g = np.linalg.svd(M, full_matrices=False)
            Fq = Fq[:,0:reduceDims]
            reduced1 = (np.matmul(Fq.T,sample1.T)).T
            reduced2 = (np.matmul(Fq.T , sample2.T)).T
            stats = ahmadStats(reduced1, reduced2)
            reducedPower = float(reducedPower + (reducedUpper>=stats>=reducedLower))/rep
            stats = ahmadStats(sample1, sample2)
            prePower = float(prePower + (preUpper>=stats>=preLower))/rep
        
        reducedPowerSet.append(reducedPower)
        prePowerSet.append(prePower)

    reducedPowerSet = np.array(reducedPowerSet)
    prePowerSet = np.array(prePowerSet)
    powerDiff = reducedPowerSet - prePowerSet
    plt.plot(reduceDimSet, powerDiff)
    
