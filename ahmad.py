import numpy as np
import math

def ahmadStats(data1, data2):
    size1 = np.size(data1, 0)
    size2 = np.size(data2, 0)
    covar1 = np.matrix(np.cov(np.transpose(sample1)))
    covar2 = np.matrix(np.cov(np.transpose(sample2)))
    e1 = eNum(covar1, data)
    e2 = eNum(covar2, data)
    e12 = np.trace(covar1*covar2)
    q = e1 + e2 -2*e12
    covar = ((size1-1)*covar1 + (size2-1)*covar2)/(size1+size2-2)
    qvar = 4*(np.trace(covar**2))**2 * (1/size1 + 1/size2)**2
    return q/math.sqrt(qvar)

def eNum(s, data):
    size = np.size(data, 0)
    coeff = (size-1)/(size*(size-2)*(size-3))
    result = coeff*((size-1)*(size-2)*np.trace(s**2)+(np.trace(s))**2-partE(data))
    return result

def partE(data):
    size = np.size(data, 0)
    mean = np.mean(data, axis=0)
    deviation = data - mean
    result = 0 
    for i in range(size):
        x = deviation[i, :]
        result = result + x*np.transpoze(x)*np.transpose(x)*x
    result = result * size/(size-1)
    return result
       
if __name__ == '__main__'
dataDim = 100
sampleSize = 12
rep = 100
sigLevel = 0.05

meanVec = np.zeros(dataDim)
sig1 = 0.99*np.eye(dataDim) + 0.01*np.ones(dataDim, dataDim)
sig2 = 0.95*np.eye(dataDim) + 0.05*np.ones(dataDim, dataDim)

testStats = []
for i in range(100):
    sample1 = np.random.multivariate_normal(mean=meanVec, cov=sig1, size=sampleSize)
    sample2 = np.random.multivariate_normal(mean=meanVec, cov=sig2, size=sampleSize)
    testStats.append(ahmadStats(sample1, sample2))

testStats.sort()
cv = np.percentile(testStats, 95)


    
sampleCov1 = np.cov(np.transpose(sample1)) 
sampleCov2 = np.cov(np.transpose(sample2)) 
M = sampleCov1 - sampleCov2
Fq, *others = np.linalg.svd(M, full_matrices=False)


    
