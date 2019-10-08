import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

homePath = "/home/ollie/Dropbox/2019/PVB304/"

def getPaths(directory, name):
    paths = []
    for file in os.listdir(directory):
        if name in file:
            paths.append(directory + file)
    return paths

paths = getPaths(homePath, "run")
pathslr = getPaths(homePath, "lr")

nt = 200
deltaT = 1

# Load data file names
def filenames(path):
    sims = []
    for filename in os.listdir(path):
        if "np" in filename:
            sims.append(filename)
    sims.sort()
    return sims

def eigenvector(data):
    '''
    Determine the eigenvector for a simulation 
    '''
    Np = np.shape(data)[0]
    DT = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            sigma = np.sum(data[:,i]*data[:,j])
            DT[i][j] = 1/(2*nt*deltaT)*1/Np*sigma
    return np.linalg.eigvals(DT)

def FA(data, sims):
    FA = np.zeros((len(sims),2))
    i = 0
    for row in data:
        D1 = row[1]
        D2 = row[2]
        D3 = row[3]
        Davg = (D1 + D2 + D3)/3
        numerator = (D1 - Davg)**2 + (D2 - Davg)**2 + (D3 - Davg)**2
        denominator = D1**2 + D2**2 + D3**2
        FA[i][1] = np.sqrt(3/2) * np.sqrt(numerator/denominator)
        FA[i][0] = row[0]
        i += 1
    return FA

def eigVals(sims,path):
    eigValues = np.zeros((len(sims),4))
    for k in range(len(sims)):
        data1 = np.loadtxt(path + "/" + sims[k], delimiter = " ")
        eigValues[k][0] = np.shape(data1)[0]
        eigValues[k][1:] = eigenvector(data1)
    return eigValues

def simulation(sims, path):
    eigVal = eigVals(sims, path)
    fa = FA(eigVal, sims)
    return eigVal, fa

simLabels = filenames(paths[0])  
simLabelslr = filenames(pathslr[0]) 
        
def rawData(pathnames, labels):
    falist = []
    evlist = []
    for data in pathnames:
        evRun, faRun = simulation(labels, data)
        falist.append(faRun)
        evlist.append(evRun)
        
    faDF = np.concatenate(np.array(falist))  
    evDF = np.concatenate(np.array(evlist))
    return faDF, evDF

 

# %% Plotting Data for small runs 

faDF, evDF = rawData(paths, simLabels)
   
NP = faDF[:,0]
fa = faDF[:,1]

x = 1/np.sqrt(NP)
slope, intercept, r_value, p_value, std_err = stats.linregress(x, fa)

regline = intercept + x * slope
plt.plot(x, fa, 'o')
plt.plot(x, regline)

d = {'NP' :NP, 'fa': fa, 'x': x}
df = pd.DataFrame(data = d)

means = df.groupby('NP').mean()
err = df.groupby('NP').std()
means.plot.scatter(x = 'x', y = 'fa')

slope, intercept, r_value, p_value, std_err = stats.linregress(means.x, means.fa)
regline = intercept + means.x * slope
plt.plot(means.x, means.fa, 'o')
plt.errorbar(means.x, means.fa, yerr = err.fa, fmt = 'o')
plt.plot(means.x, regline)


# %% Plotting data for large runs

faDFlr, evDFlr = rawData(pathslr, simLabelslr)

NPlr = faDFlr[:,0]
falr = faDFlr[:,1]

xlr = 1/np.sqrt(NPlr)
slopelr, interceptlr, r_valuelr, p_valuelr, std_err_lr = stats.linregress(xlr, falr)

reglr = interceptlr + xlr * slopelr
plt.plot(xlr, falr, 'o')
plt.plot(xlr, reglr)

dlr = {'NP' :NPlr, 'fa': falr, 'xlr': xlr}
dflr = pd.DataFrame(data = dlr)

meanslr = dflr.groupby('NP').mean()
errlr = dflr.groupby('NP').std()

Mslopelr, Minterceptlr, Mr_valuelr, Mp_valuelr, Mstd_errlr = stats.linregress(meanslr.xlr, meanslr.fa)
Mreglr = Minterceptlr + meanslr.xlr * Mslopelr
plt.plot(meanslr.xlr, meanslr.fa, 'o')
plt.errorbar(meanslr.xlr, meanslr.fa, yerr = errlr.fa, fmt = 'o')
plt.plot(meanslr.xlr, Mreglr)


#%% both datasets combined

NPc = np.concatenate([NP, NPlr])
fac = np.concatenate([fa, falr])

xc = 1/np.sqrt(NPc)
slopec, interceptc, r_valuec, p_valuec, std_errc = stats.linregress(xc, fac)
regc = interceptc + xc * slopec
plt.plot(xc, fac, 'o')
plt.plot(xc, regc)


# %% Eigenvector and value analysis
