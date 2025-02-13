# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 13:06:31 2023

@author: Oliver
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import binned_statistic_dd
from scipy.linalg import block_diag
from scipy.interpolate import interp1d
from scipy import sparse as sp
from sklearn.decomposition import PCA
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.path import Path
from matplotlib import patheffects as pe

np.set_printoptions(precision=5, linewidth=200)
textwidth = 7.16 # inch
linewidth = 3.5
cm = 1 # 1/2.54 when text_width unit = cm
scale = 2
xscale = 1e3
yscale = 1e3

textsize=10
fontfamily='Times New Roman' # 'Times New Roman' or 'sans-serif'

local_textscale = scale / ( (linewidth*cm*scale) / (textwidth*cm*scale) ) # plot with textwidth for column

#%%
def readprofile(filename, flip=False, nprofiles=50):
    def updatetitle(string):
        plt.title(string, fontsize=16)
        plt.draw()
        
    with open(filename) as f:
        npoints = int( f.readline() )
    ## read chunks of file
    print(f'\nRead file with {npoints} Points:')
    chunk, chunksize = [], 2**18
    pbar = tqdm(total=npoints)
    for df in pd.read_csv(filename, sep=' ', header=None, names=["date", "time", "zenith", "distance", "intensity"], skiprows=1, chunksize=chunksize):
        chunk.append( df )
        pbar.update(chunksize)
    pbar.close()
    df = pd.concat( chunk ) # concatenate dataframes

    turningpoint = np.where( np.abs(np.diff(df['zenith'])) > np.pi )[0] # get points at 0/ (2*np.pi)
    npoints = int(np.median( turningpoint/ np.arange(1,len(turningpoint)+1) ) + 1) # n points per profile
    yz = np.c_[ np.sin(df['zenith']) * df['distance'], np.cos(df['zenith']) * df['distance'] ]
    if flip:
        yz[:,1] = -yz[:,1] # mirroring if on the backend
    
    ## area picking
    plt.clf()
    plt.scatter(yz[:npoints*nprofiles, 0], yz[:npoints*nprofiles,1], s=.2, c=df['intensity'][:npoints*nprofiles])
    plt.axis('equal')
    plt.xlabel('Y [m]'); plt.ylabel('Z [m]')
    ph = plt.fill(0, 0)
    while True:
        pts = []
        ph.pop().remove()
        updatetitle('Select polygon and click button when finished')
        pts = np.asarray( plt.ginput(-1, timeout=-1))
        ph = plt.fill(pts[:,0], pts[:,1], 'r', alpha=.5)
        updatetitle('If happy click button again else click with mouse to restart')
        if plt.waitforbuttonpress():
            break
    plt.close()
    
    pts = list(map(lambda i: (pts[i,0], pts[i,1]), range(pts.shape[0])))
    p = Path(pts)
    checker = p.contains_points(yz)
    df = df[checker]
    
    print('\nConvert timestamps of selected points')
    time, start = [], pd.to_datetime( np.min(df['time']) ).to_pydatetime()
    pbar = tqdm(total=len(df))
    for t in df['time']:
        time.append( (pd.to_datetime( t ).to_pydatetime() - start).total_seconds() )
        pbar.update(1)
    time = np.array(time)
    pbar.close()
    
    df['time'] = time
    df['X'] = df['time']
    df['Y'] = np.sin(df['zenith']) * df['distance']
    df['Z'] = np.cos(df['zenith']) * df['distance']
    if flip:
        df['Z'] = -df['Z'] # mirroring if on the backend
    
    df = df.drop(columns=['date'])
    df = df.dropna()
    df = df.sort_values(by=['time'])
    
    # df['profileID'] = np.append(0, np.float_(np.cumsum( np.abs(np.diff(df['zenith'])) > np.abs(np.max(df['zenith']) - np.min(df['zenith']))/2 ) ))
    df['profileID'] = np.append(0, np.float_(np.cumsum( np.abs(np.diff(df['zenith'])) > (1/50/2) )) ) # time difference bigger than half a profile
    
    df.to_csv(f'{filename[:-4]}_filtered.txt', sep=',', index=False)
    return df

def set_unique(binID):
    newbin = np.zeros_like(binID)
    for ID, i in enumerate(np.unique(binID)):
        newbin[binID==i] = ID
    return newbin

def shuffleID(binID):
    np.random.seed(42)
    tmp = np.unique(binID); np.random.shuffle(tmp)
    newbin = np.zeros_like(binID)
    for ID, i in enumerate(np.unique(binID)):
        newbin[binID==i] = tmp[i]
    return newbin

def clusterts(data, raster_size=1e-3, degree=2):
    if type(data) == tuple:
        StandPoint = data[0].reshape((-1,2))
        data = data[1]
    else:
        StandPoint = np.array([[0, 0]])
    pca = PCA()
    tmp_data = np.c_[ data['Y'], data['Z'] ]
    pca_data = pca.fit_transform( tmp_data )
    # pca.fit( tmp_data )
    # pca.components_ = pca.components_ * np.sign(np.diag(pca.components_)).reshape((-1,1)) # avoid mirroring
    # pca_data = pca.transform( tmp_data )
    StandPoint = pca.transform( StandPoint )
    
    print('\nCluster observed PointCloud')
    ID = data['profileID']
    transformed_data, transformation = np.zeros_like(pca_data), []
    for i in tqdm(np.unique(ID)):
        idx = ID == i
        tmp = pca_data[ idx ]
        par = np.polyfit(tmp[:,0], tmp[:,1], degree)
        alpha = np.arctan(par[-2])
        R = np.c_[ [np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)] ]
        transformed_data[idx] = np.dot(R, ( tmp - np.array([0, par[-1]]) ).T).T
        transformation.append( [par[-1], alpha] )
    
    bins = np.arange(np.min(transformed_data[:,0]), np.max(transformed_data[:,0])+raster_size, raster_size)
    bin_stat = binned_statistic_dd(transformed_data[:,0], transformed_data[:,1], bins=[bins])
    mean_data = np.c_[ np.linspace( np.min(bins)+raster_size/2, np.max(bins)-raster_size/2, len(bin_stat[0])),
                       bin_stat[0],
                      ]
    timeseriesID = set_unique(bin_stat[-1])
    
    
    DataFrame = data.copy()
    tmp_data = pca_data - StandPoint
    DataFrame['zenith'] = np.arctan2( tmp_data[:,0], tmp_data[:,1] ) % (2*np.pi)
    DataFrame['Y'] = pca_data[:,0]; DataFrame['Z'] = pca_data[:,1]
    DataFrame['timeseriesID'] = timeseriesID
    return (StandPoint, DataFrame), mean_data, np.vstack(transformation)

def baseN(knotvec, t, i, n, order): # compute basis for segments depending on t. i are knotvec indices
    """
    Parameters
    ----------
    knotvec : numpy array
        knotvector represents the segments (range 0-1)
    t : numpy array
        x-axis value (range 0-1)
    i : int
        should run from range( 0, (len(knotvec) - order) ) as input
    n : int
        stays order all the time (lookup)
    order : int
        variable for iteration
    
    Returns
    -------
    
    """
    if order-1 == 0:
        temp1 = np.logical_and( knotvec[i] <= t, t < knotvec[i + 1] )
        ## account for rounding issues original
        # temp2 = np.logical_and( i == 0, t < knotvec[n-1])
        # temp3 = np.logical_and( i == len(knotvec) - n-1, knotvec[-n] <= t )
        
        ## account for rounding issues (last 0 and first 1 are targeted)
        # temp2 = np.logical_and( i == 0, t < knotvec[n-1])
        # temp3 = np.logical_and( i == len(knotvec)-n, knotvec[-n] <= t )
        
        ## account for rounding issues (i = 0 & degree-1; knotvec first and last value of real segments!) (virtual knotvec values are nor considered)
        temp2 = np.logical_and( i == 0, t < knotvec[n-1])
        temp3 = np.logical_and( i== len(knotvec)-n-1, knotvec[-n] <= t )
        
        N = np.logical_or( temp1, np.logical_or( temp2, temp3) ).astype(float) # 0 or 1
    else:
        denom1 = knotvec[i + order-1] - knotvec[i] ## alpha_i **(n-1)
        denom2 = knotvec[i + order] - knotvec[i + 1] ## alpha_(i+1) **(n-1)
        
        term1 = 0.
        term2 = 0.
        if denom1 != 0:
            term1 = (t - knotvec[i]) / denom1  *  baseN(knotvec, t, i, n, order-1)
        if denom2 != 0:
            # term2 = (1 - (t - knotvec[i+1])/denom2) * baseN(knotvec, t, i+1, order-1) # original
            term2 = (knotvec[i+order] - t) / denom2  *  baseN(knotvec, t, i+1, n, order-1) #rearanged
        N = term1 + term2
    return N


def createAspline(x, segments=1, degree=3, method=None): # create A-matrix for weight coefficients
    
    ###########################################################################
    if segments < 1:
        segments = 1
    
    if method == 'periodic':
        dt = np.min(np.diff(x)) / (np.max(x)-np.min(x))
        knotvec = np.r_[ -np.flip(np.arange(dt, degree*dt+dt, dt)), np.linspace(0, 1, segments+1), 1+np.arange(dt, degree*dt+dt, dt) ]
    else:
        knotvec = np.r_[ np.repeat(0, degree), np.linspace(0, 1, segments+1), np.repeat(1, degree) ]

    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    
    order = degree + 1
    N = []
    for m in range(0, len(knotvec)-order):
        N.append( baseN(knotvec, x, m, order, order) )
    return np.array(N).T, knotvec


def getPSDstat(data, df=1e-2, grid_statistic='mean', interp_kind='linear'):
    if type(data) == tuple:
        data = data[1]
    
    fs = 1/ np.median( np.diff( list(map(lambda i: np.median(data['time'][data['profileID']==i]), np.unique(data['profileID']) )) ) )
    nperseg = int(fs/df)
    
    idx = data['timeseriesID'] == np.argmax(list(map(lambda i: (data['timeseriesID'] == i).sum(), np.unique(data['timeseriesID']) )))
    container = binned_statistic_dd(data['profileID'][idx], data['time'][idx], bins=[np.append(np.unique(data['profileID'][idx]), np.max(data['profileID'][idx])+1)], statistic=grid_statistic )
    len_smallest_timeseries = len(container[0])
    if nperseg > len_smallest_timeseries:
        nperseg = len_smallest_timeseries

    print(f'\n compute half spectrum with {nperseg} segments')
    ts, f, power = [], [], []
    for i in tqdm( np.unique(data['timeseriesID']) ):
        idx = data['timeseriesID'] == i
        container = binned_statistic_dd(data['profileID'][idx], data['time'][idx], bins=[np.append(np.unique(data['profileID'][idx]), np.max(data['profileID'][idx])+1)], statistic=grid_statistic )
        val = np.c_[ binned_statistic_dd(data['profileID'][idx], data['Z'][idx], binned_statistic_result=container, statistic=grid_statistic)[0],
                     binned_statistic_dd(data['profileID'][idx], data['Y'][idx], binned_statistic_result=container, statistic=grid_statistic)[0] ]
        t = container[0]
        if len(t) >= nperseg: # not enough data points
            if ~np.any(np.diff(np.unique(data['profileID'][idx])) > 1): # check equidistance
                ID = np.int_(container[1]).flatten()[:-1]
                timeinterpolator = interp1d(ID, t, kind=interp_kind) # fill profile gaps
                valueinterpolator1 = interp1d(t, val[:,0], kind=interp_kind) # fill observation gaps data['Z']
                valueinterpolator2 = interp1d(t, val[:,1], kind=interp_kind) # fill observation gaps data['Y']
                t = timeinterpolator( np.arange(np.min(ID), np.max(ID)+1, dtype=int) )
                val = np.c_[ valueinterpolator1(t), valueinterpolator2(t) ]
                
            ts.append( (i, t, val) )
            [f, Pxx] = signal.welch(val[:,0], fs=fs, nperseg=nperseg, noverlap=nperseg//2 )
            # list(map(lambda i: signal.welch(val[:,i], fs=fs, nperseg=nperseg)[1], range(val.shape[1]) ))
            power.append( Pxx )
        else:
            ts.append( (i, [], []) )
    # power = np.nanmedian( np.vstack(power), axis=0 )
    return ts, f, power

def get_frequency(frequency, power, number_of_frequencies=1, amplitude_threshold=1e-3):
    arg = signal.argrelmax(power, order=2)[0]
    arg = arg[np.flip(np.argsort(power[arg]))]
    idx = np.sqrt(power[arg]) > amplitude_threshold
    arg = arg[idx]
    if number_of_frequencies-1 == 0:
        f_select = np.array([ frequency[arg][0] ]) ## write max amplitudes in front of the last ]
        p_select = np.array([ power[arg][0] ])
    elif number_of_frequencies <= 0:
        f_select = frequency[arg]
        p_select = power[arg]
    else:
        f_select = frequency[arg][:number_of_frequencies]
        p_select = power[arg][:number_of_frequencies]
    return f_select, p_select

def gcd_frequency(frequency, gcd=[], tol=1e-1, max_test=10):
    if len(gcd) == 0:
        ## sort and enable reverse sorting
        argrevsort = np.array(list(map(lambda i: np.where( frequency == i )[0], np.sort(frequency) ))).flatten()
        frequency = np.sort(frequency)
        
        ############################################
        # compute gcd by with given tolerance
        f = np.flip(frequency)
        gcd = []
        factors = []
        for i in range(len(frequency)-1):
            tmp = f[i] / f[i+1:]
            test = tmp - np.int_(tmp)
            test = np.logical_or( test <= tol, test > (1-tol) )
            if any(test):
                gcd.append( np.min(f[i+1:][test]) )
                factors.append( np.max(tmp[test]) )
            else:
                gcd.append( f[i] )
                factors.append( 1. )
        gcd.append( f[-1] )
        factors.append( 1. )
        
        gcd = np.flipud( np.c_[ gcd, factors ] )
        gcd[:,1] = np.round(gcd[:,1])
                
        #############################################
        ## compute numpy gcd for all frequencies
        # logrange = np.logspace(2, max_test, num=max_test-1).reshape((-1,1))
        
        # gcd = []
        # for i in range(len(frequency)-1):
        #     vec = np.empty(len(frequency))
        #     vec[:] = np.nan
        #     # if i == 0:
        #     #     gcd.append(np.copy(vec) )
        #     tmp = np.float_(list(map(lambda fac: np.gcd(np.int_(frequency[i+1:]*fac), int(frequency[i]*fac)), logrange )))
        #     # error prevention
        #     tmp[tmp <= frequency[i]*logrange/2] = 0
        #     max_val = np.max(tmp, axis=0)
        #     max_val[max_val==0] = np.nan
        #     vec[i+1:] = frequency[i+1:] / ( max_val / logrange[np.argmax(tmp, axis=0)].flatten() )
        #     gcd.append( vec )
        
        # no_multiple = np.sum( np.isnan(np.array(gcd)), axis=0) == len(frequency)-1
        # tmp = np.nancumsum(np.array(gcd)[:, ~no_multiple], axis=0)
        # tmp[tmp==0] = np.nan
        # idx = np.nanargmin(tmp, axis=0)
        # multiple = tmp[ idx, np.arange(0, tmp.shape[1])]
        
        # factors = np.ones(len(frequency))
        # factors[~no_multiple] = multiple
        
        # gcd = frequency
        # gcd[~no_multiple] = frequency[ idx ]
        # gcd = np.c_[ gcd, np.round(factors) ]
        
        # ##########################################
        # check given results
        for i in range(len(frequency)):
            if gcd[i,0] != frequency[i]:
                check = np.where( gcd[i,0] == frequency )[0]
                gcd[i,1] = gcd[i,1] * gcd[check,1]
                gcd[i,0] = gcd[check,0]
        gcd = gcd[argrevsort, :]
    else:
        # map new estimated frequency to the gcd matrix
        if len(frequency) != len(np.unique(gcd[:,0])):
            print('\nlength of the frequency vector is not equal to length of sorted gcd matrix. \nThe new frequencies will be added to the matrix.')
            # np.min(np.abs(list(map(lambda i: gcd[:,0] - i, frequency )) ), axis=1)
        idx = list(map(lambda i: gcd[:,0] == i, np.unique(gcd[:,0])))
        for i, logi in enumerate(idx):
            gcd[logi, 0] = frequency[i]
    return gcd
    
    

def estimate1d(time, data, frequency, unknown=None, max_iter=100):
    time = np.array(time); data = np.array(data)
    decx = lambda x: [ x[0], x[1:-len(frequency)].reshape((2,-1)).T, x[-len(frequency):] ]
    if unknown == None:
        val = np.outer(time, 2*np.pi*frequency)
        A = np.c_[ np.ones(len(time)), np.cos(val), np.sin(val) ]
        x = np.dot( np.linalg.inv( np.dot(A.T, A) ), np.dot(A.T, data) )
        x = np.append(x, 2*np.pi*frequency)
        
        multiplier = ((np.max(data) - np.min(data))/2) / np.sum(np.sqrt(np.sum(decx(x)[1]**2, axis=1)))
        x[1:-len(frequency)] = x[1:-len(frequency)] * multiplier
    else:
        x = np.r_[ unknown[0], unknown[1].T.flatten(), unknown[2] ]
    
    history = []
    Pbb = sp.eye(data.shape[0]) * 1/(3.6e-3**2)  #Leica P50 accuracy
    for _ in range(max_iter):
        unknown = decx(x)
        val = np.outer(time, unknown[-1])
        A = np.c_[ np.ones(len(time)), np.cos(val), np.sin(val), time.reshape((-1,1)) * (unknown[1][:,1]*np.cos(val) - unknown[1][:,0]*np.sin(val)) ]
        w = data - ( unknown[0] + np.sum( unknown[1][:,0]*np.cos(val) + unknown[1][:,1]*np.sin(val), axis=1) )
        A = sp.csr_matrix( A )
        N = sp.linalg.inv( A.T.dot(Pbb).dot(A) ).tocsc()
        x += N.dot( A.T.dot(Pbb).dot(w) )
        # N = np.linalg.inv( np.dot(A.T, np.dot(Pbb, A)) )
        # x += np.dot( N, np.dot(A.T, w) )
        history.append(np.copy(x))
    unknown = decx(x)
    return unknown, N.toarray(), np.asarray(history)

def estimate2d(time, data, timeseriesID, frequency, max_iter=100):
    """
    Parameters
    ----------
    time : Series/ numpy array
        time vector for observations in data.
    data : Series/ numpy array
        observations of the oscillation.
    timeseriesID : Series/ numpy array
        ID vector to spatially distinguish.
    frequency : list or vector
        starting frequency values.
    max_iter : int, optional
        max number of iterations. The default is 100.

    Returns
    -------
    list
        [0] = offset; [1] = fourier coefficients; [2] = estimated frequencies.
    """
    time = np.array(time); data = np.array(data); timeseriesID = np.array( timeseriesID )
    decx = lambda x: [ x[:size], x[size:-len(frequency)].reshape((2, len(frequency), size)).swapaxes(1,2), x[-len(frequency):] ]
    
    availableIDs = np.unique(timeseriesID)
    size = availableIDs.shape[0]
    val = np.outer(time, 2*np.pi*frequency)
    A_offset = np.concatenate( list(map(lambda i: np.eye(1, size, k=int(np.argwhere(availableIDs==i)) ), timeseriesID)), axis=0 )
    A = np.c_[ A_offset, 
              np.concatenate(list(map(lambda i: A_offset * np.cos(val[:,i]).reshape((-1,1)), range(val.shape[1]) )), axis=1), # a coefficients
              np.concatenate(list(map(lambda i: A_offset * np.sin(val[:,i]).reshape((-1,1)), range(val.shape[1]) )), axis=1), # b coefficients
              ]
    A = sp.csr_matrix(A)
    x = sp.linalg.inv( A.T.dot(A) ).dot( A.T.dot(data) )
    x = np.append(x, 2*np.pi*frequency)
    
    unknown = decx(x)
    min_max = np.asarray(list(map(lambda i: [ np.min(data[timeseriesID==i]), np.max(data[timeseriesID==i]) ], np.unique(timeseriesID) )))
    multiplier = np.max((min_max[:,1] - min_max[:,0])/2) / np.max(np.sum( np.sqrt(unknown[1][0,:,:]**2 + unknown[1][1,:,:]**2), axis=1))
    x[size:-len(frequency)] = x[size:-len(frequency)] * multiplier
    
    history = []
    Pbb = sp.eye(data.shape[0]) * 1/(3.6e-3**2)  #Leica P50 accuracy
    for _ in tqdm( range(max_iter) ):
        unknown = decx(x)
        val = np.outer(time, unknown[-1])
        cval = np.cos(val); sval = np.sin(val)
        Ac = np.concatenate(list(map(lambda i: A_offset * cval[:,i].reshape((-1,1)), range(val.shape[1]) )), axis=1) # a coefficients
        As = np.concatenate(list(map(lambda i: A_offset * sval[:,i].reshape((-1,1)), range(val.shape[1]) )), axis=1) # b coefficients
        Af = np.hsplit( time.reshape((-1,1)) * (Ac * unknown[1][1,:,:].T.flatten() - As * unknown[1][0,:,:].T.flatten()), len(frequency) )
        A = np.c_[ A_offset, Ac, As, np.concatenate( list(map(lambda mat: np.sum(mat, axis=1, keepdims=True), Af)), axis=1) ]

        w = data - ( np.dot(A_offset, unknown[0]) + np.dot(Ac, unknown[1][0,:,:].T.flatten()) + np.dot(As, unknown[1][1,:,:].T.flatten()) )
        A = sp.csr_matrix( A )
        N = sp.linalg.inv( A.T.dot(Pbb).dot(A) ).tocsc()
        x += N.dot( A.T.dot(Pbb).dot(w) )
        # N = np.linalg.inv(np.dot(A.T, A))
        # x += np.dot(N, np.dot(A.T, w))
        history.append(np.copy(x))
    return decx(x), N.toarray(), np.asarray(history)

def estimatepoly(time, data, frequency, degree=3, max_iter=100):
    """
    Parameters
    ----------
    time : Series/ numpy array
        time vector for observations in data.
    data : numpy array [nx2]
        observed coordinates. First column object direction and second column oscillation direction. numpy.c_[ Y, Z ]
    frequency : list or vector
        starting frequency values.
    degree : int, optional
        polynomial degree. The default is 3.
    max_iter : int, optional
        max number of iterations. The default is 100.

    Returns
    -------
    list
        [0] mean,
        [1] [a&b, par, frequency]
        [2] frequency.
    Covariance matrix of unknown
    history of unknown vector

    """
    time = np.array(time); data = np.array(data)
    y = np.arange(np.min(data[:,0]), np.max(data[:,0]), 1e-3)
    decx = lambda x: [ x[:degree+1], x[degree+1:-len(frequency)].reshape((2, len(frequency), degree+1)).swapaxes(1, 2), x[-len(frequency):] ] # poly for amplitudes (a, b) x par x frequency
    
    val = np.outer(time, 2*np.pi*frequency)
    A_poly = np.fliplr( np.stack( list(map(lambda i: data[:,0]**i, range(degree+1) )), axis=1) )
    A = np.c_[ A_poly, 
              np.concatenate(list(map(lambda i: A_poly * np.cos(val[:,i]).reshape((-1,1)), range(val.shape[1]) )), axis=1),
              np.concatenate(list(map(lambda i: A_poly * np.sin(val[:,i]).reshape((-1,1)), range(val.shape[1]) )), axis=1),
              ]
    # A = sp.csr_matrix(A)
    # x = sp.linalg.inv( A.T.dot(A) ).dot( A.T.dot(data[:,1]) )
    x = np.dot( np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, data[:,1]) )
    x = np.append(x, 2*np.pi*frequency)
    
    max_stat = binned_statistic_dd( data[:,0], data[:,1], 'max', bins=[y] )
    min_stat = binned_statistic_dd( data[:,0], data[:,1], 'min', bins=[y] )
    
    a = np.stack(list(map(lambda i: np.polyval(decx(x)[1][0][:,i], y), range(len(frequency)) ))).T # [y.shape, len(frequency)]
    b = np.stack(list(map(lambda i: np.polyval(decx(x)[1][1][:,i], y), range(len(frequency)) ))).T
    
    # bin_edge = np.array(list(map(lambda i: np.mean(max_stat[1][0][i:i+1]), range(len(max_stat[1][0])-1) )))
    multiplier = np.nanmax((max_stat[0] - min_stat[0])/2) / np.max( np.sum(np.sqrt(a**2 + b**2), axis=1) )
    x[degree+1:-len(frequency)] = x[degree+1:-len(frequency)] * multiplier
    
    history = []
    Pbb = sp.eye(data.shape[0]) * 1/(3.6e-3**2)  #Leica P50 accuracy
    for _ in tqdm( range(max_iter) ):
        unknown = decx(x)
        val = np.outer(time, unknown[-1])
        a = np.stack( list(map(lambda i: np.polyval(unknown[1][0, :, i], data[:,0]), range(len(unknown[-1])) )) ).T
        b = np.stack( list(map(lambda i: np.polyval(unknown[1][1, :, i], data[:,0]), range(len(unknown[-1])) )) ).T
        A = np.c_[ A_poly, 
                  np.concatenate(list(map(lambda i: A_poly * np.cos(val[:,i]).reshape((-1,1)), range(val.shape[1]) )), axis=1),
                  np.concatenate(list(map(lambda i: A_poly * np.sin(val[:,i]).reshape((-1,1)), range(val.shape[1]) )), axis=1),
                  time.reshape((-1,1)) * (b*np.cos(val) - a*np.sin(val)),
                  ]
        A = sp.csr_matrix(A)
        w = data[:,1] - A[:,:-len(frequency)].dot( x[:-len(frequency)] )
        N = sp.linalg.inv(A.T.dot(Pbb).dot(A) ).tocsc()
        x += N.dot( A.T.dot(Pbb).dot(w) )
        # w = data[:,1] - np.dot(A[:,:-len(frequency)], x[:-len(frequency)])
        # N = np.linalg.inv( np.dot(A.T, A) )
        # x += np.dot( N, np.dot(A.T, w) )
        history.append(np.copy(x))
    history = np.array(history)
    # x = decx(x)
    # params[-1] --> degree=0 compatible with numpy.polyval
    # x[0] = np.flip(x[0]) 
    return decx(x), N.toarray(), np.asarray(history)

def estimatespline(time, data, frequency, segments=1, degree=3, method=None, max_iter=100):
    """
    Parameters
    ----------
    time : Series/ numpy array
        time vector for observations in data.
    data : numpy array [nx2]
        observed coordinates. First column object direction and second column oscillation direction. numpy.c_[ Y, Z ]
    frequency : list or vector
        starting frequency values.
    segments : int, optional
        how many spline segments are used to approximate the object. The default is 1. (equally spaces objects here)
    degree : int, optional
        spline degree. The default is 3.
    method : string, optional
        when set to "periodic" spaces first and last value of the control vector equally. The default is None.
    max_iter : int, optional
        max number of iterations. The default is 100.

    Returns
    -------
    list
        [0] mean,
        [1] [a&b, par, frequency]
        [2] frequency.
    Covariance matrix of unknown
    history of unknown vector

    """
    time = np.array(time); data = np.array(data)
    y = np.arange(np.min(data[:,0]), np.max(data[:,0]), 1e-3)
    decx = lambda x: [ x[:degree+segments], x[degree+segments:-len(frequency)].reshape((2, len(frequency), degree+segments)).swapaxes(1, 2), x[-len(frequency):] ] # poly for amplitudes (a, b) x par x frequency
    
    val = np.outer(time, 2*np.pi*frequency)

    A_spline, knotvec = createAspline(data[:,0], segments=segments, degree=degree, method=method ) # corresponding weights to the observations
    
    A = np.c_[ A_spline, 
              np.concatenate(list(map(lambda i: A_spline * np.cos(val[:,i]).reshape((-1,1)), range(val.shape[1]) )), axis=1),
              np.concatenate(list(map(lambda i: A_spline * np.sin(val[:,i]).reshape((-1,1)), range(val.shape[1]) )), axis=1), 
              ]
    # A = sp.csr_matrix(A)
    # x = sp.linalg.inv( A.T.dot(A) ).dot( A.T.dot(data[:,1]) )
    x = np.dot( np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, data[:,1]) )
    x = np.append(x, 2*np.pi*frequency)
    
    max_stat = binned_statistic_dd( data[:,0], data[:,1], 'max', bins=[y] )
    min_stat = binned_statistic_dd( data[:,0], data[:,1], 'min', bins=[y] )
    
    a = np.stack(list(map(lambda i: np.dot(A_spline, decx(x)[1][0, :,i]), range(len(frequency)) ))).T # [y.shape, len(frequency)]
    b = np.stack(list(map(lambda i: np.dot(A_spline, decx(x)[1][1, :,i]), range(len(frequency)) ))).T
    
    multiplier = np.nanmax((max_stat[0] - min_stat[0])/2) / np.max( np.sum(np.sqrt(a**2 + b**2), axis=1) )
    x[degree+segments:-len(frequency)] = x[degree+segments:-len(frequency)] * multiplier
    
    history = []
    Pbb = sp.eye(data.shape[0]) * 1/(3.6e-3**2)  #Leica P50 accuracy
    for _ in tqdm( range(max_iter) ):
        unknown = decx(x)
        val = np.outer(time, unknown[-1])
        a = np.stack( list(map(lambda i: np.dot(A_spline, unknown[1][0, :, i]), range(len(unknown[-1])) )) ).T
        b = np.stack( list(map(lambda i: np.dot(A_spline, unknown[1][1, :, i]), range(len(unknown[-1])) )) ).T
        A = np.c_[ A_spline, 
                  np.concatenate(list(map(lambda i: A_spline * np.cos(val[:,i]).reshape((-1,1)), range(val.shape[1]) )), axis=1),
                  np.concatenate(list(map(lambda i: A_spline * np.sin(val[:,i]).reshape((-1,1)), range(val.shape[1]) )), axis=1),
                  time.reshape((-1,1)) * (b*np.cos(val) - a*np.sin(val)),
                  ]
        A = sp.csr_matrix(A)
        w = data[:,1] - A[:,:-len(frequency)].dot( x[:-len(frequency)] )
        N = sp.linalg.inv(A.T.dot(Pbb).dot(A) ).tocsc()
        x += N.dot( A.T.dot(Pbb).dot(w) )
        # w = data[:,1] - np.dot(A[:,:-len(frequency)], x[:-len(frequency)])
        # N = np.linalg.inv( np.dot(A.T, A) )
        # x += np.dot( N, np.dot(A.T, w) )
        history.append(np.copy(x))
    history = np.array(history)
    return decx(x), N.toarray(), knotvec, np.asarray(history)


def estimatepoly_harmonic(time, data, frequency, tolerance=1e-1, degree=3, max_iter=100):
    time = np.array(time); data = np.array(data)
    y = np.arange(np.min(data[:,0]), np.max(data[:,0]), 1e-3)
    gcd = gcd_frequency(frequency, tol=tolerance)
    f = np.unique(gcd[:,0])
    idx = list(map(lambda i: gcd[:,0] == i, f))
    decx = lambda x: [ x[:degree+1], x[degree+1:-len(f)].reshape((2, len(frequency), degree+1)).swapaxes(1, 2), gcd_frequency(x[-len(f):], gcd=gcd) ] # poly for amplitudes (a, b) x par x frequency
    
    val = np.outer(time, 2*np.pi*frequency)
    A_poly = np.fliplr( np.stack( list(map(lambda i: data[:,0]**i, range(degree+1) )), axis=1) )
    A = np.c_[ A_poly, 
              np.concatenate(list(map(lambda i: A_poly * np.cos(val[:,i]).reshape((-1,1)), range(val.shape[1]) )), axis=1),
              np.concatenate(list(map(lambda i: A_poly * np.sin(val[:,i]).reshape((-1,1)), range(val.shape[1]) )), axis=1),
              ]
    x = np.dot( np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, data[:,1]) )
    x = np.append(x, 2*np.pi*f)
    
    max_stat = binned_statistic_dd( data[:,0], data[:,1], 'max', bins=[y] )
    min_stat = binned_statistic_dd( data[:,0], data[:,1], 'min', bins=[y] )
    
    a = np.stack(list(map(lambda i: np.polyval(decx(x)[1][0][:,i], y), range(len(frequency)) ))).T # [y.shape, len(frequency)]
    b = np.stack(list(map(lambda i: np.polyval(decx(x)[1][1][:,i], y), range(len(frequency)) ))).T
    
    # bin_edge = np.array(list(map(lambda i: np.mean(max_stat[1][0][i:i+1]), range(len(max_stat[1][0])-1) )))
    multiplier = np.nanmax((max_stat[0] - min_stat[0])/2) / np.max( np.sum(np.sqrt(a**2 + b**2), axis=1) )
    x[degree+1:-len(f)] = x[degree+1:-len(f)] * multiplier
    
    history = []
    for _ in tqdm( range(max_iter) ):
        unknown = decx(x)
        val = np.outer(time, unknown[-1][:,0] * unknown[-1][:,1])
        a = np.stack( list(map(lambda i: np.polyval(unknown[1][0, :, i], data[:,0]), range(len(unknown[-1])) )) ).T
        b = np.stack( list(map(lambda i: np.polyval(unknown[1][1, :, i], data[:,0]), range(len(unknown[-1])) )) ).T
        A = np.c_[ A_poly, 
                  np.concatenate(list(map(lambda i: A_poly * np.cos(val[:,i]).reshape((-1,1)), range(val.shape[1]) )), axis=1),
                  np.concatenate(list(map(lambda i: A_poly * np.sin(val[:,i]).reshape((-1,1)), range(val.shape[1]) )), axis=1),
                  np.concatenate(list(map(lambda i: np.sum(time.reshape((-1,1)) * (b[:,i] * np.cos(val[:,i]) - a[:,i] * np.sin(val[:,i])), axis=1), idx )) ),
                  ]
        
        w = data[:,1] - np.dot(A[:,:-len(f)], x[:-len(f)])
        x += np.dot( np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, w) )
        history.append(np.copy(x))
    history = np.array(history)
    return decx(x), np.asarray(history)
    
def estimatespline_harmonic(time, data, frequency, segments=1, degree=3, tolerance=1e-1, method=None, max_iter=100):
    time = np.array(time); data = np.array(data)
    y = np.arange(np.min(data[:,0]), np.max(data[:,0]), 1e-3)
    gcd = gcd_frequency(frequency, tol=tolerance)
    f = np.unique(gcd[:,0])
    idx = list(map(lambda i: gcd[:,0] == i, f))
    decx = lambda x: [ x[:degree+segments], x[degree+segments:-len(f)].reshape((2, len(frequency), degree+segments)).swapaxes(1, 2), gcd_frequency(x[-len(f):], gcd=gcd) ] # poly for amplitudes (a, b) x par x frequency

    val = np.outer(time, 2*np.pi* gcd[:,0] * gcd[:,1])
    A_spline, knotvec = createAspline(data[:,0], segments=segments, degree=degree, method=method ) # corresponding weights to the observations
    
    A = np.c_[ A_spline, 
              np.concatenate(list(map(lambda i: A_spline * np.cos(val[:,i]).reshape((-1,1)), range(val.shape[1]) )), axis=1),
              np.concatenate(list(map(lambda i: A_spline * np.sin(val[:,i]).reshape((-1,1)), range(val.shape[1]) )), axis=1), 
              ]
    x = np.dot( np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, data[:,1]) )
    x = np.append(x, 2*np.pi*f)
    
    max_stat = binned_statistic_dd( data[:,0], data[:,1], 'max', bins=[y] )
    min_stat = binned_statistic_dd( data[:,0], data[:,1], 'min', bins=[y] )
    
    a = np.stack(list(map(lambda i: np.dot(A_spline, decx(x)[1][0, :,i]), range(len(frequency)) ))).T # [y.shape, len(frequency)]
    b = np.stack(list(map(lambda i: np.dot(A_spline, decx(x)[1][1, :,i]), range(len(frequency)) ))).T
    
    multiplier = np.nanmax((max_stat[0] - min_stat[0])/2) / np.max( np.sum(np.sqrt(a**2 + b**2), axis=1) )
    x[degree+segments:-len(f)] = x[degree+segments:-len(f)] * multiplier
    
    history = []
    for _ in tqdm( range(max_iter) ):
        unknown = decx(x)
        val = np.outer(time, unknown[-1][:,0] * unknown[-1][:,1])
        
        a = np.stack( list(map(lambda i: np.dot(A_spline, unknown[1][0, :, i]), range(unknown[-1].shape[0]) )) ).T
        b = np.stack( list(map(lambda i: np.dot(A_spline, unknown[1][1, :, i]), range(unknown[-1].shape[0]) )) ).T
        
        A = np.c_[ A_spline, 
                  np.concatenate(list(map(lambda i: A_spline * np.cos(val[:,i]).reshape((-1,1)), range(val.shape[1]) )), axis=1),
                  np.concatenate(list(map(lambda i: A_spline * np.sin(val[:,i]).reshape((-1,1)), range(val.shape[1]) )), axis=1),
                  np.concatenate(list(map(lambda i: np.sum(time.reshape((-1,1)) * (b[:,i] * np.cos(val[:,i]) - a[:,i] * np.sin(val[:,i])), axis=1), idx )) ),
                  ]
        w = data[:,1] - np.dot(A[:,:-len(f)], x[:-len(f)])
        x += np.dot( np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, w) )
        history.append(np.copy(x))
    # history = np.array(history)
    return decx(x), knotvec, np.asarray(history)


def bsplineval(parameters, knotvec, x, degree=3, diff=False):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    
    order = degree + 1
    N = []
    for m in range(0, len(knotvec)-(order)):
        N.append( baseN(knotvec, x, m, order, order) )
    A_spline = np.array(N).T
    if diff:
        y = np.dot(A_spline**2, parameters)
    else:
        y = np.dot(A_spline, parameters)
    return y
    
def ab2AP(a, b, a_var=[], b_var=[]):
    A = np.sqrt( a**2 + b**2 )
    P = np.arctan2( b, a )
    
    A_var, P_var = np.array([]), np.array([])
    if len(a_var) != 0 and len(b_var) != 0:
        A_var = np.abs( ( a / np.sqrt(a**2 + b**2) )**2 * a_var + (b / np.sqrt(a**2 + b**2) )**2 * b_var )
        P_var = np.abs( ( a / (a**2 + b**2) )**2 * b_var - ( b / (a**2 + b**2) )**2 * a_var ) % (2*np.pi)
    return [A, P], [A_var, P_var]

    
def validate1d(time, unknown, Q=[]):
    time = np.array(time)
    val = np.outer(time, unknown[-1])
    offset = unknown[0]
    a = unknown[1][:,0]
    b = unknown[1][:,1]
    y = offset + np.sum( a * np.cos(val) + b * np.sin(val), axis=1)
    
    A = np.sqrt( a**2 + b**2 )
    P = np.arctan2( b, a )
    
    if len(Q) != 0:
        decx = lambda x: [ x[0], x[1:-len(unknown[-1])].reshape((2,-1)).T, x[-len(unknown[-1]):] ]
        
        F = block_diag([1], np.r_[ np.c_[ np.diag( a/np.sqrt(a**2+b**2) ), np.diag( b/np.sqrt(a**2+b**2) ) ], 
                    np.c_[ np.diag( -b/(a**2+b**2) ), np.diag( a/(a**2+b**2) ) ] ], np.eye(len(unknown[-1])) )
        Qyy = np.dot(F, np.dot(Q, F.T))
        var = decx( np.diag(Qyy) )
        offset_var = var[0]
        A_var = var[1][0,:]
        P_var = var[1][1,:]
        f_var = var[-1]
    else:
        offset_var, A_var, P_var, f_var = np.zeros(offset.shape), np.zeros(a.shape), np.zeros(b.shape), np.zeros(unknown[-1].shape)
        
    variance = []
    variance.append( (offset, offset_var) )
    variance.append( (A, A_var) )
    variance.append( (P, P_var) )
    variance.append( (unknown[-1] / (2*np.pi), f_var) )
    return y, variance

def validate2d(pca, unknown, Q=[]):
    time = np.array(pca['time'])
    val = np.outer(time, unknown[-1])
    offset = unknown[0]
    a = unknown[1][0,:,:]
    b = unknown[1][1,:,:]
    validate = pca.copy()
    validate['Z'] = unknown[0][pca['timeseriesID']] + np.sum( a[pca['timeseriesID'],:] * np.cos(val) + b[pca['timeseriesID'],:] * np.sin(val), axis=1 )
    
    A = np.sqrt( a**2 + b**2 )
    P = np.arctan2( b, a )
    if len(Q) != 0:
        decx = lambda x: [ x[:len(unknown[0])], x[len(unknown[0]):-len(unknown[-1])].reshape((2, len(unknown[-1]), len(unknown[0]))).swapaxes(1,2), x[-len(unknown[-1]):] ]
        
        F1a, F1b, F2a, F2b = [], [], [], []
        for i in range(len(unknown[-1])):
            F1a = block_diag(F1a, np.diag(a[:,i] / np.sqrt(a[:,i]**2 + b[:,i]**2)) )
            F1b = block_diag(F1b, np.diag(b[:,i] / np.sqrt(a[:,i]**2 + b[:,i]**2)) )
            F2a = block_diag(F2a, np.diag(-b[:,i] / (a[:,i]**2 + b[:,i]**2)) )
            F2b = block_diag(F2b, np.diag( a[:,i] / (a[:,i]**2 + b[:,i]**2)) )
        F = np.r_[ np.c_[ F1a[1:,:], F1b[1:,:] ], np.c_[ F2a[1:,:], F2b[1:,:] ] ]
        F = block_diag( np.eye(len(unknown[0])), F, np.eye(len(unknown[-1])) )
        Qyy = np.dot(F, np.dot( Q, F.T))
        var = decx( np.diag(Qyy) )
        offset_var = np.abs( var[0] )
        A_var = np.abs( var[1][0,:,:] )
        P_var = np.abs( var[1][1,:,:] ) % (2*np.pi)
        f_var = np.abs( var[-1] )
    else:
        offset_var, A_var, P_var, f_var = np.zeros(offset.shape), np.zeros(a.shape), np.zeros(b.shape), np.zeros(unknown[-1].shape)
    # AP, AP_var = ab2AP( a, b, a_var, b_var )
    variance = []
    variance.append( [np.array(list(map(lambda i: np.mean(pca['Y'][pca['timeseriesID']==i]), np.unique(pca['timeseriesID']) ))), 
                      offset, offset_var] )
    variance.append( [A, A_var] )
    variance.append( [P, P_var] )
    variance.append( (unknown[-1] / (2*np.pi), f_var) )
    return validate, variance

def validatepoly(pca, location, unknown, Q=[]):
    time, location = np.array(pca['time']), np.array(location)
    
    offset = np.polyval( unknown[0], np.array(pca['Y']) )
    a = np.stack( list(map(lambda i: np.polyval(unknown[1][0, :, i], np.array(pca['Y'])), range(len(unknown[-1])) )) ).T
    b = np.stack( list(map(lambda i: np.polyval(unknown[1][1, :, i], np.array(pca['Y'])), range(len(unknown[-1])) )) ).T
    
    val = np.outer( time, unknown[-1] )
    validate = pca.copy()
    validate['Z'] = offset + np.sum( a * np.cos(val) + b * np.sin(val), axis=1)
    
    offset = np.polyval( unknown[0], location )
    a = np.stack( list(map(lambda i: np.polyval(unknown[1][0, :, i], location), range(len(unknown[-1])) )) ).T
    b = np.stack( list(map(lambda i: np.polyval(unknown[1][1, :, i], location), range(len(unknown[-1])) )) ).T
    A = np.sqrt( a**2 + b**2 )
    P = np.arctan2( b, a )
    if len(Q) != 0:
        order = len(unknown[0]) # degree+1
        size = len(offset)
        decx = lambda x: [ x[:size], x[size:-len(unknown[-1])].reshape((2, len(unknown[-1]), size)).swapaxes(1, 2), x[-len(unknown[-1]):] ]
        
        fac = location.reshape((-1,1))**np.repeat(np.arange(order-1,-1,-1).reshape((1, -1)), len(location), axis=0)
        F1a, F1b, F2a, F2b = [], [], [], []
        for i in range(len(unknown[-1])):
            F1a = block_diag( F1a, fac * (a[:,i] / np.sqrt( a[:,i]**2 + b[:,i]**2 )).reshape((-1,1)) )
            F1b = block_diag( F1b, fac * (b[:,i] / np.sqrt( a[:,i]**2 + b[:,i]**2 )).reshape((-1,1)) )
            F2a = block_diag( F2a, fac * (-b[:,i] / (a[:,i]**2 + b[:,i]**2)).reshape((-1,1)) )
            F2b = block_diag( F2b, fac * ( a[:,i] / (a[:,i]**2 + b[:,i]**2)).reshape((-1,1)) )
        F = sp.csr_matrix( block_diag( fac, np.r_[ np.c_[ F1a[1:,:], F1b[1:,:] ], np.c_[ F2a[1:,:], F2b[1:,:] ] ], np.eye(len(unknown[-1])) ) )
        Qxx = sp.csr_matrix( Q )
        Qyy = F.dot(Qxx).dot(F.T)
        var = decx( Qyy.diagonal() )
        offset_var = np.abs( var[0] )
        A_var = np.abs( var[1][0,:,:] )
        P_var = np.abs( var[1][1,:,:] ) % (2*np.pi)
        f_var = np.abs( var[-1] )
    else:
        offset_var, A_var, P_var = np.zeros(offset.shape), np.zeros(a.shape), np.zeros(b.shape), np.zeros(unknown[-1].shape)
        
    variance = []
    # append (y, offset, offset_var)
    variance.append(  [location,
                      offset, 
                      offset_var] )
        
    # append (A, A_var)
    variance.append( [A,
                      A_var ] )
    
    # append (P, P_var)
    variance.append( [P,
                      P_var] )
        
    # append frequency
    variance.append( (unknown[-1] / (2*np.pi), f_var ) )
    return validate, variance

def validatespline(pca, location, knotvec, unknown, Q=[], degree=3):
    time, location = np.array(pca['time']), np.array(location)
    
    offset = bsplineval(unknown[0], knotvec, np.array(pca['Y']), degree=degree)
    a = np.stack( list(map(lambda i: bsplineval(unknown[1][0, :, i], knotvec, np.array(pca['Y']), degree=degree), range(len(unknown[-1])) )) ).T
    b = np.stack( list(map(lambda i: bsplineval(unknown[1][1, :, i], knotvec, np.array(pca['Y']), degree=degree), range(len(unknown[-1])) )) ).T
    
    val = np.outer( time, unknown[-1] )
    validate = pca.copy()
    validate['Z'] = offset + np.sum( a * np.cos(val) + b * np.sin(val), axis=1)
    
    offset = bsplineval(unknown[0], knotvec, location, degree=degree)
    a = np.stack( list(map(lambda i: bsplineval(unknown[1][0, :, i], knotvec, location, degree=degree), range(len(unknown[-1])) )) ).T
    b = np.stack( list(map(lambda i: bsplineval(unknown[1][1, :, i], knotvec, location, degree=degree), range(len(unknown[-1])) )) ).T
    A = np.sqrt( a**2 + b**2 )
    P = np.arctan2( b, a )
    if len(Q) != 0:
        # size_of_splines = len(knotvec)-degree-1
        size = len(location)
        decx = lambda x: [ x[:size], x[size:-len(unknown[-1])].reshape((2, len(unknown[-1]), size)).swapaxes(1, 2), x[-len(unknown[-1]):] ]
        
        # offset_var = np.abs( bsplineval( np.diag( Q[:size_of_splines, :size_of_splines] ), knotvec, location, degree=degree, diff=True ) )
        # f_var = np.diag( Q[-len(unknown[-1]):, -len(unknown[-1]):] )
        
        x = (location - np.min(location)) / (np.max(location) - np.min(location))
        N = []
        for m in range(0, len(knotvec)-(degree+1)):
            N.append( baseN(knotvec, x, m, degree+1, degree+1) )
        fac = np.array(N).T
        F1a, F1b, F2a, F2b = [], [], [], []
        for i in range(len(unknown[-1])):
            F1a = block_diag( F1a, fac * (a[:,i] / np.sqrt( a[:,i]**2 + b[:,i]**2 )).reshape((-1,1)) )
            F1b = block_diag( F1b, fac * (b[:,i] / np.sqrt( a[:,i]**2 + b[:,i]**2 )).reshape((-1,1)) )
            F2a = block_diag( F2a, fac * (-b[:,i] / (a[:,i]**2 + b[:,i]**2)).reshape((-1,1)) )
            F2b = block_diag( F2b, fac * ( a[:,i] / (a[:,i]**2 + b[:,i]**2)).reshape((-1,1)) )
        F = sp.csr_matrix( block_diag( fac, np.r_[ np.c_[ F1a[1:,:], F1b[1:,:] ], np.c_[ F2a[1:,:], F2b[1:,:] ] ], np.eye(len(unknown[-1])) ) )
        Qxx = sp.csr_matrix( Q )
        Qyy = F.dot(Qxx).dot(F.T)
        var = decx( Qyy.diagonal() )
        offset_var = np.abs( var[0] )
        A_var = np.abs( var[1][0,:,:] )
        P_var = np.abs( var[1][1,:,:] ) % (2*np.pi)
        f_var = np.abs( var[-1] )
    else:
        offset_var, A_var, P_var, f_var = np.zeros(offset.shape), np.zeros(a.shape), np.zeros(b.shape), np.zeros(unknown[-1].shape)
        
    variance = []
    # append (y, offset, offset_var)
    variance.append( [location,
                      offset, 
                      offset_var] )
    
    # append (A, A_var)
    variance.append( [A,
                      A_var] )
    
    # append (P, P_var)
    variance.append( [P,
                      P_var] )
    
    # append frequency
    variance.append( (unknown[-1] / (2*np.pi), f_var ) )

    return validate, variance



#%% Read data
# data = readprofile('curved_plane2.txt', flip=True, nprofiles=40)
data = pd.read_csv('curved_plane2_filtered.txt')
data_filter = list(np.where( np.logical_and(data['time'] >= 20, data['time'] <= 80) )[0])
data = data.filter( items=data_filter, axis=0)
# data = pd.read_csv('train_harmonic.csv')
# data_filter = list(np.where(data['intensity'] < 0.94)[0])
# data = data.filter(items=data_filter, axis=0)


#%% cluster observations
raster_size = 1e-2 # m, cm or mm
pca, mean, transformation = clusterts(data, raster_size=raster_size)
StandPoint, pca = pca

## reduce time series when number data points is too large
## filter_idx = np.sum(np.stack( list(map(lambda i: pca['timeseriesID'] == i, np.random.choice(np.unique(pca['timeseriesID']), np.min( [400, len(np.unique(pca['timeseriesID']))]), replace=False) ))), axis=0) > 0
## pca = pca[filter_idx]
## for i, b in enumerate(list(map(lambda i: pca['timeseriesID'] == i, np.unique(pca['timeseriesID']) ))):
##     pca[b] = i

#%% display object with pca axis
pca_scale=0.2

pca_transform = PCA()
pca_transform.fit( np.c_[data['Y'], data['Z']] )
pca_mean = pca_transform.mean_
pca_first = pca_transform.components_[0,:]
pca_second = pca_transform.components_[1,:]

fig, ax = plt.subplots(figsize=(linewidth*cm*scale, linewidth*cm*scale))
# data in m
ax.scatter(data['Y'], data['Z'], s=.2, c=shuffleID(pca['timeseriesID']), cmap='tab20', alpha=.2 )
ax.quiver(pca_mean[0], pca_mean[1], pca_first[0] * 1.2, pca_first[1] * 1.2, color='r', angles='xy', scale_units='xy', scale=1/pca_scale)
ax.text(pca_mean[0] + pca_first[0]*pca_scale+0.005, pca_mean[1] + pca_first[1]*pca_scale+0.032, r'$u$', fontsize=textsize*scale, fontfamily=fontfamily, c='r', path_effects=[pe.withStroke(linewidth=0.5, foreground='k')])
ax.quiver(pca_mean[0], pca_mean[1], pca_second[0], pca_second[1], color='lime', angles='xy', scale_units='xy', scale=2/pca_scale)
ax.text(pca_mean[0] + pca_second[0]*(pca_scale/2)-0.003, pca_mean[1] + pca_second[1]*(pca_scale/2)+0.004, r'$v$', fontsize=textsize*scale, fontfamily=fontfamily, c='lime', path_effects=[pe.withStroke(linewidth=0.5, foreground='k')])
ax.scatter( pca_mean[0], pca_mean[1],  s=20*scale, c='k')

# data cm
# ax.scatter(data['Y'] *1e2, data['Z']*1e2, s=.2, c=shuffleID(pca['timeseriesID']), cmap='tab20', alpha=.2 )
# ax.quiver(pca_mean[0]*1e2, pca_mean[1]*1e2, pca_first[0] *1e2* 1.2, pca_first[1] *1e2* 1.2, color='r', angles='xy', scale_units='xy', scale=1/pca_scale)
# ax.text(pca_mean[0]*1e2 + pca_first[0]*1e2*pca_scale+0.5, pca_mean[1]*1e2 + pca_first[1]*1e2*pca_scale+3.2, r'$u$', fontsize=textsize*scale, fontfamily=fontfamily, c='r', path_effects=[pe.withStroke(linewidth=0.5, foreground='k')])
# ax.quiver(pca_mean[0]*1e2, pca_mean[1]*1e2, pca_second[0]*1e2, pca_second[1]*1e2, color='lime', angles='xy', scale_units='xy', scale=2/pca_scale)
# ax.text(pca_mean[0]*1e2 + pca_second[0]*1e2*(pca_scale/2)-0.3, pca_mean[1]*1e2 + pca_second[1]*1e2*(pca_scale/2)+0.4, r'$v$', fontsize=textsize*scale, fontfamily=fontfamily, c='lime', path_effects=[pe.withStroke(linewidth=0.5, foreground='k')])
# ax.scatter( pca_mean[0]*1e2, pca_mean[1]*1e2,  s=20*scale, c='k')

# individual
# ax.plot( pca_transform.inverse_transform(oi)[:,0], pca_transform.inverse_transform(oi)[:,1], 'k', alpha=0.3) # individual results
# ax.plot( pca_transform.inverse_transform(np.c_[ oi[:,0], oi[:,1] + np.sum(ai, axis=1) ])[:,0], pca_transform.inverse_transform(np.c_[ oi[:,0], oi[:,1] + np.sum(ai, axis=1) ])[:,1], 'r', alpha=0.3 )

# # polynomial
# ax.plot( pca_transform.inverse_transform(op)[:,0], pca_transform.inverse_transform(op)[:,1], 'tab:orange' )
# ax.plot( pca_transform.inverse_transform(np.c_[ op[:,0], op[:,1] + np.sum(ap, axis=1) ])[:,0], pca_transform.inverse_transform(np.c_[ op[:,0], op[:,1] + np.sum(ap, axis=1) ])[:,1], 'r')

# # B-spline
# ax.plot( pca_transform.inverse_transform(ob)[:,0], pca_transform.inverse_transform(ob)[:,1], 'tab:cyan' )
# ax.plot( pca_transform.inverse_transform(np.c_[ ob[:,0], ob[:,1] + np.sum(ab, axis=1) ])[:,0], pca_transform.inverse_transform(np.c_[ ob[:,0], ob[:,1] + np.sum(ab, axis=1) ])[:,1], 'tab:cyan')

ax.xaxis.set_tick_params(labelsize=textsize*scale, labelfontfamily=fontfamily)
ax.yaxis.set_tick_params(labelsize=textsize*scale, labelfontfamily=fontfamily)
ax.axis('equal'); ax.spines[['right', 'top']].set_visible(False)
ax.set_xlabel('Y (m)', fontsize=textsize*scale, fontfamily=fontfamily); ax.set_ylabel('Z (m)', fontsize=textsize*scale, fontfamily=fontfamily)
# ax.set_xticklabels([]); ax.set_yticklabels([])
fig.tight_layout()
# ax.set_xlim([208, 231])


#%% display time series aggregated over different areas
idx0 = pca['time'] <= (np.min(pca['time'])+20)
idx1 = pca['timeseriesID'] == 22


fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(textwidth*cm*scale, linewidth*cm*scale))
ax[0].scatter(pca['time'][np.logical_and(idx0, idx1)] - np.min(pca['time']), pca['Z'][np.logical_and(idx0, idx1)]*1e3, s=.05, c='k', alpha=.5)
ax[1].scatter(pca['time'][idx0] - np.min(pca['time']), pca['Z'][idx0]*1e3, s=.05, c='k', alpha=.5)

ax[0].yaxis.set_tick_params(labelsize=textsize*local_textscale, labelfontfamily=fontfamily)
# ax[0].set_ylabel(r'$v$ (mm)', fontsize=textsize*local_textscale, fontfamily=fontfamily)
ax[0].spines[['right', 'top']].set_visible(False)

ax[1].xaxis.set_tick_params(labelsize=textsize*local_textscale, labelfontfamily=fontfamily)
ax[1].yaxis.set_tick_params(labelsize=textsize*local_textscale, labelfontfamily=fontfamily)
ax[1].set_xlabel('time (s)', fontsize=textsize*local_textscale, fontfamily=fontfamily)
ax[1].set_ylabel(r'$v$ (mm)', fontsize=textsize*local_textscale, fontfamily=fontfamily)
ax[1].yaxis.set_label_coords(-0.1, 1)
ax[1].xaxis.set_label_coords(0.5, -0.25)
ax[1].spines[['right', 'top']].set_visible(False)
fig.tight_layout()

#%% compute spectrum
[ts, frequency, psd] = getPSDstat(pca, df=5e-2)
power = np.median(np.vstack(psd), axis=0)

# used_ts = []
# for x in ts:
#     if len(x[1]) != 0:
#         used_ts.append( np.array(pca['timeseriesID']==x[0]) )
# used_ts = np.sum(np.vstack(used_ts), axis=0) > 0
# fig, ax = plt.subplots(figsize=(10,10))
# ax.scatter( data['Y'], data['Z'], s=.2, c=used_ts, cmap='RdYlGn'); plt.axis('equal')
# ax.set_xlabel('Y [m]', fontsize='x-large'); ax.set_ylabel('Z [m]', fontsize='x-large')
# ax.spines[['right', 'top']].set_visible(False)
# ax.set_xticklabels([]); ax.set_yticklabels([])

max_iter = 30
number_of_frequencies = 2
f, p = get_frequency(frequency, power, number_of_frequencies=number_of_frequencies, amplitude_threshold=1e-5)
f = np.array([0.79, 1.58])
# f = np.array([0.7, 1.5])
#f = np.append(f, np.array([0.78, 0.8]) )
print(f'\nfrequencies: {f}\n')

# plot half spectrum
threshold = 2e-3
cmap = plt.get_cmap('tab20')
colors = cmap( shuffleID(np.unique(pca['timeseriesID'])) )[:,:-1]

fig, ax = plt.subplots(figsize=(linewidth*cm*scale, linewidth/2*cm*scale))
# # list(map(lambda i: ax.plot(frequency, np.sqrt(psd[i]), color=colors[i,:], label=f'timeseries {i}'), range(len(psd))))
ax.plot(frequency, np.sqrt(power)*1e3, 'k', label='median'); #ax.plot(f, np.sqrt(p)*1e3, 'rx')
# ax.hlines( threshold*scale, np.min(frequency), np.max(frequency), colors='r')
ax.fill_between( frequency, np.repeat(threshold*1e3, len(frequency)), color='r', alpha=.2 )
ax.set_yscale('log')
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_xlabel('frequency (Hz)', fontsize=textsize*scale, fontfamily=fontfamily); ax.set_ylabel('amplitude (mm)', fontsize=textsize*scale, fontfamily=fontfamily)
ax.xaxis.set_tick_params(labelsize=textsize*scale, labelfontfamily=fontfamily), ax.yaxis.set_tick_params(labelsize=textsize*scale, labelfontfamily=fontfamily)
ax.set_ylim([np.min(np.sqrt(power))*1e3, np.max(np.sqrt(power))*1e3]), ax.set_xlim([np.min(frequency), np.max(frequency)])
ax.spines[['right', 'top']].set_visible(False)
fig.tight_layout()

#%% estimate every time series (mean)

print(f'\nEstimate all timeseries with {len(f)} frequencies')
x = None
unknown, thist, Q_single, variance, red = [], [], [], [], np.zeros((pca['Z'].shape[0], 2))
for i in tqdm( np.unique(pca['timeseriesID']) ):
    idx = pca['timeseriesID'] == i
    x, Qxx, history1 = estimate1d(pca['time'][idx], pca['Z'][idx], f, unknown=None, max_iter=max_iter)
    Q_single.append( Qxx )
    thist.append( np.copy(history1) )
    
    y, var = validate1d(pca['time'][idx], x, Qxx)
    red[idx,0] = idx.sum() - history1.shape[1]
    red[idx,1] = np.array(pca['Z'][idx]) - y
    variance.append(var)
    unknown.append( (i, np.mean(pca['Y'][idx]), x) )

vari = []
for i in range(len(variance[0])):
    vari.append( ( np.stack(list(map(lambda x: x[i][0], variance ))), np.stack(list(map(lambda x: x[i][1], variance ))) ) )
vari[0] = ( np.array(list(map(lambda i: np.mean(pca['Y'][pca['timeseriesID']==i]), np.unique(pca['timeseriesID']) ))), vari[0][0], vari[0][1] )

ID = np.unique(pca['timeseriesID'])
oi = np.stack( vari[0] ).T[:,:2]
ai = vari[1][0]
pi = vari[2][0]
fi = vari[3][0]

# plot single timeseries
# idx = np.logical_and( pca['time'] <= 30, pca['timeseriesID'] == 42)
# factor = 1e3
# fig, ax = plt.subplots(figsize=(10,10))
# ax.scatter( pca['time'][idx], pca['Z'][idx] * factor, s=.1, c='k' )
# ax.plot( pca['time'][idx], validate1d( pca['time'][idx], unknown[21][-1] ) * factor, color='tab:orange' )
# ax.spines[['right', 'top']].set_visible(False)
# ax.set_xlabel('time [sec]', fontsize='x-large'); ax.set_ylabel('Amplitude [mm]', fontsize='x-large')

## plot all timeseries
# idx = pca['time'] <= 30
# factor = 1e3
# cmap = plt.get_cmap('tab20')
# colors = cmap( shuffleID(ID) )[:,:-1]
# fig, ax = plt.subplots(figsize=(10,10))
# ax.scatter( pca['time'][idx], pca['Z'][idx] * factor, s=.2, c=shuffleID(pca['timeseriesID'][idx]), cmap='tab20')
# # ax.scatter( pca['time'][idx], pca['Z'][idx] * factor, s=.1, c='k' )
# list(map(lambda x: plt.plot(pca['time'][idx], validate1d( pca['time'][idx], x[-1] ) * factor, color=colors[x[0]], label=str(x[0]) ), unknown))
# ax.spines[['right', 'top']].set_visible(False)
# ax.set_xlabel('time [sec]', fontsize='x-large'); ax.set_ylabel('Amplitude [mm]', fontsize='x-large')

## plot mean, amplitude, phase and frequency
fig, ax = plt.subplots(4, 2, sharex=True, figsize=(textwidth*cm, textwidth*cm) )
gs = ax[0,0].get_gridspec()
ax[0,0].plot(oi[:,0], fi[:,0])
ax[0,1].plot(oi[:,0], fi[:,1])

ax[1,0].plot(oi[:,0], ai[:,0])
ax[1,1].plot(oi[:,0], ai[:,1])

ax[2,0].plot(oi[:,0], pi[:,0])
ax[2,1].plot(oi[:,0], pi[:,1])

ax[-1,0].remove(); ax[-1,1].remove()
axs = fig.add_subplot( gs[-1,:] )
axs.plot(oi[:,0], oi[:,1])

## absolut deviation (observations - computed)
threshold = -0.2
max_dev_idx = pca['Y'] > threshold
max_dev = np.array(list(map(lambda i: np.max(np.abs( red[pca['timeseriesID'] == i,1] )) * 1e3, np.unique(pca['timeseriesID']) )))
print(f'maximum deviation with individual: %.4f mm' % (np.max(max_dev[oi[:,0] > threshold])) )

#%% same frequency individual amplitude
# IDs = np.arange(0, 43)
IDs = np.unique(pca['timeseriesID'])
# IDs = np.random.choice( np.unique(pca['timeseriesID']), 300, replace=False ) # 453 for mm raster size
idx = np.stack(list(map(lambda i: np.array(pca['timeseriesID']==i), np.unique(IDs) ))).sum(axis=0).astype(bool)

print(f'\nEstimate {len(IDs)} timeseries together with {len(f)} frequencies')
x, Qxx2, history2 = estimate2d(pca['time'][idx], pca['Z'][idx], pca['timeseriesID'][idx], f, max_iter=max_iter)
vali2, vari2 = validate2d(pca[idx], x, Qxx2)

oi2 = np.c_[ vari2[0][0], vari2[0][1] ]
ai2 = vari2[1][0]
pi2 = vari2[1][0]

print(f'maximum deviation with same frequency: %.4f mm' % (np.max(np.abs(vali2['Z'][max_dev_idx] - pca['Z'][max_dev_idx]))*1e3) )

# fig, ax = plt.subplots(figsize=(10,10))
# ax.scatter(pca['Y'][idx], pca['Z'][idx], s=.2, c=shuffleID(pca['timeseriesID'])[idx], cmap='tab20', alpha=.2 )
# ax.plot( est_mean[IDs,0], est_mean[IDs,1], 'k', alpha=0.3 )
# ax.plot( est_mean[IDs,0], est_mean[IDs,1] + np.sum(est_amp[IDs,:], axis=1), 'r', alpha=0.3 )
# ax.plot( est_mean[IDs,0], x[0], 'k')
# ax.plot( est_mean[IDs,0], x[0] + amp, 'r')
# ax.axis('equal'); ax.spines[['right', 'top']].set_visible(False)
# ax.set_xlabel('Y [m]', fontsize='x-large'); ax.set_ylabel('Z [m]', fontsize='x-large')
# ax.set_xticklabels([]); ax.set_yticklabels([])

#%% polynom
print(f'\nEstimate all timeseries with {len(f)} frequencies and a polynomial')
idx = pca['timeseriesID'] > 0
x, Q_poly, historyp = estimatepoly( pca['time'][idx], np.c_[ pca['Y'][idx], pca['Z'][idx] ], f, degree=3, max_iter=max_iter)
valp, varp = validatepoly( pca, oi2[:,0], x, Q_poly)

op = np.c_[ varp[0][0], varp[0][1] ]
ap = varp[1][0]
pp = varp[2][0]

print(f'maximum deviation with polynomial connection: %.4f mm' % (np.max(np.abs(valp['Z'][max_dev_idx] - pca['Z'][max_dev_idx]))*1e3) )

#%% B-Spline
segments = 3
method = None
print(f'\nEstimate all timeseries with {len(f)} frequencies and B-spline with {segments} segments')
idx = pca['timeseriesID'] > 0
x, Q_spline, knotvec, historyb = estimatespline( pca['time'][idx], np.c_[ pca['Y'][idx], pca['Z'][idx] ], f, segments=segments, degree=3, method=method, max_iter=max_iter )
valb, varb = validatespline( pca, oi2[:,0], knotvec, x, Q_spline, degree=3)


ob = np.c_[ varb[0][0], varb[0][1] ]
ab = varb[1][0]
pb = varb[2][0]

print(f'maximum deviation with B-splines connection: %.4f mm' % (np.max(np.abs(valb['Z'][max_dev_idx] - pca['Z'][max_dev_idx]))*1e3) )

#%% convergence
fig, ax = plt.subplots(figsize=(linewidth*cm*scale, linewidth/2*cm*scale))
ax.plot( np.max(np.max(np.stack(list(map(lambda x: np.abs(np.diff(x, axis=0)), thist))), axis=0), axis=1), 'k', label='individual' )
ax.plot( np.max(np.abs(np.diff(history2, axis=0)), axis=1), 'tab:blue', label='same frequency' )
ax.plot( np.max(np.abs(np.diff(historyp, axis=0)), axis=1), 'tab:orange', label='polynomial' )
ax.plot( np.max(np.abs(np.diff(historyb, axis=0)), axis=1), 'tab:cyan', label='B-spline' )

ax.spines[['right', 'top']].set_visible(False)
ax.set_yscale('log')
# ax.yaxis.set_major_formatter( ticker.ScalarFormatter() )
ax.xaxis.set_tick_params(labelsize=textsize*scale, labelfontfamily=fontfamily); ax.yaxis.set_tick_params(labelsize=textsize*scale, labelfontfamily=fontfamily)

ax.set_xlabel('iterations', fontsize=textsize*scale, fontfamily=fontfamily); ax.set_ylabel('change in variables', fontsize=textsize*scale, fontfamily=fontfamily)
ax.grid(which = 'both', axis='y')
# ax.legend(loc='upper left', bbox_to_anchor=(-0.01, 1.5), ncols=4, prop={'family':fontfamily, 'size':textsize*scale})
ax.minorticks_on()
fig.tight_layout()

#%% time series observations with estimation results
IDs = [42, 22, 0]
labels = ['top', 'mid', 'low']
ycoord = -0.014 * textsize + 0.05  # 0.014 = pt2inch, 0.03 'sans-serif', + 0.05 'TNR'
fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(textwidth*cm*scale, linewidth*1.5*cm*scale))
for i, j in enumerate(IDs):
    # idx = np.logical_and( np.logical_and(pca['time'] >= 20.5, pca['time'] <= 21.7), pca['timeseriesID'] == j )
    idx = np.logical_and( np.logical_and(pca['time'] >= 21.05, pca['time'] <= 21.2), pca['timeseriesID'] == j ) # cut to maximum
    vali, _ = validate1d(pca['time'][idx], unknown[i][-1])

    ax[i].scatter( pca['time'][idx], pca['Z'][idx] * xscale, s=.5, c='k', alpha=.5)

    ax[i].plot( pca['time'][idx], vali * xscale, 'k', label='individual')
    ax[i].plot( vali2['time'][idx], vali2['Z'][idx] * xscale,'tab:blue', label='same frequency')
    ax[i].plot( valp['time'][idx], valp['Z'][idx] * xscale, 'tab:orange', label='polynomial')
    ax[i].plot( valb['time'][idx], valb['Z'][idx] * xscale, 'tab:cyan', label='B-spline')
    
    ax[i].grid(visible=True, which='both', axis='y')
    ax[i].spines[['right', 'top']].set_visible(False)
    ax[i].yaxis.set_tick_params(labelsize=textsize*local_textscale, labelfontfamily=fontfamily)
    ax[i].set_ylabel(f'{labels[i]} (mm)', fontsize=textsize*local_textscale, fontfamily=fontfamily)
    ax[i].yaxis.set_label_coords(ycoord, .5)
ax[i].xaxis.set_tick_params(labelsize=textsize*local_textscale, labelfontfamily=fontfamily)
ax[i].set_xlabel('time (s)', fontsize=textsize*local_textscale, fontfamily=fontfamily)
ax[i].xaxis.set_label_coords(0.5, -0.25) # -0.35 'sans-serif', -0.25 'TNR'

ax[i].set_ylim([24, 35] )
# lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
# lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
# fig.legend(lines_labels[0][0], lines_labels[0][1], loc='upper center', fontsize=text_size, ncol=4)
fig.tight_layout()

# ## one vs. all
# time_min = np.min(pca['time'])
# time_length = 20
# fig, ax = plt.subplots(2, 1, sharex=True, figsize=(16*cm, 13*cm))
# idx = np.logical_and( np.logical_and(pca['time'] >= time_min, pca['time'] <= time_min+time_length), pca['timeseriesID'] == 22 )
# ax[0].scatter( pca['time'][idx] - time_min, pca['Z'][idx]*scale, s=.1, c='k', alpha=.8)
# ax[0].spines[['right', 'top']].set_visible(False)
# ax[0].yaxis.set_tick_params(labelsize=text_size)
# ax[0].set_ylabel('$v$ [mm]', fontsize=text_size)

# idx = np.logical_and(pca['time'] >= time_min, pca['time'] <= time_min+time_length)
# ax[1].scatter( pca['time'][idx] - time_min, pca['Z'][idx]*scale, s=.1, c='k', alpha=.8)
# ax[1].spines[['right', 'top']].set_visible(False)
# ax[1].yaxis.set_tick_params(labelsize=text_size)
# ax[1].set_ylabel('$v$ [mm]', fontsize=text_size)
# ax[1].xaxis.set_tick_params(labelsize=text_size)
# ax[1].set_xlabel('time [sec]', fontsize=text_size)
# fig.tight_layout()

# ## timeseries difference ( observed - computed )
# val_diff = np.array( pca['Z'][idx] - valb['Z'][idx] )
# fig, ax = plt.subplots(figsize=(10,10))
# ax.scatter( pca['time'][idx], val_diff * factor, s=.1, c='k' )
# ax.spines[['right', 'top']].set_visible(False)
# ax.set_xlabel('time [sec]', fontsize='x-large'); ax.set_ylabel('Amplitude [mm]', fontsize='x-large')


#%% estimated parameters along PCA1
ycoord = -0.014 * textsize  + 0.05 # 0.014 = pt2inch
idx = np.ones(pca.shape[0], dtype=bool)
fig, ax = plt.subplots(ai.shape[1]+1, 1, sharex=True, figsize=(textwidth*cm*scale, linewidth*1.5*cm*scale))
for i in range(ai.shape[1]):
    ax[i].plot(oi[:,0] * 1e2, ai[:,i]*yscale, 'k', label='individual')
    ax[i].plot(oi2[:,0] * 1e2, ai2[:,i]*yscale, 'tab:blue', label='same frequency')
    ax[i].plot(op[:,0] * 1e2, ap[:,i]*yscale, 'tab:orange', label='polynomial')
    ax[i].plot(ob[:,0] * 1e2, ab[:,i]*yscale, 'tab:cyan', label='B-spline')
    
    tmp_minmax = ( np.min(list([ai[:,i].min(), ai2[:,i].min(), ap[:,i].min(), ab[:,i].min()])) * yscale,
                  np.max(list([ai[:,i].max(), ai2[:,i].max(), ap[:,i].max(), ab[:,i].max()])) * yscale )
    ax[i].vlines( oi[IDs,0] * 1e2, tmp_minmax[0], tmp_minmax[1], 'r')
    ax[i].spines[['right', 'top']].set_visible(False)
    ax[i].yaxis.set_tick_params(labelsize=textsize*local_textscale, labelfontfamily=fontfamily)
    ax[i].set_ylabel(f'$A_{i+1}$ (mm)', fontsize=textsize*local_textscale, fontfamily=fontfamily)
    ax[i].yaxis.set_label_coords(ycoord, .5)
    ax[i].grid(which = 'both', axis='y')
    ax[i].minorticks_on()

ax[i+1].plot(oi[:,0] * 1e2, oi[:,1]*yscale, 'k', label='individual time series')
ax[i+1].plot(oi2[:,0] * 1e2, oi2[:,1]*yscale, 'tab:blue', label='same frequency')
ax[i+1].plot(op[:,0] * 1e2, op[:,1]*yscale, 'tab:orange', label='polynomial')
ax[i+1].plot(ob[:,0] * 1e2, ob[:,1]*yscale, 'tab:cyan', label='B-spline')
tmp_minmax = ( np.min(list([oi[:,1].min(), oi2[:,1].min(), op[:,1].min(), ob[:,1].min()])) * yscale,
              np.max(list([oi[:,1].max(), oi2[:,1].max(),op[:,1].max(), ob[:,1].max()])) * yscale )
ax[i+1].vlines( oi[IDs,0] * 1e2, tmp_minmax[0], tmp_minmax[1], 'r')
ax[i+1].spines[['right', 'top']].set_visible(False)
ax[i+1].xaxis.set_tick_params(labelsize=textsize*local_textscale, labelfontfamily=fontfamily)
ax[i+1].yaxis.set_tick_params(labelsize=textsize*local_textscale, labelfontfamily=fontfamily)
ax[i+1].set_ylabel('mean (mm)', fontsize=textsize*local_textscale, fontfamily=fontfamily)
ax[i+1].set_xlabel(f'$u$ (cm)', fontsize=textsize*local_textscale, fontfamily=fontfamily)
ax[i+1].yaxis.set_label_coords(ycoord, .5)
ax[i+1].xaxis.set_label_coords(0.5, -0.25)
ax[i+1].grid(which = 'both', axis='y')
ax[i+1].minorticks_on()
fig.tight_layout()

#%% display estimated frequencies
ycoord = -0.014 * textsize - .1 # 0.014 = pt2inch

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(textwidth*cm*scale, linewidth*cm*scale))#, figsize=(text_width*cm, text_width/1.5*cm))
ax[0].plot(oi[:,0] * 1e2, vari[-1][0][:,0] - 0.79, 'k', label='individual')
ax[0].hlines(vari2[-1][0][0] - 0.79, oi[:,0].min() * 1e2, oi[:,0].max() * 1e2, 'tab:blue', label='same frequency')
ax[0].hlines(varp[-1][0][0] - 0.79, oi[:,0].min() * 1e2, oi[:,0].max() * 1e2, 'tab:orange', label='polynomial')
ax[0].hlines(varb[-1][0][0] - 0.79, oi[:,0].min() * 1e2, oi[:,0].max() * 1e2, 'tab:cyan', label='B-spline')
ax[0].grid(visible=True, which='both', axis='y')
ax[0].spines[['right', 'top']].set_visible(False)
ax[0].set_ylabel('$f_1$ (Hz)', fontsize=textsize*local_textscale, fontfamily=fontfamily)
ax[0].yaxis.set_label_coords(ycoord, .5)
ax[0].yaxis.set_tick_params(labelsize=textsize*local_textscale, labelfontfamily=fontfamily)

ax[1].plot(oi[:,0] * 1e2, vari[-1][0][:,-1] - 1.58, 'k')
ax[1].hlines(vari2[-1][0][-1] - 1.58, oi[:,0].min() * 1e2, oi[:,0].max() * 1e2, 'tab:blue')
ax[1].hlines(varp[-1][0][-1] - 1.58, oi[:,0].min() * 1e2, oi[:,0].max() * 1e2, 'tab:orange')
ax[1].hlines(varb[-1][0][-1] - 1.58, oi[:,0].min() * 1e2, oi[:,0].max() * 1e2, 'tab:cyan')
ax[1].grid(visible=True, which='both', axis='y')
ax[1].spines[['right', 'top']].set_visible(False)
ax[1].set_ylabel('$f_2$ (Hz)', fontsize=textsize*local_textscale, fontfamily=fontfamily)
ax[1].set_xlabel('$u$ (cm)', fontsize=textsize*local_textscale, fontfamily=fontfamily)
ax[1].yaxis.set_label_coords(ycoord, .5)
ax[1].xaxis.set_label_coords(0.5, -0.25)
ax[1].xaxis.set_tick_params(labelsize=textsize*local_textscale, labelfontfamily=fontfamily)
ax[1].yaxis.set_tick_params(labelsize=textsize*local_textscale, labelfontfamily=fontfamily)
# plt.legend(['individual', 'same frequency', 'polynomial', 'B-spline'], loc='upper center', bbox_to_anchor=(1, 2.05), ncols=4, fontsize=textsize*scale)
fig.tight_layout()

#%% overall estimation wrt observations
fig, ax = plt.subplots(figsize=(textwidth*cm*scale, textwidth/2*cm*scale))
ax.scatter(pca['Y'][idx] * 1e2, pca['Z'][idx] * yscale, s=.1, c=shuffleID(pca['timeseriesID'])[idx], cmap='tab20', alpha=.1 )
# individual
# ax[3].plot( oi[:,0], oi[:,1] * scale, 'k', alpha=.3, label='individual time series' )
# ax[3].plot( oi[:,0], ( oi[:,1] + np.sum(ai, axis=1) ) * scale, 'k' )
# ax[3].fill_between( oi[:,0], ( oi[:,1] + np.sqrt(vari[0][2]) ) * scale, ( oi[:,1] - np.sqrt(vari[0][2]) ) * scale, color='k', alpha=.2)
# ax[3].fill_between( oi[:,0], ( oi[:,1] + np.sum(ai, axis=1) + np.sqrt(np.sum(vari[1][1], axis=1)) ) * scale, ( oi[:,1] + np.sum(ai, axis=1) - np.sqrt(np.sum(vari[1][1], axis=1)) ) * scale, color='k', alpha=.2 )

# same frequency
ax.plot(oi2[:,0] * 1e2, oi2[:,1] * yscale, 'tab:blue', label='same frequency')
ax.plot( oi2[:,0] * 1e2, (oi2[:,1] + np.sum(ai2, axis=1) ) * yscale, 'tab:blue')
# ax[3].fill_between( oi2[:,0], ( oi2[:,1] + np.sqrt(vari2[0][2]) ) * scale, ( oi2[:,1] - np.sqrt(vari2[0][2]) ) * scale, color='tab:blue', alpha=.2)
# ax[3].fill_between( oi2[:,0], ( oi2[:,1] + np.sum(ai2, axis=1) + np.sqrt(np.sum(vari2[1][1], axis=1)) ) * scale, ( oi2[:,1] + np.sum(ai2, axis=1) - np.sqrt(np.sum(vari2[1][1], axis=1)) ) * scale, color='tab:blue', alpha=.2)

# polynomial
ax.plot( op[:,0] * 1e2, op[:,1] * yscale, 'tab:orange', label='polynomial' )
ax.plot( op[:,0] * 1e2, ( op[:,1] + np.sum(ap, axis=1) ) * yscale, 'tab:orange')
# ax[3].fill_between( op[:,0], ( op[:,1] + np.sqrt(varp[0][2]) ) * scale, ( op[:,1] - np.sqrt(varp[0][2]) ) * scale, color='tab:orange', alpha=.2 )
# ax[3].fill_between( op[:,0], ( op[:,1] + np.sum(ap, axis=1) + np.sqrt(np.sum(varp[1][1], axis=1)) ) * scale, ( op[:,1] + np.sum(ap, axis=1) - np.sqrt(np.sum(varp[1][1], axis=1)) ) * scale, color='tab:orange', alpha=.2 )


#B-spline
ax.plot( ob[:,0] * 1e2, ob[:,1] * yscale, 'tab:cyan', label='B-spline' )
ax.plot( ob[:,0] * 1e2, ( ob[:,1] + np.sum(ab, axis=1) ) * yscale, 'tab:cyan')
# ax.fill_between( ob[:,0], ( ob[:,1] + np.sqrt(varb[0][2]) ) * scale, ( ob[:,1] - np.sqrt(varb[0][2]) ) * scale, color='tab:cyan', alpha=.2 )
# ax.fill_between( ob[:,0], ( ob[:,1] + np.sum(ab, axis=1) + np.sqrt(np.sum(varb[1][1], axis=1)) ) * scale, ( ob[:,1] + np.sum(ab, axis=1) - np.sqrt(np.sum(varb[1][1], axis=1)) ) * scale, color='tab:cyan', alpha=.2 )

ax.spines[['right', 'top']].set_visible(False)
ax.yaxis.set_tick_params(labelsize=textsize*scale, labelfontfamily=fontfamily)
ax.xaxis.set_tick_params(labelsize=textsize*scale, labelfontfamily=fontfamily)
ax.set_ylabel('$v$ (mm)', fontsize=textsize*scale, fontfamily=fontfamily)
ax.set_xlabel('$u$ (cm)', fontsize=textsize*scale, fontfamily=fontfamily)
x = np.array([np.min(oi[:,0]), np.min(np.abs(oi[:,0])), np.max(oi[:,0])])
# ax.set_xticks( x, [np.round(x[0]*1e2, 1), 0, np.round(x[-1]*1e2, 1)] )
# ax.yaxis.set_label_coords(ycoord, .5)
ax.grid(which = 'major', axis='y')

# lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
# #lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
# fig.legend(lines_labels[0][0], lines_labels[0][1], loc='upper center', fontsize=text_size*2, ncol=4)

fig.tight_layout()
# ax.set_xticklabels([]); ax.set_yticklabels([])


#%% standard deviation
ycoord = -0.014 * textsize - 0.01  # 0.014 = pt2inch
stdo = {
        'individual': tuple(np.round(np.sqrt(vari[0][2])*yscale, 2)),
        'same frequency': tuple(np.round(np.sqrt(vari2[0][2])*yscale, 2)),
        'polynomial': tuple(np.round(np.sqrt(varp[0][2])*yscale, 2)),
        'B-spline':   tuple(np.round(np.sqrt(varb[0][2])*yscale, 2)),
        }

stda1 = {
        'individual': tuple(np.round(np.sqrt(vari[1][1][:,0])*yscale, 2)),
        'same frequency': tuple(np.round(np.sqrt(vari2[1][1][:,0])*yscale, 2)),
        'polynomial': tuple(np.round(np.sqrt(varp[1][1][:,0])*yscale, 2)),
        'B-spline':   tuple(np.round(np.sqrt(varb[1][1][:,0])*yscale, 2)),
        }

stda2 = {
        'individual': tuple(np.round(np.sqrt(vari[1][1][:,1])*yscale, 2)),
        'same frequency': tuple(np.round(np.sqrt(vari2[1][1][:,1])*yscale, 2)),
        'polynomial': tuple(np.round(np.sqrt(varp[1][1][:,1])*yscale, 2)),
        'B-spline':   tuple(np.round(np.sqrt(varb[1][1][:,1])*yscale, 2)),
        }

# std = {
        # 'individual Amplitude 1': tuple(np.round(np.sqrt(vari[1][1][:,0])*scale, 2)),
        # 'individual Amplitude 2': tuple(np.round(np.sqrt(vari[1][1][:,1])*scale, 2)),
        # 'individual Amplitude 3': tuple(np.round(np.sqrt(vari[1][1][:,2])*scale, 2)),
        # 'individual Amplitude 4': tuple(np.round(np.sqrt(vari[1][1][:,3])*scale, 2)),
        # 'polynomial Amplitude 1': tuple(np.round(np.sqrt(varp[1][1][:,0])*scale, 2)),
        # 'polynomial Amplitude 2': tuple(np.round(np.sqrt(varp[1][1][:,1])*scale, 2)),
        # 'B-spline Amplitude 1':   tuple(np.round(np.sqrt(varb[1][1][:,0])*scale, 2)),
        # 'B-spline Amplitude 2':   tuple(np.round(np.sqrt(varb[1][1][:,1])*scale, 2)),
        # 'B-spline Amplitude 3':   tuple(np.round(np.sqrt(varb[1][1][:,2])*scale, 2)),
        # 'B-spline Amplitude 4':   tuple(np.round(np.sqrt(varb[1][1][:,3])*scale, 2)),
        # }

label = np.round(oi[:,0] * 1e2, 1)
x = np.arange(len(label))

fig, ax = plt.subplots(ai.shape[1]+1, 1, sharex=True, figsize=(textwidth*cm*scale, linewidth*1.5*cm*scale) )
width = 1/(len(stda1)+1)
multiplier = 0
for attribute, measurement in stda1.items():
    offset = width * multiplier
    if attribute == 'individual':
        color = 'k'; alpha=1
    elif attribute == 'same frequency':
        color = 'tab:blue'; alpha=1
    elif attribute == 'polynomial':
        color = 'tab:orange'; alpha=1
    elif attribute == 'B-spline':
        color = 'tab:cyan'; alpha=1
    else:
        color = 'tab:red'; alpha=1
    ax[0].bar(x + offset, measurement, width, color=color, label=attribute)
    multiplier += 1
ax[0].spines[['right', 'top']].set_visible(False)
ax[0].set_ylabel('$\sigma_{A_1}$ (mm)', fontsize=textsize*local_textscale, fontfamily=fontfamily)
ax[0].yaxis.set_tick_params(labelsize=textsize*local_textscale, labelfontfamily=fontfamily)
ax[0].yaxis.set_label_coords(ycoord, .5)
ax[0].grid(which='both', axis='y')
ax[0].minorticks_on()

multiplier = 0
for attribute, measurement in stda2.items():
    offset = width * multiplier
    if attribute == 'individual':
        color = 'k'; alpha=1
    elif attribute == 'same frequency':
        color = 'tab:blue'; alpha=1
    elif attribute == 'polynomial':
        color = 'tab:orange'; alpha=1
    elif attribute == 'B-spline':
        color = 'tab:cyan'; alpha=1
    else:
        color = 'tab:red'; alpha=1
    ax[1].bar(x + offset, measurement, width, color=color, label=attribute)
    multiplier += 1
ax[1].spines[['right', 'top']].set_visible(False)
ax[1].set_ylabel('$\sigma_{A_2}$ (mm)', fontsize=textsize*local_textscale, fontfamily=fontfamily)
ax[1].yaxis.set_tick_params(labelsize=textsize*local_textscale, labelfontfamily=fontfamily)
ax[1].yaxis.set_label_coords(ycoord, .5)
ax[1].grid(which='both', axis='y')
ax[1].minorticks_on()
    

multiplier = 0
for attribute, measurement in stdo.items():
    offset = width * multiplier
    if attribute == 'individual':
        color = 'k'; alpha=1
    elif attribute == 'same frequency':
        color = 'tab:blue'; alpha=1
    elif attribute == 'polynomial':
        color = 'tab:orange'; alpha=1
    elif attribute == 'B-spline':
        color = 'tab:cyan'; alpha=1
    else:
        color = 'tab:red'; alpha=1
    ax[2].bar(x + offset, measurement, width, color=color, label=attribute)
    multiplier += 1
ax[2].spines[['right', 'top']].set_visible(False)
ax[2].set_ylabel('$\sigma_{mean}$ (mm)', fontsize=textsize*local_textscale, fontfamily=fontfamily)
ax[2].yaxis.set_tick_params(labelsize=textsize*local_textscale, labelfontfamily=fontfamily)
ax[2].yaxis.set_label_coords(ycoord, .5)
ax[2].grid(which='both', axis='y')
ax[2].minorticks_on()
    
ax[2].set_xlabel('$u$ (cm)', fontsize=textsize*local_textscale, fontfamily=fontfamily)
ax[2].xaxis.set_label_coords(0.5, -0.25)
ax[2].xaxis.set_tick_params(labelsize=textsize*local_textscale, labelfontfamily=fontfamily)
# ax.set_title('standard deviation offset', fontsize='xx-large')
# ax.set_xticks(x[::3] + width, label[::3])
ax[2].set_xticks( np.array([x[0], x[np.argmin(np.abs(label))], x[-1]]) + width, np.array([label[0], 0, label[-1]]))
# ax.legend(loc='upper left', fontsize='x-large')
fig.tight_layout()

#%% residuals by location and time
location_time_space = np.c_[ pca['time'], pca['Y'] ]
bins = [ np.arange(np.min(pca['time']), np.max(pca['time']), 0.05), oi[:,0] ]
statistic = 'mean'
resi =  binned_statistic_dd( location_time_space, red[:,1], statistic='mean', bins=bins )[0].T * scale
resi2 = binned_statistic_dd( location_time_space, np.array(pca['Z'] - vali2['Z']), statistic=statistic, bins=bins )[0].T * scale
resp =  binned_statistic_dd( location_time_space, np.array(pca['Z'] - valp['Z']), statistic=statistic, bins=bins )[0].T * scale
resb =  binned_statistic_dd( location_time_space, np.array(pca['Z'] - valb['Z']), statistic=statistic, bins=bins )[0].T * scale
T, S = np.meshgrid(bins[0][:-1] - np.min(pca['time']), bins[1][:-1])
S = S * 1e2
maxlim = np.round(np.max(np.abs([resi, resi2, resp, resb]))*10) / 10
maxlim = np.floor(np.max(np.abs([resi, resi2, resp, resb])))
fig, ax = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(16*cm, 21*cm))
pm0 = ax[0].pcolormesh(T, S, resi, clim=(-maxlim, maxlim))
ax[0].set_ylabel('$u$ [cm]', fontsize=textsize)
pm1 = ax[1].pcolormesh(T, S, resi2, clim=(-maxlim, maxlim))
ax[1].set_ylabel('$u$ [cm]', fontsize=textsize)
pm2 = ax[2].pcolormesh(T, S, resp, clim=(-maxlim, maxlim))
ax[2].set_ylabel('$u$ [cm]', fontsize=textsize)
pm3 = ax[3].pcolormesh(T, S, resb, clim=(-maxlim, maxlim))
ax[3].set_ylabel('$u$ [cm]', fontsize=textsize)
ax[3].set_xlabel('time [sec]', fontsize=textsize)
cb = fig.colorbar(pm3, ax=ax.ravel().tolist(), location='right')
cb.set_label('residual [mm]', fontsize=textsize)

# ## run time
# run = np.array([ [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
#                 [2, 4, 6, 7, 10, 11, 12, 14, 17, 18, 21, 23],
#                 [24, 49, 75, 100, 125, 158, 181, 204, 226, 252, 277, 300],
#                 [3, 9, 14, 18, 22, 27, 31, 36, 40, 44, 49, 54],
#                 [3, 8, 13, 17, 20, 25, 28, 33, 37, 40, 45, 48],
#                 ]).T

# list(map(lambda i: plt.plot(run[:,0], run[:,i]), range(1,5)))
# plt.legend(['individual', 'same frequency', 'polynomial', 'B-spline'])
# plt.xlabel('timeseries duration [sec]'); plt.ylabel('run time [sec]')

#%% residuals
_, ax = plt.subplots(figsize=(textwidth*cm, textwidth/1.5*cm))
ax.plot( np.unique(S), resi.max(1), 'k', label='individual' )
ax.plot( np.unique(S), resi2.max(1), 'tab:blue', label='same frequency' )
ax.plot( np.unique(S), resp.max(1), 'tab:orange', label='polynomial' )
ax.plot( np.unique(S), resb.max(1), 'tab:cyan', label='B-splines' )
ax.set_xlabel('$u$ [cm]', fontsize=textsize)
ax.set_ylabel('$v$ [mm]', fontsize=textsize)
ax.xaxis.set_tick_params(labelsize=textsize)
ax.yaxis.set_tick_params(labelsize=textsize)
ax.grid(visible=True, which='both', axis='y')
ax.spines[['right', 'top']].set_visible(False)
ax.legend(fontsize=textsize)
fig.tight_layout()

#%% Sensitivity
## Frequency starting value, number of frequencies, timeseries length, noise,
## cluster size, segments, observation accuracy (covariances?)

# segments = 3
# max_iter = 15
# f = np.array([0.79, 1.58, 0.78, 0.79])
# frequency = np.array([0.79, 1.58, 0.78])#, 0.79])

# time_period = np.arange(5, 1, -1)
# raster_size = np.array([1e-2])
# noise_increase = np.linspace(0, 1e-2, 3)
# number_of_frequencies = np.arange(1, 3)
# frequency_change = np.arange(0, 1e-1 + 1e-3, 1e-3)



# data = pd.read_csv('curved_plane2_filtered.txt')
# parameters, deviation = [], []
# for length in time_period:
#     for r in raster_size:
#         for noise in noise_increase:
#             for nof in number_of_frequencies:
#                 for cf in frequency_change:
#                     parameters.append([length, r, noise, nof, cf])
#                     print(f'time length {length} with cluster size {r}, increased noise of {noise*1e3} mm, using {nof} frequencies and changed by {np.round(cf, 3)} Hz')
                    
                    
#                     data_filter = list(np.where( np.logical_and(data['time'] >= 20, data['time'] <= 20+length) )[0])
#                     data_tmp = data.filter( items=data_filter, axis=0)
                    
#                     pca, mean, transformation = clusterts(data_tmp, raster_size=r)
#                     StandPoint, pca = pca
#                     pca['Z'] = pca['Z'] + np.random.randn(len(pca)) * noise
                    
#                     f = frequency[:nof] + cf

#                     # single timeseries
#                     unknown, thist, variance, error = [], [], [], np.zeros((pca['Z'].shape[0]))
#                     for i in tqdm( np.unique(pca['timeseriesID']) ):
#                         idx = pca['timeseriesID'] == i
#                         x, Qxx, history1 = estimate1d(pca['time'][idx], pca['Z'][idx], f, unknown=None, max_iter=max_iter)
#                         Q_single.append( Qxx )
#                         thist.append( np.copy(history1) )
                        
#                         y, var = validate1d(pca['time'][idx], x, Qxx)
#                         error[idx] = pca['Z'][idx] - y
                    
#                     # same frequency
#                     x, Qxx2, history2 = estimate2d(pca['time'], pca['Z'], pca['timeseriesID'], f, max_iter=max_iter)
#                     vali2, vari2 = validate2d(pca, x, Qxx2)
                    
#                     ## polynom
#                     x, Q_poly, historyp = estimatepoly( pca['time'], np.c_[ pca['Y'], pca['Z'] ], f, degree=3, max_iter=max_iter)
#                     valp, varp = validatepoly( pca, x, Q_poly)

#                     ## B-spline
#                     x, Q_spline, knotvec, historyb = estimatespline( pca['time'], np.c_[ pca['Y'], pca['Z'] ], f, segments=segments, degree=3, method=method, max_iter=max_iter )
#                     valb, varb = validatespline(pca, knotvec, x, Q_spline, degree=3)
                    
#                     deviation.append( [np.median(np.abs(error)), np.median(np.abs(pca['Z'] - vali2['Z'])), np.median(np.abs(pca['Z'] - valp['Z'])), np.median(np.abs(pca['Z'] - valb['Z']))] )


#%% load sensitivity
df = pd.read_csv('sensitivity.csv')

idx = np.logical_and( df['numfreq'] == 2, df['noise'] == 0 )
x, y = df['length'][idx], df['freqeuncystart'][idx]
X, Y = np.meshgrid( np.unique(x), np.unique(y) )
X = np.fliplr(X)

midx = np.array(list(map( lambda i: list(np.array( np.where(np.logical_and(X == i[0], Y == i[1])) ).flatten()), zip(x, y) )))
Zi = np.zeros(X.shape); Zi[midx[:,0], midx[:,1]] = np.sqrt(df['individual'][idx]**2) * yscale
Zs = np.zeros(X.shape); Zs[midx[:,0], midx[:,1]] = np.sqrt(df['same'][idx]**2) * yscale
Zp = np.zeros(X.shape); Zp[midx[:,0], midx[:,1]] = np.sqrt(df['polynomial'][idx]**2) * yscale
Zb = np.zeros(X.shape); Zb[midx[:,0], midx[:,1]] = np.sqrt(df['B-spline'][idx]**2) * yscale

Za = np.mean(np.stack([Zi, Zs, Zp, Zb]), axis=0)
fig, ax = plt.subplots(figsize=(linewidth*cm*scale, linewidth*cm*scale))
# ax.imshow(np.fliplr(np.flipud(Za)), aspect='auto')
# ax.set_xticks([0, 1, 2, 3], np.unique(X))
# ax.set_yticks( np.arange(0, Y.shape[0], 25), np.flip(np.unique(Y)[::25]) )
pm = ax.pcolormesh(X, Y, Za, clim=(0, 1)); 
cb = fig.colorbar(pm)
cb.set_label('MSE (mm)', fontsize=textsize*scale, fontfamily=fontfamily)
cb.ax.set_yticks([0, 0.5, 1], labels=['0', '0.5', '>1'], fontsize=textsize*scale, fontfamily=fontfamily)
ax.set_xticks( np.unique(X) )
ax.xaxis.set_tick_params(labelsize=textsize*scale, labelfontfamily=fontfamily)
ax.yaxis.set_tick_params(labelsize=textsize*scale, labelfontfamily=fontfamily)
ax.set_xlabel('observation time (s)', fontsize=textsize*scale, fontfamily=fontfamily)
ax.set_ylabel('frequency deviation (Hz)', fontsize=textsize*scale, fontfamily=fontfamily)
ax.yaxis.set_label_coords(-0.2, 0.5)
ax.xaxis.set_label_coords(0.5, -0.1)
fig.tight_layout()

# fig, ax = plt.subplots(2,2, sharex=True, sharey=True, figsize=(20*cm, 12*cm))
# cset1 = ax[0,0].pcolor(X, Y, Zi, cmap='coolwarm', vmin=0, vmax=np.max([Zi, Zs, Zp, Zb]))
# # ax[0,0].set_xticks(ticks=np.arange(X.shape[1]), labels=np.unique(X))
# # ax[0,0].set_yticks(ticks=np.arange(Y.shape[0])[::20], labels=np.unique(Y)[::20])

# cset2 = ax[0,1].pcolor(X, Y, Zs, cmap='coolwarm', vmin=0, vmax=np.max([Zi, Zs, Zp, Zb]))
# # ax[0,1].set_xticks(ticks=np.arange(X.shape[1]), labels=np.unique(X))
# # ax[0,1].set_yticks(ticks=np.arange(Y.shape[0])[::20], labels=np.unique(Y)[::20])

# cset3 = ax[1,0].pcolor(X, Y, Zp, cmap='coolwarm', vmin=0, vmax=np.max([Zi, Zs, Zp, Zb]))
# # ax[1,0].set_xticks(ticks=np.arange(X.shape[1]), labels=np.unique(X))
# # ax[1,0].set_yticks(ticks=np.arange(Y.shape[0])[::20], labels=np.unique(Y)[::20])

# cset4 = ax[1,1].pcolor(X, Y, Zb, cmap='coolwarm', vmin=0, vmax=np.max([Zi, Zs, Zp, Zb]))
# # ax[1,1].set_xticks(ticks=np.arange(X.shape[1]), labels=np.unique(X))
# # ax[1,1].set_yticks(ticks=np.arange(Y.shape[0])[::20], labels=np.unique(Y)[::20])
# fig.tight_layout()
# fig.colorbar(cset4, ax=ax.ravel().tolist(), shrink=0.95)

# fig, ax = plt.subplots(figsize=(12*cm, 12*cm), subplot_kw={'projection':'3d'})
# # ax.plot_wireframe(X, Y, Zi, color='k', label='individual')
# # ax.plot_wireframe(X, Y, Zs, color='tab:blue', alpha=.6, label='same frequency')
# # ax.plot_wireframe(X, Y, Zp, color='tab:orange', alpha=.6, label='polynomial')
# # ax.plot_wireframe(X, Y, Zb, color='tab:cyan', alpha=.6, label='B-spline')

# ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
# ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
# ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

# surf1 = ax.plot_surface(X, Y, Zi, color='k', alpha=.3, label='individual')
# surf2 = ax.plot_surface(X, Y, Zs, color='tab:blue', label='same frequency')
# surf3 = ax.plot_surface(X, Y, Zp, color='tab:orange', label='polynomial')
# surf4 = ax.plot_surface(X, Y, Zb, color='tab:cyan', label='B-spline')

# surf1._edgecolors2d = surf1._edgecolor3d; surf1._facecolors2d = surf1._facecolor3d
# surf2._edgecolors2d = surf2._edgecolor3d; surf2._facecolors2d = surf2._facecolor3d
# surf3._edgecolors2d = surf3._edgecolor3d; surf3._facecolors2d = surf3._facecolor3d
# surf4._edgecolors2d = surf4._edgecolor3d; surf4._facecolors2d = surf4._facecolor3d

# ax.legend(loc='upper right', fontsize=text_size)
# ax.tick_params(labelsize=text_size)
# # ax.tick_params(axis='y', labelrotation=-0)
# ax.set_xlabel('time duration [sec]', labelpad=5, fontsize=text_size)
# ax.set_ylabel('frequency deviation [Hz]', labelpad=8, fontsize=text_size, rotation='horizontal')
# ax.set_zlabel('MSE [mm]', labelpad=2, fontsize=text_size, rotation='horizontal')
# ax.view_init(elev=30, azim=-60)



##### visualization in time
# a = varb.copy()
# time = np.arange(0, 5, 1/100)
# i = 0

# max_amp = np.max(np.abs(a[1][0][:,i])) + 1e-3
# plt.figure()
# for t in tqdm( time) :
#     plt.cla()
#     plt.plot( a[0][0], a[1][0][:,i] * np.cos(2*np.pi*a[3][0][i]*t + a[2][0][:,i] ) )
#     # plt.ylim([-max_amp, max_amp])
#     plt.title(f'time: {np.round(t, 2)} sec')
#     plt.pause(1e-6)
    