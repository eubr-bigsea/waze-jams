#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

from pycompss.api.task         import task
from pycompss.api.parameter    import *
from pycompss.functions.reduce import mergeReduce
from pycompss.functions.data   import chunks

import time
import numpy as np

@task (filename = FILE_IN, returns = list)
def prepare(filename, Ntrain):
    """
    Forming adjacency-related covariate: proportion of jams on neighboring cells
    """

    np.set_printoptions(threshold=np.nan)
    result = {}

    start = time.time()

    ytab = np.loadtxt(filename, delimiter=' ')

    N, M =  ytab.shape
    N = int(N)
    print "[{} {}]".format(N,M)

    #cria uma nova matriz por linha
    sqrt_M = int(np.sqrt(M))
    yg = ytab.reshape(N, sqrt_M ,sqrt_M )

    adj =  np.zeros((N,M), dtype=float).reshape(N, sqrt_M ,sqrt_M )
    nsqrt_M = sqrt_M-1

    for jj in xrange(0, sqrt_M):
        for ii in xrange(0, sqrt_M):
            c = 0;
            if (ii > 0 and jj > 0):
                adj[:,ii,jj] = adj[:,ii,jj] + yg[:,ii-1,jj-1]
                c +=1.0
            if (ii > 0):
                adj[:,ii,jj] = adj[:,ii,jj] + yg[:,ii-1,jj]
                c +=1.0
            if (ii > 0 and jj < nsqrt_M):
                adj[:,ii,jj] = adj[:,ii,jj] +  yg[:,ii-1,jj+1]
                c +=1.0
            if (jj > 0):
                adj[:,ii,jj] = adj[:,ii,jj] + yg[:,ii,jj-1]
                c +=1.0
            if (jj < nsqrt_M ):
                adj[:,ii,jj] = adj[:,ii,jj]+ yg[:,ii,jj+1]
                c +=1.0
            if (ii < nsqrt_M  and jj > 0):
                adj[:,ii,jj] = adj[:,ii,jj] + yg[:,ii+1,jj-1]
                c +=1.0
            if (ii < nsqrt_M ):
                adj[:,ii,jj] = adj[:,ii,jj] + yg[:,ii+1,jj]
                c +=1.0
            if (ii < nsqrt_M  and jj < nsqrt_M):
                adj[:,ii,jj] = adj[:,ii,jj] + yg[:,ii+1,jj+1]
                c +=1.0

            adj[:,ii,jj] = map(lambda x: x/c, adj[:,ii,jj])


    adj    = adj.ravel()
    yg     = [ y*2-1 for y in yg.ravel()] #Fixing labels to be +/- 1
    if Ntrain == -1:
        Ntrain = N
        Ntest  = 1
    else:
        if Ntrain>N:
            print 'Dataset has too few instances!'
            Ntrain = N
        Ntest = N - Ntrain + 1

    end = time.time()
    print "Elapsed {} seconds".format(end-start)
    return [adj, yg, M, Ntrain, Ntest]

@task (returns = list)
def GP(script,config,cellnums,train_op):
    result = []
    adj, yg, M, Ntrain, Ntest = config
    import oct2py
    if not train_op:
        for cellnum in cellnums:
            if cellnum < M:
                start = time.time()
                print "Creating the model and predicting the next hour of the grid #{}".format(cellnum)
                result_i  = oct2py.octave.feval(script, adj, yg, M, cellnum, Ntrain, Ntest)
                result.append([cellnum, result_i ])
                end = time.time()
                print "Elapsed {} seconds".format(end-start)
    else:
        for cellnum, hypers in cellnums:
            if cellnum < M:
                start = time.time()
                print "Predicting the next hour of the grid #{}".format(cellnum)
                result_i  = oct2py.octave.feval(script, adj, yg, M, cellnum, Ntrain, Ntest, hypers)
                result.append([cellnum, result_i ])
                end = time.time()
                print "Elapsed {} seconds".format(end-start)

    return result

@task (returns = list)
def mergelists(list1,list2):
    return list1+list2

def load_hypers(hypers,frag_cells):
    for i in range(len(frag_cells)):
        hyper = np.loadtxt("hypers_{}.txt".format(frag_cells[i]), delimiter=' ', dtype=float)
        hyper = hyper.reshape((len(hyper), 1))
        frag_cells[i] = [frag_cells[i], hyper]
    return frag_cells

def  waze_jams(trainfile, hypers, Ntrain, script, ngrids, grid, numFrag, output):
    """
    prepare():
        It contains both the data to be used for hyperparameter learning and
        inference as information regarding the GP prior distribution.

    trainGP():
        It outputs two items per cell: forecasts and hypers. The first items
        contains a Tx2 matrix with predictive mean and variance, where T is
        the number of time intervals required for testing.

        Predictions are in the interval [-1,+1], where predictions closer to -1
        indicate greater probability of being associated with label -1 and
        predictions closer to +1 indicate the opposite scenario.

        These predictions can be turned into probabilities by turning them into
        the interval [0,1]. The second item consists of a vector with learned
        hyperparameters.

    """
    import time
    timestr = str(time.strftime("%Y%m%d_%Hh"))

    config = prepare(trainfile, Ntrain)
    
    if grid == -1:
        frag_cells = np.array_split(np.arange(1,ngrids+1), numFrag)
        if len(hypers)>0:
            frag_cells    = [load_hypers(hypers,frag_cells[i]) for i in range(numFrag)]
            partialResult = [GP(script, config, frag, True ) for frag  in frag_cells ]
        else:
            partialResult = [GP(script, config, frag, False) for frag  in frag_cells ]
        results       = mergeReduce(mergelists,partialResult)
    else:
        frag_cells = [grid]
        if len(hypers)>0:
            frag_cells = load_hypers(hypers, frag_cells)
            results  = GP(script, config, frag_cells, True)
        else:
            results  = GP(script, config, frag_cells, False)


    from pycompss.api.api import compss_wait_on
    results = compss_wait_on(results)
    for r in results:
        cellnum   = r[0]
        print cellnum
        forecasts = r[1]['Forecasts']
        hypers    = r[1]['hyp']
        print np.array(forecasts).shape
        print np.array(hypers).shape
        np.savetxt('forecasts_{}_{}.txt'.format(cellnum,timestr), forecasts, delimiter=',', fmt='%f')
        np.savetxt('hypers_{}.txt'.format(cellnum),    hypers,    delimiter=',', fmt='%f')


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description='Waze-jams - PyCompss')

    p.add_argument('-t','--trainfile',required=True, help='Filename of the training set.')
    p.add_argument('-r','--script',   required=True, help='File path to the script to the second stage (runGP or trainAndRunGP)')
    p.add_argument('-g','--grid',     required=False,help='Number of a cell grid (1 <= N <= ngrid), -1 to all.', type=int,  default=-1)
    p.add_argument('-s','--Ntrain',   required=True, help='Size of Training Set. -1 to use all training set. (default, -1)', type=int, default=-1)
    p.add_argument('-f','--numFrag',  required=False,help='Number of cores', type=int,  default=4)
    p.add_argument('-n','--ngrids',   required=False,help='Number of grids. (default, 2500)', type=int, default=2500)
    p.add_argument('-p','--hypers',   required=False,help='Path of the previous hyperparameters.', type=str, default='')
    p.add_argument('-o','--output',   required=True, help='Output file directory')
    arg = vars(p.parse_args())

    trainfile       = arg['trainfile']
    hypers          = arg['hypers']
    script          = arg['script']
    output          = arg['output']
    ngrids          = arg['ngrids']
    grid            = arg['grid']
    Ntrain          = arg['Ntrain']
    numFrag         = arg['numFrag']
    print """
        Running Traffic-jams in PyCOMPSs with the parameters:
         - Training File:   {}
         - hypers:          {}
         - Training size:   {}
         - script:          {}
         - Number of grids: {}
         - grid:            {}
         - numFrag:         {}
         - Output Path:     {}

    """.format(trainfile, hypers, Ntrain, script,ngrids, grid, numFrag, output)

    start = time.time()
    waze_jams(trainfile, hypers, Ntrain, script, ngrids, grid, numFrag, output)
    end = time.time()
    print "Elapsed {} seconds".format(int(end-start))
