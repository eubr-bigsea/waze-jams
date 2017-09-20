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
from datetime import datetime, timedelta

@task (filename = FILE_IN, returns = list)
def prepare(filename, Ntrain):
    """
    Forming adjacency-related covariate: proportion of jams on neighboring cells
    """

    #np.set_printoptions(threshold=np.nan)
    #result = {}

    start = time.time()

    ytab = np.loadtxt(filename, delimiter=',', dtype=str)
    N, M =  ytab.shape
    #N = int(N)
    print "[{} {}]".format(N,M)

    if Ntrain == -1:
        Ntrain = N
        Ntest  = 1
        last_date = ytab[-1,0]
    else:
        if Ntrain>N:
            print 'Dataset has too few instances!'
            Ntrain = N
        Ntest = N - Ntrain + 1
        last_date = ytab[Ntrain-1,0]


    ytab = ytab[:, 1:].astype(int)
    M-=1

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


    adj = adj.ravel()
    yg  = [ y*2-1 for y in yg.ravel()] #Fixing labels to be +/- 1


    end = time.time()
    print "Elapsed {} seconds".format(end-start)
    return [adj, yg, M, Ntrain, Ntest, last_date]

@task (output_forecast=FILE_OUT)
def GP(script,config,cellnums,output_forecast):
    result = []
    adj, yg, M, Ntrain, Ntest,last_date = config
    import oct2py

    last_date_i = datetime.strptime(last_date, "%Y-%m-%d %H:%M:%S")
    for cellnum, hypers in cellnums:
        start = time.time()
        last_date = last_date_i
        print "Predicting the next hour of the grid #{}".format(cellnum)
        try:
            result_i  = oct2py.octave.feval(script, adj, yg, M, cellnum, Ntrain, Ntest, hypers)
            prediction =  result_i['Forecasts']
            for p in prediction:
                last_date+= timedelta(hours=1)
                result.append([cellnum, str(last_date),p[0], p[1] ])
        except Exception as e:
            print "[ERROR] - Error predicting the grid #",str(cellnum)
            print e
        end = time.time()
        print "Elapsed {} seconds".format(end-start)


    np.savetxt(output_forecast,result, delimiter=',', fmt='%s,%s,%s,%s')

@task (output_forecast=FILE_OUT, returns=list)
def GP_hyper(script,config,cellnums,output_forecast):
    result = []
    adj, yg, M, Ntrain, Ntest, last_date = config
    import oct2py
    hypers = []
    last_date_i = datetime.strptime(last_date, "%Y-%m-%d %H:%M:%S")
    for cellnum in cellnums:
        start = time.time()
        last_date = last_date_i
        print "Creating the model and predicting the next hour of the grid #{}".format(cellnum)
        try:
            result_i   = oct2py.octave.feval(script, adj, yg, M, cellnum, Ntrain, Ntest)
            prediction = result_i['Forecasts']
            for p in prediction:
                last_date+= timedelta(hours=1)
                result.append([cellnum, str(last_date) ,p[0], p[1] ])
            r = np.insert(result_i['hyp'].flatten(), 0, cellnum)
            hypers.append(r)
        except Exception as e:
            print "[ERROR] - Error predicting the grid #",str(cellnum)
            print e

        end = time.time()
        print "Elapsed {} seconds".format(end-start)

    np.savetxt(output_forecast, result, delimiter=',', fmt='%s,%s,%s,%s')

    return hypers



@task (returns = list)
def mergelists(list1,list2):
    return list1+list2

def load_hypers(hypers,frag_cells):
    hyper = np.loadtxt(hypers, delimiter=',', dtype=float)
    frag_cells = frag_cells.tolist()
    for i in range(len(frag_cells)):
        grid = frag_cells[i]
        row = hyper[:,1:][hyper[:, 0] == grid][0]
        row = row.reshape((7, 1))
        frag_cells[i] = [grid, row]

    return frag_cells

def  waze_jams(trainfile, hypers, Ntrain, script, gridsList, grid, numFrag, output):
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
    gridsList = np.loadtxt(gridsList, delimiter=',', dtype=(int,int), skiprows=1, usecols = (4,5))
    gridsList = gridsList[:,0][gridsList[:, 1] == 1]
    print "[INFO] - {} valid grids".format(len(gridsList))
    config = prepare(trainfile, Ntrain)

    if grid == -1:
        frag_cells = np.array_split(gridsList, numFrag)
    else:
        if grid in gridsList:
            frag_cells = np.array([[grid]])
        else:
            print "[INFO] - Grid #{} is not valid".format(grid)
            return

    output_forecast = ['{}forecasts_part{}_{}.txt'.format(output,f,timestr) for f in range(len(frag_cells))]

    if len(hypers)>0:
        frag_cells    = [load_hypers(hypers,frag_cells[i]) for i in range(len(frag_cells))]
        for f  in range(len(frag_cells)):
            GP(script, config, frag_cells[f],  output_forecast[f])
    else:
        output_hyper  = [GP_hyper(script, config, frag_cells[f],  output_forecast[f]) for f  in range(len(frag_cells))]
        results       = mergeReduce(mergelists,output_hyper)

        from pycompss.api.api import compss_wait_on
        results = compss_wait_on(results)
        np.savetxt( '{}hypers_{}.txt'.format(output,timestr),
                    np.asarray(results), delimiter=',',
                    fmt="%i,%f,%f,%f,%f,%f,%f,%f")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description='Waze-jams - PyCompss')

    p.add_argument('-t','--trainfile',required=True, help='Filename of the training set.')
    p.add_argument('-r','--script',   required=True, help='File path to the script to the second stage (runGP or trainAndRunGP)')
    p.add_argument('-g','--grid',     required=False,help='Number of a cell grid (1 <= N <= ngrid), -1 to all.', type=int,  default=-1)
    p.add_argument('-s','--Ntrain',   required=True, help='Size of Training Set. -1 to use all training set. (default, -1)', type=int, default=-1)
    p.add_argument('-f','--numFrag',  required=False,help='Number of cores', type=int,  default=4)
    p.add_argument('-l','--gridslist',required=True, help='File with the grids list.')
    p.add_argument('-p','--hypers',   required=False,help='File of the previous hyperparameters.', type=str, default='')
    p.add_argument('-o','--output',   required=True, help='Output file directory')
    arg = vars(p.parse_args())

    trainfile       = arg['trainfile']
    hypers          = arg['hypers']
    script          = arg['script']
    output          = arg['output']
    gridsList       = arg['gridslist']
    grid            = arg['grid']
    Ntrain          = arg['Ntrain']
    numFrag         = arg['numFrag']
    print """
        Running Traffic-jams in PyCOMPSs with the parameters:
         - Training File:   {}
         - hypers:          {}
         - Training size:   {}
         - script:          {}
         - Grids File:      {}
         - grid:            {}
         - numFrag:         {}
         - Output Path:     {}

    """.format(trainfile, hypers, Ntrain, script,gridsList, grid, numFrag, output)

    start = time.time()
    waze_jams(trainfile, hypers, Ntrain, script, gridsList, grid, numFrag, output)
    end = time.time()
    print "Elapsed {} seconds".format(int(end-start))
