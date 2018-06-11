#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Waze-Jams - FS version.

This application was designed to train and test a spatio-temporal
Gaussian process-based model for forecasting traffic congestions
using Waze data.
"""

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce
from datetime import datetime, timedelta
import time
import numpy as np


def prepare(filename, Ntrain):
    """Create adjacency-related covariate: jams's proportion on neighboring."""
    start = time.time()

    ytab = np.loadtxt(filename, delimiter=',', dtype=str)
    N, M = ytab.shape
    print "[{} {}]".format(N, M)

    if Ntrain == -1:
        Ntrain = N
        Ntest = 1
        last_date = ytab[-1, 0]
    else:
        if Ntrain > N:
            print 'Dataset has too few instances!'
            Ntrain = N
        Ntest = N - Ntrain + 1
        last_date = ytab[Ntrain-1, 0]

    ytab = ytab[:, 1:].astype(int)
    M -= 1

    sqrt_M = int(np.sqrt(M))
    yg = ytab.reshape(N, sqrt_M, sqrt_M)

    adj = np.zeros((N, M), dtype=float).reshape(N, sqrt_M, sqrt_M)
    nsqrt_M = sqrt_M-1

    for jj in xrange(0, sqrt_M):
        for ii in xrange(0, sqrt_M):
            c = 0
            if (ii > 0 and jj > 0):
                adj[:, ii, jj] = adj[:, ii, jj] + yg[:, ii-1, jj-1]
                c += 1.0
            if (ii > 0):
                adj[:, ii, jj] = adj[:, ii, jj] + yg[:, ii-1, jj]
                c += 1.0
            if (ii > 0 and jj < nsqrt_M):
                adj[:, ii, jj] = adj[:, ii, jj] + yg[:, ii-1, jj+1]
                c += 1.0
            if (jj > 0):
                adj[:, ii, jj] = adj[:, ii, jj] + yg[:, ii, jj-1]
                c += 1.0
            if (jj < nsqrt_M):
                adj[:, ii, jj] = adj[:, ii, jj] + yg[:, ii, jj+1]
                c += 1.0
            if (ii < nsqrt_M and jj > 0):
                adj[:, ii, jj] = adj[:, ii, jj] + yg[:, ii+1, jj-1]
                c += 1.0
            if (ii < nsqrt_M):
                adj[:, ii, jj] = adj[:, ii, jj] + yg[:, ii+1, jj]
                c += 1.0
            if (ii < nsqrt_M and jj < nsqrt_M):
                adj[:, ii, jj] = adj[:, ii, jj] + yg[:, ii+1, jj+1]
                c += 1.0

            adj[:, ii, jj] = map(lambda x: x/c, adj[:, ii, jj])

    adj = adj.ravel()
    yg = [y * 2 - 1 for y in yg.ravel()]  # Fixing labels to be +/- 1

    settings = dict()
    settings['adj'] = adj
    settings['yg'] = yg
    settings['M'] = M
    settings['Ntrain'] = Ntrain
    settings['Ntest'] = Ntest
    settings['last_date'] = last_date

    end = time.time()
    print "Elapsed {} seconds".format(int(end-start))
    return settings


@task(returns=list)
def GP(script, settings, hypers, cellnums):
    """Predict wheater a cell will be traffic jam or not."""
    result = []
    adj = settings['adj']
    yg = settings['yg']
    M = settings['M']
    Ntrain = settings['Ntrain']
    Ntest = settings['Ntest']
    last_date = settings['last_date']

    import oct2py

    last_date = datetime.strptime(last_date, "%Y-%m-%d %H:%M:%S")

    start = time.time()
    print "Predicting jams to grids: #{} to #{}".format(cellnums[0],
                                                        cellnums[-1])

    result_i = oct2py.octave.feval(script, adj, yg, M,
                                   Ntrain, Ntest, hypers, cellnums)

    for cellnum, prediction in zip(cellnums, result_i['Forecasts']):
        mu, s2 = prediction
        # append the  95% confidence interval
        std = np.sqrt(s2)
        ci_95 = [mu-2*std, mu+2*std]
        ci_95 = np.clip(ci_95, -1, 1)
        percentage = (mu+1)/2
        row = [cellnum, str(last_date), round(mu, 4),
               round(s2, 4), round(ci_95[0], 4), round(ci_95[1], 4),
               round(percentage, 4)]
        result.append(row)

    end = time.time()
    print "Elapsed {} seconds".format(int(end-start))

    return [result, []]


@task(returns=list)
def GP_hyper(script, settings, cellnums):
    """Predict the possibility of traffic jam and update the hyper file."""

    adj = settings['adj']
    yg = settings['yg']
    M = settings['M']
    Ntrain = settings['Ntrain']
    Ntest = settings['Ntest']
    last_date = settings['last_date']
    last_date = datetime.strptime(last_date, "%Y-%m-%d %H:%M:%S")

    import oct2py

    result = []
    start = time.time()
    print "Predicting jams to grids: #{} to #{}".format(cellnums[0],
                                                        cellnums[-1])

    result_i = oct2py.octave.feval(script, adj, yg, M,
                                   Ntrain, Ntest, cellnums)

    for cellnum, prediction in zip(cellnums, result_i['Forecasts']):
        mu, s2 = prediction
        # append the  95% confidence interval
        std = np.sqrt(s2)
        ci_95 = [mu-2*std, mu+2*std]
        ci_95 = np.clip(ci_95, -1, 1)
        percentage = (mu+1)/2
        row = [cellnum, str(last_date), round(mu, 4),
               round(s2, 4), round(ci_95[0], 4), round(ci_95[1], 4),
               round(percentage, 4)]
        result.append(row)

    cellnums = np.array(cellnums).reshape((len(cellnums), 1))
    hypers = np.hstack((cellnums, result_i['hyp']))

    end = time.time()
    print "Elapsed {} seconds".format(int(end-start))
    return [result, hypers]


@task(returns=list)
def mergelists(list1, list2):
    """Merge the partial results."""
    list1[0] = np.concatenate((list1[0], list2[0]), axis=0)
    list1[1] = np.concatenate((list1[1], list2[1]), axis=0)
    return list1

def waze_jams(trainfile, hypers, Ntrain, script,
              gridsList, grid, numFrag, output):
    """
    Waze-Jams.

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
    gridsList = np.loadtxt(gridsList, delimiter=',',
                           dtype=(int, int), skiprows=1, usecols=(4, 5))
    gridsList = gridsList[:, 0][gridsList[:, 1] == 1]
    print "[INFO] - {} valid grids".format(len(gridsList))
    config = prepare(trainfile, Ntrain)

    if grid == -1:
        frag_cells = np.array_split(gridsList, numFrag)
    else:
        if grid in gridsList:
            frag_cells = np.array([[grid]])
        else:
            raise "[INFO] - Grid #{} is not valid".format(grid)

    nfrag = len(frag_cells)
    outputs = [[] for i in range(nfrag)]
    if len(hypers) > 0:
        hyper = np.loadtxt(hypers, delimiter=',', dtype=np.float64)
        # when you inform a hyperparameters's file
        for i in range(nfrag):
            tmp_hypers = hyper[np.in1d(hyper[:,0], frag_cells[i])]
            outputs[i] = GP(script, config, tmp_hypers, frag_cells[i])
    else:
        # when you also need to training the model
        for i in range(nfrag):
            outputs[i] = GP_hyper(script, config, frag_cells[i])

    results = mergeReduce(mergelists, outputs)
    from pycompss.api.api import compss_wait_on as sync
    results = sync(results)

    if len(hypers) == 0:
        output_hyper = '{}hypers_{}.txt'.format(output, timestr)
        np.savetxt(output_hyper, np.asarray(results[1]), delimiter=',',
                   fmt="%i,%1.20f,%1.20f,%1.20f,%1.20f,%1.20f,%1.20f,%1.20f")
    output_forecast = '{}forecasts_{}.txt'.format(output, timestr)
    np.savetxt(output_forecast, results[0], delimiter=',',
               header='IDgrid,pred_timestamp,average,variance,'
               'ci_95_1,ci_95_2,prediction', fmt='%s')


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description='Waze-jams - PyCompss')

    p.add_argument('-t', '--trainfile', required=True,
                   help='Filename of the training set.')
    p.add_argument('-r', '--script', required=True,
                   help='File path to the script to the '
                   'second stage (runGP or trainAndRunGP)')
    p.add_argument('-g', '--grid', required=False, type=int,  default=-1,
                   help='Number of a cell (1 <= N <= ngrid), -1 to all.')
    p.add_argument('-s', '--Ntrain', required=True,
                   help='Size of Training Set. -1 to use '
                   'all training set. (default, -1)', type=int, default=-1)
    p.add_argument('-f', '--numFrag', required=False,
                   help='Number of cores', type=int,  default=4)
    p.add_argument('-l', '--gridslist', required=True,
                   help='File with the grids list.')
    p.add_argument('-p', '--hypers', required=False, type=str, default='',
                   help='File of the previous hyperparameters.')
    p.add_argument('-o', '--output', required=True,
                   help='Output file directory')
    arg = vars(p.parse_args())

    trainfile = arg['trainfile']
    hypers = arg['hypers']
    script = arg['script']
    output = arg['output']
    gridsList = arg['gridslist']
    grid = arg['grid']
    Ntrain = arg['Ntrain']
    numFrag = arg['numFrag']
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

    """.format(trainfile, hypers, Ntrain, script,
               gridsList, grid, numFrag, output)

    start = time.time()
    waze_jams(trainfile, hypers, Ntrain, script, gridsList,
              grid, numFrag, output)
    end = time.time()
    print "Elapsed {} seconds".format(int(end-start))
