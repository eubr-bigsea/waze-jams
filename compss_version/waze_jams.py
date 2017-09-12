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
def prepare(filename):
    """
    Forming adjacency-related covariate: proportion of jams on neighboring cells
    """

    np.set_printoptions(threshold=np.nan)
    result = {}

    start = time.time()

    ytab = np.loadtxt(filename, delimiter=' ')

    N,M =  ytab.shape
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



    Ntrain = int(N)
    Ntest  = N - Ntrain + 1
    adj    = adj.ravel()
    y      = [ y*2-1 for y in yg.ravel()] #Fixing labels to be +/- 1

    end = time.time()
    print "Elapsed {} seconds".format(end-start)
    return [adj,y,M,N,Ntrain, Ntest]

@task (returns = list)
def GP(script_runGP,path,config,cellnums):
    result = []
    adj, y, M, N, Ntrain, Ntest = config
    import oct2py
    for cellnum in cellnums:
        start = time.time()
        result_i  = oct2py.octave.feval(script_runGP, adj,y, M, N, Ntrain, Ntest, cellnum)
        result.append([cellnum, result_i ])
        end = time.time()
        print "Elapsed {} seconds".format(end-start)

    return result

def mergelists(list1,list2):
    return list1+list2


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def  waze_jams(grid,script_runGP,filename,output,numFrag):

    config = prepare(filename)

    if grid == -1:
        cells = [i for i in xrange(0,2500)]
        frag_cells = chunks(cells, int(float(len(cells))/numFrag+1))

        partialResult = [GP(script_runGP, output, config, cellnums) for cellnums  in frag_cells ]
        results       = mergeReduce(mergelists,partialResult)

    else:
        cellnums = [grid]
        results = GP(script_runGP, output, config, cellnums)

    from pycompss.api.api import compss_wait_on
    results = compss_wait_on(results)
    for r in results:
        print r


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description='Waze-jams - PyCompss')

    p.add_argument('-i','--input', required=True,help='Filename of the input')
    p.add_argument('-r','--script_runGP',  required=True,
                    help='File path to the script to the second stage (runGP)')
    p.add_argument('-o','--output',required=True, help='Output file directory')
    p.add_argument('-g','--grid',   required=False,
                    help='Number of a cell grid (1 <= N <= 2500), -1 to all.',
                    type=int,  default=-1)
    p.add_argument('-n','--numFrag',required=False,
                                  help='Number of nodes', type=int,  default=4)

    arg = vars(p.parse_args())

    script_runGP    = arg['script_runGP']
    filename        = arg['input']
    numFrag         = arg['numFrag']
    output          = arg['output']
    grid            = arg['grid']

    print """
        Running Waze-jams-compss with the parameters:
         - Filename:    {}
         - script_runGP:{}
         - grid:        {}
         - numFrag:     {}
         - Output Path: {}

    """.format(filename,script_runGP,grid,numFrag, output)

    start = time.time()
    waze_jams(grid,script_runGP,filename,output,numFrag)
    end = time.time()
    print "Elapsed {} seconds".format(int(end-start))
