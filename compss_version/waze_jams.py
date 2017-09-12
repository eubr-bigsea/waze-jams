#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter    import *
from pycompss.functions.reduce import mergeReduce
from pycompss.functions.data   import chunks

import string
import re
import unicodedata
import sys
import math
reload(sys)
sys.setdefaultencoding("utf-8")
import time
import numpy as np
from oct2py import octave


#@task(returns=list)
def compute(script_runGP,path,config,cellnums):
    from oct2py import octave
#    octave.addpath(path)

    out1 = ""
    out2 = ""
    for cellnum in cellnums:
        start = time.time()
        out1 = "{}forecasts_{}".format(path,cellnum)
        out2 = "{}hypers_{}".format(path,cellnum)

        code = octave.runGP(config,cellnum,out1,out2)
        #code  = octave.feval(script_runGP, config,cellnum,out1,out2)
        print code
        end = time.time()
        print "Elapsed {} seconds".format(end-start)

    return [out1,out2]

#@task(returns = dict)
def prepare(script_prepare, filename):
    """
    In order to use an m-file in Oct2Py you can call feval with the full path.
    Thread-safety: each Oct2Py object uses an independent Octave session.
    """
    result = {}


    from oct2py import Oct2Py
    start = time.time()

    ytab = np.loadtxt(filename,delimiter=' ')
    #print ytab

    #path = "/home/lucasmsp/workspace/BigSea/waze-jams/compss/"

    #octave.addpath(path)
    #result = octave.prepare(ytab)

    #or (data, covf, meanfunc, likf, infm, hyp, Ntrain, Ntest, M)
    result = octave.feval(script_prepare, ytab)
    #print result
    end = time.time()
    print "Elapsed {} seconds".format(end-start)
    return result


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def  waze_jams(script_prepare,script_runGP,filename,output,numFrag):
    """
        script_prepare: path to the first stage;
        script_runGP:   path to the second stage;
        path:           path to workspace in octave

    """
    config = prepare(script_prepare,filename)


    cells = [i for i in xrange(1,4)]
    frag_cells = chunks(cells, int(float(len(cells))/numFrag+1))

    from pycompss.api.api import compss_wait_on
    partialResult = [compute(script_runGP,path, config, cellnums) for cellnums  in frag_cells ]
    partialResult = compss_wait_on(partialResult)




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Waze-jams - PyCompss')
    parser.add_argument('-p', '--script_prepare',   required=True, help='Script to the first stage (prepare)')
    parser.add_argument('-r', '--script_runGP',     required=True, help='Script to the second stage (runGP)')
    parser.add_argument('-f', '--filename',         required=True, help='Filename of the input')
    parser.add_argument('-o', '--output',           required=True, help='output path')
    parser.add_argument('-n', '--numFrag',       type=int,  default=4, required=False, help='Number of nodes')

    arg = vars(parser.parse_args())

    script_prepare  = arg['script_prepare']
    script_runGP    = arg['script_runGP']
    filename        = arg['filename']
    numFrag         = arg['numFrag']
    output          = arg['output']

    path = "/home/lucasmsp/workspace/BigSea/waze-jams/compss/"


    print "Running Waze-jams-compss with the parameters:\n\t- Filename:{}\n\t- numFrag: {}".format(filename,numFrag)

    start = time.time()
    waze_jams(script_prepare,script_runGP,filename,output,numFrag)
    end = time.time()
    print "Elapsed {} seconds".format(int(end-start))
