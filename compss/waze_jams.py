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




@task(returns=list)
def compute(script_runGP,path,config,cellnums):
    from oct2py import octave
    #octave.addpath(path)

    for cellnum in cellnums:
        start = time.time()
        out1 = "{}forecasts_{}".format(path,cellnum)
        out2 = "{}hypers_{}".format(path,cellnum)

        #code = octave.runGP(config,cellnum,out1,out2)
        code  = octave.feval(script_runGP, config,cellnum,out1,out2)

        end = time.time()
        print "Elapsed {} seconds".format(end-start)

    return [out1,out2]

@task(returns=FILE_OUT)
def prepare(script_prepare,filename,output):
    from oct2py import octave
    #from oct2py import Oct2Py

    start = time.time()

    ytab = np.loadtxt(filename,delimiter=' ')


    #octave.addpath(path)
    #code = octave.prepare(ytab,output)

    #or
    code = octave.feval(script_prepare, ytab,output)

    end = time.time()

    print "Elapsed {} seconds".format(end-start)
    return output


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def  waze_jams(script_prepare,
               script_runGP,
               path,
               filename,
               output_config,
               numFrag):

    config = prepare(script_prepare,filename,output_config)


    cells = [i for i in xrange(1,4)]
    frag_cells = chunks(cells, int(float(len(cells))/numFrag+1))

    from pycompss.api.api import compss_wait_on
    partialResult = [compute(script_runGP,path,config, cellnums) for cellnums  in frag_cells ]
    partialResult = compss_wait_on(partialResult)

# olhar como é nao retornar nada ... talvez n precisaria



if __name__ == "__main__":
    script_prepare  = "/home/lucasmsp/workspace/BigSea/waze-jams/compss/prepare.m"
    script_runGP    = "/home/lucasmsp/workspace/BigSea/waze-jams/compss/runGP.m"
    path = "/home/lucasmsp/workspace/BigSea/waze-jams/compss/"
    filename = "/home/lucasmsp/workspace/BigSea/waze-jams/sample"
    output_config = '/home/lucasmsp/workspace/BigSea/waze-jams/compss/config.txt'
    numFrag = 4

    print "Running Waze-jams-compss with the parameters:\n\t- Filename:{}\n\t- numFrag: {}".format(filename,numFrag)

    start = time.time()
    waze_jams(script_prepare,script_runGP,path,filename,output_config,numFrag)
    end = time.time()
    print "Elapsed {} seconds".format(int(end-start))
