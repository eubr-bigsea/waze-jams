import glob

import re
import numpy as np

def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s

def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]

def join_results(path,window):
    result = []
    grid = 1
    for name in sorted(glob.glob(path), key=alphanum_key):

        for line in open(name,'r'):
            line = line.split(',')
            result.append([grid,window ,float(line[0]),float(line[1])])
        grid+=1


    import pandas as pd
    df = pd.DataFrame(result,columns=['grid','window time','score','variance'])
    df.to_csv('predictions_curitiba.csv',index=False)


def join_hypers(path):
    result = []
    grid = 1
    for name in sorted(glob.glob(path), key=alphanum_key):
        hyper = np.loadtxt(name, delimiter=',', dtype=float)
        hyper = np.insert(hyper, 0, grid)
        result.append(hyper)
        grid+=1


    np.savetxt( 'hypers_curitiba.txt',
                np.asarray(result), delimiter=',',
                fmt="%i,%f,%f,%f,%f,%f,%f,%f")

#join_results('/home/lucasmsp/workspace/BigSea/waze-jams/Results_20160919_14h-15h/forecasts_*.txt',"2016-09-19 14:00:00")
join_hypers('/home/lucasmsp/workspace/BigSea/waze-jams/Results_20160919_14h-15h/hypers_*.txt')
