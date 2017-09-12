# -*- coding: utf-8 -*-
#!/usr/bin/env python

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce
from pycompss.api.api import compss_wait_on

from datetime import datetime, timedelta
import numpy as np
from shapely.geometry import Polygon, LineString


def group_datetime(d, interval):
    seconds = d.second + d.hour*3600 + d.minute*60
    k = d - timedelta(seconds=seconds % interval)
    return datetime(k.year, k.month, k.day, k.hour, k.minute, k.second)


def create_grid(ngrid, bounds):
    ncols     = ngrid[0] * ngrid[1]
    grids     =  [[] for f in range(ncols)]
    div_y =  np.sqrt((bounds[0][1] - bounds[0][0])**2) /ngrid[0]
    div_x =  np.sqrt((bounds[1][1] - bounds[1][0])**2) /ngrid[1]

    pos_y = bounds[0][0]
    pos_x = bounds[1][0]

    tmp_y = pos_y
    for c in range(ncols):
        if (c % ngrid[0] == 0):
            pos_y  = tmp_y
            tmp_y += div_y
            pos_x = bounds[1][0]

        tmp_x = pos_x+div_x
        grid = [(pos_y, pos_x), (pos_y,tmp_x), (tmp_y,tmp_x),(tmp_y,pos_x) ]
        pos_x = tmp_x
        grids[c] = grid


    return np.array(grids)

@task(returns=list, filename=IN)
def preprocessing(filename,grids,ngrid,nday_sample,window_time, mapper, jam_grids):
    import json
    ncols     = ngrid[0] * ngrid[1]

    for i, line in  enumerate(open(filename,'r')):
        record = json.loads(line)
        points = record['line']
        currentTime = record['pubMillis']["$numberLong"]
        currentTime = datetime.utcfromtimestamp(float(currentTime)/ 1000.0)
        currentTime = group_datetime(currentTime, window_time)

        index = mapper.get(str(currentTime), None)

        if (i% 10000 == 0):
            print currentTime


        if index != None:

            line_y = [ float(pair['y'])  for pair in points]
            line_x = [ float(pair['x'])  for pair in points]

            min_y = min(line_y)
            max_y = max(line_y)
            i_min = 0
            i_max = ngrid[1]
            for e, g in enumerate(grids):
                if (g[0][0] < min_y) and (min_y != g[0][0]):
                    i_min = 0

            line = [(y,x) for y,x in zip(line_y,line_x)]
            shapely_line = LineString(line)

            for c in xrange(i_min,ncols):
                    polygon = grids[c]
                    shapely_poly = Polygon(polygon)
                    intersection_line = not shapely_poly.disjoint(shapely_line)
                    if intersection_line:
                        jam_grids[index,c]  += 1


    return jam_grids



@task(returns=list)
def mergeMatrix( matrix1,matrix2):
    r1 = matrix1+matrix2
    return r1


def updateJamGrid(jam_grids,mapper):
    events = jam_grids.sum(axis=1)
    mapper = {v: k for k, v in mapper.iteritems()}
    count = [[mapper[i], v[0,0]] for i,v in enumerate(events)]
    np.clip( jam_grids, 0, 1, out=jam_grids)
    return jam_grids, count



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Waze-jams's preprocessing script")

    parser.add_argument('-i','--input',  required=True, help='the input file path.')
    parser.add_argument('-b','--bounds', required=True, type=float, nargs='+',
        help='the bounding box around a city, state, country, or zipcode using '
        'geocoding. Format: min_y, max_y min_x max_x  '
        '(where y=Latitude and x=Longitude)')
    parser.add_argument('-w','--window',    type=int,   default=3600,
        help='The window time to take in count')
    parser.add_argument('-l','--nday',      type=int,   default=28,
        help='The number of days to generate the sample.')
    parser.add_argument('-g','--ngrid',     type=int,   default=[50,50],
        nargs='+',  help='The organization of the grid')
    parser.add_argument('-f','--numFrag',   type=int,   default=4,
        help='Number of workers(core)')
    parser.add_argument('-s','--init_time', type=float, default=1468738813423.0,
        help='Initial time (in milis)')
    arg = vars(parser.parse_args())

    filename    = arg['input']
    bounds      = arg['bounds']
    ngrid       = arg['ngrid']
    window_time = arg['window']
    nday_sample = arg['nday']
    init_time   = arg['init_time']
    numFrag     = arg['numFrag']

    bounds     = np.reshape(bounds, (2, 2)).tolist()
    ncols      = ngrid[0] * ngrid[1]
    init_time  = datetime.utcfromtimestamp(init_time/ 1000.0)
    init_time  = group_datetime(init_time, window_time)

    print """
        Running: Waze-jams's preprocessing script with the following parameters:
         - input file:   {}
         - bounding box:      {}
         - window time:       {} seconds
         - number of days:    {} days
         - grid:              {}
         - inital time:       {}
         - number of workers: {}

    """.format(filename,bounds,window_time,nday_sample,ngrid,init_time,numFrag)


    grids     = create_grid(ngrid, bounds)
    np.savetxt('output_grids.csv', grids, delimiter=',',fmt='%s,%s,%s,%s')

    jam_grids = np.matrix(np.zeros( (nday_sample*24,ncols), dtype=np.int) )
    mapper = dict()
    for d in range(nday_sample*24):
        c = init_time + timedelta(hours=d)
        mapper[str(c)] = d

    partial_grid = [  preprocessing("{}_{}".format(filename,f),
                                    grids,
                                    ngrid,
                                    nday_sample,
                                    window_time,
                                    mapper,
                                    jam_grids) for f in range(numFrag)]


    jam_grids_p = mergeReduce(mergeMatrix, partial_grid)
    jam_grids_p = compss_wait_on(jam_grids_p)
    jam_grids, events = updateJamGrid(jam_grids_p,mapper)

    np.savetxt('output_training.csv', jam_grids, delimiter=',',fmt='%d')
    np.savetxt('output_events.csv',   events,   delimiter=',', fmt='%s,%s')
