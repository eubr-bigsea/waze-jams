#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PreProcessing-hdfs.

This version (using HDFS) assumes that the input file to
be read is in HDFS. The result output is stored in the commom filesystem.
"""
__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.functions.reduce import mergeReduce
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from shapely.geometry import Polygon, LineString


def group_datetime(d, interval):
    """Group datetime in bins."""
    seconds = d.second + d.hour*3600 + d.minute*60
    k = d - timedelta(seconds=seconds % interval)
    return datetime(k.year, k.month, k.day, k.hour, k.minute, k.second)


@task(returns=list)
def preprocessing(grids, window_time, blk, target):
    import json
    from hdfspycompss.Block import Block
    ncols = len(grids)

    index_c = -1     # Index of the current instante T
    labels = ["instante"] + [i for i in xrange(1, ncols+1)]
    zeros = np.zeros(ncols).tolist()
    jam_grids = pd.DataFrame([], columns=labels)  # Partial result

    # West Longitude,South Latitude,East Longitude,North Latitude,IDgrid,Valid
    WEST = 0
    SOUTH = 1
    EAST = 2
    NORTH = 3
    VALID = 5

    div_y = grids[0][NORTH] - grids[0][SOUTH]
    init_y = grids[0][SOUTH]
    records = Block(blk).readBlock()
    for i, line in enumerate(records):
        try:
            record = json.loads(line)
            if record['city'].lower() == target:
                points = record['line']
                cTime = record['pubMillis']["$numberLong"]
                currentTime = datetime.utcfromtimestamp(float(cTime) / 1000.0)
                currentTime = group_datetime(currentTime, window_time)

                index_c = jam_grids['instante']\
                    .loc[jam_grids['instante'] == str(currentTime)]\
                    .index.tolist()

                if index_c == []:
                    row = [[str(currentTime)] + zeros]
                    index_c = len(jam_grids)
                    jam_grids = jam_grids.append(
                        pd.DataFrame(row, columns=labels, index=[index_c]))
                else:
                    index_c = index_c[0]

                # if (i % 10000 == 0):
                #     print "Line {} at {}".format(i, currentTime)

                line = [(float(pair['y']),
                        float(pair['x'])) for pair in points]
                shapely_line = LineString(line)

                # pruning the list of grids
                bound = shapely_line.bounds
                miny, minx, maxy, maxx = bound
                # print "LINE: miny {} and maxy {}".format(miny,maxy)

                p = abs(miny - init_y)
                i_min = int(p/div_y)*50
                p = abs(maxy - init_y)
                i_max = int(p/div_y)*50+49

                if i_min >= ncols:
                    # print "Line #{} - ({},{}]".format(i, i_min, i_max)
                    i_min = ncols-1
                if i_max >= ncols:
                    # print "Line #{} - ({},{}]".format(i, i_min, i_max)
                    i_max = ncols-1

                for col in xrange(i_min, i_max+1):
                    row = grids[col]
                    if row[VALID]:
                        polygon = Polygon([(row[SOUTH], row[WEST]),
                                          (row[NORTH], row[WEST]),
                                          (row[NORTH], row[EAST]),
                                          (row[SOUTH], row[EAST])])

                        shapely_poly = Polygon(polygon)
                        intersection_line = \
                            not shapely_poly.disjoint(shapely_line)
                        if intersection_line:
                            jam_grids.ix[index_c, col] += 1

        except Exception as e:
            print "Error at Line #{}. Skipping this line: {}".format(i, line)
            print e

    return jam_grids


@task(returns=list)
def mergeMatrix(jam1, jam2):
    """Merge partial matrix."""
    r1 = pd.concat([jam1, jam2]).groupby(['instante']).sum()
    return r1


def updateJamGrid(jam_grids):
    """Generate some statistical analysis."""
    events = jam_grids.sum(axis=1)
    events.ix['Total'] = events.sum()
    jam_grids = jam_grids.astype(int).clip(0, 1)

    return jam_grids, events


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
                description="Waze-jams's preprocessing script.")

    parser.add_argument('-i', '--input', required=True,
                        help='The input file path.')
    parser.add_argument('-g', '--grids', required=True,
                        help='The input of the grids list file.')
    parser.add_argument('-c', '--city', required=True,
                        help='Name of city.')
    parser.add_argument('-w', '--window', required=False,
                        help='The window time (in seconds)'
                        ' to take in count (default, 3600)',
                        type=int, default=3600)
    parser.add_argument('-f', '--numFrag', required=False,
                        help='Number of workers (cores)', type=int, default=4)
    arg = vars(parser.parse_args())

    filename = arg['input']
    grids = arg['grids']
    window_time = arg['window']
    numFrag = arg['numFrag']
    city = arg['city'].lower()

    print """
        Waze-jams's preprocessing script with the following parameters:
         - City: {}
         - Input file: {}
         - Grids file: {}
         - Window time: {} seconds
         - Number of workers: {}

    """.format(city, filename, grids, window_time, numFrag)

    grids = np.genfromtxt(grids, delimiter=',', dtype=None, names=True)

    from hdfspycompss.HDFS import HDFS
    dfs = HDFS('localhost', 9000)
    HDFS_BLOCKS = dfs.findNBlocks(filename, numFrag)
    partial_grid = [preprocessing(grids, window_time, blk, city)
                    for blk in HDFS_BLOCKS]

    from pycompss.api.api import compss_wait_on
    jam_grids_p = mergeReduce(mergeMatrix, partial_grid)
    jam_grids_p = compss_wait_on(jam_grids_p)
    jam_grids, events = updateJamGrid(jam_grids_p)

    jam_grids.to_csv("output_training.csv", sep=",", index=True, header=False)
    events.to_csv("output_counts.csv", sep=",")
