# -*- coding: utf-8 -*-
#!/usr/bin/env python

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"

import re
import pandas as pd
import numpy as np
import shapefile
from math import atan, tan, sin, cos, pi, sqrt, atan2, asin, radians
from shapely.geometry import shape,Polygon

WEST  = 0
SOUTH = 1
EAST  = 2
NORTH = 3

def CreateGrids(ngrid, bounds):
    ncols     = ngrid[0] * ngrid[1]
    grids     =  [[] for f in range(ncols)]
    div_y =  np.sqrt((bounds[NORTH] - bounds[SOUTH])**2) /ngrid[0]
    div_x =  np.sqrt((bounds[EAST] - bounds[WEST])**2) /ngrid[1]

    distance_y = great_circle((bounds[NORTH], 0), (bounds[SOUTH], 0))/ngrid[0]
    distance_x = great_circle((bounds[EAST], 0), (bounds[WEST], 0))/ngrid[1]

    pos_y = bounds[SOUTH]
    pos_x = bounds[WEST]
    id_grid = 1
    tmp_y = pos_y
    for c in range(ncols):
        if (c % ngrid[0] == 0):
            pos_y  = tmp_y
            tmp_y += div_y
            pos_x = bounds[WEST]

        tmp_x = pos_x+div_x
        grids[c] = [pos_x, pos_y, tmp_x, tmp_y, id_grid]
        pos_x = tmp_x
        id_grid+=1



    return grids,distance_x,distance_y

def CheckPointInPolygon(grids,sf):

    feature = sf.shapeRecords()[0]
    first = feature.shape.__geo_interface__
    shp_geom = shape(first)

    for i, g in enumerate(grids):

        grid = Polygon([ (g[WEST],g[SOUTH]),
                         (g[WEST],g[NORTH]),
                         (g[EAST],g[NORTH]),
                         (g[EAST],g[SOUTH])
                        ])

        if shp_geom.disjoint(grid):
            grids[i].append(0)

        else:
            grids[i].append(1)

    return grids


def great_circle(a, b):
    """
        The great-circle distance or orthodromic distance is the shortest
        distance between two points on the surface of a sphere, measured
        along the surface of the sphere (as opposed to a straight line
        through the sphere's interior).

        :Note: use cython in the future
        :returns: distance in meters.
    """

    EARTH_RADIUS = 6371.009
    lat1, lng1 = radians(a[0]), radians(a[1])
    lat2, lng2 = radians(b[0]), radians(b[1])

    sin_lat1, cos_lat1 = sin(lat1), cos(lat1)
    sin_lat2, cos_lat2 = sin(lat2), cos(lat2)

    delta_lng = lng2 - lng1
    cos_delta_lng, sin_delta_lng = cos(delta_lng), sin(delta_lng)

    d = atan2(sqrt((cos_lat2 * sin_delta_lng) ** 2 +
                   (cos_lat1 * sin_lat2 -
                    sin_lat1 * cos_lat2 * cos_delta_lng) ** 2),
              sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng)

    return (EARTH_RADIUS * d)*1000

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Waze-jams's preprocessing script - Create Grids")
    parser.add_argument('-g','--ngrid', type=int, default=[50,50], nargs='+',  help='The organization of the grid')
    parser.add_argument('-s','--shapefile', type=str, required=True, help='The Filepath of the shapefile (*.shp and *.dbf)')
    arg = vars(parser.parse_args())

    ngrid    = arg['ngrid']
    shp_file = arg['shapefile']
    dbf_file = re.sub('.shp$', '.dbf', shp_file)
    shp = open(shp_file, "rb")
    dbf = open(dbf_file, "rb")
    sf = shapefile.Reader(shp= shp, dbf= dbf)
    bounds     = sf.bbox

    print """
        Running: Waze-jams's preprocessing script with the following parameters:
         - bounding box:
            * North Latitude:   {}
            * South Latitude:   {}
            * East  Longitude:  {}
            * West  Longitude:  {}
         - grid:                {}
         - shapefile:           {}

    """.format(bounds[NORTH], bounds[SOUTH], bounds[EAST], bounds[WEST], ngrid, shp_file)


    grids, distance_x, distance_y = CreateGrids(ngrid, bounds)
    grids = CheckPointInPolygon(grids, sf)

    area = (distance_y*distance_x)/(1000**2)
    print """
        Distance in each edge: {:.0f}m x {:.0f}m
        Area in each grid    : {:.2f}km²
        """.format(distance_x, distance_y, area )


    grids = pd.DataFrame(grids,columns=['West Longitude','South Latitude','East Longitude','North Latitude','IDgrid','Valid'])
    grids.to_csv("output_grids.csv",sep=",",index=False)
