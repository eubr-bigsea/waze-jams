# -*- coding: utf-8 -*-
#!/usr/bin/env python

__author__ = "Lucas Miguel S Ponce"
__email__  = "lucasmsp@gmail.com"


import numpy as np
import pandas as pd
def create_mean_grid(ngrid, bounds):
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

        m_y = (tmp_y + pos_y)/2
        m_x = (tmp_x + pos_x)/2
        grid = [m_y, m_x ,np.random.random()]
        pos_x = tmp_x
        grids[c] = grid


    return np.array(grids)


ngrid  = [50,50]
ncols  = ngrid[0] * ngrid[1]
bounds = [-25.645386, -25.346736, -49.389339, -49.185225]
bounds = np.reshape(bounds, (2, 2)).tolist()

grids     = create_mean_grid(ngrid, bounds)

df = pd.DataFrame(grids,columns=['lat','long','value'])
df.to_csv("df_meanPoints.csv")
print df
