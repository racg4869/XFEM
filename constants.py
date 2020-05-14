# -*- coding: UTF-8 -*-

#  first or second order approximations
APPR_ORDER=1

# 
EPS=1e-10

# max time of iteration 
ITMAX=40

#  desired convergence tolerance
CONVTOL=1e-3

# labeling or points used in cubic interpolation
VoxelPointOrder={
    1:[(0,),(1,)],
    2:[(0,0),(0,1),(1,0),(1,1)],
    3:[(0, 0, 0),(0, 0, 1),(0, 1, 0),(0, 1, 1),
       (1, 0, 0),(1, 0, 1),(1, 1, 0),(1, 1, 1)],
}

VoxelPointOrder={
    1:[(0,),(1,)],
    # Figure 3.6.1 in Numerical recips
    2:[(0,0),(1,0),(1,1),(0,1)], 
    # 
    3:[(0,0,0),(1,0,0),(1,1,0),(0,1,0),
       (0,0,1),(1,0,1),(1,1,1),(0,1,1),],
}

# logger
import logging
logger=logging.getLogger("XFEM")
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler("./FastMarching.log",mode="w")
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.handlers.clear()
logger.addHandler(ch)

import numpy as np
np.set_printoptions(linewidth=np.nan)