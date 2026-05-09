#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 17:03:59 2026

@author: javier
"""

import numpy as np
from wind_v1 import wind_speed_manual
from quant_func_v1 import emission_quantification


#Input Values

pbase = '/home/javier/Desktop/After_PhD/github_code/' #base path for inputs and outputs

#Scene timestamp and location (it can be obtained from acquisition metadata)
ts = 20220827060753 #ts:#YYYYMMDDHHMMSS format
lat, lon = 36.25758,61.73344 #lat,lon representing approximately the scene location

#Our methane enhancement map (dxch4) is obtained from an EMIT scene (saved as .npy format)
mission = 'EMIT' #other alternatives: EnMAP, PRISMA, GF5, ZY1
n_dxch4 = 'EMIT_L1B_RAD_001_20220827T060753_2223904_013_dXCH4'
dxch4 = np.load(pbase + n_dxch4 + '.npy') #Import dxch4 (must be in ppb)


#Outputs
u10 = wind_speed_manual(ts, lat, lon, pbase) #Obtain wind speed (m/s) from GEOS-FP
Q, err_Q = emission_quantification (dxch4, u10, mission) #Obtain emission flux rate (kg/h) with uncertainty






