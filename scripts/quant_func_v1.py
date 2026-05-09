#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 17:03:59 2026

@author: javier
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path
from matplotlib_scalebar.scalebar import ScaleBar


def select_plume(plot_xch4, mission):
    
    if mission == 'EMIT':
        pix_res = 60 #m
    elif mission == 'EnMAP' or mission == 'PRISMA' or mission == 'GF5' or mission == 'ZY1':
        pix_res = 30 #m
        
    lim_sup = 2 * np.nanstd(plot_xch4) #units of xch4_im
    
    # Función para dibujar una línea entre dos puntos
    def draw_line(ax, start, end):
        ax.plot([start[0], end[0]], [start[1], end[1]], color='r')
        
    # set these values to
    vminval = 0  # ppm
    vmaxval = lim_sup  # ppm
    fig, ax = plt.subplots(figsize=(10,8))
        
    mappable = ax.imshow(plot_xch4, vmin=vminval, vmax=vmaxval, cmap='plasma')
    fig.suptitle('1) Zoom into plume area (use the magnifying less icon). Press enter when done.\n2) Delineate plume shape by left-clicking on the edges. Press enter when done.', fontsize=17)
    cbar = plt.colorbar(mappable)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('$\mathregular{\Delta XCH_4}$ [ppm]', rotation=270, fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    scalebar = ScaleBar(pix_res, "m", length_fraction=0.25)
    ax.add_artist(scalebar)
    ax.axis('off')
    plt.rc('legend', fontsize=14)
    
    while True:
        key_pressed = plt.waitforbuttonpress()
    
        if key_pressed:
            break
        
    # Save zoom boundaries
    xlim = ax.get_xlim(); x0,xe = int(xlim[0]), int(xlim[1])
    ylim = ax.get_ylim(); y0,ye = int(ylim[1]), int(ylim[0])
    
    #print("Current zoom:")
    #print('x0,xe,y0,ye =', x0,xe,y0,ye)
    
    polygon = plt.ginput(n=-1, timeout=0, show_clicks=True, mouse_add=1, mouse_pop=3, mouse_stop=2)

    # Dibujar la línea que conecta los puntos
    for i in range(len(polygon)-1):
        draw_line(ax, polygon[i], polygon[i+1])

    plt.draw()  # Show the polygon
    plt.show()
    
    xv, yv = np.meshgrid(range(plot_xch4.shape[1]), range(plot_xch4.shape[0]), indexing='xy')
    points = np.hstack((xv.reshape((-1, 1)), yv.reshape((-1, 1))))

    path = matplotlib.path.Path(polygon)
    mask = path.contains_points(points)
    mask.shape = xv.shape
    plt.pause(0.5)
    plt.close()
    
    
    fig, ax = plt.subplots(figsize=(10,8)); 
    mappable = ax.imshow((mask*plot_xch4)[y0:ye,x0:xe], vmin=vminval, vmax=vmaxval, cmap='plasma')
    fig.suptitle('Masked Plume. \n Please close the window when done.', fontsize=20)
    cbar = plt.colorbar(mappable)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('$\mathregular{\Delta XCH_4}$ [ppm]', rotation=270, fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    scalebar = ScaleBar(pix_res, "m", length_fraction=0.25)
    ax.add_artist(scalebar)
    ax.axis('off')
    plt.rc('legend', fontsize=14)
    while plt.fignum_exists(fig.number):
        plt.pause(0.1)  

    return mask

def extract_ueff (u10, mission):
    
    if u10 < 4:
        bool_thres = False
        err_u10 = 0.5 * u10  # 50% error (<4 m/s)
    else: #Carvalho et al., (2019)
        bool_thres = True
        err_u10 = 2.0  # 2 m/s (≥4 m/s)
        
    if mission == 'EMIT':
        a, b = 0.31, 0.4 #Guanter et al., (2024)
        pix_res = 60 #m
    elif mission == 'EnMAP' or mission == 'PRISMA' or mission == 'GF5' or mission == 'ZY1':
        a, b = 0.34, 0.44 #Guanter et al., (2021); Roger et al., (2024)
        pix_res = 30 #m
        
    ueff = a*u10+b
        
    return ueff, a, err_u10, pix_res, bool_thres

def extract_Q(xch4_im, mask, u10, mission):
    
    #Ueff and other wind related parameters (+ pix_res)
    ueff, a, err_u10, pix_res, bool_thres = extract_ueff (u10, mission)
    
    #Important factor to get the IME
    M_ch4 = 0.01604246 #kg/mol - molar mass of ch4
    M_air = 0.0289644 #kg/mol - molar mass of air
    omega_air = 10332 #kg/m2 - column of dry air (generalization)
    conv_IME = (M_ch4 * omega_air) / M_air
    
    IME = np.sum(mask*xch4_im)*conv_IME*(1.e-9)*(pix_res**2) #kg, we assume that xch4_im is in ppb
    L = np.sqrt(np.sum(mask)*pix_res**2) #m
    
    Q = IME * 3600 * ueff / L #3600 to convert to kg/h
    
    
    if bool_thres: 
        err_Q = np.sqrt((IME*3600*a*2/L)**2 + (ueff*(pix_res**2)*conv_IME*(1.e-9)*3600*np.sqrt(np.sum(mask))*np.std(xch4_im)/L)**2)#err(U10) = 2 m/s
    else:
        err_Q = np.sqrt((IME*3600*a*u10*(1/2)/L)**2 + (ueff*(pix_res**2)*conv_IME*(1.e-9)*3600*np.sqrt(np.sum(mask))*np.std(xch4_im)/L)**2)#50% err U10
    
    print(f'Q = {round(Q,2)} ± {round(err_Q,2)} kg/h')
    return Q, err_Q

def emission_quantification (dxch4, u10, mission):
    
    mask = select_plume(dxch4, mission)
    Q, err_Q = extract_Q(dxch4, mask, u10, mission)
    
    return Q, err_Q









