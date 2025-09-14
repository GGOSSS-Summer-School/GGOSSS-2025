#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 05:37:44 2024

@author: fernand
"""

#%% importation de librairie

import numpy as np

#%% operations sur les matrices
a = np.array([4,7,9])
type(a)

b = np.array([[1,2,3], [4,5,6]])

np.size(b)
np.shape(b)

b.size
b.shape

# multiplication matricielle
c = np.array([[2,1,3], [3,2,1]])

m = b*c

# produit matriciel 
pp1 = np.dot(b,a)

pp2 = b @ a

# transposee
b2 = b.T

# comptage
x = np.arange(10)
print(x)

x1 = np.arange(1, 11)
x2 = np.arange(1, 10.1)
print(x)


x2 = np.array([int(x) for x in x2])

x1 = np.array([float(x) for x in x1])

# compter avec un increment
y = np.arange(1,10,0.5)    # 0.5 = .5
print(y)


# matrice de zeros
zz = np.zeros(10)
print(zz)

nn = np.ones(10)
print(nn)

nn2 = np.ones((100,300))*np.nan
print(nn2)

# Slicing
bs = b[:, 0:2]
print(bs)

b1 = b[0,:]
b2 = b[1, 1:3]

# reshaping
b = np.arange(25)
c = np.reshape(b, (5,5))

b2 = np.arange(24)
d = np.reshape(b2, (2,3,4))

#%%%
number = np.arange(10)
alphabet = ['a','b','c','d','e','f','g','h','i','j']


#1
for i in number:
    print(i)
    print(i**2)
    print('bonjour')
    b2[i] = i+2
    

#2
for i in np.arange(10):

    if i==2 or i==4 or i==6:
        print('bonjour')
    else:  
        print(i)
    
#3
for i,alpha in enumerate(alphabet):
    print(f'index is: {i}, letter is {alpha}')


#4  
for _,alpha in enumerate(alphabet):
    print(f'index is: , letter is {alpha}')

#5
for num,alpha in zip(number,alphabet):
    print(f'number is: {num}, letter is {alpha}')
#6
for i,(num,alpha) in enumerate(zip(number, alphabet)):
    print(f'index is {i}, number is: {num}, letter is {alpha}')


#%% plotting avec python
import matplotlib.pyplot as plt 
import numpy as np

repertoire = "/home/fernand/Documents/formation_python/"
x = np.arange(100)
y = np.sin(x)
z = np.cos(x)

fig = plt.figure(figsize=(10,6), dpi=300)

plt.plot(x,y, '--',color='k',label="sin(X)")
plt.plot(x,z, '-*',color='m',label="cos(X)")

plt.title("Sinus et Cosinus d'un nombre X", fontsize=16)
plt.xlabel("X", fontsize=12)
plt.ylabel("sin(x) and cos(x)", fontsize=12)
plt.legend(loc=1, ncols=2)

plt.savefig(repertoire+'sinus_cosinus_x.png', dpi = fig.dpi, bbox_inches='tight')
plt.show()


#%%
# lettre
"""
yellow    y
green     g
blue      b
red       r
black     k
cyan      c
magenta   m
white     w"""

# matrice RGB (Red, Green, Blue)
[0, 0, 0] #noir
[1, 0, 0] #rouge
[0, 0.5, 0] #vert
[0, 0, 1] #bleu
[0, 1, 1] #cyan
[0.7, 0, 0.7] #magenta
[1, 1, 1] #blanc

# code
# voir google


#%% les fonctions dans python

def polynome(x):
    
    # polynome 2nd dégré
    y = (x**2) + 2*x - 1
    
    return y

number = np.arange(10)
polynom_number = polynome(number)


#%% creer une fonction et appeler dans une boucle
import numpy as np 
import matplotlib.pyplot as plt

def plot_line(x,i,repertoire):
    y = np.sin(x)
    z = np.cos(x)
    
    fig = plt.figure(figsize=(10,6), dpi=300)
    
    plt.plot(x,y, '--',color='k',label="sin(X)")
    plt.plot(x,z, '-*',color='m',label="cos(X)")
    
    plt.title(f"Sinus et Cosinus d'un nombre ex n_{i}", fontsize=16)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("sin(x) and cos(x)", fontsize=12)
    plt.legend(loc=1, ncols=2)
    
    plt.savefig(repertoire+f'sinus_cosinus_x_n_{i}.png', dpi = fig.dpi, bbox_inches='tight')
    
    return 

repertoire = "/home/fernand/Documents/formation_python/"
number_x = (np.arange(1,10,0.5), np.arange(30,60,0.5), np.arange(-20,-5,0.5))

for i,num in enumerate(number_x):
    plot_line(num, i+1, repertoire)
    
    
#%% diagramme ombrothermique
from datetime import date
import matplotlib.dates as mdates
import numpy as np 
import matplotlib.pyplot as plt

repertoire = "/home/fernand/Documents/formation_python/"

times = [date(2024, month, 1).strftime("%b") for month in range(1, 13)] 
# %b ==> 3 caracteres : Jan, Fev, Mar, ...
# %B ==> toutes les lettres : Janvier, Fevrier, Mars, ...

precip = np.array([32,95,164,247,289,364,530,600,556,397,195,43])

temp_max   = np.array([32,33,33,32,31,30,28,28,29,30,31,32])
temp_min   = np.array([23,24,24,24,23,23,22,22,23,22,23,23])

temp = [(i+j)/2 for i,j in zip(temp_max, temp_min)] # temperature moyenne

### diagramme

fig, ax1 = plt.subplots(figsize=(9,5), dpi=150)
ax2 = ax1.twinx()

ax1.bar(times,precip,color='b')
ax1.set_ylabel('Mean precip. (mm)', fontsize= 13)
ax1.set_xlabel('Time (month)', fontsize= 13)
   
ax2.plot(times,temp,'-*',color='r')
ax2.set_ylabel('max. Temp. (°C)', color='r', fontsize= 13)

ax2.tick_params(axis='y', color='r') # mettre les marqueurs du second axe en rouge

yticklabels = ax2.get_yticklabels() # mettre les valeurs de l'axe (ticklabels) en rouge
for label in yticklabels:
    label.set_color('red')
ax2.spines['right'].set_color('red') # mettre le second axe en rouge

ax1.spines['top'].set_visible(False) # retirer la barre supérieure de l'axe
ax2.spines['top'].set_visible(False)
    
plt.title('Temperature-Precipitation diagramm, Douala', fontsize= 14)

#plt.grid(which='both', axis='both')

plt.savefig(repertoire+'diag_ombro_douala.png',dpi=fig.dpi, bbox_inches='tight')

plt.show()


#%% graphique 2D de Température
repertoire = "/home/fernand/Documents/formation_python/"
file = "GG_NCEI-L4_GHRSST-SSTblend-AVHRR_OI-GLOB-v02.0-fv02.0__1981_2024_C.nc"

# pip install netCDF4

import netCDF4 as nc
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

netc = nc.Dataset(repertoire+file,'r')
sst   = netc.variables['analysed_sst'][:]
lon   = netc.variables['lon'][:]
lat   = netc.variables['lat'][:]

ds = xr.open_dataset(repertoire+file, decode_times=True)
temps = netc.variables['time'][:]

#%% distribution des donnees
import sys
sys.path.insert(1,"/home/fernand/Documents/py_files")

from distribution import distr as dist

sst2 = sst-273.15
dist(sst2[10000:], 'Temperature values (°C)', 'SST Distribution Gulf of Guinea [1981-2024]', 
     'sst_distribution_GG_1981_2024', repertoire)


#%% moyenne SST
mean_sst = np.nanmean(sst,axis=0)
mean_sst = mean_sst - 273.15 # convert °K to °C

#%% carte 2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature

fig = plt.figure(dpi=300,facecolor='w',transparent=True)
axe =plt.axes(projection=ccrs.PlateCarree(),)
ax = plt.gca()
ax.set_facecolor('w')
axe.set_extent([-7, 14, -6, 8.5])
axe.add_feature(cfeature.COASTLINE)

g = axe.gridlines(draw_labels=True, linewidth=0.2)
g.top_labels = False
g.right_labels = False
g.xlabel_style = {'size':9, 'color':'b'}
g.ylabel_style = {'size':9, 'color':'b'}

im = axe.contourf(lon,lat, mean_sst, extend="both", cmap='jet', 
                  levels=np.linspace(25,28,25))
im2 = axe.contour(lon, lat, mean_sst, np.linspace(25,28,7), linestyles='--',
                  colors='k',linewidths=0.5)
#cl = ['25°C' , '25.5°C', '26°C' , '26.5°C', '27°C' , '27.5°C', '28°C' ]
cl = [str(i)+'°C' for i in  np.linspace(25,28,7)]
fmt={}

for l, s in zip(im2.levels,cl):
    fmt[l]=s
plt.clabel(im2, im2.levels, fmt=fmt, fontsize=9, inline =True,
           inline_spacing=5)

land_hires = cfeature.NaturalEarthFeature('physical','land','10m',
                                           edgecolor='k',facecolor=[0.8, 0.8, 0.82])
axe.add_feature(land_hires)
axe.add_feature(cfeature.RIVERS)
axe.add_feature(cfeature.LAKES)
axe.add_feature(cfeature.BORDERS)

plt.title('Mean Temperature Gulf of Guinea [1981-2024]',fontsize=14)

cbar_ax = fig.add_axes([0.92, 0.13, 0.015, 0.71])
cb = fig.colorbar(im, cax=cbar_ax, orientation='vertical', 
                  drawedges=True, ticks=np.linspace(25,28,7))

cb.ax.set_title("T (°C)", fontsize=11, loc='left')
cb.ax.tick_params(labelsize=10)

plt.savefig(repertoire+'mean_sst_1981_2024_GG.png',dpi=fig.dpi, 
            bbox_inches='tight', transparent=False)

plt.show()



#%% climatologie

# importation des donnees
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import date

repertoire = "/home/fernand/Documents/formation_python/"

directory = "/home/fernand/Documents/formation_linux_nco_cdo/resultat_operation/"
file = "climatology_sst_gg_1981_2024.nc"

data = nc.Dataset(directory+file, 'r')
sst = data.variables['analysed_sst'][:]
lon = data.variables['lon'][:]
lat = data.variables['lat'][:]
data.close()

month = [date(2024, month, 1).strftime("%B") for month in range(1, 13)] 

# figure
fig, axes = plt.subplots(nrows=4, ncols=3, tight_layout=True, figsize=(6, 7), dpi=300,
                         sharex=True, sharey=True, subplot_kw={'projection': ccrs.PlateCarree()})
vmin = 22
vmax = 30

for i in range(4):
    for j in range(3):
        temp  = sst[i * 3 + j ,:,:] 
        month_name = month[i * 3 + j]
        ax = axes[i, j]
        im = ax.pcolormesh(lon, lat, temp, 
                           transform=ccrs.PlateCarree(), cmap='jet',
                           vmin=vmin, vmax=vmax)
        ax.set_title('$'+month_name+'$',fontsize=10)
        ax.set_xlabel("",fontsize=6)
        ax.set_ylabel("",fontsize=6)
        ax.set_extent([-7, 14, -6, 8.5])
        
        gl = ax.gridlines(draw_labels=True, linewidth=.08,color='k')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'fontsize': 7}
        gl.ylabel_style = {'fontsize': 7}
        
for ax in axes.flat:
    ax.add_feature(cfeature.COASTLINE, edgecolor='k')
    ax.add_feature(cfeature.LAND, color=[0.8, 0.8, 0.82])
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.LAKES)
    ax.add_feature(cfeature.BORDERS,color='k')

plt.suptitle('SST Climatology [1981-2024], Gulf of Guinea ',fontsize=14)

cbar_ax = fig.add_axes([0.1, -0.045, 0.8, 0.02]) 
cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')

cbar.ax.set_title('$T$ $($°$C)$',fontsize=9)
cbar.ax.tick_params(axis='x', labelsize=10)  

plt.savefig(repertoire+'clim_sst_GG.png', dpi=fig.dpi, bbox_inches='tight')

plt.show()

#%% anomalie + std
# repertoire = "/home/fernand/Documents/formation_python/"
# file = "GG_NCEI-L4_GHRSST-SSTblend-AVHRR_OI-GLOB-v02.0-fv02.0__1981_2024_C.nc"

directory = "/home/fernand/Documents/formation_linux_nco_cdo/resultat_operation/"
file_m = "year_monthmean_sst_gg_1981_2024.nc"
file_s = "year_monthstd_sst_gg_1981_2024.nc"

netc = nc.Dataset(directory+file_m,'r')
sst   = netc.variables['analysed_sst'][:]
sst   = sst -273.15

sst_spatial_mean = np.nanmean(sst, axis=(1,2))
sst_spatial_std  = np.nanstd(sst, axis=(1,2))
global_mean = np.nanmean(sst, axis=(0,1,2))

anom = sst_spatial_mean - global_mean

import xarray as xr
ds = xr.open_dataset(directory+file_m, decode_times=True)
temps = ds['time'][:]

#%% plot

fig = plt.figure(figsize=(14,5),dpi=250)
ax = plt.gca()

plt.plot(temps, anom, 'b',linewidth=.5)
plt.fill_between(temps, anom-sst_spatial_std, anom+sst_spatial_std, alpha=0.2, color='b')
plt.xlabel("Time")
plt.ylabel("Temp. anom. (°C)")
plt.title("Gulf of Guinea Temperature Anomaly [1981--2024]")
plt.hlines(y=0, xmin=temps[0], xmax=temps[-1:], color='k',linewidth=.5)
ax.set_xlim(xmin=temps[0], xmax=temps[-1:])

ax.xaxis.set_minor_locator(mdates.YearLocator())
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))  # Minor tick every 0.5 units
plt.grid(linewidth=.15, which='both',axis='both')

plt.savefig(repertoire+'anomalie_sst_GG.png', dpi=fig.dpi, bbox_inches='tight')
plt.show()


#%% hist
fig = plt.figure(figsize=(10,5),dpi=250)
ax = plt.gca()
positive = anom.copy()
positive = np.where(anom<0, np.nan, positive)

negative = anom.copy()
negative = np.where(anom>0, np.nan, negative)

plt.bar(temps, positive, color='r')
plt.bar(temps, negative, color='b')
plt.xlabel("Time")
plt.ylabel("Temp. anom. (°C)")
plt.title("Gulf of Guinea Temperature Anomaly [1981--2024]")
plt.hlines(y=0, xmin=temps[0], xmax=temps[-1:], color='k',linewidth=.5)
ax.set_xlim(xmin=temps[0], xmax=temps[-1:])

ax.xaxis.set_minor_locator(mdates.YearLocator())
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))  # Minor tick every 0.5 units
plt.grid(linewidth=.15, which='both',axis='both')

plt.savefig(repertoire+'anomalie_sst_GG_hist.png', dpi=fig.dpi, bbox_inches='tight')
plt.show()

#%%






#%% Animation de la temperature
"""import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# Sample data (replace with your actual data)
lats = np.linspace(-90, 90, 180)
lons = np.linspace(-180, 180, 360)
data = np.random.rand(180, 360, 100)  # Sample data with time dimension
time_steps = np.arange(data.shape[2])  # Example time steps

# Create the figure and axes
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines()
gl = ax.gridlines(draw_labels=True)
gl.right_labels = False
gl.top_labels = False

# Initialize contourf and contour plots with empty data
contourf_plot = ax.contourf([], [], [], levels=10, cmap='viridis', transform=ccrs.PlateCarree(), animated=True)
contour_plot = ax.contour([], [], [], levels=10, colors='k', transform=ccrs.PlateCarree(), animated=True)
# Add a text object for the title (initialized with an empty string)
title_text = ax.text(0.5, 1.05, "", transform=ax.transAxes, ha="center", va="center") 

# Function to initialize the animation
def init():
    for c in contourf_plot.collections:
        c.remove()
    for c in contour_plot.collections:
        c.remove()
    title_text.set_text("")  # Set the initial title to an empty string
    return contourf_plot, contour_plot, title_text


# Function to update the animation for each frame
def update(frame):
    # Remove old contours
    for c in contourf_plot.collections:
        c.remove()
    for c in contour_plot.collections:
        c.remove()

    # Plot new contours
    contourf_plot = ax.contourf(lons, lats, data[:, :, frame], levels=10, cmap='viridis', transform=ccrs.PlateCarree(), animated=True)
    contour_plot = ax.contour(lons, lats, data[:, :, frame], levels=10, colors='k', transform=ccrs.PlateCarree(), animated=True)

    # Update the title
    title_text.set_text(f"Time Step: {time_steps[frame]}")  

    return contourf_plot, contour_plot, title_text

# Create the animation
ani = FuncAnimation(fig, update, frames=data.shape[2], init_func=init, blit=False, interval=100)


# Save the animation as a GIF
writer_gif = PillowWriter(fps=10)
ani.save("cartopy_contour_animation.gif", writer=writer_gif)

# Save the animation as an MP4 (requires ffmpeg)
writer_mp4 = ani.writers['ffmpeg'](fps=10)
ani.save("cartopy_contour_animation.mp4", writer=writer_mp4)

plt.show()

"""

#%%



    




































