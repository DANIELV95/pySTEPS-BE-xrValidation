# Functions and scripts to:
# - read in RMI-precitation data and transform to xarray structure
# - precipitation validation using xarray and dask
# Author: Michiel Van Ginderachter (michiel.vanginderachter@meteo.be)

# This scripts assumes that:
# - pySTEPS nowcasts are in CF1.7 compliant netCDF-format, arranged by case-number in a tarball
# - Radar QPE products are stored in an HDF5-file according to OPERA specifications
# - INCA-BE nowcasts are stored in .gz compressed GRIB(I)-files

# This scipts does:
# - Calculate deterministic (RMSE, BIAS and MAPE), probabilistic (BRIER, CRPS, Rank hist, Reliabity and FSS)
#   and contingency-based (POD, FAR, CSI and ETS) scores for every nowcast that is part of a specific case

# This script does not:
# - Average and visualize the results
#   An example of you to use the results of this script can be found in the jupyter notebook:
#   case_discussion.ipynb 

# Ricardo: As INCA-BE is not the purpose of evaluation, I commented associated lines.
# DanielV & Ricardo: Masked non finite values for evaluation, Compute Reliability, histogram, FSS,
#   and Contingency-based metrics (POD, FAR, CSI and ETS)

import xarray as xr
# import xradar as xd
import glob
import os
import tarfile
import numpy as np
import pandas as pd
import time
# import xesmf as xe
# import dask
import xskillscore as xs
import sys
import netCDF4 as nc
from pysteps import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pickle

from pysteps import rcparams
from pysteps.io.importers import import_odim_hdf5 as importer
from pysteps.io.archive import find_by_date
from pysteps.io.readers import read_timeseries
from pandas import to_datetime
import datetime as dt
import pyproj
from wradlib.io import read_opera_hdf5


# Open RMI QPE HDF-file and transform to xarray
# TODO: Still some problems with the x,y coordinates in the 15th decimal
def read_hdf5_to_xarray(fname):
    # Read the content
    fcontent = read_opera_hdf5(fname)
    # Determine the quantity
    try:
        quantity = fcontent['dataset1/data1/what']['quantity'].decode()
    except:
        quantity = fcontent['dataset1/data1/what']['quantity']
    if quantity == 'RATE':
        short_name = 'precip_intensity'
        long_name = 'instantaneous precipitation rate'
        units = 'mm h-1'
    else:
        #TODO: implement other quantities
        raise Exception(f"Quantity {quantity} not yet implemented.")
    # Create the grid
    try:
        projection = fcontent["where"]["projdef"].decode()
    except:
        projection = fcontent["where"]["projdef"]
    gridspec = fcontent["dataset1/where"]
    # - X and Y coordinates
    x = np.linspace(
        gridspec['UL_x'],
        gridspec['UL_x']+gridspec['xsize']*gridspec['xscale'],
        num = gridspec['xsize'],
        endpoint=False
        )
    x += gridspec['xscale']
    y = np.linspace(
        gridspec['UL_y'],
        gridspec['UL_y']-gridspec['ysize']*gridspec['yscale'],
        num=gridspec['ysize'],
        endpoint=False
        )
    y -= gridspec['yscale']/2
    x_2d, y_2d = np.meshgrid(x, y)
    pr=pyproj.Proj(projection)
    # - Lon and Lat coordinates
    lon, lat = pr(x_2d.flatten(),y_2d.flatten(), inverse=True)
    lon = lon.reshape(gridspec['ysize'],gridspec['xsize'])
    lat = lat.reshape(gridspec['ysize'],gridspec['xsize'])
    
    # Build the xarray dataset
    ds = xr.Dataset(
        # - data
        data_vars = {
            short_name : (
                #FIXME: is it really ['x','y'] or is should it be ['y','x']
                ['x','y'],
                fcontent["dataset1/data1/data"],
                {'long_name' : long_name, 'units': units}
                )
            },
        # - coordinates
        coords = {
            'x' : (
                ['x'],
                x,
                {
                    'axis' : 'X', 
                    'standard_name': 'projection_x_coordinate',
                    'long_name': 'x-coordinate in Cartesian system',
                    'units': 'm'
                    }
                ),
            'y' : (
                ['y'], 
                y,
                {
                    'axis' : 'Y', 
                    'standard_name': 'projection_y_coordinate',
                    'long_name': 'y-coordinate in Cartesian system',
                    'units': 'm'
                    }
                ),
            'lon' : (
                ['y','x'],
                lon,
                {
                    'standard_name': 'longitude',
                    'long_name': 'longitude coordinate',
                    'units': 'degrees_east'
                    }
                ),
            'lat' : (
                ['y','x'],
                lat,
                {
                    'standard_name': 'latitude',
                    'long_name': 'latitude coordinate',
                    'units': 'degrees_north'

                    }
                )
            }
    )
    return(ds)


# Find the tarball based on the case number
def find_tarball(pysteps_path,ncase):
    tarballs = glob.glob(
        os.path.join(pysteps_path,f"case_{ncase:02}_*[!_qpe].tar")
        )
    if len(tarballs) > 1:
        raise Exception(f"More than 1 file has case id {ncase:02}")
    return(tarballs[0])

# Get the filenames and some basic information from the tarball
def get_filenames(tarfname):
    tarball = tarfile.open(tarfname,"r")
    # filenames in tarball
    fnames = tarball.getnames()
    # case number
    ncase = int(tarfname.split("case_")[1].split("_")[0])
    # number of files
    nfiles = len(fnames)
    # time difference between nowcasts
    dates = [pd.Timestamp(fname.split("_")[2].split(".")[0]) for fname in fnames]
    dates.sort()
    startdate = dates[0]
    # - check multiple difference between nowcasts (in case of missing nowcasts)
    m_inc1 = (dates[1]-dates[0]).total_seconds()/60
    m_inc2 = (dates[-1]-dates[-2]).total_seconds()/60
    if m_inc1 == m_inc2:
        m_inc=int(m_inc1)
    else:
        m_inc3 = (dates[2]-dates[1]).total_seconds()/60
        if m_inc1 == m_inc3:
            m_inc=int(m_inc1)
        elif m_inc2 == m_inc3:
            m_inc=int(m_inc2)
        else:
            raise Exception("Could not find consitent time difference between nowcasts")
    enddate = dates[-1]+pd.Timedelta(minutes=m_inc)
    # print some information
    print(f"Case {ncase:02} contains {nfiles} pySTEPS-nowcast between:")
    print(f"    startdate: {startdate}")
    print(f"    enddate: {enddate}")
    print(f"    with a nowcast every {m_inc} minutes")
    tarball.close()
    return(fnames)

# ncname = 'C:/Workdir/Develop/pysteps/pystepsval/data/nwc/case_19_20210704.nc'
# def get_filenames(ncname):
#     ncfile = nc.Dataset(ncname,"r")
#     # get starting time
#     timezero = ncfile.variables['time'].units.split("since ")[1]
#     # case number
#     ncase = int(ncname.split("case_")[1].split("_")[0])
#     # number of files
#     # nfiles = len(fnames)
#     nfiles = 1
#     # time difference between nowcasts
#     # dates = [pd.Timestamp(timezero)]
#     # dates.sort()
#     startdate = pd.Timestamp(timezero)
#     # - check multiple difference between nowcasts (in case of missing nowcasts)
#     m_inc1 = (dates[1]-dates[0]).total_seconds()/60
#     m_inc2 = (dates[-1]-dates[-2]).total_seconds()/60
#     if m_inc1 == m_inc2:
#         m_inc=int(m_inc1)
#     else:
#         m_inc3 = (dates[2]-dates[1]).total_seconds()/60
#         if m_inc1 == m_inc3:
#             m_inc=int(m_inc1)
#         elif m_inc2 == m_inc3:
#             m_inc=int(m_inc2)
#         else:
#             raise Exception("Could not find consitent time difference between nowcasts")
#     enddate = dates[-1]+pd.Timedelta(minutes=m_inc)
#     # print some information
#     print(f"Case {ncase:02} contains {nfiles} pySTEPS-nowcast between:")
#     print(f"    startdate: {startdate}")
#     print(f"    enddate: {enddate}")
#     print(f"    with a nowcast every {m_inc} minutes")
#     tarball.close()
#     return(fnames)
# ncfile.close()

# Read a particular (pySTEPS) netCDF file from a tarball 
# and add some additional information to the Dataset
def read_netcdf(filename,tarball):
    print("Reading netCDF-file...",end="",flush=True)
    t0 = time.time()
    # get startdate from filename
    startdate = filename.split("_")[-1].split(".")[0]
    # open Dataset
    ds = xr.open_dataset(
        tarball.extractfile(tarball.getmember(filename)),
        engine = 'h5netcdf'
        )
    # move lon and lat from data variables to coordinates and add startdate
    ds = ds.assign_coords(
        lon = (("y","x"),ds.lon.data),
        lat = (("y","x"),ds.lat.data),
        startdate = ((),pd.to_datetime(startdate))
        ).rename_vars({"time":"validtime"})
    # add leadtimes to coordinates
    step = ds.validtime.data[1]-ds.validtime.data[0]
    steps = np.array([step+i*step for i in range(ds.sizes["time"])])
    ds = ds.assign_coords(leadtime = (("time"), steps))
    # chunk the data for use with dask
    ds = ds.chunk(dict(time=6,ens_number=6,y=-1,x=-1))
    print(f"done ({(time.time()-t0)/60:.2f}m)")
    return(ds)

# Read the RMI radar-QPE files given a DataArray of validdates
# The location and filename format of the radar files is taken from the pysteps.rcparams specifications
# validdates = pysteps.validtime
def read_radar(validdates):
    t0 = time.time()
    print("Reading RADAR data...",end="",flush=True)
    # get the radar paths and filename formats
    rmi = rcparams.data_sources.rmi
    root_path = 'C:/Workdir/Develop/pysteps/pystepsval/data/radar' #rmi.root_path
    path_fmt = rmi.path_fmt
    fn_pattern = '%Y%m%d%H%M%S.rad.best.comp.rate.qpe' #rmi.fn_pattern
    fn_ext = rmi.fn_ext
    importer_kwargs = rmi.importer_kwargs
    startdate = validdates.isel(time=0).data
    enddate = validdates.isel(time=len(validdates)-1).data
    # find the number of previouis timesteps
    timestep = (validdates.isel(time=1).data - startdate)/np.timedelta64(1,"m")
    nprev = int((enddate-startdate)/np.timedelta64(1,"m")/timestep)
    # build the filenames
    fns = find_by_date(
        dt.datetime.utcfromtimestamp(int(enddate)/1e9),
        root_path,
        path_fmt,
        fn_pattern,
        fn_ext,
        timestep,
        num_prev_files = nprev
           )
    
    # read the data
    r, _, meta = read_timeseries(fns, importer, **importer_kwargs)
    # Temporal #
    # print(meta["timestamps"])
    
    # convert to xarray
    # - X and Y coordinates
    x = np.linspace(meta['x1'], meta['x2'], r.shape[2]+1)[:-1]
    x += 0.5 * (x[1] - x[0]) 
    y = np.linspace(meta["y2"], meta["y1"], r.shape[1]+1)[:-1]
    y += 0.5 * (y[1] - y[0])
    # - lon and lat coordinates
    # FIXME: still some problems with lon-lat coordinates in 15th decimal
    x_2d, y_2d = np.meshgrid(x, y)
    pr=pyproj.Proj(meta['projection'])
    lon, lat = pr(x_2d.flatten(),y_2d.flatten(), inverse=True)
    lon = lon.reshape(r.shape[1],r.shape[2])
    lat = lat.reshape(r.shape[1],r.shape[2])

    if meta['unit'] == 'mm/h':
        short_name = 'precip_intensity'
        long_name = 'instantaneous precipitation rate'
        units = 'mm h-1'
    # else:
    #     #TODO: add other quantities
    #     raise Exception(f"Quantity {quantity} not yet implemented.")
    # - build the Dataset
    ds = xr.Dataset(
        # data variables
        data_vars = {
            short_name : (
                ['time','y','x'],
                r,
                {'long_name' : long_name, 'units': units}
                )
            },
        # coordinates
        coords = {
            'validtime' : (
                ['time'],
                to_datetime(meta['timestamps']),
                ),
            'x' : (
                ['x'],
                x.round(),
                {
                    'axis' : 'X',
                    'standard_name': 'projection_x_coordinate',
                    'long_name': 'x-coordinate in Cartesian system',
                    'units': 'm'
                    }
                ),
            'y' : (
                ['y'],
                y.round(),
                {
                    'axis' : 'Y',
                    'standard_name': 'projection_y_coordinate',
                    'long_name': 'y-coordinate in Cartesian system',
                    'units': 'm'
                    }
                ),
            'lon' : (
                ['y','x'],
                lon,
                {
                    'standard_name': 'longitude',
                    'long_name': 'longitude coordinate',
                    'units': 'degrees_east'
                    }
                ),
            'lat' : (
                ['y','x'],
                lat,
                {
                    'standard_name': 'latitude',
                    'long_name': 'latitude coordinate',
                    'units': 'degrees_north'

                    }
                )
            }
    )
    # chunk for use with dask
    ds = ds.chunk(dict(time=6,y=-1,x=-1))
    print(f"done ({(time.time()-t0)/60:.2f}m)")
    return(ds)

# Read an RMI-INCA .gz compressed GRIB-file given the startdate of the INCA-nowcast
# INCA root-path and path-format can be given as arguments.
# This function gives a warning concerning the STEP parameter
# and is related with the inability of GRIB-I to represent timesteps smaller than 1 hour.

# def read_inca(startdate,inca_root,path_fmt="%Y/%m/%d/precip"):
#     import gzip
#     import tempfile
#     t0 = time.time()
#     print("Reading INCA nowcast...",end="",flush=True)
#     # build the full path
#     if type(startdate) is str:
#         startdate = pd.to_datetime(startdate)
#     fname = os.path.join(
#         inca_path,
#         startdate.strftime(path_fmt),
#         startdate.strftime("%Y%m%d%H%M") + \
#             "_RR_FC_INCA.grb.gz"
#         )
#     if os.path.exists(fname):
#         # unzip the file and save as temporary file
#         with gzip.open(fname, 'rb') as gzipfile:
#             with tempfile.NamedTemporaryFile(delete=False,suffix=".grib") as tmp:
#                 tmp.write(gzipfile.read())
#         # load data with xarray
#         ds = xr.open_dataset(tmp.name,filter_by_keys={'stepUnits' : 0}).load()
#         # transform from raindepth (mm or kg m-2) to rainrate (mm h-1)
#         accutime =  np.round(ds.step.data[0]/np.timedelta64(1,'m'))
#         ds['tp'] = ds.tp / accutime * 60.0 
#         # replace some attributes
#         replacements = {'long_name': 'instantaneous precipitation rate', 'units': 'mm h-1'}
#         for key, item in replacements.items():
#             ds.tp.attrs[key] = item
#         # rename some variables and coordinates
#         ds = ds.rename_vars({"tp": "precip_intensity"})
#         ds = ds.rename_vars(
#             {'step' : 'leadtime','time':'startdate'}
#                 ).set_index(
#                     step="valid_time"
#                     ).rename_dims(
#                         {"step":'time'}
#                         ).rename_vars({'step':"validtime"})
#         step = ds.validtime.data[1]-ds.validtime.data[0]
#         steps = np.array([step+i*step for i in range(ds.dims["time"])])
#         # rearange some coordinates to unify with pySTEPS and Radar xarrays
#         ds = ds.drop_vars("leadtime").assign_coords(leadtime = (("time"), steps))
#         # chunk for use with dask
#         ds = ds.chunk(dict(time=6,x=-1,y=-1))
#         # remove temporary file
#         os.remove(tmp.name)
#     else:
#         ds = None
#     print(f"done ({(time.time()-t0)/60:.2f}m)")
#     return(ds)

# Compute Fraction Skill Score (FSS)
def frac_skill_score(observations, forecasts, dim=['y','x'], window_size=10, noise_0=1e-10):
    #Smooth data with average over window
    obs_smooth = observations.rolling(y=window_size, x=window_size, center=True).mean()
    fct_smooth = forecasts.rolling(y=window_size, x=window_size, center=True).mean()
    
    #Compute FSS
    num = (fct_smooth - obs_smooth) ** 2
    den = fct_smooth ** 2 + obs_smooth ** 2
    num_sum = num.sum(dim)
    den_sum = den.sum(dim) + noise_0
    fss = 1 - (num_sum / den_sum)
    return fss

def histogram_only(data, bins):
    return np.histogram(data, bins=bins)[0]

def usage():
    print("xrValidation [ncase]")
    print("    ncase: case number")
    print("Run the validation for case <ncase>.")
    sys.exit(1)

if __name__ == "__main__":
    # arguments = sys.argv
    # if len(arguments) != 2:
    #     usage()
    # else:
    #     ncase = int(arguments[1])
    
    #Temporal
    ncase = int(19)
    
    # General variables and path locations
    dataset = "nwc" #{"fct":"nowcast", "nwc":"nowcast blended NWP", "inca":"inca"}
    # data_path = "/home/michielv/pySTEPS-BE/data"
    data_path = "C:/Workdir/Develop/pysteps/pystepsval/data"
    # data_path = r"C:\Users\u0168535\OneDrive - KU Leuven\PhD\Shared\Data\2021-07-04"
    pysteps_path = os.path.join(data_path,dataset)
    # pysteps_path = os.path.join(data_path,'case_19_20210704')
    # inca_path = os.path.join(data_path,"inca")
    # inca_pfmt = "%Y/%m/%d"
    vali_path = os.path.join(data_path,"validation")

    # Make the directory where validation results will be saved.
    os.makedirs(vali_path,exist_ok=True)

    # Find the tarball for the specified case
    tarfname = find_tarball(pysteps_path,ncase)
    # Get the filenames inside the tarball
    filenames = get_filenames(tarfname)
    # Open the tarball
    tarball = tarfile.open(tarfname,"r")

    # Prepare the validation lists
    pysteps_det_final = []
    pysteps_prob_final = []
    pysteps_cont_final = []
    # inca_det_final = []
    # inca_prob_final = []

    # Get number of files (only needed for tracking the progress)
    nfile = 1
    nfiles = len(filenames)
    
    # Define precipitation thresholds [mm]
    thresholds = [0.1,0.5,1.0,5.0]
    
    # Define bin edges for histogram and reliability diagram
    bin_edges = np.linspace(-0.000001,1.000001,11)
    bins_x = np.arange(0.05,1,0.1)
    bn_len = len(bins_x)
    
    # Leadtimes to keep
    leadtimes = np.array([10,30,60,120,180,360])
    ileadtimes = (leadtimes/5-1).astype(int)
    # leadtimes = [np.timedelta64(i,"m") for i in leadtimes]
    
    # Start loop over the pySTEPS-nowcasts inside the tarball
    # filename = filenames[0]
    for filename in filenames:
        tstart=time.time()
        print(f"Processing file {nfile}/{nfiles} ({filename}):")
        nfile += 1

        # - get the startdate for reading the INCA-nowcast
        startdate = pd.to_datetime(filename.split("_")[-1].split(".")[0])
        # - load the pySTEPS nowcast
        pysteps = read_netcdf(filename,tarball)
        # - load the radar-observations
        radar = read_radar(pysteps.validtime)
        # TODO: lon and lat differ in the 15th decimal (~machine precision??)
        radar['lon'] = pysteps.lon
        radar['lat'] = pysteps.lat
        
        # # - load the corresponding INCA-nowcast
        # inca = read_inca(startdate,inca_path,inca_pfmt)
        # # - specify the chunks for rechunking 
        # chunks = {
        #     'ens_number' : 6,
        #     'time' : 6,
        #     'y' : -1,
        #     'x' : -1,
        #     }
            
        # t0 = time.time()
        # # - regridding
        # if filename == filenames[0]:
        #     # -- if first time regridding save the weights
        #     regridder = xe.Regridder(pysteps,inca,'nearest_s2d')
        #     fn = regridder.to_netcdf()
        # else:
        #     # -- if not first time regridding load the weights
        #     regridder = xe.Regridder(pysteps,inca,'nearest_s2d',weights=fn)
        # # -- regrid pySTEPS nowcast to INCA-BE grid and rechunk result
        # pysteps = regridder(
        #     pysteps
        #     ).drop_vars(
        #         "surface"
        #          ).compute(
        #             ).chunk(
        #                 {k:chunks[k] for k in pysteps.dims.keys()}
        #                 )
        # # -- regrid radar to INCA-BE grid and rechunk result
        # radar = regridder(
        #     radar
        #     ).drop_vars(
        #         "surface"
        #         ).compute(
        #             ).chunk(
        #                 {k:chunks[k] for k in radar.dims.keys()}
        #                 )
        # # - chunk INCA dataset (still needed?) 
        # inca = inca.chunk({k:chunks[k] for k in inca.dims.keys()})
        # tdiff = (time.time()-t0)
        # print(f"Regridding took {tdiff:.3f} seconds")
        
        #Original data
        observations = radar.precip_intensity.chunk({"time":-1})[ileadtimes]
        forecasts = pysteps.chunk({"ens_number":-1}).chunk({"time":-1}).precip_intensity[:,ileadtimes]
        
        #Masked data for finite values in both, observations and nowcasts
        nan_mask = ((np.isfinite(observations))&(np.isfinite(forecasts.mean('ens_number')))).chunk({"time":-1})
        
        obs_masked = observations.where(nan_mask).chunk({"time":-1})
        fct_masked = forecasts.where(nan_mask).chunk({"time":-1})
        
        #Boolean data when compared with thresholds
        obs_thold_bool = [obs_masked >= thold for thold in thresholds]
        obs_thold_bool = [
            obs_thold_bool[i].assign_coords(
                {"threshold":((),thresholds[i])}
                ) for i in range(len(thresholds))
            ]
        obs_thold_bool = xr.concat(obs_thold_bool,"threshold").chunk({"threshold":-1})
        
        fct_thold_bool = [fct_masked >= thold for thold in thresholds]
        fct_thold_bool = [
            fct_thold_bool[i].assign_coords(
                {"threshold":((),thresholds[i])}
                ) for i in range(len(thresholds))
            ]
        fct_thold_bool = xr.concat(fct_thold_bool,"threshold").chunk({"threshold":-1})
        
        #Data when compared with thresholds
        obs_thold = obs_masked.where(obs_thold_bool)
        obs_thold = obs_thold.transpose('threshold', 'time', 'y', 'x')
        
        fct_thold = fct_masked.where(fct_thold_bool)
        fct_thold = fct_thold.transpose('threshold', 'ens_number', 'time', 'y', 'x')
        
        # Deterministic scores
        # - RMSE
        pysteps_rmse = xs.rmse(
            obs_masked,
            fct_masked,
            ["y","x"],
            skipna=True
            )
        pysteps_rmse = pysteps_rmse.transpose('ens_number', 'time')
        
        # inca_rmse = xs.rmse(
        #     inca.precip_intensity.set_index(time="validtime"),
        #     radar.precip_intensity.set_index(time="validtime"),
        #     ["y","x"],
        #     skipna=True
        #     )
    
        # - BIAS 
        pysteps_bias = xs.me(
            obs_masked,
            fct_masked,
            ["y","x"],
            skipna=True
            )
        pysteps_bias = pysteps_bias.transpose('ens_number', 'time')
        
        # inca_bias = xs.me(
        #     radar.precip_intensity.set_index(time="validtime"),
        #     inca.precip_intensity.set_index(time="validtime"),
        #     ["y","x"],
        #     skipna=True
        #     )
    
        # - MAPE
        pysteps_mape = xs.mape(
            obs_masked,
            fct_masked,
            ["y","x"],
            skipna=True
            )
        pysteps_mape = pysteps_mape.transpose('ens_number', 'time')
        
        # inca_mape = xs.mape(
        #     radar.precip_intensity.set_index(time="validtime"),
        #     inca.precip_intensity.set_index(time="validtime"),
        #     ["y","x"],
        #     skipna=True
        #     )
        
        # - Perform calculations
        t0 = time.time()
        pysteps_rmse = pysteps_rmse.compute() 
        pysteps_bias = pysteps_bias.compute()
        pysteps_mape = pysteps_mape.compute()
    
        # inca_rmse = inca_rmse.compute()
        # inca_bias = inca_bias.compute()
        # inca_mape = inca_mape.compute()
        tdiff = (time.time()-t0)
        print(f"Calculating DETERMINISTIC took {tdiff:.3f} seconds")
    
        # - Merge Datasets
        names = ['rmse','bias','mape']
        pysteps_det = [pysteps_rmse,pysteps_bias,pysteps_mape] 
        # inca_det = [inca_rmse,inca_bias,inca_mape]
        for i,_ in enumerate(pysteps_det):
            pysteps_det[i].name = names[i]
            # inca_det[i].name = names[i]
    
        pysteps_det = xr.merge(pysteps_det)
        # inca_det = xr.merge(inca_det).rename_vars({"time":"validtime"})
        # inca_det["leadtime"] = pysteps_det["leadtime"]
        pysteps_det_final.append(pysteps_det)
        # inca_det_final.append(inca_det)
        del pysteps_det
        # del inca_det
        tdiff = (time.time()-tstart)
        print(f"Total time processing the file took {tdiff/60:.2f} minutes")
        
        # Probabilistic scores
        # - Brier
        pysteps_brier = xs.threshold_brier_score(
            obs_masked,
            fct_masked,
            thresholds,
            issorted=False,
            member_dim="ens_number",
            dim=['y','x'],
            )
        pysteps_brier = pysteps_brier.transpose('threshold', 'time')
        # inca_brier = xs.threshold_brier_score(
        #     radar.precip_intensity.set_index(time="validtime").sel(time=inca.validtime),
        #     inca.expand_dims("ens_number",axis=0).precip_intensity.set_index(time="validtime"),
        #     thresholds,
        #     issorted=False,
        #     member_dim="ens_number",
        #     dim=['y','x'],
        #     )
        
        # - CRPS
        pysteps_crps = xs.crps_ensemble(
            obs_masked,
            fct_masked,
            issorted=False,
            member_dim="ens_number",
            dim=['y','x'],
            )
        # inca_crps = xs.crps_ensemble(
        #     radar.precip_intensity.set_index(time="validtime").sel(time=inca.validtime),
        #     inca.expand_dims("ens_number",axis=0).precip_intensity.set_index(time="validtime"),
        #     issorted=False,
        #     member_dim="ens_number",
        #     dim=['y','x'],
        #     )
        
        # - Rank histogram without thresholds
        # pysteps_rhist = xs.rank_histogram(
        #     obs_masked,
        #     fct_masked,
        #     dim=['y','x'],
        #     member_dim='ens_number'
        #     )
        
        # Rank histogram by threshold
        pysteps_rhist = xs.rank_histogram(
            obs_thold,
            fct_thold,
            dim=['y','x'],
            member_dim='ens_number'
            )
        
        # - Reliability
        pysteps_rel = xs.reliability(
                obs_thold_bool,
                fct_thold_bool.mean('ens_number'),
                dim=['y','x'],
                probability_bin_edges=bin_edges
                )
                
        # - Histogram of forecast over threshold to plot in Reliability Diagram using xarray        
        fct_thold_bool_mean = fct_thold_bool.mean('ens_number')
        hist = xr.apply_ufunc(histogram_only,
                              fct_thold_bool_mean,
                              kwargs={'bins':bin_edges},
                              input_core_dims=[['x', 'y']],
                              output_core_dims=[['bins_x']],
                              dask_gufunc_kwargs={'output_sizes': {'bins_x':bn_len}},
                              dask='parallelized',
                              vectorize=True)
        hist = hist.assign_coords(bins_x=bins_x)
        
        # - Fraction Skill Score (FSS)
        pysteps_fss = frac_skill_score(
                obs_thold_bool,
                fct_thold_bool,
                dim=['y','x'],
                window_size = 10
                )

        # - Perform calculations
        t0 = time.time()
        pysteps_brier = pysteps_brier.compute()
        pysteps_crps = pysteps_crps.compute()
        pysteps_rhist = pysteps_rhist.compute()
        pysteps_rel = pysteps_rel.compute()
        pysteps_hist = hist.compute()
        pysteps_fss = pysteps_fss.compute()
        
        # inca_brier = inca_brier.compute()
        # inca_crps = inca_crps.compute()
        tdiff = (time.time()-t0)
        print(f"Calculating PROBABILISTIC took {tdiff:.3f} seconds")
        
        # - Merge Datasets
        names = ['brier','crps','rhist','rel', 'hist','fss']
        pysteps_prob = [pysteps_brier,pysteps_crps,pysteps_rhist,pysteps_rel,pysteps_hist,pysteps_fss]
        # inca_prob = [inca_brier,inca_crps]
        for i in range(len(pysteps_prob)):
            pysteps_prob[i].name = names[i]
        # for i in range(len(inca_prob)):
        #     inca_prob[i].name = names[i]
        
        pysteps_prob = xr.merge(pysteps_prob)
        # inca_prob = xr.merge(inca_prob).drop_vars("time")
        pysteps_prob_final.append(pysteps_prob)
        # inca_prob_final.append(inca_prob)
        del pysteps_prob
        # del inca_prob
        tdiff = (time.time()-tstart)
        print(f"Total time processing the file took {tdiff/60:.2f} minutes")

        #Contingency Table, ETS and CSI
        pysteps_contingency = xs.Contingency(
            obs_thold_bool,
            fct_thold_bool,
            observation_category_edges=np.array([0,0.5,1]),
            forecast_category_edges=np.array([0,0.5,1]),
            dim=["y","x"]
            )
        
        # - Perform calculations
        t0 = time.time()
        # pysteps_contingency_table = pysteps_contingency.table.assign_coords(time=fct_thold_bool['leadtime'].values).compute()
        pysteps_pod = pysteps_contingency.hit_rate().assign_coords(time=fct_thold_bool['leadtime'].values).compute()
        pysteps_far = pysteps_contingency.false_alarm_ratio().assign_coords(time=fct_thold_bool['leadtime'].values).compute()
        pysteps_ets = pysteps_contingency.equit_threat_score().assign_coords(time=fct_thold_bool['leadtime'].values).compute()
        pysteps_csi = pysteps_contingency.threat_score().assign_coords(time=fct_thold_bool['leadtime'].values).compute()
        
        tdiff = (time.time()-t0)
        print(f"Calculating CONTINGENCY took {tdiff:.3f} seconds")
        
        # - Merge Datasets
        names = ['pod','far','ets','csi']
        pysteps_cont = [pysteps_pod,pysteps_far,pysteps_ets,pysteps_csi]
        # inca_prob = [inca_brier,inca_crps]
        for i in range(len(pysteps_cont)):
            pysteps_cont[i].name = names[i]
        # for i in range(len(inca_prob)):
        #     inca_prob[i].name = names[i]
        
        pysteps_cont = xr.merge(pysteps_cont)
        # inca_prob = xr.merge(inca_prob).drop_vars("time")
        pysteps_cont_final.append(pysteps_cont)
        # inca_prob_final.append(inca_prob)
        del pysteps_cont
        # del inca_prob
        tdiff = (time.time()-tstart)
        print(f"Total time processing the file took {tdiff/60:.2f} minutes")
        
    # End of loop over filenames, close the tarball
    tarball.close()
    
    # Concatenate the validation lists to xarray Dataset and save the results.
    xr.concat(pysteps_det_final,'startdate').to_netcdf(
        os.path.join(vali_path,f"{ncase:02}_pysteps_det_{dataset}.nc")
        )
    xr.concat([ds.drop_indexes("validtime") for ds in pysteps_prob_final],"startdate").to_netcdf(
        os.path.join(vali_path,f"{ncase:02}_pysteps_prob_{dataset}.nc")
        )
    xr.concat(pysteps_cont_final,"startdate").to_netcdf(
        os.path.join(vali_path,f"{ncase:02}_pysteps_cont_{dataset}.nc")
        )
    # xr.concat([ds.drop_indexes("validtime") for ds in inca_det_final],"startdate").to_netcdf(
    #     os.path.join(vali_path,f"{ncase:02}_inca_det.nc")
    #     )
    # xr.concat(inca_prob_final,"startdate").to_netcdf(
    #     os.path.join(vali_path,f"{ncase:02}_inca_prob.nc")
    #     )
    

    # #################################################
    # # Define lead times to plot
    # lead_times = [10, 30, 60] #, 120, 180, 360] #minutes range(5,60,5) #
    # timestep = 5
    # lead_times_id = (np.array(lead_times)/timestep-1).astype(int) #convert minutes to index
    
    # #Set probability (x axis) to plot
    # # prob_x = np.linspace(0.05,0.95,10) #center of each bin
    # prob_x = [0.,0.1,0.2,0.3,0.4,0.55,0.7,0.8,0.9,1.] #Edge of each bin
    
    # # # Make default plot for reldiag
    # # fig, ax = plt.subplots()
    # # ax.plot([0,1],[0,1],ls='--', c='k')
    # # ax.set_xlim(0,1)
    # # ax.set_ylim(0,1)
    # # ax.set_xlabel('Forecast probability')
    # # ax.set_ylabel('Observed relative frequency')
    # # ax.grid(visible=True, alpha=0.5)
    # # fig.savefig('./Pictures/base_plot_reldiag.png', dpi=300)
    
    # # # Save default plot for reldiag
    # # with open('./Pictures/base_plot_reldiag.pkl', 'wb') as f:
    # #     pickle.dump(fig, f)
    
    # # #Comparison time reliability plot xarray vs pysteps
    # # # Plot using xarray
    # # t0 = time.time()
    # # pysteps_rel = xs.reliability(
    # #         obs_thold_bool,
    # #         fct_thold_bool.mean('ens_number'),
    # #         dim=['y','x'],
    # #         probability_bin_edges=bin_edges
    # #         )
    # # pysteps_rel = pysteps_rel.compute()
    
    # # P_fx = fct_thold_bool.mean('ens_number')
    # # P_fx[0,0].plot.hist(bins=bin_edges, color='grey', edgecolor='black')
    
    # # t0 = time.time()
    # for th in range(len(thresholds)):
    #     with open('./Pictures/base_plot_reldiag.pkl', 'rb') as f:
    #         fig = pickle.load(f)
    #     ax = fig.axes[0]
        
    #     ibox, jbox = 0, 0
    #     for lt in lead_times_id:
    #         ax.plot(prob_x, pysteps_rel[th,lt], marker='o', alpha=0.7)
            
    #         # Plot sharpness diagram into an inset figure.
    #         if len(lead_times_id) == 1:
    #             if pysteps_rel[th,lt].mean() <= 0.5:
    #                 iax = inset_axes(ax, width="30%", height="20%", loc='upper left', borderpad=0.5) #, bbox_to_anchor=(60,210,80,40))
    #                 iax.yaxis.tick_right()
    #                 iax.yaxis.set_label_position("right")
    #             else:
    #                 iax = inset_axes(ax, width="30%", height="20%", loc='bottom right', borderpad=1.5)
    #         else:
    #             ibox += 1
    #             iax = inset_axes(ax, width="100%", height="100%", bbox_to_anchor=(420+100*jbox,270-55*ibox,80,40))
            
    #         P_fx[th,lt].plot.hist(bins=bin_edges, color='grey', edgecolor='black', ax=iax)
    #         iax.set_title(str((lt+1)*5), y=1.0, pad=-7, size=7)
    #         iax.set_yscale("log")
    #         iax.set_xlabel("")
    #         iax.set_xticks(bin_edges)
    #         iax.set_xticklabels(["%.1f" % max(v, 1e-6) for v in bin_edges])
    #         iax.set_xlim(0.0, 1.0)
    #         if jbox == 0:
    #             iax.set_ylabel("log(samples)", size=7, labelpad=1)
    #         else:
    #             iax.set_ylabel("")
    #         iax.tick_params(axis="both", which="major", labelsize=5, pad=2)
            
    #         if ibox == 4:
    #             ibox = 0
    #             jbox += 1
            
    # tdiff = (time.time()-t0)
    # print(f"Total time processing the file took {tdiff:.3f} seconds")
    
    # t0 = time.time()
    # # fig, ax = plt.subplots()
    # for thold in thresholds:
    #     fig, ax = plt.subplots()
    #     for lt in lead_times_id:
    #     # for i in range(0,12):
    #         # t0 = time.time()
    #         # fig, ax = plt.subplots()
    #         reldiag = verification.reldiag_init(thold)
    #         P_f = postprocessing.ensemblestats.excprob(fct_masked[:, lt, :, :], thold, ignore_nan=True)
    #         R_o = obs_masked[lt, :, :].values
    #         verification.reldiag_accum(reldiag, P_f, R_o)
    #         verification.plot_reldiag(reldiag, ax)
    #     plt.show()
    # # reldiag["Y_sum"] / reldiag["num_idx"]
    # # reldiag["X_sum"] / reldiag["num_idx"]
    # tdiff = (time.time()-t0)
    # print(f"Total time processing the file took {tdiff:.3f} seconds")
    
    # #Plot Rank histogram
    # for th in range(len(thresholds)):
    #     fig, ax = plt.subplots()
    #     for lt in lead_times_id:
    #         pysteps_rhist[th,lt].plot(marker='_', ms=20, ls='', label=lt)
    #         # pysteps_rhist[lt].plot(marker='_', ms=20, ls='', label=lt)
    #         plt.bar(pysteps_rhist['rank'], pysteps_rhist[th,lt], alpha=0.5, log=True)
    #         plt.legend(loc='upper right')
    #         plt.xlabel('Rank')
    #         plt.ylabel('Count')
    
    # #Plot CSI
    # for th in range(len(thresholds)):
    #     fig, ax = plt.subplots()
    #     for lt in lead_times_id:
    #         pysteps_csi[th,lt].plot(label=lt)
    #     plt.legend(loc='lower right')
    #     plt.xlabel('Member')
    #     plt.ylabel('CSI')
    



