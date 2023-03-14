import os
import numpy as np
import pandas as pd

# For displaying warnings
import warnings

# For finding paths
from . import config

# To import and export netCDF files
import netCDF4 as nc

# To convert arrays in a format that can be exported to netCDF files
import xarray as xr

# To plot heatmaps of the event
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# To make a gif from a series of jpeg files
import PIL

# To fit the return period using the GAM method
from pygam import LinearGAM, s

class Hazard:
	'''
	Hazard Class:

	:attribute lon: array, list of the longitudes covered by the hazard
	:attribute lat: array, list of the latitude covered by the hazard
	:attribute time: array, list of the datetimes covered by the hazard
	:attribute intensity: array, The intensity of the hazard event for each datetime, lat and lon

	:method read_nc(filename):
	:method geospatial_generator(geodata1, geodata2, delta_km):
	:method datetime_generator(sdate,edate,frequency):
	:method geospatialGrid_generator():
	:method epicenterTrajectory_generator(epicenter_radius, epicenter_intensity, epicenter_lat, epicenter_lon):
	:method epicenterTrajectory_reader(filename):
	:method intensityGrid_generator(max_intensity, max_radius):
	:method grid_generator(max_intensity, max_radius):
	:method to_nc(df, filename, intensity_unit='m/s', intensity_name='wind speed', title='Sythetic storm'):
	:method get_intensity(lon, lat, startTime=None, endTime=None):
	:method plot(time_idx):
	:method plot_gif(speed=3)
	'''

	def __init__(self, simulationName):
		# variables read from the ncdf file
		self.intensity = None
		self.lon = None
		self.lat = None
		self.time = None

		# internal/private variables
		self.latitudes = []
		self.longitudes = []
		self.datetimes = []
		self.df_trajectory = pd.DataFrame()
		self.simulationName = simulationName

	def hazardFromNC(self, filename):
		'''
		:param filename: string, filename of the nc file. Must end with .nc and must be in the directory: file/input/*project name*/hazards/
		
		Initializes the Hazard object using a .nc file
		'''
		self.clear()
		self.read_nc(filename)
	
	def hazardFromTrajectory(self, filename, 
								max_intensity, max_radius,
								sdate, edate, geodata1, geodata2, 
								delta_km, frequency='1H'):
		'''
		:param filename: string, filename of the csv file. Must end with .csv and must be in the directory: file/input/*project name*/hazards/
		:param max_intensity: float, the rated intensity used for p.u. values in the trajectory data.
		:param max_radius: float, the rated radius used for p.u. values in the trajectory data.
		:param sdate: datetime, start time of the hazard event, can be generated using datetime.date(year,month,day)
		:param edate: datetime, end time of the hazard event, can be generated using datetime.date(year,month,day)
		:param geodata1: network.Geodata, first corner of the geospactial range covered by the hazard
		:param geodata2: network.Geodata, second corner of the geospactial range covered by the hazard
		:param delta_km: float, geospatial resolution of the intensity data in km
		:param frequency: string, time intervals, default='1H'

		Initializes the Hazard object using a .csv trajectory file
		'''
		self.clear()

		self.datetime_generator(sdate,edate,frequency)
		self.geospatial_generator(geodata1, geodata2, delta_km)
		self.epicenterTrajectory_reader(filename)

		hazard_grid = self.grid_generator(max_intensity, max_radius)
		nc_filename = filename.split('.')[0] + '.nc'
		self.to_nc(hazard_grid, nc_filename)
		self.read_nc(nc_filename)

	def hazardFromStaticInput(self, filename, 
								max_intensity, max_radius,
								sdate, edate, geodata1, geodata2, delta_km,
								epicenter_lat, epicenter_lon, 
								frequency='1H', epicenter_radius=1, epicenter_intensity=1):
		'''
		:param filename: string, filename of the csv file. Must end with .csv and must be in the directory: file/input/*project name*/hazards/
		:param max_intensity: float, the rated intensity used for p.u. values in the trajectory data.
		:param max_radius: float, the rated radius used for p.u. values in the trajectory data.
		:param sdate: datetime, start time of the hazard event, can be generated using datetime.date(year,month,day)
		:param edate: datetime, end time of the hazard event, can be generated using datetime.date(year,month,day)
		:param geodata1: network.Geodata, first corner of the geospactial range covered by the hazard
		:param geodata2: network.Geodata, second corner of the geospactial range covered by the hazard
		:param delta_km: float, geospatial resolution of the intensity data in km
		:param epicenter_lat: float, latitude of the epicenter of the hazard event
		:param epicenter_lon: float, longitude of the epicenter of the hazard event
		:param frequency: string, time intervals, default='1H'
		:param epicenter_radius: float, radius of the hazard event from the epicenter in p.u., default = 1 p.u.
		:param epicenter_intensity: float, intensity of the hazard event at the epicenter in p.u., default = 1 p.u.

		Initializes the Hazard object using the inputs provided.
		'''
		self.clear()

		self.datetime_generator(sdate,edate,frequency)
		self.geospatial_generator(geodata1, geodata2, delta_km)
		self.epicenterTrajectory_generator(epicenter_radius, epicenter_intensity, epicenter_lat, epicenter_lon)

		hazard_grid = self.grid_generator(max_intensity, max_radius)
		self.to_nc(hazard_grid, filename)
		self.read_nc(filename)

	def clear(self):
		'''
		Clears all class attributes
		'''
		self.intensity = None
		self.lon = None
		self.lat = None
		self.time = None
		self.latitudes = []
		self.longitudes = []
		self.datetimes = []
		self.df_trajectory = pd.DataFrame()

	def read_nc(self, filename):
		'''
		:param filename: string, filename of the nc file. Must end with .nc and must be in the directory: file/input/*project name*/hazards/
		:return cols: list,
		:return Hazard.lon: array, list of the longitudes covered by the hazard
		:return Hazard.lat: array, list of the latitude covered by the hazard
		:return Hazard.time: array, list of the datetimes covered by the hazard
		:return Hazard.intensity: array, The intensity of the hazard event for each datetime, lat and lon

		Reads the nc file provided in file/input/*project name*/hazards/ and populates the hazards attributes.
		'''
		ncdf = nc.Dataset(config.path.hazardFile(self.simulationName, filename), mode='r')
		cols = []
		for att in ncdf.variables:
			if att == 'longitude':
				self.lon = ncdf[att][:]
			elif att == 'latitude':
				self.lat = ncdf[att][:]
			elif att == 'DateTime':
				time_tmp = ncdf.variables[att]
				self.time = nc.num2date(
					time_tmp[:], time_tmp.units, only_use_cftime_datetimes=False)
				self.time = pd.to_datetime(self.time)
			elif att == 'intensity':
				self.intensity = ncdf[att][:]
			else:
				cols.append(att)
		ncdf.close()
		return cols, self.lon, self.lat, self.time, self.intensity

	def geospatial_generator(self, geodata1, geodata2, delta_km):
		'''
		:param geodata1: network.Geodata, first corner of the geospactial range covered by the hazard
		:param geodata2: network.Geodata, second corner of the geospactial range covered by the hazard
		:param delta_km: float, geospatial resolution of the intensity data in km
		:return Hazard.latitudes: np.array, range of latitudes covered by the hazard
		:return Hazard.longitudes: np.array, range of longitudes covered by the hazard

		Calculated the geospatial ranges covered by the hazard
		'''
		delta = delta_km/111 # delta in degrees for lattitude every and longitude near the equator
		self.latitudes = np.arange(min(geodata1.latitude, geodata2.latitude),max(geodata1.latitude, geodata2.latitude), delta)
		self.longitudes = np.arange(min(geodata1.longitude, geodata2.longitude),max(geodata1.longitude, geodata2.longitude), delta)
		return self.latitudes, self.longitudes

	def datetime_generator(self, sdate,edate,frequency='1H'):
		'''
		:param sdate: datetime, start time of the hazard event, can be generated using datetime.date(year,month,day)
		:param edate: datetime, end time of the hazard event, can be generated using datetime.date(year,month,day)
		:param frequency: string, time intervals, default='1H'
		:return Hazard.datetimes: list, list datetime objects

		Generates the time range over which the hazard intensity is provided
		'''	
		self.datetimes = pd.date_range(sdate,edate,freq=frequency)
		return self.datetimes

	def geospatialGrid_generator(self):
		'''
		:return grid_latitudes: list, list of latitudes for generatiing the ncdf file
		:return grid_longitudes: list, list of longitudes for generatiing the ncdf file
		:return grid_datetimes: list, list of datetimes for generatiing the ncdf file

		Generates the geospacial and temporal grid of generation the ncdf file
		'''	
		grid_latitudes, grid_longitudes = [],[]
		for lat in self.latitudes:
			grid_latitudes.append([lat]*len(self.longitudes))
			grid_longitudes.append(self.longitudes)
		grid_latitudes = np.array(grid_latitudes).flatten()
		grid_longitudes = np.array(grid_longitudes).flatten()
		
		grid_datetimes = []
		for dt in self.datetimes:
		    grid_datetimes.append([dt]*len(grid_longitudes))
		grid_datetimes = np.array(grid_datetimes).flatten()
		grid_latitudes = np.array([grid_latitudes]*len(self.datetimes)).flatten()
		grid_longitudes = np.array([grid_longitudes]*len(self.datetimes)).flatten()

		return grid_latitudes, grid_longitudes, grid_datetimes

	def epicenterTrajectory_generator(self, epicenter_radius, epicenter_intensity,
							  				epicenter_lat, epicenter_lon):
		'''
		:param epicenter_radius: float, radius of the hazard event from the epicenter in p.u.
		:param epicenter_intensity: float, intensity of the hazard event at the epicenter in p.u.
		:param epicenter_lat: float, latitude of the epicenter of the hazard event
		:param epicenter_lon: float, longitude of the epicenter of the hazard event
		:return Hazard.df_trajectory: pandas.Dataframe, trajectory, and intensity of the epicenter of the hazard event.

		Calculates the df_trajectory attribute from the provided inputs
		'''	
		self.df_trajectory = pd.DataFrame()
		self.df_trajectory['time'] = self.datetimes
		self.df_trajectory['lat'] = [epicenter_lat]*len(self.df_trajectory.index)
		self.df_trajectory['lon'] = [epicenter_lon]*len(self.df_trajectory.index)
		self.df_trajectory['intensity'] = [epicenter_intensity]*len(self.df_trajectory.index)
		self.df_trajectory['radius'] = [epicenter_radius]*len(self.df_trajectory.index)
		return self.df_trajectory

	def epicenterTrajectory_reader(self, filename):
		'''
		:param filename: string, filename of the csv file. Must end with .csv and in the directory: file/input/*project name*/hazards/
		:return Hazard.df_trajectory: pandas.Dataframe, trajectory, and intensity of the epicenter of the hazard event.

		Reads the df_trajectory attribute from the csv file provided in file/input/*project name*/hazards/
		'''	
		self.df_trajectory = pd.read_csv(config.path.hazardFile(self.simulationName, filename), 
										sep=';',header=0)
		self.df_trajectory['time'].apply(pd.to_datetime)
		return self.df_trajectory

	def intensityGrid_generator(self, max_intensity, max_radius):
		'''
		:param max_intensity: float, the rated intensity used for p.u. values in the trajectory data.
		:param max_radius: float, the rated radius used for p.u. values in the trajectory data.
		:return grid_intensity: list, intensity values from each geospatial and temporal point

		Calculates the intensity values from the df_trajectory attribute.
		'''	
		grid_intensity = []
		for time in self.df_trajectory['time']:
		    t_lon = self.df_trajectory.loc[self.df_trajectory['time'] == time, 'lon'].iloc[0]
		    t_lat = self.df_trajectory.loc[self.df_trajectory['time'] == time, 'lat'].iloc[0]
		    intensity = max_intensity*self.df_trajectory.loc[self.df_trajectory['time'] == time, 'intensity'].iloc[0]
		    radius = max_radius/111*self.df_trajectory.loc[self.df_trajectory['time'] == time, 'radius'].iloc[0]

		    for lat in self.latitudes:
		        for lon in self.longitudes:
		            dist = (lon - t_lon)**2 + (lat - t_lat)**2
		            if dist == 0:
		                grid_intensity.append(intensity)
		            elif dist <= radius**2:
		                grid_intensity.append(intensity*(1-(dist/radius**2)))
		            else:
		                grid_intensity.append(0)
		return grid_intensity

	def grid_generator(self, max_intensity, max_radius):
		'''
		:param max_intensity: float, the rated intensity used for p.u. values in the trajectory data.
		:param max_radius: float, value given in km. the rated radius used for p.u. values in the trajectory data.
		:return df: pandas.DataFrame, contains all the geospatial and temporal data with the intensity values.

		Creates an .nc file from the pandas dataframe containing the intensity for different geospacial and temporal points.
		the file will be created in: file/input/*project name*/hazards/
		'''	
		colnames = ['DateTime', 'latitude', 'longitude', 'intensity']
		grid_latitudes, grid_longitudes, grid_datetimes = self.geospatialGrid_generator()
		grid_intensity = self.intensityGrid_generator(max_intensity, max_radius)
		df = pd.DataFrame(list(zip(grid_datetimes, grid_latitudes, grid_longitudes, grid_intensity)), columns=colnames)
		return df

	def to_nc(self, df, filename, intensity_unit='m/s', intensity_name='wind speed', title='Sythetic storm'):
		'''
		:param df: pandas.DataFrame, use the function hazard.grid_generator() to generate a dataframe with all the data needed to generate the ncdf file
		:param filename: string, ncdf filename, must end with .nc
		:param intensity_unit: string, the unit of the intensity used for the ncdf file, default='m/s'
		:param intensity_name: string, the attribute name of the intensity used for the ncdf file, default='wind speed'
		:param title: string, title used for the ncdf file, default='Sythetic storm'

		Creates an .nc file from the pandas dataframe containing the intensity for different geospacial and temporal points.
		the file will be created in: file/input/*project name*/hazards/
		'''		

		# create xray Dataset from Pandas DataFrame
		xr1 = df.set_index(['DateTime', 'latitude', 'longitude']).to_xarray()
		xr1['latitude'].attrs={'units':'degrees', 'long_name':'Latitude'}
		xr1['longitude'].attrs={'units':'degrees', 'long_name':'Longitude'}
		xr1['intensity'].attrs={'units':intensity_unit, 'long_name':intensity_name}
		xr1.attrs={'Conventions':'CF-1.6', 'title':title, 'summary':'Syntheticly generate hazard'}

		# save to netCDF usinf the filename provided as input
		xr1.to_netcdf(config.path.hazardFile(self.simulationName, filename))

	def get_intensity(self, lon, lat, startTime=None, endTime=None):
		'''
		:param lon: float, in degrees
		:param lat: float, in degrees
		:param startTime: DateTime, default= start of hazard
		:param endTime: DateTime, default= end of hazard
		:return t: list, DateTime list
		:return att: list, float list of the intensity values for the given time frame

		The nearest longitude at lattitude values available in the ncdf file will be returned.
		if the lon and lat are outside the range provided in the ncdf file a value of zero is returned.
		
		It is possible to convert strings to the correct datetime format using pd.to_datetime()
		'''

		# if startTime and Endtime are not provided use the entire hazard time by default
		if startTime == None:
			idx_startTime = self.time.get_loc(self.datetimes[0], method='nearest')
		else:
			idx_startTime = self.time.get_loc(startTime, method='nearest')
		if endTime == None:
			idx_endTime = self.time.get_loc(self.datetimes[-1], method='nearest')
		else:
			idx_endTime = self.time.get_loc(endTime, method='nearest')
		
		t = self.time[idx_startTime:idx_endTime]

		# if the longitude and lattitude provided are beyond the defined limits return an intensity of zero.
		if lon < min(self.lon) or lon > max(self.lon) or lat < min(self.lat) or lat > max(self.lat):
			att = np.zeros(t.size)
		else:
			lon_approx_idx = min(range(len(self.lon)),
								key=lambda i: abs(self.lon[i]-lon))
			lat_approx_idx = min(range(len(self.lat)),
								key=lambda i: abs(self.lat[i]-lat))
			att = self.intensity[idx_startTime:idx_endTime, lat_approx_idx, lon_approx_idx]
		
		return t, att

	def plot(self, time_idx, projection='cyl', edge_pad=0):
		'''
		:param time_idx: int, should be between 0 and the maximum number of timesteps defined in the Hazard
		:param projection: string, projection used by Basemap, default='cyl' espg projections can be used as well
		:param edge_pad: int, shows more of the map by adding additional lat and lon degrees
		:return fig: matplotlib.pyplot figure
		:return ax: matplotlib.pyplot axis

		Plots the the hazard intensity on a heatmap for a given timestep.
		'''

		mp = Basemap(llcrnrlon=min(self.lon)-edge_pad,   # lower longitude
		             llcrnrlat=min(self.lat)-edge_pad,    # lower latitude
		             urcrnrlon=max(self.lon)+edge_pad,   # uppper longitude
		             urcrnrlat=max(self.lat)+edge_pad,
		             projection=projection, resolution ='i',area_thresh=1000.)

		# Here you can plot the intensity over the map for a specific time
		longs, lats = np.meshgrid(self.lon, self.lat)  #this converts coordinates into 2D arrray
		x,y = mp(longs, lats) #mapping them together 
		
		fig, ax = plt.subplots(tight_layout=True)
		
		intensity_matrix = np.copy(np.squeeze(self.intensity[time_idx,:,:]))
		mask = np.ones((intensity_matrix.shape[0],intensity_matrix.shape[1])).astype(bool)
		mask = np.logical_and(mask, intensity_matrix == 0)
		intensity_matrix[mask] = np.nan
		c_scheme = mp.pcolormesh(x,y,intensity_matrix,cmap = 'turbo',zorder=10,shading='gouraud', vmin=0, vmax=np.around(self.intensity.max()+1))
		# consider this as the outline for the map that is to be created 
		# draw coastlines, meridians and parallels.
		mp.drawcoastlines(color='w')
		mp.drawcountries(color='w')
		mp.drawmapboundary(fill_color='#99ffff')
		mp.fillcontinents(color='#cc9966',lake_color='#99ffff')
		mp.drawparallels(np.arange(-90,90,10),labels=[1,1,0,0],zorder=11)
		mp.drawmeridians(np.arange(-180,180,10),labels=[0,0,0,1],zorder=12)
		plt.title('Intensity of hazard event')

		cbar = mp.colorbar(c_scheme,location='right',pad = '15%') # map information

		return fig, ax

	def plot_gif(self, filename, speed=3, projection='cyl', edge_pad=0):
		'''
		:param filename: sting, filename of the generated gif file. must end with .gif
		:param speed: int, default=3, the reply speed of the gif
		:param projection: string, projection used by Basemap, default='cyl' espg projections can be used as well
		:param edge_pad: int, shows more of the map by adding additional lat and lon degrees
		
		Generate a .gif file if the file/input/*project name*/hazards/gif directory of the hazard
		'''
		time_steps = np.arange(0,len(self.time))
		for i in time_steps:
			fig,ax = self.plot(i, projection=projection, edge_pad=edge_pad)
			plt.title('Intensity '+str(i)+' time')

			jpg_filename = config.path.hazardGifFile(self.simulationName, str(i)+'.jpg')
			plt.savefig(jpg_filename)
			
			fig.clear()
			ax.clear()
			plt.close(fig)

		# put all the frames together and create the gif file
		image_frames = [] # creating a empty list to be appended later on
		for k in time_steps:
			jpg_filename = config.path.hazardGifFile(self.simulationName, str(k)+'.jpg')
			new_fram = PIL.Image.open(jpg_filename) 
			image_frames.append(new_fram)

		image_frames[0].save(config.path.hazardGifFile(self.simulationName, filename),format='GIF',
							append_images = image_frames[1:],
							save_all = True, duration = len(self.datetimes)*speed,loop = 0)

		# cleanup, removing the jpg files generated
		for i in time_steps:
			jpg_filename = config.path.hazardGifFile(self.simulationName, str(i)+'.jpg')
			os.remove(jpg_filename)

def readReturnPeriods(simulationName):
	'''
	:param simulationName: string, name of the simulation case.
	:return return_period_list: list, list of tuples containing the original data from the fragility curves
	'''
	return_period_list = []
	directory = config.path.returnPeriodFolder(simulationName)

	exclude = set(['.ipynb_checkpoints'])
	for _,dirs,files in os.walk(directory):
		dirs[:] = [d for d in dirs if d not in exclude]
		for file in files:
			if file.endswith(".csv"):
				df = pd.read_csv(os.path.join(directory,file), sep=';')
				x = df["time_yr"].values
				for col in df.columns[1:]:
					return_period_list.append((col,x,df[col].values))
	return return_period_list

def build_return_period_database(simulationName):
	rps = readReturnPeriods(simulationName)
	dict_return_periods = {}
	
	for rp in rps:
		dict_return_periods[rp[0]] = ReturnPeriod(rp[0], list(rp[1]), list(rp[2]))
	return dict_return_periods

class ReturnPeriod:
	'''
	ReturnPeriod Class:

	:attribute df_return_period: dataFrame, intensity and return period to be used for the analysis

	:method interpolate_return_period(newX):
	'''

	def __init__(self, name, x, y):
		self.x_data = x
		self.y_data = y
		self.name = name

		X = np.array([[i] for i in self.y_data])
		logY = np.log(self.x_data)
		self.log_gam = LinearGAM(s(0, n_splines=len(X))).gridsearch(X, logY)

		y = self.y_data
		logx = np.log(np.array([[i] for i in self.x_data]))
		self.inv_log_gam = LinearGAM(s(0, n_splines=len(logx))).gridsearch(logx, y)

	def interpolate_return_period(self, newX):
		return np.exp(self.log_gam.predict(newX))

	def interpolate_inv_return_period(self, newX):
		return self.inv_log_gam.predict(np.log(newX))

	def generate_samples(self, x_min, x_max, N):
		X = np.linspace(x_min,x_max,N)
		
		CDF = [1-1/r for r in self.interpolate_return_period(X)]
		CDF_new_samples = np.linspace(CDF[0], CDF[-1], N)

		return_period_resampled = [1/(1-cdf) for cdf in CDF_new_samples]
		samples = self.interpolate_inv_return_period(return_period_resampled)

		return samples