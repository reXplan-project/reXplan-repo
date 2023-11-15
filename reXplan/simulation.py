
import pandas as pd
import numpy as np

from . import network
from . import config
from . import utils
from.const import *

import random
import warnings

DECIMAL_PRECISION = 1

def convert_index_to_internal_time(df, df_int_ext_time):
	map = df_int_ext_time.dt.strftime(
		'%Y-%m-%d %hh:%mm:%ss').to_dict()  # need to convert to string
	map = {y: x for x, y in map.items()}
	return df.rename(index=map)  # otherwise renaming won't work

def convert_index_to_external_time(df, df_int_ext_time):
	return df.rename(index=df_int_ext_time.to_dict())

'''
def build_database(iterations, databases, df_int_ext_time):
	# TODO: call const.py for names
	# TODO: too many things at the same time
	return convert_index_to_external_time(pd.concat(dict(zip(iterations, databases)), names=['iteration'], axis=1), df_int_ext_time).T
'''
def build_database(iterations, databases, df_int_ext_time):
	# TODO: call const.py for names
	# TODO: too many things at the same time
	stratas = []
	iter_offset = 0
	db = []
	for i, df in enumerate(databases):
		if len(df)>0:
			sub_iterations = iterations[iter_offset:len(df)+iter_offset]
			db.append(pd.concat(dict(zip(sub_iterations, df)), names=['iteration'], axis=1))
			stratas.append(i)
			iter_offset += len(df)

	return convert_index_to_external_time(pd.concat(dict(zip(stratas, db)), names=['strata'], axis=1), df_int_ext_time).T

'''
def read_database(dababaseFile):
	# TODO: call const.py for index_col
	return pd.read_csv(dababaseFile,  index_col=[0, 1, 2, 3]).T
'''
def read_database(dababaseFile):
	# TODO: call const.py for index_col
	return pd.read_csv(dababaseFile,  index_col=[0, 1, 2, 3, 4]).T

def allocate_column_values(object, df_):
	# .to_frame().T is needed because the function acts on the df's columns
	df = utils.df_to_internal_fields(df_.to_frame().T)
	# 'value' is needed since we are working on a dataFrame
	for key, value in df.loc[COL_NAME_VALUE].to_dict().items():
		if hasattr(object, key):
			setattr(object, key, value)
		else:
			warnings.warn(f'Input parameter "{key}" unknown in dataframe.')

def get_index_as_dataSeries(df):
	return df.index.to_frame(index=False)[0]

def enrich_database(df):
	#TODO: put this in another library?

	def get_number_of_hours_per_timesteps(df):
		hours = (df.columns[1:] - df.columns[:-1])/np.timedelta64(1,'h')
		if hours.size == 0:
			hours = np.append(hours,1) # assumes duration of 1 hour
		else:
			hours = np.append(hours, hours[-1]) # assumes last timestep's duration is the same as the previous one
		return hours

	def format_to_multiindex(df, keys, names):
		return pd.concat({tuple(keys): df}, names=names).reorder_levels(['strata','iteration', 'field','type', 'id'])

	def loss_of_load():
		field = 'loss_of_load_p_mw'
		type = 'load'
		content = df.loc[:,:, 'max_p_mw','load',:] - df.loc[:,:, 'p_mw','load',:]
		return format_to_multiindex(content, [field, type], ['field', 'type'])
		# return pd.concat({(field, type): content}, names=['field', 'type']).reorder_levels(['iteration', 'field','type', 'id'])

	def loss_of_load_percentage():
		field = 'loss_of_load_p_percentage'
		type = 'load'
		# content = (df.loc[:, 'max_p_mw','load',:] - df.loc[:, 'p_mw','load',:]) / df.loc[:, 'max_p_mw','load',:]*100
		content = loss_of_load().groupby(['strata','iteration', 'id']).sum()/df.loc[:,:, 'max_p_mw','load',:]*100
		content.fillna(0, inplace = True) # load of zero considered as supplied
		return format_to_multiindex(content, [field, type], ['field', 'type'])
	
	def loss_of_load_duration():
		field = 'loss_of_load_p_duration_h'
		type = 'load'
		hours = get_number_of_hours_per_timesteps(df)
		filter = (df.loc[:,:, 'max_p_mw','load',:] - df.loc[:,:, 'p_mw','load',:]) / df.loc[:,:, 'max_p_mw','load',:]*100
		filter.fillna(0, inplace = True)
		filter = loss_of_load_percentage().groupby(['strata','iteration', 'id']).sum()  # get rid of unused fields
		filter = filter.round(DECIMAL_PRECISION)!=0 # DECIMAL_PRECISION = 1 means that less than 0.1% is NOT cosnidered as a loss of load.
		content = filter.multiply(hours, axis = 1)
		return format_to_multiindex(content, [field, type], ['field', 'type'])

	def energy_not_served():
		field = 'energy_not_served_mwh'
		type = 'load'
		# hours = get_number_of_hours_per_timesteps(df)  # todo: change to loss_of_load_duration
		hours = loss_of_load_duration().groupby(['strata','iteration', 'id']).sum()
		# content = loss_of_load().loc[:, 'loss_of_load_p_mw','load',:].multiply(hours, axis = 1)
		content = loss_of_load().groupby(['strata','iteration', 'id']).sum().multiply(hours, axis = 1)
		return format_to_multiindex(content, [field, type], ['field', 'type'])

	def total_loss_of_load():
		field = 'loss_of_load_p_mw'
		type = 'network'
		id = ''
		# content = (df.loc[:, 'max_p_mw','load',:] - df.loc[:, 'p_mw','load',:]).groupby('iteration').sum()
		content = loss_of_load().groupby(['strata','iteration']).sum()
		return format_to_multiindex(content, [field, type, id], ['field', 'type', 'id'])

	def total_loss_of_load_percentage():
		field = 'loss_of_load_p_percentage'
		type = 'network'
		id = ''
		# content = (df.loc[:, 'max_p_mw','load',:] - df.loc[:, 'p_mw','load',:]).groupby('iteration').sum() / df.loc[:, 'max_p_mw','load',:].groupby('iteration').sum() * 100 
		content = total_loss_of_load().groupby(['strata','iteration']).sum() / df.loc[:,:, 'max_p_mw','load',:].groupby(['strata','iteration']).sum() * 100 
		content.fillna(0, inplace = True) # load of zero considered as supplied
		return format_to_multiindex(content, [field, type, id], ['field', 'type', 'id'])

	def total_loss_of_load_duration():
		field = 'loss_of_load_p_duration_h'
		type = 'network'
		id = ''
		hours = get_number_of_hours_per_timesteps(df)
		# filter = (df.loc[:, 'max_p_mw','load',:] - df.loc[:, 'p_mw','load',:]).groupby('iteration').sum() / df.loc[:, 'max_p_mw','load',:].groupby('iteration').sum() * 100
		filter = total_loss_of_load_percentage().groupby(['strata','iteration']).sum() # get rid of unused fields
		filter = filter.round(DECIMAL_PRECISION)!=0 # DECIMAL_PRECISION = 1 means that less than 0.1% is NOT cosnidered as a loss of load.
		content = filter.multiply(hours, axis = 1)
		return format_to_multiindex(content, [field, type, id], ['field', 'type', 'id'])

	def total_energy_not_served():
		field = 'energy_not_served_mwh'
		type = 'network'
		id = ''
		# hours = get_number_of_hours_per_timesteps(df) # todo: change to total_loss_of_load_duration
		hours = total_loss_of_load_duration().groupby(['strata','iteration']).sum()
		# content = loss_of_load().loc[:, 'loss_of_load_p_mw','load',:].multiply(hours, axis = 1)
		content = energy_not_served().groupby(['strata','iteration']).sum()
		return format_to_multiindex(content, [field, type, id], ['field', 'type', 'id'])

	to_concat = [
					df,
					loss_of_load(),
					loss_of_load_percentage(),
					loss_of_load_duration(),
					energy_not_served(),
					total_loss_of_load(),
					total_loss_of_load_percentage(),
					total_loss_of_load_duration(),
					total_energy_not_served()
				]
	
	return pd.concat(to_concat).sort_index(level = ['strata','iteration', 'type'])

class Sim:
	'''
	Add description of Sim class here
	'''
	def __init__(self,
				 simulationName):

		self.simulationName = simulationName
		self.startTime = None
		self.duration = None
		self.hazardStartTime = None
		self.hazardDuration = None
		self.results = None
		self.stratResults = None
		self.failureProbs = pd.DataFrame(columns=['iteration','strata','event intensity','element type','power element','failure probability','status'])
		self.samples = None

		df_simulation = pd.read_excel(config.path.networkFile(simulationName), sheet_name=SHEET_NAME_SIMULATION, index_col=0)
		allocate_column_values(self, df_simulation[COL_NAME_VALUE])
		self.externalTimeInterval = get_index_as_dataSeries(pd.read_excel(config.path.networkFile(simulationName),
																		 sheet_name=SHEET_NAME_PROFILES, index_col=0, header=[0, 1]))
		self.time = self.to_internal_time(self.startTime, self.duration)
		print(f'Duration of simulation is {self.time.duration} timesteps.') # TODO: Find way to specify timesteps?
		self.hazardTime = self.to_internal_time(self.hazardStartTime, self.hazardDuration)
		print(f'Start of hazard at timestep {self.hazardTime.start}, end at timestep {self.hazardTime.stop} (duration: {self.hazardTime.duration} timesteps)')

	def to_internal_time(self, startTime, duration):
		# TODO: very important to check!
		# TODO: valid for startTime within self.externalTimeInterval. Extend for other cases (usefuls for hazards happening before simulation interval)
		delta = self.externalTimeInterval.loc[1] - self.externalTimeInterval.loc[0]
		filter = (self.externalTimeInterval >= startTime) & (self.externalTimeInterval < startTime + delta*duration)
		start, duration = self.externalTimeInterval[filter].dropna().index[0], self.externalTimeInterval[filter].dropna().index.size
		return Time(start, duration)

	def initialize_model_sh(self, network, mc_iterations):
		"""
		The `initialize_model_sh()` function calculates the probability of failure for the electrical components in the network and applies
		the Monte Carlo Method to create/update the **outage schedule** for the Montecarlo Analysis `(montecarlo_database.csv)`.
		This schedule holds the equipment status information (`in-service`, `out-of-service`, `awating repair`, etc.) for the simulation timeseries.
		Increasing the number of Monte Carlo iterations reduces uncertainty, but increases computational load.\n
		The file is stored at */file/output/project name/*.
		
		:param network: Class variable with network topology, elements, and their properties.
		:param mc_iterations: Number of Monte Carlo iterations.

		:network type: reXplan.network.Network
		:mc_iterations type: int
		:return: outages schedule of the network
		:rtype: pandas.core.frame.DataFrame
		"""

		databases = []
		iterations = range(mc_iterations)
		network.update_failure_probability()
		for _ in iterations:
			network.calculate_outages_schedule(self.time, self.hazardTime)
			network.calculate_switches_schedule(self.time)
			network.propagate_schedules_to_network_elements()
			databases.append(network.build_montecarlo_database(self.time))
		out = build_database(iterations, [databases], self.externalTimeInterval)
		out.to_csv(config.path.montecarloDatabaseFile(self.simulationName))
		return network.outagesSchedule

	def initialize_model_rp(self, network, mc_iteration_factor, ref_return_period, cv=0.1, max_mc_iterations=10000, nStrataSamples=10000, min_intensity=None, max_intensity=None, maxStrata=10):
		"""
		The `initialize_model_rp()` function utilizes **return periods**, which needs to be defined in */file/input/project name/returnPeriods/*. 
		It uses the R module `StratifiedSampling` to divide the total event intensity range into strata to reduce uncertainty and computational load.
		A number of outage schedules is created for the Monte Carlo analysis per stratum,
		which are collected and stored in the **outage schedule** file (*montecarlo_database.csv*).
		The schedule holds the equipment status information (`in-service`, `out-of-service`, `awating repair`, etc.) for the simulation timeseries.
		The number of Monte Carlo iterations is optimized by the R module.\n
		The file is stored at */file/output/project name/*.
		
		:param network: Class variable with network topology, elements, and their properties.
		:param mc_iteration_factor: Number of Monte Carlo iterations per stratum sample. Strata samples are calculated using the `StratifiedSampling` module of R.
		:param ref_return_period: Reference return period for fragility curves. Used to generate samples for the Monte Carlo simulation.
		:param cv: Coefficient of variation. Measure of variability of a dataset relative to its mean. Used to control the precision of the Monte Carlo simulation.
					A smaller value of cv will result in a more precise simulation, but will also increase computational time.
		:param max_mc_iterations: Limit of Monte Carlo iterations performed during the simulation.
		:param nStrataSamples: Number of intensity samples from return period used for the stratification.
		:param min_intensity: Minimum intensity of return periods
		:param max_intensity: Maximum intensity of return periods
		:param maxStrata: Maximum number of strata samples

		:network type: reXplan.network.Network
		:mc_iteration_factor type: int
		:ref_return_period type: str
		:cv type: float
		:max_mc_iterations type: int
		:nStrataSamples type: int
		:min_intensity type: float
		:max_intensity type: float
		"""
		for j, (key, rp) in enumerate(network.returnPeriods.items()): # TODO. Check this function
			if j == 0:
				xmin = min(rp.y_data)
				xmax = max(rp.y_data)
			else:
				xmin = min(xmin, min(rp.y_data))
				xmax = max(xmax, max(rp.y_data))

		# TODO: Find naming solution
		if min_intensity==None:
			min_intensity = xmin
			print(f'Lowest return period intensity: {min_intensity}')
		elif min_intensity < xmin:
			warnings.warn(f'Warning: selected min_intensity is lower than the data provided for the fragility curves: {min_intensity} < {xmin}')

		if max_intensity==None:
			max_intensity = xmax
			print(f'Highest return period intensity: {max_intensity}')
		elif max_intensity > xmax:
			warnings.warn(f'Warning: selected max_intensity is greater than the data provided for the fragility curves: {max_intensity} > {xmax}')
		
		self.samples = network.returnPeriods[ref_return_period].generate_samples(min_intensity, max_intensity, nStrataSamples)
		self.stratResults = network.calc_stratas(
			self.samples, network.returnPeriods[ref_return_period], xmin=min_intensity, xmax=max_intensity, cv=cv, maxStrata=maxStrata)

		if self.stratResults["Allocation"].sum()*mc_iteration_factor >  max_mc_iterations:
			warnings.warn(f'Warning: Estimated needed strata samples to reach cv = {cv} are greater than max_mc_iterations = {max_mc_iterations}')

		iteration_number = 0
		df_temp = pd.DataFrame()
		databases = []
		for strata in range(len(self.stratResults.index)):
			strata_db = []
			sample_pool = self.samples[(self.samples >= self.stratResults["Lower_X1"].values[strata]) & (self.samples <= self.stratResults["Upper_X1"].values[strata])]
			
			if self.stratResults["Allocation"].sum()*mc_iteration_factor <=  max_mc_iterations:
				nsamples = self.stratResults["Allocation"].values[strata]*mc_iteration_factor
			else:
				nsamples = round(self.stratResults["Allocation"].values[strata]*max_mc_iterations/self.stratResults["Allocation"].sum())
					
			print(f'\nStrata = {strata}')
			print(f'Number of samples = {nsamples}')
			print(f'Intensity samples between {self.stratResults["Lower_X1"].values[strata]} and {self.stratResults["Upper_X1"].values[strata]}')
			df_temp["strata"] = [strata]

			for i in range(int(nsamples)):			
				event_intensity = sample_pool[random.randint(0, len(sample_pool)-1)]
				network.update_failure_probability(intensity=event_intensity, ref_return_period=ref_return_period)
				iteration_number += 1

				network.calculate_outages_schedule(self.time, self.hazardTime)
				network.calculate_switches_schedule(self.time)
				network.propagate_schedules_to_network_elements()
				strata_db.append(network.build_montecarlo_database(self.time))

				df_temp["iteration"] = [iteration_number]

				for key, value in network.powerElements.items():
					if value.return_period != None:
						df_temp["event intensity"] = network.fragilityCurves[value.fragilityCurve].projected_intensity(rp=network.returnPeriods[value.return_period],
																														ref_rp=network.returnPeriods[ref_return_period],
																														x=event_intensity)
					else:	
						df_temp["event intensity"] = [event_intensity]
					df_temp["power element"] = [key]
					df_temp["element type"] = [value.__class__.__name__]
					df_temp["failure probability"] = [value.failureProb]
					self.failureProbs = pd.concat([self.failureProbs, df_temp], ignore_index=True)
			databases.append(strata_db)

		self.failureProbs.reset_index()
		out = build_database(range(iteration_number+1), databases, self.externalTimeInterval)
		out.to_csv(config.path.montecarloDatabaseFile(self.simulationName))
		
	def run(self, network, iterationSet = None, saveOutput = True, time = None, debug=None, **kwargs):
		# TODO: call const.py instead of 'iteration'
		"""
		The `run()` function uses the outage schedule `(montecarlo_database.csv)`, generated using one of the `initialize functions`.
		A timeseries OPF is executed for the provided network object and updates `engine_database.csv` with the OPF results.\n
		The file is stored *file/output/project name/*, if **saveOutput** = True.
		
		:param network: Contains information about the network topology, elements, and their properties. See `Network` class.
		:param iterationSet: If provided, selected Monte Carlo iterations will be selected. If not provided, all iterations will be executed.
		:param saveOutput: Determines whether the output database should be saved to a file `(engine_database.csv)` or not.
		:param time: Specifies the duration of the simulation. If provided, it will override the default simulation time set in the `self.time` variable. See `Time` class.
		:param run_type: (of `**kwargs`) can utilize different OPF approaches:\n
			-> **dc_opf**	 - pypower (Python)\n
			-> **ac_opf**	 - pypower (Python)\n
			-> **pm_dc_opf** - PandaModels (Julia)\n
			-> **pm_ac_opf** - PandaModels (Julia)\n
			PandaModels is recommended, as PandaModels has better convergence properties.
		:network type: reXplan.network.Network object
		:iterationSet type: list
		:saveOutput type: bool	
		:time type: class
		:run_type type: string

		"""
		time_ = self.time
		if time:
			time_ = time
		df_montecarlo = convert_index_to_internal_time(read_database(config.path.montecarloDatabaseFile(self.simulationName)), self.externalTimeInterval)
		stratas = df_montecarlo.columns.get_level_values(level='strata').drop_duplicates()
		databases = []
		total_iteration = []
		for s in stratas:
			strata_db = []
			iterations = df_montecarlo[s].columns.get_level_values(level='iteration').drop_duplicates()
			if iterationSet:
				iterations = [i for i in iterations if i in iterationSet]
			total_iteration.extend(iterations)

			for i in iterations:
				print(f'Strata = {s}; Iteration = {i}')
				network.update_grid(df_montecarlo[s][i], debug=debug)
				try:
					strata_db.append(network.run(time_, debug=debug, **kwargs))
				except Exception as e:
					print(f'Iteration {i} did not execute successfully. {str(e)}')
			databases.append(strata_db)
		
		self.results = enrich_database(build_database(total_iteration, databases, self.externalTimeInterval))
		if saveOutput:
			print ('Saving output database...')
			self.results.to_csv(config.path.engineDatabaseFile(self.simulationName))
			print ('done!')

class Time():
	# TODO: error raising for uncompatible times // STARTS AT 0 OR 1 ?
	"""
	Contains information about the simulation time.
	:param start: Starting point of the simulation interval. It is an integer value that indicates the first value in the interval
	:param duration: Represents the length of the interval in units of time
	:start type: int
	:duration type: int
	"""
	def __init__(self, start, duration):
		"""
		Initializes an object with start and duration attributes, calculates the stop time, creates an interval list.
		"""
		self.start = start
		self.duration = duration
		self.maxduration = duration
		self.stop = duration + start
		self.interval = list(range(start, duration + start))

# TODO SimpleControl inheriting ConstCrontol and overriding set_recycle()
# class SimpleControl(ConstControl):