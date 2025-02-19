
import pandas as pd
import numpy as np

from . import network
from . import config

import random
import warnings

DECIMAL_PRECISION = 1

def convert_index_to_internal_time(df, df_int_ext_time):
	map = df_int_ext_time.dt.strftime('%Y-%m-%d %hh:%mm:%ss').to_dict()  # need to convert to string
	map = {y: x for x, y in map.items()}
	return df.rename(index=map)  # otherwise renaming will not work

def convert_index_to_external_time(df, df_int_ext_time):
	return df.rename(index=df_int_ext_time.to_dict())

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

def read_database(databaseFile):
	# TODO: call const.py for index_col
	for sep in [",", ";"]:
		try:
			return pd.read_csv(databaseFile, index_col=[0, 1, 2, 3, 4], sep=sep).T
		except Exception:
			continue
	raise ValueError("Error: Unable to read montecarlo file.")

def allocate_column_values(object, df_):
	# .to_frame().T is needed because the function acts on of the df's columns
	df = config.df_to_internal_fields(df_.to_frame().T)
	# 'value' is needed since we are working on a dataFrame
	for key, value in df.loc[config.COL_NAME_VALUE].to_dict().items():
		if hasattr(object, key):
			setattr(object, key, value)
		else:
			warnings.warn(f'Input parameter "{key}" unknown in dataframe.')

def get_index_as_dataSeries(df):
	if df.index.empty:
		print("Sheet Profiles is empty!")
	#return df.index.to_frame(index=False)[0]
	return df.index.to_frame(index=False).iloc[:, 0]

def enrich_database(df):
	"""
	This function enriches the given DataFrame `df` with additional calculated fields related to loss of load and energy not served.
	It calculates various metrics such as loss of load, loss of load percentage, loss of load duration, energy not served, and their total values for the network.
	The enriched DataFrame is then returned with these additional metrics.

	Args:
		df (pd.DataFrame): The input DataFrame containing the simulation data.

	Returns:
		pd.DataFrame: The enriched DataFrame with additional calculated fields.
	"""

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
		"""
		Calculate the loss of load in megawatts (MW).

		This function computes the difference between the maximum load power (max_p_mw) 
		and the actual load power (p_mw) from a DataFrame `df`. The result is then 
		formatted into a multi-index DataFrame with specified field and type labels.

		Returns:
			pd.DataFrame: A multi-index DataFrame containing the loss of load values 
			with the specified field and type labels.
		"""
		field = 'loss_of_load_p_mw'
		type = 'load'
		content = df.loc[:,:, 'max_p_mw','load',:] - df.loc[:,:, 'p_mw','load',:]
		return format_to_multiindex(content, [field, type], ['field', 'type'])

	def loss_of_load_percentage():
		"""
		Calculate the loss of load percentage.

		This function calculates the loss of load percentage by grouping the 
		loss of load data by 'strata', 'iteration', and 'id', summing the values, 
		and then dividing by the maximum load in megawatts (MW) to get the percentage. 
		It fills any NaN values with 0, considering a load of zero as supplied.

		Returns:
			pd.DataFrame: A multi-index DataFrame with the calculated loss of load 
			percentage, formatted with the specified field and type.
		"""
		field = 'loss_of_load_p_percentage'
		type = 'load'
		content = loss_of_load().groupby(['strata','iteration', 'id']).sum()/df.loc[:,:, 'max_p_mw','load',:]*100
		content.fillna(0, inplace = True) # load of zero considered as supplied
		return format_to_multiindex(content, [field, type], ['field', 'type'])
	
	def loss_of_load_duration():
		"""
		Calculate the loss of load duration for each time step.

		This function computes the duration of load loss by comparing the maximum 
		power capacity with the actual power load. It filters out the periods where 
		the load loss is less than a specified precision and then multiplies the 
		filtered results by the number of hours per time step to get the duration 
		of load loss.

		Returns:
			pd.DataFrame: A multi-index DataFrame with the calculated loss of load 
			duration, formatted with specified field and type.
		"""
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
		"""
		Calculate the energy not served in megawatt-hours (MWh).

		This function computes the energy not served by multiplying the loss of load 
		duration by the loss of load for each time step. The result is then formatted 
		into a multi-index DataFrame with specified field and type labels.

		Returns:
			pd.DataFrame: A multi-index DataFrame containing the energy not served 
			values with the specified field and type labels.
		"""
		field = 'energy_not_served_mwh'
		type = 'load'
		hours = loss_of_load_duration().groupby(['strata','iteration', 'id']).sum()
		content = loss_of_load().groupby(['strata','iteration', 'id']).sum().multiply(hours, axis = 1)
		return format_to_multiindex(content, [field, type], ['field', 'type'])

	def total_loss_of_load():
		"""
		Calculate the total loss of load in megawatts (MW) for the entire network.

		This function computes the total loss of load by summing the loss of load values 
		across all iterations and strata. The result is then formatted into a multi-index 
		DataFrame with specified field, type, and id labels.

		Returns:
			pd.DataFrame: A multi-index DataFrame containing the total loss of load values 
			with the specified field, type, and id labels.
		"""
		field = 'loss_of_load_p_mw'
		type = 'network'
		id = ''
		content = loss_of_load().groupby(['strata','iteration']).sum()
		return format_to_multiindex(content, [field, type, id], ['field', 'type', 'id'])

	def total_loss_of_load_percentage():
		"""
		Calculate the total loss of load percentage for the entire network.

		This function calculates the total loss of load percentage by summing the loss of load values 
		across all iterations and strata, and then dividing by the maximum load in megawatts (MW) 
		to get the percentage. It fills any NaN values with 0, considering a load of zero as supplied.

		Returns:
			pd.DataFrame: A multi-index DataFrame with the calculated total loss of load 
			percentage, formatted with the specified field, type, and id.
		"""
		field = 'loss_of_load_p_percentage'
		type = 'network'
		id = ''
		content = total_loss_of_load().groupby(['strata','iteration']).sum() / df.loc[:,:, 'max_p_mw','load',:].groupby(['strata','iteration']).sum() * 100 
		content.fillna(0, inplace = True) # load of zero considered as supplied
		return format_to_multiindex(content, [field, type, id], ['field', 'type', 'id'])

	def total_loss_of_load_duration():
		"""
		Calculate the total loss of load duration.

		This function calculates the total duration of loss of load events by 
		multiplying the filtered loss of load percentage by the number of hours 
		per timestep. The result is formatted into a multi-index DataFrame.

		Returns:
			pd.DataFrame: A multi-index DataFrame with the total loss of load 
			duration, including the fields 'field', 'type', and 'id'.
		"""
		field = 'loss_of_load_p_duration_h'
		type = 'network'
		id = ''
		hours = get_number_of_hours_per_timesteps(df)
		filter = total_loss_of_load_percentage().groupby(['strata','iteration']).sum() # get rid of unused fields
		filter = filter.round(DECIMAL_PRECISION)!=0 # DECIMAL_PRECISION = 1 means that less than 0.1% is NOT cosnidered as a loss of load.
		content = filter.multiply(hours, axis = 1)
		return format_to_multiindex(content, [field, type, id], ['field', 'type', 'id'])

	def total_energy_not_served():
		"""
		Calculate the total energy not served.

		This function calculates the total energy not served by summing up the 
		energy not served (in MWh) grouped by strata and iteration. It formats 
		the result into a multi-index DataFrame.

		Returns:
			pd.DataFrame: A DataFrame with the total energy not served, formatted 
			with multi-index columns ['field', 'type', 'id'].
		"""
		field = 'energy_not_served_mwh'
		type = 'network'
		id = ''
		hours = total_loss_of_load_duration().groupby(['strata','iteration']).sum() # TODO Not using variable?
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
	The Sim class contains information of the updated power grid, including the outage schedule of power elements, based
	on the probability of failure and hazard intensity.
	The Sim class contains information of the updated power grid, including the outage schedule of power elements, based
	on the probability of failure and hazard intensity.
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

		df_simulation = pd.read_excel(config.path.networkFile(simulationName), sheet_name=config.get_input_sheetname('Simulation'), index_col=0)
		allocate_column_values(self, df_simulation[config.COL_NAME_VALUE])
		self.externalTimeInterval = get_index_as_dataSeries(pd.read_excel(config.path.networkFile(simulationName), sheet_name=config.get_input_sheetname('Profile'), index_col=0, header=[0, 1]))
		self.time = self.to_internal_time(self.startTime, self.duration)
		print(f'\nSimulation:	Start = {self.time.start:>3}; Stop = {self.time.stop:>3}; Duration = {self.time.duration:>3} timesteps.')
		self.hazardTime = self.to_internal_time(self.hazardStartTime, self.hazardDuration)
		print(f'Hazard:		Start = {self.hazardTime.start:>3}; Stop = {self.hazardTime.stop:>3}; Duration = {self.hazardTime.duration:>3} timesteps.')

	def to_internal_time(self, startTime, duration):
		# TODO: very important to check!
		# TODO: valid for startTime within self.externalTimeInterval. Extend for other cases (usefuls for hazards happening before simulation interval)
		delta = self.externalTimeInterval.loc[1] - self.externalTimeInterval.loc[0]
		filter = (self.externalTimeInterval >= startTime) & (self.externalTimeInterval < startTime + delta*duration)
		start, duration = self.externalTimeInterval[filter].dropna().index[0], self.externalTimeInterval[filter].dropna().index.size
		return Time(start, duration)

	# To be updated to consider stratas!
	def initialize_model_sh(self, network, mc_iteration_factor):
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
		iterations = range(mc_iteration_factor)
		network.update_failure_probability()
		for _ in iterations:
			network.calculate_outages_schedule(self.time, self.hazardTime)
			network.calculate_switches_schedule(self.time)
			network.propagate_schedules_to_network_elements()
			databases.append(network.build_montecarlo_database(self.time))
		out = build_database(iterations, [databases], self.externalTimeInterval)
		out.to_csv(config.path.montecarloDatabaseFile(self.simulationName))
		return network.outagesSchedule
	
	def get_intensity_boundaries(self, network, min_intensity, max_intensity, ref_return_period):
		# TODO Add better Feedback if rp issues (file missing, column value not matching, etc)
		max_intensity_from_rp = max(network.returnPeriods[ref_return_period].y_data)
		min_intensity_from_rp = min(network.returnPeriods[ref_return_period].y_data)

		if min_intensity == None:
			min_intensity = min_intensity_from_rp
		elif min_intensity < min_intensity_from_rp:
			warnings.warn(f'Selected minimum intensity is lower than reference return period: {min_intensity} < {min_intensity_from_rp}')

		if max_intensity == None:
			max_intensity = max_intensity_from_rp
		elif max_intensity > max_intensity_from_rp:
			warnings.warn(f'Selected maximum intensity is greater than reference the return periods: {max_intensity} > {max_intensity_from_rp}')

		return min_intensity, max_intensity

	def populate_failure_prob_df(self, network, mc_iteration_factor, max_mc_iterations, ref_return_period):
		iteration_number = 0
		self.databases = []
		# Iterate over strata
		print("------------------------------------------------------------------")
		for strata in range(len(self.stratResults.index)):
			strata_db = []
			# get samples within the stratum boundaries
			sample_pool = self.samples[(self.samples >= self.stratResults["Lower_X1"].values[strata]) & (self.samples <= self.stratResults["Upper_X1"].values[strata])]
			# calculate number of samples allocated to the stratum
			if self.stratResults["Allocation"].sum()*mc_iteration_factor <= max_mc_iterations:
				nsamples = self.stratResults["Allocation"].values[strata]*mc_iteration_factor
			else:
				nsamples = round(self.stratResults["Allocation"].values[strata]*max_mc_iterations/self.stratResults["Allocation"].sum())					
			
			print(f'Strata: {strata}')
			print(f'Number of samples: {int(nsamples)}')
			print(f'Hazard intensity range: {self.stratResults["Lower_X1"].values[strata]:.2f} - {self.stratResults["Upper_X1"].values[strata]:.2f}')
			# Iterate over each montecarlo iteration
			for _ in range(int(nsamples)):
				iteration_number += 1
				# pick a value for the event inntensity from the sample pool
				event_intensity = sample_pool[random.randint(0, len(sample_pool)-1)]
				# Update the network element for the given intensity
				network.update_failure_probability(intensity=event_intensity, ref_return_period=ref_return_period)
				network.calculate_outages_schedule(self.time, self.hazardTime)
				network.calculate_switches_schedule(self.time)
				network.propagate_schedules_to_network_elements()
				strata_db.append(network.build_montecarlo_database(self.time))
				# iterate of each power element
				for key, value in network.powerElements.items():
					if value.return_period != None:
						elm_intensity = network.fragilityCurves[value.fragilityCurve].projected_intensity(
																rp=network.returnPeriods[value.return_period],
																ref_rp=network.returnPeriods[ref_return_period],
																x=event_intensity)
					else:	
						elm_intensity = event_intensity

					yield iteration_number, strata, elm_intensity, value.__class__.__name__, key, value.failureProb, float('nan')
			self.databases.append(strata_db)
			self.iteration_number = iteration_number

	def initialize_model_rp(self, network, mc_iteration_factor, ref_return_period, cv=0.1, max_mc_iterations=1000, nStrataSamples=10000, min_intensity=None, max_intensity=None, maxStrata=10):
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
		:maxStrata type: int
		"""
		# If no boudaries are provided, min_intensity and max_intensity are calculated from the return periods
		min_intensity, max_intensity = self.get_intensity_boundaries(network, min_intensity, max_intensity, ref_return_period)
		# Generate intensity samples from the reference return period between the provided boudaries 
		self.samples = network.returnPeriods[ref_return_period].generate_samples(min_intensity, max_intensity, nStrataSamples)
		# Perform the stratification of the samples
		self.stratResults = network.calc_stratas(
								self.samples, 
								network.returnPeriods[ref_return_period], 
								xmin = min_intensity,
								xmax = max_intensity, 
								cv = cv,
								maxStrata = maxStrata)
		# Generate warning if max_mc_iterations is too low to achieve desired error
		if self.stratResults["Allocation"].sum()*mc_iteration_factor >  max_mc_iterations:
			warnings.warn(f'Warning: Estimated needed strata samples to reach cv = {cv} are greater than max_mc_iterations ({max_mc_iterations})!')
		# Generate failureProbs dataframe
		self.failureProbs = pd.DataFrame(self.populate_failure_prob_df(network, mc_iteration_factor, max_mc_iterations, ref_return_period), columns = self.failureProbs.columns)
		# Generate and export the montecarlo database file
		out = build_database(range(self.iteration_number + 1), self.databases, self.externalTimeInterval)
		out.to_csv(config.path.montecarloDatabaseFile(self.simulationName))

	def run(self, network, iterationSet = None, saveOutput = True, time = None, debug = None, **kwargs):
		# TODO: call const.py instead of 'iteration'
		"""
		The `run()` function uses the outage schedule `(montecarlo_database.csv)`, generated using one of the `initialize functions`.
		A timeseries OPF is executed for the provided network object and updates `engine_database.csv` with the OPF results.\n
		The file is stored *file/output/project name/*, if **saveOutput** = True.
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
	# TODO: error raising for uncompatible times
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
		# print(f'Start = {self.start}, Stop = {self.stop}')

# TODO SimpleControl inheriting ConstCrontol and overriding set_recycle()
# class SimpleControl(ConstControl):