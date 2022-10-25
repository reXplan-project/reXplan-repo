
import pandas as pd
import numpy as np

from . import network
from . import config
from . import utils
from.const import *

DECIMAL_PRECISION = 1

def convert_index_to_internal_time(df, df_int_ext_time):
	map = df_int_ext_time.dt.strftime(
		'%Y-%m-%d %hh:%mm:%ss').to_dict()  # need to conver to string
	map = {y: x for x, y in map.items()}
	return df.rename(index=map)  # otherwise renaiming won't work

def convert_index_to_external_time(df, df_int_ext_time):
	return df.rename(index=df_int_ext_time.to_dict())

def build_database(iterations, databases, df_int_ext_time):
	# TODO: call const.py for names
	# TODO: too many things at the same time
	return convert_index_to_external_time(pd.concat(dict(zip(iterations, databases)), names=['iteration'], axis=1), df_int_ext_time).T

def read_database(dababaseFile):
	# TODO: call const.py for index_col
	return pd.read_csv(dababaseFile,  index_col=[0, 1, 2, 3]).T


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
		return pd.concat({tuple(keys): df}, names=names).reorder_levels(['iteration', 'field','type', 'id'])

	def loss_of_load():
		field = 'loss_of_load_p_mw'
		type = 'load'
		content = df.loc[:, 'max_p_mw','load',:] - df.loc[:, 'p_mw','load',:]
		return format_to_multiindex(content, [field, type], ['field', 'type'])
		# return pd.concat({(field, type): content}, names=['field', 'type']).reorder_levels(['iteration', 'field','type', 'id'])

	def loss_of_load_percentage():
		field = 'loss_of_load_p_percentage'
		type = 'load'
		# content = (df.loc[:, 'max_p_mw','load',:] - df.loc[:, 'p_mw','load',:]) / df.loc[:, 'max_p_mw','load',:]*100
		content = loss_of_load().groupby(['iteration', 'id']).sum()  / df.loc[:, 'max_p_mw','load',:]*100
		content.fillna(0, inplace = True) # load of zero considered as supplied
		return format_to_multiindex(content, [field, type], ['field', 'type'])
	
	def loss_of_load_duration():
		field = 'loss_of_load_p_duration_h'
		type = 'load'
		hours = get_number_of_hours_per_timesteps(df)
		filter = (df.loc[:, 'max_p_mw','load',:] - df.loc[:, 'p_mw','load',:]) / df.loc[:, 'max_p_mw','load',:]*100
		filter.fillna(0, inplace = True)
		filter = loss_of_load_percentage().groupby(['iteration', 'id']).sum()  # get rid of unused fields
		filter = filter.round(DECIMAL_PRECISION)!=0 # DECIMAL_PRECISION = 1 means that less than 0.1% is NOT cosnidered as a loss of load.
		content = filter.multiply(hours, axis = 1)
		return format_to_multiindex(content, [field, type], ['field', 'type'])

	def energy_not_served():
		field = 'energy_not_served_wh'
		type = 'load'
		# hours = get_number_of_hours_per_timesteps(df)  # todo: change to loss_of_load_duration
		hours = loss_of_load_duration().groupby(['iteration', 'id']).sum()
		# content = loss_of_load().loc[:, 'loss_of_load_p_mw','load',:].multiply(hours, axis = 1)
		content = loss_of_load().groupby(['iteration', 'id']).sum().multiply(hours, axis = 1)
		return format_to_multiindex(content, [field, type], ['field', 'type'])

	def total_loss_of_load():
		field = 'loss_of_load_p_mw'
		type = 'network'
		id = ''
		# content = (df.loc[:, 'max_p_mw','load',:] - df.loc[:, 'p_mw','load',:]).groupby('iteration').sum()
		content = loss_of_load().groupby('iteration').sum()
		return format_to_multiindex(content, [field, type, id], ['field', 'type', 'id'])

	def total_loss_of_load_percentage():
		field = 'loss_of_load_p_percentage'
		type = 'network'
		id = ''
		# content = (df.loc[:, 'max_p_mw','load',:] - df.loc[:, 'p_mw','load',:]).groupby('iteration').sum() / df.loc[:, 'max_p_mw','load',:].groupby('iteration').sum() * 100 
		content = total_loss_of_load().groupby('iteration').sum() / df.loc[:, 'max_p_mw','load',:].groupby('iteration').sum() * 100 
		content.fillna(0, inplace = True) # load of zero considered as supplied
		return format_to_multiindex(content, [field, type, id], ['field', 'type', 'id'])

	def total_loss_of_load_duration():
		field = 'loss_of_load_p_duration_h'
		type = 'network'
		id = ''
		hours = get_number_of_hours_per_timesteps(df)
		# filter = (df.loc[:, 'max_p_mw','load',:] - df.loc[:, 'p_mw','load',:]).groupby('iteration').sum() / df.loc[:, 'max_p_mw','load',:].groupby('iteration').sum() * 100
		filter = total_loss_of_load_percentage().groupby('iteration').sum() # get rid of unused fields
		filter = filter.round(DECIMAL_PRECISION)!=0 # DECIMAL_PRECISION = 1 means that less than 0.1% is NOT cosnidered as a loss of load.
		content = filter.multiply(hours, axis = 1)
		return format_to_multiindex(content, [field, type, id], ['field', 'type', 'id'])

	def total_energy_not_served():
		field = 'energy_not_served_mwh'
		type = 'network'
		id = ''
		# hours = get_number_of_hours_per_timesteps(df) # todo: change to total_loss_of_load_duration
		hours = total_loss_of_load_duration().groupby('iteration').sum()
		# content = loss_of_load().loc[:, 'loss_of_load_p_mw','load',:].multiply(hours, axis = 1)
		content = energy_not_served().groupby('iteration').sum().multiply(hours, axis = 1)
		return format_to_multiindex(content, [field, type, id], ['field', 'type', 'id'])


	# def total(field):
	# 	# deprecated
	# 	id = 'network'
	# 	content = (df[field].groupby('type',axis = 1).sum())
	# 	return pd.concat({(field, id):content}, names=['field', 'type','id'], axis=1).reorder_levels(['field','type', 'id'], axis = 1)

	to_concat = [df,
			loss_of_load(),
			loss_of_load_percentage(),
			loss_of_load_duration(),
			energy_not_served(),
			total_loss_of_load(),
			total_loss_of_load_percentage(),
			total_loss_of_load_duration(),
			total_energy_not_served()]
	return pd.concat(to_concat).sort_index(level = ['iteration', 'type'])

class Sim:
	'''
	Add description of Sim class here
	'''

	def __init__(self,
				 simulationName,
				 # history,
				 # start,
				 # duration,
				 # hazardTime
				 ):

		self.simulationName = simulationName
		self.startTime = None
		self.duration = None
		self.hazardStartTime = None
		self.hazardDuration = None
		self.results = None
		df_simulation = pd.read_excel(config.path.networkFile(
			simulationName), sheet_name=SHEET_NAME_SIMULATION, index_col=0)
		allocate_column_values(self, df_simulation[COL_NAME_VALUE])
		self.externalTimeInterval = get_index_as_dataSeries(pd.read_excel(config.path.networkFile(
			simulationName), sheet_name=SHEET_NAME_PROFILES, index_col=0, header=[0, 1]))
		self.time = self.to_internal_time(self.startTime, self.duration)
		self.hazardTime = self.to_internal_time(
			self.hazardStartTime, self.hazardDuration)

	def to_internal_time(self, startTime, duration):
		# TODO: very important to check!
		# TODO: valid for startTime within self.externalTimeInterval. Extend for other cases (usefuls for hazards happening before simulation interval)
		delta = self.externalTimeInterval.loc[1] - \
			self.externalTimeInterval.loc[0]
		filter = (self.externalTimeInterval >= startTime) & (
			self.externalTimeInterval < startTime + delta*duration)
		start, duration = self.externalTimeInterval[filter].dropna(
		).index[0], self.externalTimeInterval[filter].dropna().index.size
		return Time(start, duration)

	def pre_run(self, network, iterationNumber):
		databases = []
		iterations = range(iterationNumber)
		for i in iterations:
			print(f'Iteration = {i}')
			network.calculate_outages_schedule(self.time, self.hazardTime)
			network.propagate_outages_to_network_elements()
			databases.append(network.build_montecarlo_database(self.time))
		out = build_database(iterations, databases, self.externalTimeInterval)
		out.to_csv(config.path.montecarloDatabaseFile(self.simulationName))

	def run(self, network, iterationSet = None, saveOutput = True, time = None, **kwargs):
		# TODO: call const.py instead of 'iteration'
		time_ = self.time
		if time:
			time_ = time
		
		df_montecarlo = convert_index_to_internal_time(read_database(
			config.path.montecarloDatabaseFile(self.simulationName)), self.externalTimeInterval)
		print(df_montecarlo)
		iterations = df_montecarlo.columns.get_level_values(
			level='iteration').drop_duplicates()
		if iterationSet:
			iterations = [i for i in iterations if i in iterationSet]
		
		databases = []
		for i in iterations:
			print(f'Iteration = {i}')
			network.updateGrid(df_montecarlo[i])
			databases.append(network.run(time_,**kwargs))
		self.results = enrich_database(build_database(iterations, databases, self.externalTimeInterval))
		if saveOutput:
			print ('Saving output database...')
			self.results.to_csv(config.path.engineDatabaseFile(self.simulationName))
			print ('done!')


class Time():
	# TODO: error raising for uncompatible times
	def __init__(self, start, duration):
		self.start = start
		self.duration = duration
		self.stop = duration + start
		self.interval = list(range(start, duration + start))
		print(f'start= {self.start}, stop= {self.stop}')

# TODO SimpleControl inheriting ConstCrontol and overriding set_recycle()
# class SimpleControl(ConstControl):
