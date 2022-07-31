from pandapower.toolbox import _detect_read_write_flag, write_to_net
from pandapower.control.basic_controller import Controller
import pandas as pd
import numpy as np

from . import network
from . import config
from . import utils
from.const import *


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

def index_to_dataSeries(df):
	return df.index.to_frame(index=False)[0]

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
		df_simulation = pd.read_excel(config.path.networkFile(
			simulationName), sheet_name=SHEET_NAME_SIMULATION, index_col=0)
		allocate_column_values(self, df_simulation[COL_NAME_VALUE])
		self.externalTimeInterval = index_to_dataSeries(pd.read_excel(config.path.networkFile(
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

	def run(self, network, iterationNumber = None):
		# TODO: call const.py instead of 'iteration'
		databases = []
		df_montecarlo = convert_index_to_internal_time(read_database(
			config.path.montecarloDatabaseFile(self.simulationName)), self.externalTimeInterval)
		iterations = df_montecarlo.columns.get_level_values(level='iteration').drop_duplicates()
		if iterationNumber:
			iterations = iterations[0:iterationNumber]
		for i in iterations:
			print(f'Iteration = {i}')
			network.updateGrid(df_montecarlo[i])
			databases.append(network.run(self.time))
		out = build_database(iterations, databases, self.externalTimeInterval)
		out.to_csv(config.path.engineDatabaseFile(self.simulationName))

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


class SimpleControl(Controller):

	def __init__(self, net, element, variable, element_index, profile_name=None, data_source=None,
				 scale_factor=1.0, in_service=True, recycle=False, order=-1, level=-1,
				 drop_same_existing_ctrl=False, matching_params=None,
				 initial_run=False, **kwargs):
		# just calling init of the parent
		if matching_params is None:
			matching_params = {"element": element, "variable": variable,
							   "element_index": element_index}
		super().__init__(net, in_service=in_service, recycle=recycle, order=order, level=level,
						 drop_same_existing_ctrl=drop_same_existing_ctrl,
						 matching_params=matching_params, initial_run=initial_run,
						 **kwargs)

		# data source for time series values
		self.data_source = data_source
		# ids of sgens or loads
		self.element_index = element_index
		# element type
		self.element = element
		self.values = None
		self.profile_name = profile_name
		self.scale_factor = scale_factor
		self.applied = False
		self.write_flag, self.variable = _detect_read_write_flag(
			net, element, element_index, variable)
		# self.set_recycle(net)

	def time_step(self, net, time):
		"""
		Get the values of the element from data source
		Write to pandapower net by calling write_to_net()
		If ConstControl is used without a data_source, it will reset the controlled values to the initial values,
		preserving the initial net state.
		"""
		self.applied = False
		if self.data_source is None:
			self.values = net[self.element][self.variable].loc[self.element_index]
		else:
			self.values = self.data_source.get_time_step_value(time_step=time,
															   profile_name=self.profile_name,
															   scale_factor=self.scale_factor)
		if self.values is not None:
			write_to_net(net, self.element, self.element_index,
						 self.variable, self.values, self.write_flag)

	def is_converged(self, net):
		"""
		Actual implementation of the convergence criteria: If controller is applied, it can stop
		"""
		return self.applied

	def control_step(self, net):
		"""
		Set applied to True, which means that the values set in time_step have been included in the load flow calculation.
		"""
		self.applied = True

	def __str__(self):
		return super().__str__() + " [%s.%s]" % (self.element, self.variable)
