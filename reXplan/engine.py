import re
import pandas as pd
import pandapower as pp
from pandapower.timeseries import DFData
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.timeseries import OutputWriter

from pandapower.toolbox import _detect_read_write_flag, write_to_net
from pandapower.control.basic_controller import Controller

from pandapower.control import ConstControl

from . import simulation
from . import config


rt_pp_map = {'Generator': 'gen', 'Line': 'line',
             'Load': 'load', 'Transformer': 'trafo', 'Switch': 'switch', 'Bus': 'bus'}

REGEX = r"(?:res_)?(\w+)\."


class pandapower():
	# TODO: @TIM add description
	def __init__(self, network):
		self.network = network  # pandapower network type

	def create_controllers(self, df_timeseries):
		# TODO call cont.py
		def get_network_df_intersection_dataframe(network, type, dataframe):
			# Intersection between network.type and df based on names and ids.
			out = getattr(network, rt_pp_map[type])
			return out.loc[out.name.isin(dataframe.columns)]

		for (field, type), df in df_timeseries.groupby(level=['field', 'type'], axis=1):
			# TODO: find better way to avoid level drop
			df.columns = df.columns.droplevel(['field', 'type'])
			df_elements = get_network_df_intersection_dataframe(
				self.network, type, df)
			self.SimpleControl(self.network, element=rt_pp_map[type], variable=field, element_index=df_elements.index, data_source=self.DFData_boolean_adapted(
				df), profile_name=df_elements.name, drop_same_existing_ctrl=True)

	def configure_output_writer(self, time_steps):
		ow = OutputWriter(self.network, time_steps)
		ow.log_variable('res_ext_grid', 'p_mw')
		ow.log_variable('res_ext_grid', 'q_mvar')
		ow.log_variable('res_load', 'p_mw')
		ow.log_variable('res_load', 'q_mvar')
		ow.log_variable('res_gen', 'p_mw')
		ow.log_variable('res_gen', 'q_mvar')
		ow.log_variable('res_bus', 'vm_pu')
		ow.log_variable('res_bus', 'va_degree')
		ow.log_variable('res_bus', 'p_mw')
		ow.log_variable('res_bus', 'q_mvar')
		ow.log_variable('res_line', 'loading_percent')
		ow.log_variable('res_line', 'i_ka')
		ow.log_variable('res_line', 'pl_mw')
		ow.log_variable('res_line', 'ql_mvar')
		ow.log_variable('load', 'max_p_mw')
		ow.log_variable('load', 'max_q_mvar')
		ow.log_variable('load', 'in_service')
		ow.log_variable('gen', 'max_p_mw')
		ow.log_variable('gen', 'max_q_mvar')
		ow.log_variable('gen', 'in_service')
		ow.log_variable('line', 'in_service')
		ow.log_variable('trafo', 'in_service')
		ow.log_variable('bus', 'in_service')
		ow.log_variable('switch', 'closed')

		return ow

	def run_time_series(self, df_timeseries, **kwargs):
		# TODO: @TIM add description
		run_timeseries(self.network, df_timeseries.index, continue_on_divergence=True, **kwargs)

	def format(self, output):
		out = []
		for key, df in output.output.items():
			type = re.search(REGEX, key)
			if type:
				type = type.group(1)
				field = key.split('.')[1]
				df_ = df.rename(columns=getattr(self.network, type)['name'].to_dict())
				df_.columns.name = 'id'
				out.append(pd.concat([df_], keys=[(field, type)], names=['field', 'type'], axis=1))
		return pd.concat(out, axis=1)

	def run(self, df_timeseries, **kwargs):
		"""
		Runs the time series simulation for the given timeseries data.

		Parameters:
		df_timeseries (DataFrame): The timeseries data for the simulation.
		kwargs: Additional keyword arguments for the simulation configuration.

		Returns:
		DataFrame: The formatted output of the simulation.
		"""
		self.create_controllers(df_timeseries)
		output = self.configure_output_writer(df_timeseries.index)
		run_type = pp.runopp
		if 'run_type' in kwargs:
			if kwargs['run_type'] == 'pf' or kwargs['run_type'] == 'ac_pf':
				# Exectues an AC power flow calculation from pandapower
				run_type = pp.runpp

			elif kwargs['run_type'] == 'dc_pf':
				# Executes a DC power flow calculation from pandapower
				run_type = pp.rundcpp

			elif kwargs['run_type'] == 'ac_opf':
				# Executes an AC optimal power flow calculation from pandapower
				run_type = pp.runopp

			elif kwargs['run_type'] == 'dc_opf':
				# Executes a DC optimal power flow calculation from pandapower
				run_type = pp.rundcopp

			elif kwargs['run_type'] == 'pm_ac_opf':
				# Executes an AC optimal power flow calculation from pandapower via PandaModels (Julia)
				run_type = pp.runpm_ac_opf

			elif kwargs['run_type'] == 'pm_dc_opf':
				# Executes a DC optimal power flow calculation from pandapower via PandaModels (Julia)
				run_type = pp.runpm_dc_opf

			else:
				print('No run_type specified. Defaulting to runopp (OPF)')

		self.run_time_series(df_timeseries, run=run_type, **kwargs)
		return self.format(output)

	class SimpleControl(ConstControl):
		"""
		ConstControl does not consider "in_service" parameter for timeseries manipulation. 
		set_recycle routine (from ConstControl) will therefore define the recycle variable for this parameter as False.
		As an outcome, it will not be considered during the timeseries simulation (it still note clear why though).
		We therefore overwrite the routine to do nothing. The default behaviour will be therefore that no variables are recycled, 
		i.e. everything is re-mapped as an input before launching the actual simulation.
		"""
		# TODO: For improving performance, set_recycle must be constructed to consider the specific variables that will be changing during a timeseries simulation

		def __init__(self, net, element, variable, element_index, profile_name=None, data_source=None,
					 scale_factor=1.0, in_service=True, recycle=False, order=-1, level=-1, drop_same_existing_ctrl=False,
					 matching_params=None, initial_run=False, **kwargs):

			super().__init__(net, element, variable, element_index, profile_name, data_source, scale_factor,
							 in_service, recycle, order, level, drop_same_existing_ctrl, matching_params, initial_run, **kwargs)

		def set_recycle(self, net):
			# it overrides set_recycle function from ConstControl
			pass

	class DFData_boolean_adapted(DFData):
		def __init__(self, df, multi=False):
			super().__init__(df, multi=multi)

		def get_time_step_value(self, time_step, profile_name, scale_factor=1.0):
			res = self.df.loc[time_step, profile_name]
			if hasattr(res, 'values'):
				res = res.values
			if any(isinstance(x, bool) for x in res):
				# without explicit conversion, array's dtype remains as 'object', which will make numba fail
				res = res.astype(dtype=bool)
			else:
				res = res*scale_factor
			return res
