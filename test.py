import resiliencyTool as rt
import pandapower as pp

import pandas as pd
import time
# import matplotlib.pyplot as plt
# import datetime

# run_time_series
import os
import numpy as np
import pandapower as pp
from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.control import ConstControl
# end of run_time_series

# plotting
from pandapower.plotting.plotly import simple_plotly
# end plotting


simulationName = 'Simulation_1'
network = rt.network.Network(simulationName)
# print(network.pp_network)
#print(network.bus)
#print(network.line)
#print(network.load)

def run_power_flows(network):
	tic = time.time()
	pp.rundcopp(network.pp_network, delta=1e-16)
	pp.runopp(network.pp_network, delta=1e-16)
	pp.runpp(network.pp_network, delta=1e-16)
	tac = time.time()
	print(f"Duration of runopp = {tac-tic}")
	print(network.pp_network.res_bus.vm_pu)
	print(network.pp_network.res_line.loading_percent)

def run_time_series(net, lines_to_disconnect):
	def create_data_source(n_timesteps, lines_to_disconnect):
		profiles = pd.DataFrame()
		# load,gen = [],[]
		for index, row in net.load.iterrows():
			profiles[row['name']] = np.random.random(n_timesteps) * row['p_mw']
			# load.append(row['name'])
		for index, row in net.gen.iterrows():
			profiles[row['name']] = np.random.random(n_timesteps) * row['p_mw']
			# gen.append(row['name'])
		for index, row in net.line.iterrows():
			not_disconnect = not (row['name'] in lines_to_disconnect)
			profiles[row['name']] = np.random.choice([True, not_disconnect],n_timesteps)#.astype(int)*100
		print(profiles.loc[:,profiles.columns.str.contains('line')])
		# breakpoint()
		return DFData(profiles)

	def create_output_writer(net, time_steps, output_dir):
		ow = OutputWriter(net, time_steps, output_path=output_dir, output_file_type=".xlsx", log_variables=list())
		# these variables are saved to the harddisk after / during the time series loop
		ow.log_variable('res_load', 'p_mw')
		ow.log_variable('res_gen', 'p_mw')
		ow.log_variable('res_bus', 'vm_pu')
		ow.log_variable('res_line', 'loading_percent')
		ow.log_variable('res_line', 'i_ka')
		return ow

	def create_controllers(net, ds):
		ConstControl(net, element='load', variable='p_mw', element_index=net.load.index,
					 data_source=ds, profile_name=net.load.name)
		ConstControl(net, element='gen', variable='p_mw', element_index=net.gen.index,
					 data_source=ds, profile_name=net.gen.name)
		# net.line.loc[net.line['name'] == 'line1','in_service'] = False
		# net.line.loc[net.line['name'] == 'line2','in_service'] = False
		# net.line.loc[net.line['name'] == 'line3','in_service'] = False
		rt.simulation.SimpleControl(net, element='line', variable='in_service', element_index=net.line.index,
					 data_source=ds, profile_name=net.line.name) # max_loading_percent
		return net


	n_timesteps = 48
	time_steps = range(0, n_timesteps)
	create_controllers(net, create_data_source(n_timesteps, lines_to_disconnect))
	create_output_writer(net, time_steps, output_dir=rt.config.path.outputFolderPath(simulationName))
	run_timeseries(net, time_steps)


if __name__ == "__main__":
	# run_power_flows(network)
	lines_to_disconnect = ['line3', 'line7']
	# lines_to_disconnect = []
	# run_time_series(network.pp_network, lines_to_disconnect)
	simple_plotly(network.pp_network)
