

import sys
import pandas as pd
import os
import shutil
# sys.path.insert(0, '..')
import resiliencyTool as rt

path = rt.config.path

# def launch_simulation(simulationName):
# 	network = rt.network.Network(simulationName)
# 	simulation = rt.simulation.Sim(simulationName)
# 	simulation.time = rt.simulation.Time(start = 11, duration = 2)
# 	# simulation.pre_run(network, 100)
# 	simulation.run(network, delta = 1e-16, distributed_slack = True)

def generate_control_test_1():
	# launch just once
	simulationName = 'control_test_1'
	iterationSet = range(5)
	network = rt.network.Network(simulationName)
	simulation = rt.simulation.Sim(simulationName)
	simulation.pre_run(network, 100)
	simulation.run(network, iterationSet=iterationSet, run_type = 'pf')

def test_1():
	# generate_control_test_1()
	def get_df(iterationSet, simulationName):
		df = pd.read_csv(path.engineDatabaseFile(simulationName), index_col=[0, 1, 2, 3]).T
		return df[iterationSet]
	iterationSet = range(5)
	control_test = 'control_test_1'
	test = 'test_1'
	# copy network.xlsx and montecarlo_database.csv from control_test_1 to test_1's folder before launching it
	rt.config.checkPath(os.path.join(path.inputFolder, test))
	shutil.copyfile(path.networkFile(control_test), path.networkFile(test))
	shutil.copyfile(path.montecarloDatabaseFile(control_test),
					path.montecarloDatabaseFile(test))
	
	network = rt.network.Network(test)
	simulation = rt.simulation.Sim(test)
	simulation.run(network, iterationSet=iterationSet, run_type = 'pf')
	df_output = get_df(iterationSet, test)
	df_control = get_df(iterationSet, control_test)
	return df_output.equals(df_control)

def test_2():
	simulationName = 'test_2'
	network = rt.network.Network(simulationName)
	simulation = rt.simulation.Sim(simulationName)
	# simulation.time = rt.simulation.Time(start = 11, duration = 3)
	# simulation.pre_run(network, 100)
	simulation.run(network, delta = 1e-16, iterationSet=[0], distributed_slack = False)

if __name__ == '__main__':
	# print (test_1())
	test_2()	

