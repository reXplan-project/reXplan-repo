

import sys
import pandas as pd
import os
import shutil
sys.path.insert(0, '..')
import resiliencyTool as rt
import argparse

path = rt.config.path

parser = argparse.ArgumentParser()
parser.add_argument("-tn", "--test_number", type=int, default=-1, help="test number")
args = parser.parse_args()

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
	# shutil.copyfile(path.networkFile(control_test), path.networkFile(test))
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
	# simulation.time = rt.simulation.Time(start = 26, duration = 1)
	# simulation.pre_run(network, 100)
	simulation.run(network, iterationSet = [0, 1, 2, 3, 4, 5], run_type = 'opf', delta = 1e-16, saveOutput = False)
	# simulation.run(network, delta = 1e-16)

def test_3():
	simulationName = 'test_3'
	network = rt.network.Network(simulationName)
	simulation = rt.simulation.Sim(simulationName)
	simulation.time = rt.simulation.Time(start = 14, duration = 1)
	# simulation.time = rt.simulation.Time(start = 26, duration = 1)
	# simulation.run(network, iterationSet = [0], run_type = 'opf', delta = 1e-16, saveOutput = True)
	# simulation.time = rt.simulation.Time(start = 28, duration = 1)	
	# simulation.run(network, iterationSet = [1], run_type = 'opf', delta = 1e-16, saveOutput = True)
	simulation.run(network, iterationSet = [2], run_type = 'opf', delta = 1e-15, saveOutput = True)
	
	filter = network.pp_network.ext_grid['name'] == 'g16_slack' # already in the input file but not in service
	index = filter.index[filter].values[0]
	network.pp_network.ext_grid.at[index, 'in_service'] = True

	filter = network.pp_network.ext_grid['name'] == 'eg'
	index = filter.index[filter].values[0]
	network.pp_network.ext_grid.at[index, 'in_service'] = False

	filter = network.pp_network.gen['name'] == 'g16' 
	index = filter.index[filter].values[0]
	network.pp_network.gen.at[index, 'in_service'] = False
	breakpoint()
	simulation.run(network, iterationSet = [2], run_type = 'opf', delta = 1e-15, saveOutput = True)



def test_4():
	simulationName = 'test_4'
	network = rt.network.Network(simulationName)
	simulation = rt.simulation.Sim(simulationName)
	simulation.run(network, iterationSet = [2], run_type = 'opf', delta = 1e-15, saveOutput = True)

def test_5():
	simulationName = 'test_5'
	network = rt.network.Network(simulationName)
	simulation = rt.simulation.Sim(simulationName)
	simulation.time = rt.simulation.Time(start = 13, duration = 1)
	simulation.run(network, iterationSet = [2], run_type = 'opf', max_iteration = 1000, delta = 1e-15, saveOutput = True)

if __name__ == '__main__':
	if args.test_number == 1:
		test_1()
	elif args.test_number == 2:
		test_2()	
	elif args.test_number == 3:
		test_3()
	elif args.test_number == 4:
		test_4()
	elif args.test_number == 5:
		test_5()

