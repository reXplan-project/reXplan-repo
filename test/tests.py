

import sys
import pandas as pd
import os
import shutil
sys.path.insert(0, '..')
import resiliencyTool as rt

path = rt.config.path

def generate_control_test_1():
	# launch just once
	simulationName = 'control_test_1'
	network = rt.network.Network(simulationName)
	simulation = rt.simulation.Sim(simulationName)
	simulation.pre_run(network, 100)
	simulation.run(network)

def test_1():
	def get_df(iterationNumber, simulationName):
		df = pd.read_csv(path.engineDatabaseFile(simulationName), index_col=[0, 1, 2, 3]).T
		return df[range(iterationNumber)]
	iterationNumber = 5 # compare first 5 montecarlo iterations
	control_test = 'control_test_1'
	test = 'test_1'
	# copy network.xlsx and montecarlo_database.csv from control_test_1 to test_1's folder before lauching it
	rt.config.checkPath(os.path.join(path.inputFolder, test))
	shutil.copyfile(path.networkFile(control_test), path.networkFile(test))
	shutil.copyfile(path.montecarloDatabaseFile(control_test),
					path.montecarloDatabaseFile(test))
	
	network = rt.network.Network(test)
	simulation = rt.simulation.Sim(test)
	simulation.run(network, iterationNumber=iterationNumber)
	df_output = get_df(iterationNumber, test)
	df_control = get_df(iterationNumber, control_test)

	return df_output.equals(df_control)

if __name__ == '__main__':
	print (test_1())
