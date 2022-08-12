import resiliencyTool as rt
import pandas as pd

simulationName = 'Simulation_1'
network = rt.network.Network(simulationName)
simulation = rt.simulation.Sim(simulationName)
# simulation.pre_run(network, 100)
#TODO: bug if pre_run and run are launched one right after the other
# iteration_set = None
# time_set = pd.date_range("2022-01-01 12:00:00", periods = 10, freq = 'H')
# breakpoint()
simulation.run(network, iterationSet=[0,1,2,3,4,5])




