import resiliencyTool as rt

simulationName = 'Simulation_1'
network = rt.network.Network(simulationName)
simulation = rt.simulation.Sim(simulationName)
# simulation.pre_run(network, 100)
#TODO: bug if pre_run and run are launched one right after the other
simulation.run(network)




