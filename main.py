import resiliencyTool as rt
import numpy as np
from datetime import datetime, timedelta

# simStart=10
# simDuration = 48
# eventStart = 30
# eventDuration =5
# nLines = 10
# nCrew = 3

simulationName = 'Simulation_1'
network = rt.network.Network(simulationName)
simulation = rt.simulation.Sim(simulationName)
# simulation.pre_run(network, 100)
simulation.run(network)


# for i in range(nLines):
# 	network.lines.append(rt.network.Line(ID = 'line_{}'.format(i),
# 					fragilityCurve = None,
# 					kf = None,
# 					resilienceFull = None,
# 					lineSpan = None,
# 					towers = None,
# 					failureProb = np.random.rand(),
# 					normalTTR = np.random.randint(0,eventDuration*0.5)),
# 					)


# for i in range(nCrew):
	# network.crews.append(rt.network.Crew(f'crew_{i}'))

# simulation = rt.simulation.Sim(
# 				simulationName = simulationName,
				# history = None,
				# start = simStart,
				# duration = simDuration, 
				# hazardTime = rt.simulation.Time(eventStart,eventDuration),
				# )
# for line in network.lines:
# 	line.failureProb = np.random.rand()
# 	line.normalTTR = np.random.randint(0,eventDuration*0.5)
# breakpoint()
# simulation.time =  rt.simulation.Time(simStart, simDuration)
# simulation.hazardTime =  rt.simulation.Time(eventStart, eventDuration)




