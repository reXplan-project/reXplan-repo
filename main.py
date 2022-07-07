import resiliencyTool as rt
import numpy as np
from datetime import datetime, timedelta

simStart=0
simDuration = 48
eventStart = 12
eventDuration = 12
nLines = 10
nCrew = 3

simulationName = 'Simulation_1'
network = rt.network.Network(1,'spain',rt.config.path.networkFile(simulationName)) #todo: better manage input parameters and handling of simulation name

for i in range(nLines):
	network.lines.append(rt.network.Line(ID = 'line_{}'.format(i),
					fragilityCurve = None,
					kf = None,
					resilienceFull = None,
					lineSpan = None,
					towers = None,
					failureProb = np.random.rand(),
					normalTTR = np.random.randint(0,eventDuration*0.5)),
					)

for i in range(nCrew):
	network.crews.append(rt.network.Crew(f'crew_{i}'))

simulation = rt.simulation.Sim(
				simulationName = simulationName,
				history = None,
				start = simStart,
				duration = simDuration, 
				hazardTime = rt.simulation.Time(eventStart,eventDuration),
				)
simulation.run(network, 100)


