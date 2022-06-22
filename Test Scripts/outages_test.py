import sys
sys.path.insert(0,'..')

import resiliencyTool as rt
from const import *

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

network_filename = 'testNetwork4nodes'
network = rt.Network(1,'spain',network_filename)


simulationDuration = 48
eventDuration = 24
timeStep = 1
nLines = 10
nCrew = 3

for i in range(nLines):
	network.lines.append(rt.Line(ID = 'line_{}'.format(i),
					fragilityCurve = None,
					kf = None,
					resilienceFull = None,
					lineSpan = None,
					towers = None,
					failureProb = np.random.rand(),
					normalTTR = np.random.randint(0,eventDuration*0.5)),
					)

for i in range(nCrew):
	network.crews.append(rt.Crew(f'crew_{i}'))

simulation = rt.Sim(history = None,
				simStartTime  = pd.to_datetime(datetime(2020, 4, 20)),
				simTime = simulationDuration, # hours
				eventDuration = eventDuration, # hours timedelta(hours = 24), 
				timeStep = timeStep
				)
simulation.create_outages_schedule(network)



