import sys
sys.path.insert(0,'..')

import resiliencyTool as rt
from const import *

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

network_filename = 'testNetwork4nodes'
network = rt.Network(1,'spain',network_filename)

simStart=0
simDuration = 48
eventStart = 12
eventDuration = 12
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
				start = simStart,
				duration = simDuration, 
				hazardTime = rt.Time(eventStart,eventDuration),
				)
simulation.run(network, 2)
# simulation.post_process(network)

