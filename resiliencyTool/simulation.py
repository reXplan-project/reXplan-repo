import pandas as pd
import numpy as np
from . import config

def build_global_database(iterations, databases):
	return pd.concat(dict(zip(iterations, databases)), names = ['iteration'], axis = 1)
class Sim:
	'''
	Add description of Sim class here
	'''

	def __init__(self,
				simulationName,
				history,
				start,
				duration,
				hazardTime
				):
		self.simulationName = simulationName
		self.history = history
		self.time = Time(start, duration)
		self.hazardTime = hazardTime
	

	def run(self, network, iterationNumber):
		
		databases = []
		iterations = range(iterationNumber)
		for i in iterations:
			print(i)
			network.calculate_outages_schedule(self.time, self.hazardTime)
			network.calculate_metrics()
			databases.append(network.build_metrics_database())
		out = build_global_database(iterations, databases)
		out.to_csv(config.path.globalDatabaseFile(self.simulationName))


# class Event():
# 	def __init__(self, start, duration):
# 		self.time = Time(start, duration)


class Time():
	# TODO: error raising for uncompatible times
	def __init__(self,
				 start,
				 duration
				 ):
		self.start = start
		self.duration = duration
		self.stop = duration + start
		self.interval = list(range(start, duration + start))
