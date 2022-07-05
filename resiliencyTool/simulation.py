import pandas as pd
import numpy as np
from resiliencyTool import GeoData




class Sim:
	'''
	Add description of Sim class here
	'''

	def __init__(self,
				 history,
				 start,
				 duration,
				 hazardTime
				 ):

		self.history = history
		self.time = Time(start, duration)
		self.hazardTime = hazardTime

	# def create_outages_schedule(self, network):
	# 	'''
	# 	Add description of create_outages function
	# 	'''
	# 	failureCandidates = network.get_failure_candidates()
	# 	# crews = network.crews

	# 	failureProbability = np.array(
	# 		[x.failureProb for x in failureCandidates.values()])
	# 	randomNumber = 1
	# 	while (randomNumber > failureProbability).all():  # random until a failure happens
	# 		randomNumber = np.random.rand()
	# 	failure = np.where((randomNumber <= failureProbability), np.random.randint(
	# 		self.hazardTime.start, [self.hazardTime.stop]*len(failureCandidates)), self.time.stop)
	# 	crewSchedule = pd.DataFrame([[1]*len(network.crews)]*self.time.duration, columns=[
	# 								x.ID for x in network.crews], index=self.time.interval)
	# 	outagesSchedule = pd.DataFrame([[STATUS['on']]*len(failureCandidates)] *
	# 								   self.time.duration, columns=failureCandidates.keys(), index=self.time.interval)
	# 	for index, column in zip(failure, outagesSchedule):
	# 		# outagesSchedule[column].loc[(outagesSchedule.index >= failure[i])] = STATUS['off']
	# 		outagesSchedule[column].loc[index:] = STATUS['off']

	# 	for index, row in outagesSchedule.iterrows():
	# 		failureElements = row.index[row == 0].tolist()
	# 		availableCrews = crewSchedule.loc[index][crewSchedule.loc[index] == 1].index.tolist(
	# 		)
	# 		elementsToRepair, repairingCrews = network.get_closest_available_crews(
	# 			availableCrews, failureElements)
	# 		crewsTravelingTime = network.get_crews_traveling_time(
	# 			repairingCrews, elementsToRepair)
	# 		repairingTime = network.get_reparing_time(
	# 			elementsToRepair, failureCandidates)

	# 		for t_0, t_1, e, c in zip(crewsTravelingTime, repairingTime, elementsToRepair, repairingCrews):
	# 			outagesSchedule.loc[index+1:index+t_0, e] = STATUS['waiting']
	# 			outagesSchedule.loc[index+t_0+1:index +
	# 								t_0+t_1, e] = STATUS['reparing']
	# 			# TODO: this line can be removed if outagesSchedule is set to 1 on at failure time
	# 			outagesSchedule.loc[index+t_0+t_1+1:, e] = STATUS['on']
	# 			crewSchedule.loc[index+1:index+t_0+t_1, c] = e
	# 		'''
	# 		print(f'Time step: {index}')
	# 		print(f'Failure elements: {failureElements}')
	# 		print(f'Avaiable crews: {availableCrews}')
	# 		print(f'Reparing crews: {repairingCrews}')
	# 		print(f'Crews traveling time: {crewsTravelingTime}')
	# 		print(f'Elements reparing time {repairingTime}')
	# 		print(outagesSchedule.join(crewSchedule))
	# 		if (len(elementsToRepair)>0):
	# 			breakpoint()			
	# 		'''
	# 		if not outagesSchedule.loc[index+1:].isin([STATUS['on']]).any().any():
	# 			print(failureCandidates.keys())
	# 			print(failure)
	# 			print(f'Finished at time step: {index}')
	# 			print(outagesSchedule.join(crewSchedule))

	# 			return outagesSchedule
	# 	return outagesSchedule

	def create_global_database(self, iterations, databases):
		return pd.concat(dict(zip(iterations, databases)), names = ['Iteration'], axis = 1)

	def run(self, network, iterationNumber):
		network.calculate_outages_schedule(self.time, self.hazardTime)
		databases = []
		iterations = range(iterationNumber) 
		for i in iterations:
			network.calculate_metrics()
			databases.append(network.create_metrics_database())
		out = pd.concat(dict(zip(iterations, databases)), names = ['Iteration'], axis = 1)
		breakpoint()
	# def get_output_database(self, network, iteration):
	# 		timeSeries = []
	# 		scalar = []
	# 		for metric in network.metrics:
	# 			df = metric.value.copy()
	# 			breakpoint()
	# 			keys, values = zip(*[(key, value) for key, value in metric.__dict__.items() if key is not 'value'])
	# 			# keys = keys + [iteration]

	# 			df.columns = pd.MultiIndex.from_tuples([values], names=keys)
	# 			breakpoint()
	# 			if df.shape[0] == 1:
	# 				scalar.append(df)
	# 			else:
	# 				timeSeries.append(df)

	# 		dfTimeSeries = pd.concat(timeSeries, axis = 1)
	# 		# dfScalar = pd.concat(scalar, axis = 1)
	# 		dfTimeSeries.to_csv(f'timeSeries_{network.ID}.csv')
	# 		# dfScalar.to_csv(f'scalars_{network.ID}.csv')
	# 		return dfTimeSeries
	
	# def post_process(self, network):

	# 	build_output_database(network)

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
