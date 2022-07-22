import pandas as pd
import numpy as np
from . import config

from pandapower.control.basic_controller import Controller
from pandapower.toolbox import _detect_read_write_flag, write_to_net
# from pandapower.control import ConstControl

def build_global_database(iterations, databases):
	out = pd.concat(dict(zip(iterations, databases)), names = ['iteration'], axis = 1).T
	out.columns.name = 'timestep'
	return out
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

#todo SimpleControl heriting ConstCrontol and overriding set_recycle()
# class SimpleControl(ConstControl):

class SimpleControl(Controller):

    def __init__(self, net, element, variable, element_index, profile_name=None, data_source=None,
                 scale_factor=1.0, in_service=True, recycle=False, order=-1, level=-1,
                 drop_same_existing_ctrl=False, matching_params=None,
                 initial_run=False, **kwargs):
        # just calling init of the parent
        if matching_params is None:
            matching_params = {"element": element, "variable": variable,
                               "element_index": element_index}
        super().__init__(net, in_service=in_service, recycle=recycle, order=order, level=level,
                         drop_same_existing_ctrl=drop_same_existing_ctrl,
                         matching_params=matching_params, initial_run=initial_run,
                         **kwargs)

        # data source for time series values
        self.data_source = data_source
        # ids of sgens or loads
        self.element_index = element_index
        # element type
        self.element = element
        self.values = None
        self.profile_name = profile_name
        self.scale_factor = scale_factor
        self.applied = False
        self.write_flag, self.variable = _detect_read_write_flag(net, element, element_index, variable)
        # self.set_recycle(net)

    def time_step(self, net, time):
        """
        Get the values of the element from data source
        Write to pandapower net by calling write_to_net()
        If ConstControl is used without a data_source, it will reset the controlled values to the initial values,
        preserving the initial net state.
        """
        self.applied = False
        if self.data_source is None:
            self.values = net[self.element][self.variable].loc[self.element_index]
        else:
            self.values = self.data_source.get_time_step_value(time_step=time,
                                                               profile_name=self.profile_name,
                                                               scale_factor=self.scale_factor)
        if self.values is not None:
            write_to_net(net, self.element, self.element_index, self.variable, self.values, self.write_flag)

    def is_converged(self, net):
        """
        Actual implementation of the convergence criteria: If controller is applied, it can stop
        """
        return self.applied

    def control_step(self, net):
        """
        Set applied to True, which means that the values set in time_step have been included in the load flow calculation.
        """
        self.applied = True

    def __str__(self):
        return super().__str__() + " [%s.%s]" % (self.element, self.variable)
