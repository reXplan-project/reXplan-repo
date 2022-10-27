import os
import pandas as pd

# For displaying warnings
import warnings

# For finding paths
from . import config

# For plotting
import matplotlib.pyplot as plt

# For interpolation using spline
from scipy.interpolate import interp1d, splrep, splev

def readFragilityCurves(simulationName):
	'''
	:param simulationName: string, name of the simulation case.
	:return fragility_curve_list: list, list of tuples containing the original data from the fragility curves
	'''
	fragility_curve_list = []
	directory = config.path.fragilityCurveFolder(simulationName)

	exclude = set(['.ipynb_checkpoints'])
	for _,dirs,files in os.walk(directory):
		dirs[:] = [d for d in dirs if d not in exclude]
		for file in files:
			if file.endswith(".csv"):
				df = pd.read_csv(os.path.join(directory,file), sep=';')
				x = df["intensity"].values
				for col in df.columns[1:]:
					fragility_curve_list.append((col,x,df[col].values))
	return fragility_curve_list

def build_fragility_curve_database(simulationName):
	'''
	:param simulationName: string, name of the simulation case.
	:param xnew: list, new intensity vector
	:return dict_fragility_curves: dictionary with fragility curve objects

	builds a dictionary with the name of the fragility curve as a key and the fragility curve objects as values
	'''

	fcs = readFragilityCurves(simulationName)
	
	dict_fragility_curves = {}
	
	for fc in fcs:
		dict_fragility_curves[fc[0]] = FragilityCurve(fc[0], list(fc[1]), list(fc[2]))
	return dict_fragility_curves


def plotFragilityCurves(dict_fc, xnew, k=3):
	'''
	:param dict_fc: dict of Fragility curve objects, can be generated using build_fragility_curve_database(simulationName) 
	:param xnew: list, new intensity vector
	:param k: int, must be between 1 and 5, default=3, interpolation order
	:return fig: matplotlib.pyplot figure
	:return ax: matplotlib.pyplot axis

	Uses spline to interpolate the fragility curve data to a new x array. values above 1 are cliped.
	Plots the original data and the interpolated data for all fragility curves

	k = 1 -> linear interpolation
	k = 2 -> quadratic interpolation
	k = 3 -> cubic interpolation
	'''
	fig, ax = plt.subplots(tight_layout=True)

	for fc in dict_fc.values():
		ynew = fc.interpolate(xnew, k)
		plt.plot(xnew, ynew)
	
	plt.gca().set_prop_cycle(None)
	
	for fc in dict_fc.values():
		plt.plot(fc.x_data, fc.y_data, 'o')

	legend = dict_fc.keys()
	plt.legend(legend, loc='best')
	plt.xlabel("intensity")
	plt.ylabel("probability of failure")
	return fig, ax

class FragilityCurve:
	'''
	FragilityCurve Class:

	:attribute name: string, fragility curve name/type
	:attribute x_data: list, intensity data
	:attribute y_data: list, probability of failure data

	:method interpolate(xnew, k=3):
	:method plot_fc(xnew, k=3):
	'''
	def __init__(self, name, x, y):
		'''
		FragilityCurve Class Constructor:

		:param name: string, fragility curve name/type
		:param x: list, intensity data
		:param y: list, probability of failure data
		'''
		self.x_data = x
		self.y_data = y
		self.name = name

	def interpolate(self, xnew, k=3):
		'''
		:param xnew: list, new intensity vector
		:param k: int, must be between 1 and 5, default=3, interpolation order
		:return ynew: interpolated failure probabilities

		Uses spline to interpolate the fragility curve data to a new x array. values above 1 are cliped.

		k = 1 -> linear interpolation
		k = 2 -> quadratic interpolation
		k = 3 -> cubic interpolation
		...
		'''
		if k > 0 and k < 6:
			tck = splrep(self.x_data, self.y_data)
			ynew = splev(xnew, (tck[0],tck[1],k), der=0)
		else:
			warnings.warn(f'k = {k} is invalid, please choose a value for k between 1 and 5')
			return
		return ynew.clip(0,1)

	def plot_fc(self, xnew, k=3):
		'''
		:param xnew: list, new intensity vector
		:param k: int, must be between 1 and 5, default=3, interpolation order
		:return fig: matplotlib.pyplot figure
		:return ax: matplotlib.pyplot axis

		Plots the interpolated and original data of the fragility curve
		'''
		fig, ax = plt.subplots(tight_layout=True)
		ynew = self.interpolate(xnew, k)
		plt.plot(xnew, ynew)
		plt.plot(self.x_data, self.y_data, 'o')

		plt.xlabel("intensity")
		plt.ylabel("probability of failure")
		ax.set_title(self.name)
		plt.legend(['interpolation','data'], loc='best')
		return fig, ax