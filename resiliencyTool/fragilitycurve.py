import os
import pandas as pd
import numpy as np

# For displaying warnings
import warnings

# For finding paths
from . import config

# For plotting
import matplotlib.pyplot as plt

# To fit the fragility curve using the GAM method
from pygam import LinearGAM, s

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
	:return fig: matplotlib.pyplot figure
	:return ax: matplotlib.pyplot axis

	Uses spline to interpolate the fragility curve data to a new x array. values above 1 are cliped.
	Plots the original data and the interpolated data for all fragility curves

	'''
	fig, ax = plt.subplots(tight_layout=True)

	for fc in dict_fc.values():
		ynew = fc.interpolate(xnew)
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

		X = np.array([[i] for i in self.x_data])
		self.gam = LinearGAM(s(0, n_splines=len(X))).gridsearch(X, self.y_data)

	def interpolate(self, xnew):
		'''
		:param xnew: list, new intensity vector
		:return ynew: interpolated failure probabilities

		Uses gam to interpolate the fragility curve data to a new x array. values above 1 are cliped.
		'''
		return self.gam.predict(xnew).clip(0,1)

	def projected_fc(self, rp, ref_rp, xnew):
		'''
		:param fc: reference fragility curve
		:return ynew: interpolated failure probabilities for the projected fragility into the reference return period intensities
		'''
		if rp == ref_rp:
			return self.interpolate(xnew)
		else:
			return self.interpolate(rp.interpolate_inv_return_period(ref_rp.interpolate_return_period(xnew)))

	def plot_fc(self, xnew):
		'''
		:param xnew: list, new intensity vector
		:param k: int, must be between 1 and 5, default=3, interpolation order
		:return fig: matplotlib.pyplot figure
		:return ax: matplotlib.pyplot axis

		Plots the interpolated and original data of the fragility curve
		'''
		fig, ax = plt.subplots(tight_layout=True)
		ynew = self.interpolate(xnew)
		plt.plot(xnew, ynew)
		plt.plot(self.x_data, self.y_data, 'o')

		plt.xlabel("intensity")
		plt.ylabel("probability of failure")
		ax.set_title(self.name)
		plt.legend(['interpolation','data'], loc='best')
		return fig, ax