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

def interpolate_fc(datax,datay,xnew, k=3):
	'''
	:param datax: list, intensity data
	:param datay: list, failure probability data
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
		tck = splrep(datax, datay)
		ynew = splev(xnew, (tck[0],tck[1],k), der=0)
	else:
		warnings.warn(f'k = {k} is invalid, please choose a value for k between 1 and 5')
		return
	return ynew.clip(0,1)

def populateFragilityCurveDatabase(fragility_curve_list, xnew, k=3):
	'''
	:param fragility_curve_list: list, list of tuples containing the original data from the fragility curves, use fragilitycurve.readFragilityCurves(simulationName) to generate
	:param xnew: list, new intensity vector 
	:param k: int, must be between 1 and 5, default=3, interpolation order
	:return df: pandas.DataFrame, contains all the interpolated data of the fragility curves

	Uses spline to interpolate all fragility curves to a new x array. values above 1 are cliped.
	Populates a dataframe with the interpolated data.

	k = 1 -> linear interpolation
	k = 2 -> quadratic interpolation
	k = 3 -> cubic interpolation
	'''
	df = pd.DataFrame()
	df["intensity"] = xnew
	for fc in fragility_curve_list:
		df[fc[0]] = interpolate_fc(fc[1], fc[2], xnew, k)
	return df

def plotFragilityCurves(fragilityCurveDatabase=pd.DataFrame(), fcs=[]):
	'''
	:param fragilityCurveDatabase: pandas.DataFrame, interpolated fragility curves, use fragilitycurve.populateFragilityCurveDatabase(fragility_curve_list, xnew, k=3) to generate
	:param fcs: list, list of tuples containing the original data from the fragility curves, use fragilitycurve.readFragilityCurves(simulationName) to generate
	:return fig: matplotlib.pyplot figure
	:return ax: matplotlib.pyplot axis

	Plots the interpolated and original data of the fragility curve
	'''
	fig, ax = plt.subplots(tight_layout=True)
	legend = []
	if not fragilityCurveDatabase.empty:
		legend = fragilityCurveDatabase.columns[1:]
		for col in fragilityCurveDatabase.columns[1:]:
			plt.plot(fragilityCurveDatabase["intensity"], fragilityCurveDatabase[col])
	
	plt.gca().set_prop_cycle(None)
	
	if fcs:
		for fc in fcs:
			plt.plot(fc[1], fc[2], 'o')
			if fragilityCurveDatabase.empty:
				legend.append(fc[0])

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