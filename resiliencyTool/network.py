from const import *
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import netCDF4 as nc
import pandapower as pp

from mpl_toolkits.basemap import Basemap

COL_NAME_FRAGILITY_CURVE = 'fragility_curve'
COL_NAME_KF = 'kf'
COL_NAME_RESILIENCE_FULL = 'resilienceFull'
COL_NAME_WEATHER_TTR = 'weatherTTR'


# Network element status
STATUS = {'on': 1, 'off': 0, 'reparing': -1, 'waiting': -2}

GLOBAL_ID = -1
def get_GLOBAL_ID():
	global GLOBAL_ID
	GLOBAL_ID +=1
	return 'ID_{}'.format(GLOBAL_ID)


class Network:
	'''
	Add description of Newtork class here
	'''
	def __init__(self, ID, country, filename):
		self.ID = ID
		self.country = country
		self.name = ''
		self.totalInstalledPower = 0
		self.totalConventionalPower = 0
		self.totalRenewablePower = 0
		self.totalStoragePower= 0
		self.f_hz = 50
		self.nodes = []
		self.generators = []
		self.transformers = []
		self.tr_type = []
		self.lines = []
		self.ln_type = []
		self.loads = []
		self.crews = []
		self.pp_network = self.build_pp_network(filename)

		# self.outagesSchedule = None
		self.metrics = []
		# self.metrics = metrics()

	def read_network_attributes(self, filename):
		df_network = pd.read_excel(f'{filename}', sheet_name=SHEET_NAME_NETWORK)
		self.name = df_network[COL_NAME_NAME].values[0]
		self.f_hz = df_network[COL_NAME_FREQUENCY].values[0]
		self.sn_mva = df_network[COL_NAME_REF_POWER].values[0]

	def read_bus_attributes(self, filename):
		df_bus = pd.read_excel(f'{filename}', sheet_name=SHEET_NAME_NODES)
		self.name = df_bus[COL_NAME_NAME].values[0]
		self.f_hz = df_bus[COL_NAME_FREQUENCY].values[0]
		self.sn_mva = df_bus[COL_NAME_REF_POWER].values[0]

	def build_network(self, filename):
		self.read_network_attributes(filename)
		self.read_bus_attributes(filename)

		df_nodes = pd.read_excel(f'{filename}', sheet_name=SHEET_NAME_NODES)
		df_generators = pd.read_excel(f'{filename}', sheet_name=SHEET_NAME_GENERATORS)
		df_transformers = pd.read_excel(f'{filename}', sheet_name=SHEET_NAME_TRANSFORMER)
		df_tr_type = pd.read_excel(f'{filename}', sheet_name=SHEET_NAME_TR_TYPE)
		df_lines = pd.read_excel(f'{filename}', sheet_name=SHEET_NAME_LINES)
		df_ln_type = pd.read_excel(f'{filename}', sheet_name=SHEET_NAME_LN_TYPE)
		df_loads = pd.read_excel(f'{filename}', sheet_name=SHEET_NAME_LOADS)

	def updateGrid(self):
		'''
		Updates the network class elements
		'''
		print('Grid updated')
		return True

	def build_pp_network(self, filename):
		'''
		This Function takes as imput the input file name without the extention
		and gives as output the pandapower network object.
		'''
		# Creating the network elements
		df_network = pd.read_excel(f'{filename}', sheet_name=SHEET_NAME_NETWORK)
		df_network = df_network.where(pd.notnull(df_network), None)

		name = df_network[COL_NAME_NAME].values[0]
		f_hz = df_network[COL_NAME_FREQUENCY].values[0]
		sn_mva = df_network[COL_NAME_REF_POWER].values[0]
		kwargs_network = dict(f_hz=f_hz, name=name, sn_mva=sn_mva)
		network = pp.create_empty_network(**{key: value for key, value in kwargs_network.items() if value is not None})
		
		# Creating the bus elements
		df_bus = pd.read_excel(f'{filename}', sheet_name=SHEET_NAME_NODES)
		df_bus = df_bus.where(pd.notnull(df_bus), None) #TODO: update pandas version and replace this line of code by df_bus = df_bus.replace({np.nan: None}). Same for similar lines
		

		bus_ids = {}
		for index, row in df_bus.iterrows():
			if (row[COL_NAME_LONGITUDE] is not None) and (row[COL_NAME_LATITUDE] is not None):
				geodata = (row[COL_NAME_LATITUDE], row[COL_NAME_LONGITUDE])
			else:
				geodata = None
			kwargs_bus = dict(net=network, vn_kv=row[COL_NAME_VOLTAGE],
								name=row[COL_NAME_NAME],
								index=row[COL_NAME_INDEX],
								geodata=geodata, zone=row[COL_NAME_ZONE],
								in_service=row[COL_NAME_SERVICE],
								max_vm_pu=row[COL_NAME_MAX_VOLTAGE],
								min_vm_pu=row[COL_NAME_MIN_VOLTAGE])
			self.nodes.append(Bus(ID=row[COL_NAME_INDEX],
									geodata = GeoData(row[COL_NAME_LATITUDE],row[COL_NAME_LONGITUDE]),
									fragilityCurve=row[COL_NAME_FRAGILITY_CURVE],
									kf=row[COL_NAME_KF],
									resilienceFull=row[COL_NAME_RESILIENCE_FULL],
									weatherTTR=row[COL_NAME_WEATHER_TTR]))
			bus_ids[row[COL_NAME_NAME]] = pp.create_bus(**{key: value for key, value in kwargs_bus.items() if value is not None})
		# Creating the transformer elements
		df_tr = pd.read_excel(f'{filename}', sheet_name=SHEET_NAME_TRANSFORMER)
		df_tr = df_tr.where(pd.notnull(df_tr), None)
		df_tr_type = pd.read_excel(f'{filename}', sheet_name=SHEET_NAME_TR_TYPE)
		df_tr_type = df_tr_type.where(pd.notnull(df_tr_type), None)
		for index, row in df_tr.iterrows():
			tr_type = df_tr_type.loc[df_tr_type[COL_NAME_NAME] == row[COL_NAME_TYPE]]
			kwargs_tr = dict(net=network, hv_bus=bus_ids[row[COL_NAME_NODE_P]],
								lv_bus=bus_ids[row[COL_NAME_NODE_S]],
								sn_mva=tr_type[COL_NAME_REF_POWER].values[0],
								vn_hv_kv=tr_type[COL_NAME_VN_HV].values[0],
								vn_lv_kv=tr_type[COL_NAME_VN_LV].values[0],
								vkr_percent=tr_type[COL_NAME_VKR].values[0],
								vk_percent=tr_type[COL_NAME_VK].values[0],
								pfe_kw=tr_type[COL_NAME_PFE].values[0],
								i0_percent=tr_type[COL_NAME_I0].values[0],
								shift_degree=tr_type[COL_NAME_SHIFT].values[0],
								tap_side=tr_type[COL_NAME_TAP_SIDE].values[0],
								tap_neutral=tr_type[COL_NAME_TAP_NEUTRAL].values[0],
								tap_max=tr_type[COL_NAME_TAP_MAX].values[0],
								tap_min=tr_type[COL_NAME_TAP_MIN].values[0],
								tap_step_percent=tr_type[COL_NAME_TAP_STEP].values[0],
								tap_step_degree=tr_type[COL_NAME_TAP_STEP_ANGLE].values[0],
								tap_pos=row[COL_NAME_TAP_POS],
								tap_phase_shifter=tr_type[COL_NAME_TAP_PHASE_SHIFTER].values[0],
								in_service=row[COL_NAME_SERVICE],
								name=row[COL_NAME_NAME],
								vector_group=row[COL_NAME_VECTOR_GROUP],
								max_loading_percent=row[COL_NAME_MAX_LOADING],
								parallel=row[COL_NAME_PARALLEL],
								df=row[COL_NAME_DF])
			pp.create_transformer_from_parameters(**{key: value for key, value in kwargs_tr.items() if value is not None})
		
		# Creating the line elements
		df_ln = pd.read_excel(f'{filename}', sheet_name=SHEET_NAME_LINES)
		df_ln = df_ln.where(pd.notnull(df_ln), None)
		df_ln_type = pd.read_excel(f'{filename}', sheet_name=SHEET_NAME_LN_TYPE)
		df_ln_type = df_ln_type.where(pd.notnull(df_ln_type), None) 
		for index, row in df_ln.iterrows():
			if (row[COL_NAME_FROM_LONGITUDE] is not None) and (row[COL_NAME_FROM_LATITUDE] is not None) and (row[COL_NAME_TO_LONGITUDE] is not None) and (row[COL_NAME_TO_LATITUDE] is not None):
				geodata = [[row[COL_NAME_FROM_LATITUDE], row[COL_NAME_FROM_LONGITUDE]],
							[row[COL_NAME_TO_LATITUDE], row[COL_NAME_TO_LONGITUDE]]]
			else:
				geodata = None
			
			ln_type = df_ln_type.loc[df_ln_type[COL_NAME_NAME] == row[COL_NAME_TYPE]]
			kwargs_ln = dict(net=network,
								from_bus=bus_ids[row[COL_NAME_FROM_BUS]],
								to_bus=bus_ids[row[COL_NAME_TO_BUS]],
								length_km=row[COL_NAME_LENGTH],
								r_ohm_per_km=ln_type[COL_NAME_R1].values[0],
								x_ohm_per_km=ln_type[COL_NAME_X1].values[0],
								c_nf_per_km=ln_type[COL_NAME_C1].values[0],
								max_i_ka=ln_type[COL_NAME_MAX_I].values[0], 
								name=row[COL_NAME_NAME],
								index=row[COL_NAME_INDEX],
								type=ln_type[COL_NAME_TYPE].values[0],
								geodata=geodata,
								in_service=row[COL_NAME_SERVICE],
								df=row[COL_NAME_DF],
								parallel=row[COL_NAME_PARALLEL],
								g_us_per_km=ln_type[COL_NAME_G1].values[0],
								max_loading_percent=row[COL_NAME_MAX_LOADING],
								r0_ohm_per_km=ln_type[COL_NAME_R0].values[0],
								x0_ohm_per_km=ln_type[COL_NAME_X0].values[0],
								c0_nf_per_km=ln_type[COL_NAME_C0].values[0],
								g0_us_per_km=ln_type[COL_NAME_G0].values[0])
			pp.create_line_from_parameters(**{key: value for key, value in kwargs_ln.items() if value is not None})
		
		# Creating the load elements
		df_load = pd.read_excel(f'{filename}', sheet_name=SHEET_NAME_LOADS)
		df_load = df_load.where(pd.notnull(df_load), None)  
		load_ids = {}
		for index, row in df_load.iterrows():
			kwargs_load = dict(net=network,
								bus=bus_ids[row[COL_NAME_BUS]],
								p_mw=row[COL_NAME_P],
								q_mvar=row[COL_NAME_Q],
								const_z_percent=row[COL_NAME_CONST_Z],
								const_i_percent=row[COL_NAME_CONST_I],
								sn_mva=row[COL_NAME_REF_POWER],
								name=row[COL_NAME_NAME],
								scaling=row[COL_NAME_SCALING],
								in_service=row[COL_NAME_SERVICE],
								type=row[COL_NAME_TYPE], 
								max_p_mw=row[COL_NAME_MAX_P], 
								min_p_mw=row[COL_NAME_MIN_P],
								max_q_mvar=row[COL_NAME_MAX_Q],
								min_q_mvar=row[COL_NAME_MIN_Q],
								controllable=row[COL_NAME_CONTROLLABLE])
			load_ids[row[COL_NAME_NAME]] = pp.create_load(**{key: value for key, value in kwargs_load.items() if value is not None})

		# Creating the external gen elements
		df_ex_gen = pd.read_excel(f'{filename}', sheet_name=SHEET_NAME_EXTERNAL_GEN)
		df_ex_gen = df_ex_gen.where(pd.notnull(df_ex_gen), None)
		ex_gen_ids = {}
		for index, row in df_ex_gen.iterrows():
			kwargs_ex_gen = dict(net=network,
									bus=bus_ids[row[COL_NAME_BUS]],
									vm_pu=row[COL_NAME_VM],
									va_degree=row[COL_NAME_VA],
									name=row[COL_NAME_NAME],
									in_service=row[COL_NAME_SERVICE],
									s_sc_max_mva=row[COL_NAME_MAX_S_SC],
									s_sc_min_mva=row[COL_NAME_MIN_S_SC],
									rx_max=row[COL_NAME_MAX_RX],
									rx_min=row[COL_NAME_MIN_RX],
									max_p_mw=row[COL_NAME_MAX_P],
									min_p_mw=row[COL_NAME_MIN_P],
									max_q_mvar=row[COL_NAME_MAX_Q],
									min_q_mvar=row[COL_NAME_MIN_Q],
									r0x0_max=row[COL_NAME_MAX_R0X0],
									x0x_max=row[COL_NAME_MAX_X0X],
									controllable=row[COL_NAME_CONTROLLABLE],
									slack_weight=row[COL_NAME_SLACK_WEIGHT])
			ex_gen_ids[row[COL_NAME_NAME]] = pp.create_ext_grid(**{key: value for key, value in kwargs_ex_gen.items() if value is not None})

		# Creating the generator elements
		df_gen = pd.read_excel(f'{filename}', sheet_name=SHEET_NAME_GENERATORS)
		df_gen = df_gen.where(pd.notnull(df_gen), None)
		gen_ids = {}
		for index, row in df_gen.iterrows():
			kwargs_gen = dict(net=network,
								bus=bus_ids[row[COL_NAME_BUS]],
								p_mw=row[COL_NAME_P],
								vm_pu=row[COL_NAME_VM],
								sn_mva=row[COL_NAME_REF_POWER],
								name=row[COL_NAME_NAME],
								max_q_mvar=row[COL_NAME_MAX_Q],
								min_q_mvar=row[COL_NAME_MIN_Q],
								min_p_mw=row[COL_NAME_MIN_P],
								max_p_mw=row[COL_NAME_MAX_P],
								min_vm_pu=row[COL_NAME_MIN_VOLTAGE],
								max_vm_pu=row[COL_NAME_MAX_VOLTAGE],
								scaling=row[COL_NAME_SCALING],
								type=row[COL_NAME_TYPE], 
								lack=row[COL_NAME_LACK],
								controllable=row[COL_NAME_CONTROLLABLE],
								vn_kv=row[COL_NAME_VOLTAGE],
								xdss_pu=row[COL_NAME_XDSS],
								rdss_ohm=row[COL_NAME_RDSS],
								cos_phi=row[COL_NAME_COS_PHI],
								pg_percent=row[COL_NAME_PG],
								power_station_trafo=row[COL_NAME_PS_TRAFO],
								in_service=row[COL_NAME_SERVICE],
								slack_weight=row[COL_NAME_SLACK_WEIGHT])
			gen_ids[row[COL_NAME_NAME]] = pp.create_gen(**{key: value for key, value in kwargs_gen.items() if value is not None})
		
		# Creating the cost function
		df_cost = pd.read_excel(f'{filename}', sheet_name=SHEET_NAME_COST)
		df_cost = df_cost.where(pd.notnull(df_cost), None)
		for index, row in df_cost.iterrows():
			if row[COL_NAME_TYPE] == 'load':
				element=load_ids[row[COL_NAME_NAME]]
			elif row[COL_NAME_TYPE] == 'ext_grid':
				element=ex_gen_ids[row[COL_NAME_NAME]]
			elif row[COL_NAME_TYPE] == 'gen':
				element=gen_ids[row[COL_NAME_NAME]]
			kwargs_cost = dict(net=network,
								element=element,
								et=row[COL_NAME_TYPE],
								cp1_eur_per_mw=row[COL_NAME_CP1], 
								cp0_eur=row[COL_NAME_CP0], 
								cq1_eur_per_mvar=row[COL_NAME_CQ1], 
								cq0_eur=row[COL_NAME_CQ0], 
								cp2_eur_per_mw2=row[COL_NAME_CP2], 
								cq2_eur_per_mvar2=row[COL_NAME_CQ2])
			pp.create_poly_cost(**{key: value for key, value in kwargs_cost.items() if value is not None})
		return network

	def get_failure_candidates_backup(self):
		out = []
		for x in [x for x in self.__dict__.values() if type(x) is list]:
			out+=[y for y in x if hasattr(y,'failureProb')]
		return out

	def get_failure_candidates(self):
		keys = []
		values = []
		for x in [x for x in self.__dict__.values() if type(x) is list]:
			values+= [y for y in x if hasattr(y,'failureProb')]
			keys+= [y.ID for y in x if hasattr(y,'failureProb')]
		return dict(zip(keys,values))

	def get_closest_available_crews(self, availableCrew, powerElements):
		aux = len(powerElements)
		return powerElements[0:aux], availableCrew[0:aux]

	def get_crews_traveling_time(self, crew, powerElements):
		time = np.random.randint(10)
		return [time]*len(crew)

	def get_reparing_time(self, powerElementsID, powerElements):
		return [powerElements[x].normalTTR for x in powerElementsID] #TODO: correct formula for repairing time

	def calculate_outages_schedule(self, simulationTime, hazardTime):
		'''
		Add description of create_outages function
		'''
		failureCandidates = self.get_failure_candidates()
		# crews = self.crews
		failureProbability = np.array(
			[x.failureProb for x in failureCandidates.values()])
		randomNumber = 1
		while (randomNumber > failureProbability).all():  # random until a failure happens
			randomNumber = np.random.rand()
		failure = np.where((randomNumber <= failureProbability), np.random.randint(
			hazardTime.start, [hazardTime.stop]*len(failureCandidates)), simulationTime.stop)
		crewSchedule = pd.DataFrame([[1]*len(self.crews)]*simulationTime.duration, columns=[
									x.ID for x in self.crews], index=simulationTime.interval)
		outagesSchedule = pd.DataFrame([[STATUS['on']]*len(failureCandidates)] *
									   simulationTime.duration, columns=failureCandidates.keys(), index=simulationTime.interval)
		for index, column in zip(failure, outagesSchedule):
			# outagesSchedule[column].loc[(outagesSchedule.index >= failure[i])] = STATUS['off']
			outagesSchedule[column].loc[index:] = STATUS['off']

		for index, row in outagesSchedule.iterrows():
			failureElements = row.index[row == 0].tolist()
			availableCrews = crewSchedule.loc[index][crewSchedule.loc[index] == 1].index.tolist(
			)
			elementsToRepair, repairingCrews = self.get_closest_available_crews(
				availableCrews, failureElements)
			crewsTravelingTime = self.get_crews_traveling_time(
				repairingCrews, elementsToRepair)
			repairingTime = self.get_reparing_time(
				elementsToRepair, failureCandidates)

			for t_0, t_1, e, c in zip(crewsTravelingTime, repairingTime, elementsToRepair, repairingCrews):
				outagesSchedule.loc[index+1:index+t_0, e] = STATUS['waiting']
				outagesSchedule.loc[index+t_0+1:index +
									t_0+t_1, e] = STATUS['reparing']
				# TODO: this line can be removed if outagesSchedule is set to 1 on at failure time
				outagesSchedule.loc[index+t_0+t_1+1:, e] = STATUS['on']
				crewSchedule.loc[index+1:index+t_0+t_1, c] = e
			'''
			print(f'Time step: {index}')
			print(f'Failure elements: {failureElements}')
			print(f'Avaiable crews: {availableCrews}')
			print(f'Reparing crews: {repairingCrews}')
			print(f'Crews traveling time: {crewsTravelingTime}')
			print(f'Elements reparing time {repairingTime}')
			print(outagesSchedule.join(crewSchedule))
			if (len(elementsToRepair)>0):
				breakpoint()			
			'''
			if not outagesSchedule.loc[index+1:].isin([STATUS['on']]).any().any():
				# print(failureCandidates.keys())
				# print(failure)
				# print(f'Finished at time step: {index}')
				# print(outagesSchedule.join(crewSchedule))
				self.outagesSchedule =  outagesSchedule
				self.crewSchedule = crewSchedule
				return
		self.outagesSchedule  = outagesSchedule
		self.crewSchedule = crewSchedule
	
	def calculate_metrics(self):
		self.metrics = []
		def elements_in_service():
			idToClassMap = {id:type(f'{self.get_failure_candidates()[id]} in service').__name__ for id in self.outagesSchedule}
			out = self.outagesSchedule.rename(idToClassMap, axis = 1).groupby(level = 0, axis = 1).sum()
			return out
		def crews_in_service():
			return (self.crewSchedule!=1).sum(axis = 1)

		self.metrics.append(Metric('Network', 'crews_in_service',crews_in_service()))
		self.metrics.append(Metric('Network', 'elements_in_service',elements_in_service()))
		self.metrics.append(Metric('Network', 'total_elements_in_service', elements_in_service().sum().values, subfield = 'subfield_1', unit = 'unit_1'))


	def build_metrics_database(self):
		#TODO: make it recurrent
			out = []
			for metric in self.metrics:
				keys, values = zip(*[(key, value) for key, value in metric.__dict__.items() if key is not 'value'])
				if isinstance(metric.value, pd.DataFrame) or isinstance(metric.value, pd.Series):
					value = metric.value.values
					index = metric.value.index
				else:
					value = metric.value
					index = [0]
				df = pd.DataFrame(value, columns = pd.MultiIndex.from_tuples([values], names=keys), index = index)
				out.append(df)
			return pd.concat(out, axis = 1)
	

class Metric:
	def __init__(self, network_element, field, value, subfield = None, unit = None):
		self.network_element = network_element
		self.field = field
		self.value = value
		self.subfield = subfield
		self.unit = unit

class History:
	'''
	Add description of History class here
	'''
	def __init__(self, label):
		self.label = label
		self.ENS = []
		self.lineLoading = []
		self.minPower = 0
		self.loadPower = 0
		self.lineOutages = []
		self.transformerOutages = []
		self.busOutages = []
		self.generatorOutages = []
		self.LOEF = 0
		self.totENS = 0

	def plot(self):
		'''
		Add description of plot function
		'''
		pass

	def export(self):
		'''
		Add description of export function
		'''
		pass

	def print(self):
		'''
		Add description of print function
		'''
		pass

class FragilityCurve:
	'''
	Builds the Fragility curve element from a csv input file.
	The name of the csv file will be used as the name
	of the fragility curve element
	'''
	def __init__(self, filename):
		self.fc = self.readFromCsv(filename)
		self.name = filename.split('.')[0]

	def readFromCsv(self, filename):
		'''
		Takes a csv file name as input and outputs a list of lists
		with the first column being the intensity and the
		second columns being the probablity
		'''
		fc = []
		with open(filename, "r") as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			for row in csv_reader:
				 fc.append([float(i) for i in row])
		return list(zip(*fc))

	def plot(self):
		'''
		Returns the figure and axis elements from 
		matplotlib of the fragility curve. Use plot.show()
		to see the graph.
		'''
		fig, ax = plt.subplots(tight_layout=True)
		plt.plot(self.fc[0],self.fc[1])
		ax.set_xlabel('intensity')
		ax.set_ylabel('probability')
		ax.set_title(self.name)
		return fig, ax

class Hazard:
	'''
	Add description of Hazard class
	'''
	def __init__(self, name, filename):
		self.name = filename.split('.')[0]
		self.filename = filename
		self.attributes, self.lon, self.lat, self.time = self.read_attributes()

	def read_attributes(self):
		'''
		Takes as input the name of the ncdf file (with the .nc extention)
		Returns the attributes in the ncdf file
		'''
		ncdf = nc.Dataset(self.filename, mode='r')
		cols = []
		for att in ncdf.variables:
			if att == 'lon':
				lon = ncdf[att][:]
			elif att == 'lat':
				lat = ncdf[att][:]
			elif att == 'time':
				time_tmp = ncdf.variables[att]
				time = nc.num2date(time_tmp[:],time_tmp.units, only_use_cftime_datetimes=False)
				time = pd.to_datetime(time)
			else:
				cols.append(att)
		ncdf.close()
		return cols, lon, lat, time

	def get_attributes(self):
		return self.attributes

	def get_lon(self):
		return self.lon

	def get_lat(self):
		return self.lat

	def get_time(self):
		return self.time

	def read_attribute(self, attribute, lon, lat, startTime, endTime):
		'''
		Takes as input the attribute name. All avaialble attributes can be listed using the get_attributes() method.
		The coordinates lon(longitude) and lat(lattitude) must be provided as input. 
		The nearest longitude at lattitude values available in the ncdf file will be used
		A time window must also be provided (startTime and endTime) this must be a datime object from pandas.
		use the pd.to_datetime() method to convert any other time object.
		Returns the datetime and values for the requested attribute at a given longitude and lattitude.
		'''
		idx_startTime = self.time.get_loc(startTime, method='nearest')
		idx_endTime = self.time.get_loc(endTime, method='nearest')
		
		lon_approx_idx = min(range(len(self.lon)), key=lambda i: abs(self.lon[i]-lon))
		lat_approx_idx = min(range(len(self.lat)), key=lambda i: abs(self.lat[i]-lat))

		ncdf = nc.Dataset(self.filename, mode='r')
		tmp = ncdf[attribute][:]
		att = tmp[idx_startTime:idx_endTime,lat_approx_idx,lon_approx_idx]
		ncdf.close()

		return self.time[idx_startTime:idx_endTime], att

	def plot(self, attribute, time):
		'''
		Add description of plot function
		'''
		mp = Basemap(llcrnrlon= min(self.lon),   # lower longitude
			 llcrnrlat=min(self.lat),    # lower latitude
			 urcrnrlon=max(self.lon),   # uppper longitude
			 urcrnrlat=max(self.lat))

		lons, lats = np.meshgrid(self.lon,self.lat) 
		x,y = mp(lons, lats)
		fig = plt.figure(figsize=(6,8))
		#c_scheme = mp.pcolor(x,y,np.squeeze(pr[0,:,:]),cmap = 'jet') # [0,:,:] is for the first day of the year
		mp.bluemarble()
		return fig



class GeoData:
	'''
	Add description of GeoData class
	'''
	def __init__(self, latitude, longitude):
		self.latitude = latitude
		self.longitude = longitude

class PowerElement:
	'''
	Add description of PowerElement class here
	'''
	def __init__(self,
				 ID,
				 fragilityCurve,
				 kf,
				 resilienceFull,
				 geodata = GeoData(0,0),
				 failureProb=0,
				 inReparation=False,
				 normalTTR=0,
				 distTTR=0,
				 inService=True,
				 kw=4):
		if ID is None:
			self.ID = get_GLOBAL_ID()
		else:
			self.ID = ID
		self.geodata = geodata
		# self.lat = lat
		# self.lon = lon
		self.fragilityCurve = fragilityCurve
		self.kf = kf
		self.resilienceFull = resilienceFull
		self.failureProb = failureProb
		self.inReparation = inReparation
		self.normalTTR = normalTTR
		self.distTTR = distTTR
		self.inService = inService
		self.kw = kw

	def fail(self):
		'''
		Add description of fail class
		'''
		pass

	def startRepair(self):
		'''
		Add description of startRepair class
		'''
		pass

	def calculateWeatherTTR(self):
		'''
		Add description of calculateWeatherTTR class
		'''
		pass

class Bus(PowerElement):
	'''
	Add description of Bus class here
	'''   
	def __init__(self,
				 ID,
				 fragilityCurve,
				 kf,
				 resilienceFull,
				 weatherTTR,
				 geodata = None,
				 failureProb=0,
				 inReparation=False,
				 normalTTR=0,
				 distTTR=0,
				 inService=True,
				 kw=4):
		super().__init__(ID, fragilityCurve, kf, resilienceFull, geodata, failureProb, inReparation, normalTTR, distTTR, inService, kw)
		self.weatherTTR = weatherTTR
		self.elapsedReparationTime = 0

class Generator(PowerElement):
	'''
	Add description of Generator class here
	'''   
	def __init__(self,
				 ID,
				 fragilityCurve,
				 kf,
				 resilienceFull,
				 weatherTTR,
				 elapsedReparationTime,
				 geodata = None,
				 failureProb = 0,
				 inReparation=False,
				 normalTTR=0,
				 distTTR=0,
				 inService=True,
				 kw=4):
		super().__init__(ID, fragilityCurve, kf, resilienceFull, geodata, failureProb, inReparation, normalTTR, distTTR, inService, kw)
		self.weatherTTR = weatherTTR
		self.elapsedReparationTime = elapsedReparationTime

class Transformer(PowerElement):
	'''
	Add description of Transformer class here
	'''   
	def __init__(self,
				 ID,
				 fragilityCurve,
				 kf,
				 resilienceFull,
				 weatherTTR,
				 elapsedReparationTime,
				 geodata = None,
				 failureProb = 0,
				 inReparation=False,
				 normalTTR=0,
				 distTTR=0,
				 inService=True,
				 kw=4):
		super().__init__(ID, fragilityCurve, kf, resilienceFull, geodata, failureProb, inReparation, normalTTR, distTTR, inService, kw)
		self.weatherTTR = weatherTTR
		self.elapsedReparationTime = elapsedReparationTime

class Line(PowerElement):
	'''
	Add description of Line class here
	'''   
	def __init__(self,
				 ID,
				 fragilityCurve,
				 kf,
				 resilienceFull,
				 lineSpan,
				 towers,
				 length = 1,
				 span = 0.3,
				 geodata = None,
				 failureProb = 0,
				 inReparation=False,
				 normalTTR=0,
				 distTTR=0,
				 inService=True,
				 kw=4):
		super().__init__(ID, fragilityCurve, kf, resilienceFull, geodata, failureProb, inReparation, normalTTR, distTTR, inService, kw)
		self.length = length
		self.span = span
		self.lineSpan = lineSpan
		self.towers = towers

class Crew:
	'''
	Add description of Crew class here
	'''

	def __init__(self, ID, geodata=None):
		self.ID = ID
		# self.available = number
		if geodata is None:
			self.geodata = GeoData(0, 0)
		else:
			self.geodata = geodata
	# def get_crew_waiting_time():
		# return 0