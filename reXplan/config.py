import os
import pandas as pd

GLOBAL_ID = -1
COL_NAME_DATA_TYPE = "data type"
COL_NAME_INTERNAL_FIELD = "internal field"
COL_NAME_INPUT_FILE_FIELD = "input file field"
COL_NAME_PANDAPOWER_FIELD = "pandapower field"

COL_NAME_NAME = 'name'
COL_NAME_FIELD = 'field'
COL_NAME_VALUE = 'value'
COL_NAME_UNIT = 'unit'

TOOL_FOLDER = os.path.abspath(os.path.dirname(__file__ )) # absolute file path of config.py
INPUT_FIELDS_MAP_FILE = os.path.join(TOOL_FOLDER, 'fields_map.csv')
INPUT_SHEETS_MAP_FILE = os.path.join(TOOL_FOLDER, 'sheets_map.csv')

class Path():
    def __init__(self, workspace = 'Examples'):
        self.workspace = workspace
        self.inputFolder = os.path.join(self.workspace,'input')
        self.outputFolder = os.path.join(self.workspace,'output')
        
        self.network = 'network.xlsx'
        self.metricDatabase = 'metric_database.csv'
        self.montecarloDatabase = 'montecarlo_database.csv'
        self.engineDatabase = 'engine_database.csv'
        self.fcDatabase = 'fragilityCurves'
        self.hazardFolder = 'hazards'
        self.hazardGifFolder = 'gif'
        self.rpDatabase = 'returnPeriods'

    def SetWorkspaceFolder(self, workspace):
        self.workspace = workspace
        self.inputFolder = os.path.join(workspace,'input')
        self.outputFolder = os.path.join(workspace,'output')
        return True
    
    def networkFile(self, simulationName):
        return os.path.join(self.inputFolder, simulationName, self.network)
    def fragilityCurveFolder(self, simulationName):
        return os.path.join(self.inputFolder, simulationName, self.fcDatabase)
    def returnPeriodFolder(self, simulationName):
        return os.path.join(self.inputFolder, simulationName, self.rpDatabase)
    def hazardFile(self, simulationName, filename):
        return os.path.join(self.inputFolder, simulationName, self.hazardFolder, filename)
    def hazardGifFile(self, simulationName, filename):
        checkPath(os.path.join(self.inputFolder, simulationName, self.hazardFolder, self.hazardGifFolder))
        return os.path.join(self.inputFolder, simulationName, self.hazardFolder, self.hazardGifFolder, filename)
    def outputFolderPath(self, simulationName):
        checkPath(os.path.join(self.outputFolder, simulationName))
        return os.path.join(self.outputFolder, simulationName) 
    def metricDatabaseFile(self, simulationName):
        checkPath(os.path.join(self.outputFolder, simulationName))
        return os.path.join(self.outputFolder, simulationName, self.metricDatabase)
    def engineDatabaseFile(self, simulationName):
        checkPath(os.path.join(self.outputFolder, simulationName))
        return os.path.join(self.outputFolder, simulationName, self.engineDatabase)
    def montecarloDatabaseFile(self, simulationName):
        checkPath(os.path.join(self.outputFolder, simulationName))
        return os.path.join(self.outputFolder, simulationName, self.montecarloDatabase)

INPUT_FIELD_MAP =  pd.read_csv(INPUT_FIELDS_MAP_FILE, index_col = COL_NAME_INPUT_FILE_FIELD)
INPUT_SHEET_MAP =  pd.read_csv(INPUT_SHEETS_MAP_FILE, index_col = COL_NAME_INPUT_FILE_FIELD)

path = Path('Examples')

def get_GLOBAL_ID():
	global GLOBAL_ID
	GLOBAL_ID += 1
	return 'ID_{}'.format(GLOBAL_ID)

def df_to_internal_fields(df):
	return df.rename(columns=INPUT_FIELD_MAP[COL_NAME_INTERNAL_FIELD].to_dict())

def df_to_pandapower_object(df):
	return df.rename(columns=INPUT_FIELD_MAP[COL_NAME_PANDAPOWER_FIELD].to_dict())

def get_pandapower_object(field_name):
	return INPUT_FIELD_MAP[(INPUT_FIELD_MAP[COL_NAME_INTERNAL_FIELD] == field_name)][COL_NAME_PANDAPOWER_FIELD].values[0]

def get_input_sheetname(sheetname):
	return INPUT_SHEET_MAP[(INPUT_SHEET_MAP[COL_NAME_INTERNAL_FIELD] == sheetname)].index[0]

def get_sheetname_from_input_to_pandapower(sheetname):
	return INPUT_SHEET_MAP[(INPUT_SHEET_MAP.index == sheetname)][COL_NAME_PANDAPOWER_FIELD].values[0]

def get_pandapower_sheetname(sheetname):
	return INPUT_SHEET_MAP[(INPUT_SHEET_MAP[COL_NAME_INTERNAL_FIELD] == sheetname)][COL_NAME_PANDAPOWER_FIELD].values[0]

def checkPath(path):
    if not os.path.exists(path):
        os.mkdir(path)