import pandas as pd
from . import config

GLOBAL_ID = -1
COL_NAME_DATA_TYPE = "data type"
COL_NAME_INTERNAL_FIELD = "internal field"
COL_NAME_INPUT_FILE_FIELD = "input file field"
COL_NAME_PANDAPOWER_FIELD = "pandapower field"

INPUT_FIELD_MAP =  pd.read_csv(config.path.inputFieldsMapFile, index_col = COL_NAME_INPUT_FILE_FIELD)

def get_GLOBAL_ID():
	global GLOBAL_ID
	GLOBAL_ID += 1
	return 'ID_{}'.format(GLOBAL_ID)

def df_to_internal_fields(df):
	return df.rename(columns=INPUT_FIELD_MAP[COL_NAME_INTERNAL_FIELD].to_dict())

def df_to_pandapower_object(df):
	return df.rename(columns=INPUT_FIELD_MAP[COL_NAME_PANDAPOWER_FIELD].to_dict())