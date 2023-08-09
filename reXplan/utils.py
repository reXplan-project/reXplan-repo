import pandas as pd
import itertools
from . import config

GLOBAL_ID = -1
COL_NAME_DATA_TYPE = "data type"
COL_NAME_INTERNAL_FIELD = "internal field"
COL_NAME_INPUT_FILE_FIELD = "input file field"

INPUT_FIELD_MAP =  pd.read_csv(config.path.inputFieldsMapFile, index_col = COL_NAME_INPUT_FILE_FIELD)

def get_GLOBAL_ID():
	global GLOBAL_ID
	GLOBAL_ID += 1
	return 'ID_{}'.format(GLOBAL_ID)

def format_data_type(df):
	# DEPRECATED
	# Acts on columns (need to tranpose df for indices)
	df_dataTypes = INPUT_FIELD_MAP[COL_NAME_DATA_TYPE].dropna()
	return df.astype(df_dataTypes[df_dataTypes.index.intersection(df.columns)].to_dict())

def df_to_internal_fields(df):
	return df.rename(columns=INPUT_FIELD_MAP[COL_NAME_INTERNAL_FIELD].to_dict())