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


# def get_elements_with_field(object, field):
# 	# DEPRECATED
# 	if isinstance(object, list):
# 		# commented code can be replaced by itertools.chain.from_iterable
# 		# out = []
# 		# for x in object:
# 		# 	out += get_element_with_field_(x, field)
# 		# return out
# 		return list(itertools.chain(*[get_element_with_field(x, field) for x in object])) 	
# 	else:
# 		if hasattr(object, "__dict__"):
# 			return [x for x in [object] if hasattr(x, field)] + get_element_with_field([x for x in vars(object).values() if isinstance(x, list)], field)
# 			# out = []		
# 			# if hasattr(object, field):
# 			# 	out += [object]
# 			# return out + get_element_with_field([x for x in vars(object).values() if isinstance(x, list)], field)
# 		else:
# 			return []



