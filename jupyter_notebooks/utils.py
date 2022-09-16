import pandas as pd

def invert(df):
    aux_ = list(set(df.index.names)-{'field'})
    return df.unstack(level = aux_).T.reset_index(level = aux_)
    
def filter(df, **kwargs):
    filter = True
    for k,v in kwargs.items():
        filter = filter & (df.index.get_level_values(level = k) == v)
    return df.loc[filter]

def group_by(df, operation = 'sum', *args):
    if operation == 'sum':
        return df.groupby(level = args).sum()
    elif operation == 'mean':
        return df.groupby(level = args).mean()
    else:
        return df
    
def get_quantiles_on_iterations(df, q_list):
    group = list(set(df.index.names) & set(['field','type', 'id']))
    out = df.groupby(group).quantile(q = q_list)
    out.index.names = group + ['quantile']
    return out
                                           
def enrich_database(df):
    def format_to_multiindex(df, keys, names):
        return pd.concat({tuple(keys): df}, names=names).reorder_levels(['iteration', 'field','type', 'id'])
    
    def total_over_id():
        type = 'network'
        id = 'network'
        content = df.groupby(level = ['iteration','field', 'type']).sum()
        field_index = 'total_' + content.index.get_level_values('type') + '_'+content.index.get_level_values('field')
        iteration_index = content.index.get_level_values('iteration')
        content.index = pd.MultiIndex.from_tuples(list(zip(*[iteration_index, field_index])), names=["iteration", "field"])            
        # content = (df.loc[:, 'max_p_mw','load',:] - df.loc[:, 'p_mw','load',:]).groupby('iteration').sum()
        return pd.concat({(type, id): content}, names=['type', 'id']).reorder_levels(['iteration', 'field', 'type', 'id'])
    
    concat = [df, total_over_id()]
    return pd.concat(concat).sort_index(level = ['iteration', 'type'])

def filter_non_converged_iterations(df):
    df = df.copy()
    i_dropped = []
    for iteration in df.index.get_level_values(level = 'iteration').drop_duplicates():
        if (df.loc[iteration] == 0).all(axis = 0).any():
            df.drop(index = iteration, level = 'iteration', inplace = True)
            i_dropped.append(iteration)
    print(f'iterations droped : {i_dropped}')
    return df
