import resiliencyTool as rt
import pandas as pd
# import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pandapower.plotting.plotly import simple_plotly, vlevel_plotly, pf_res_plotly



def filter_non_converged_iterations(df):
    df = df.copy()
    i_dropped = []
    for iteration in df.index.get_level_values(level = 'iteration').drop_duplicates():
        breakpoint()
        if (df.loc[iteration] == 0).all(axis = 0).any():
            df.drop(index = iteration, level = 'iteration', inplace = True)
            i_dropped.append(iteration)
    print(f'iterations droped : {i_dropped}')
    return df

def filter_non_converged_timesteps(df):
    df = df.copy()
    for iteration in df.index.get_level_values(level = 'iteration').drop_duplicates():
        breakpoint()
        filter =  (df.loc[iteration] == 0).all(axis = 0)
        time_stamps = filter[filter].index
        df.loc[iteration][time_stamps] = np.nan
    return 
simulationName = 'example_1';
df = pd.read_csv(rt.config.path.engineDatabaseFile(simulationName), index_col = [0, 1, 2, 3])
filter_non_converged_timesteps(df)
breakpoint()