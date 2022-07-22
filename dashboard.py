import dash
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import resiliencyTool as rt
from dash import dcc, html
from plotly.subplots import make_subplots

TIMEINDEX = 'timestep'
DATA_TYPES = ['raw', 'quantile']
COLOR = dict(zip(DATA_TYPES, ['iteration', 'quantile']))

def index_to_columns_tranposed(df):
    return df.unstack(level = list(set(df.index.names)-{'field'})).T.reset_index()


def get_timestep_ocurrences(df, field, value):
    return ((df.loc[:,:,field] == value).replace({False:np.nan})*df.columns).values.flatten()

external_stylesheets = [dbc.themes.DARKLY]
app = dash.Dash(__name__, title='Interactive Model Dashboard', external_stylesheets=[external_stylesheets])

simulationName = 'Simulation_1'

df = pd.read_csv(rt.config.path.globalDatabaseFile(simulationName), index_col = [0,1,2,3,4])
df.columns = df.columns.astype(int)
df = df.dropna() # get rid of scalar values
df.columns.name = TIMEINDEX #TODO: initialize this while before saving the dataframe ?
# units = dict(df.index.droplevel(list(set(df.index.names)-{'field','unit'})).drop_duplicates())
df.index = df.index.droplevel(level = 'unit')
fields_y = set(df.index.get_level_values(level = 'field'))


df_raw = index_to_columns_tranposed(df)

# Quantiles
df_quantiles = df.groupby('field').quantile(q = [0.05,0.5,0.95])
df_quantiles.index.names = ['field',DATA_TYPES[1]]
df_quantiles = index_to_columns_tranposed(df_quantiles)


# value = 12
# field = 'elements_in_service'

# get_timestep_ocurrences(df, field, value)
# breakpoint()

# converting for plotting
# df_raw = index_to_columns_tranposed(df)
# FIELDS_X = set(df_.columns) - fields_y 

# features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicatessen']


app.layout = html.Div([
    html.Div([

        html.Div([

            html.Div([
                html.Label('Model selection'),], style={'font-size': '18px'}),

            dcc.Dropdown(
                id='crossfilter-fields_y',
                options = [{'label':x, 'value':x} for x in fields_y],
                value=min(fields_y),
                clearable=False

            )], style={'width': '49%', 'display': 'inline-block'}
        ),

        html.Div([

            html.Div([
                html.Label('Feature selection'),], style={'font-size': '18px', 'width': '40%', 'display': 'inline-block'}),

            # html.Div([
            #     dcc.RadioItems(
            #         id='gradient-scheme',
            #         options=[
            #             {'label': 'Orange to Red', 'value': 'OrRd'},
            #             {'label': 'Viridis', 'value': 'Viridis'},
            #             {'label': 'Plasma', 'value': 'Plasma'}
            #         ],
            #         value='Plasma',
            #         labelStyle={'float': 'right', 'display': 'inline-block', 'margin-right': 10}
            #     ),
            # ], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),

            dcc.Dropdown(
                id='crossfilter-data_types',
                options = [{'label':x, 'value':x} for x in DATA_TYPES],
                value=min(DATA_TYPES),
                clearable=False
            )], style={'width': '49%', 'float': 'right', 'display': 'inline-block'}

        )], style={'backgroundColor': 'rgb(17, 17, 17)', 'padding': '10px 5px'}
    ),

    html.Div([

        dcc.Graph(
            id='upper_graph',
            hoverData={'points': [{'x': 0, 'y':0}]}
        )

    ], style={'width': '100%', 'height':'80%', 'display': 'inline-block', 'padding': '0 20'}),

    html.Div([
        dcc.Graph(id='histogram_graph_left')
    ], style={'display': 'inline-block', 'width': '49%'}),

    html.Div([
        dcc.Graph(id='histogram_graph_right')
    ], style={'display': 'inline-block', 'width': '49%'}),

    ], style={'backgroundColor': 'rgb(17, 17, 17)'},

)


@app.callback(
    dash.dependencies.Output('upper_graph', 'figure'),
    dash.dependencies.Input('crossfilter-fields_y', 'value'),
    dash.dependencies.Input('crossfilter-data_types', 'value'),
    # dash.dependencies.Input('gradient-scheme', 'value')
    
)
def update_graph(field_y, data_type):
    if data_type == DATA_TYPES[0]:
        df = df_raw
    elif data_type == DATA_TYPES[1]:
        df = df_quantiles
    #TODO: color in term of other parameters
    # if feature == 'None':
    #     cols = None
    #     sizes = None
    #     hover_names = [f'Customer {ix:03d}' for ix in df.index.values]
    # elif feature in ['Region', 'Channel']:
    #     cols = df[feature].astype(str)
    #     sizes = None
    #     hover_names = [f'Customer {ix:03d}' for ix in df.index.values]
    # else:
    #     cols = df[feature].values #df[feature].values
    #     sizes = [np.max([max_val/10, x]) for x in df[feature].values]
    #     hover_names = []
    #     for ix, val in zip(df.index.values, df[feature].values):
    #         hover_names.append(f'Customer {ix:03d}<br>{feature} value of {val}')
    fig = px.line(
        df,
        x = TIMEINDEX, 
        y = field_y,
        color = COLOR[data_type],
        markers = True,
        # size=sizes,
        # opacity=0.8,
        # hover_name=hover_names,
        # hover_data=features,
        template='plotly_dark',
        # color_continuous_scale=gradient,
    )

    fig.update_traces(customdata=df.index)

    fig.update_layout(
        # coloraxis_colorbar={'title': f'{feature}'},
        # coloraxis_showscale=False,
        # legend_title_text=COLOR[data_type],
        # height=650, 
        # margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
        legend=dict(yanchor="top", y=1., xanchor="right", x=1),
        hovermode='closest',
        template='plotly_dark'
    )

    fig.update_xaxes(showticklabels=True)
    fig.update_yaxes(showticklabels=True)

    return fig


# def create_histogram(field, timestep):

#     fig = px.histogram(df.loc[:,:,field][timestep], nbins=20)

#     fig.update_layout(
#         # barmode='group',
#         height=225,
#         margin={'l': 20, 'b': 30, 'r': 10, 't': 10},
#         template='plotly_dark'
#     )

#     fig.update_xaxes(showgrid=True)
#     fig.update_yaxes(showgrid=True)
#     # fig.update_yaxes(type="log", range=[0,5])

#     return fig

# def create_cumulative_histogram(field, timestep):

#     fig = px.histogram(df.loc[:,:,field][timestep], nbins=20)

#     fig.update_layout(
#         # barmode='group',
#         height=225,
#         margin={'l': 20, 'b': 30, 'r': 10, 't': 10},
#         template='plotly_dark'
#     )

#     fig.update_xaxes(showgrid=True)
#     fig.update_yaxes(showgrid=True)
#     # fig.update_yaxes(type="log", range=[0,5])

#     return fig

@app.callback(
    dash.dependencies.Output('histogram_graph_left', 'figure'),
    # dash.dependencies.Output('histogram_graph_right', 'figure'),
    # [
    dash.dependencies.Input('crossfilter-fields_y', 'value'),
    dash.dependencies.Input('upper_graph', 'hoverData')
    # ]
)

def update_graph(field, hoverData):
    timestep = hoverData['points'][0]['x']
    range_x = [df.loc[:,:,field].to_numpy().min(), df.loc[:,:,field].to_numpy().max()+.5]
    x = df.loc[:,:,field][timestep]
    legend_title=f"timestep = {timestep}"
    x_title = field
    return update_histogram(field, legend_title, x_title, range_x, x)

def update_histogram(field, legend_title, x_title,  range_x, x):
    # timestep = hoverData['points'][0]['customdata']
    # range_x = [df.loc[:,:,field].to_numpy().min(), df.loc[:,:,field].to_numpy().max()+.5]
    # df_ = df.loc[:,:,field][timestep]
    
    # fig = go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Histogram(x = x, histnorm='probability', name = 'histogram'))
    fig.add_trace(go.Histogram(x = x, histnorm='probability', cumulative_enabled = True, name = 'cumulated histogram'), secondary_y = True)
    fig.update_traces(opacity=0.6)
    fig.update_layout(
        # legend_title= legend_title,
        barmode='overlay',
        # height=225,
        # margin={'l': 20, 'b': 30, 'r': 10, 't': 30},
        # xaxis_title_text='Value',
        legend=dict(title = legend_title, orientation="h", yanchor="top", y=1.1, xanchor="left", x=0),
        template='plotly_dark'
    )
    fig.update_xaxes(showgrid=True, range = range_x, title_text = x_title, )
    fig.update_yaxes(showgrid=True)
    # fig.update_yaxes(type="log", range=[0,5])
    return fig


@app.callback(
    dash.dependencies.Output('histogram_graph_right', 'figure'),
    dash.dependencies.Input('crossfilter-fields_y', 'value'),
    dash.dependencies.Input('upper_graph', 'hoverData')
)

def update_graph(field, hoverData):
    value = hoverData['points'][0]['y']
    range_x = [df.columns.min(), df.columns.max()+.5]
    x = get_timestep_ocurrences(df, field, value)
    legend_title=f"{field} = {value}"
    x_title = 'timestep'
    return update_histogram(field, legend_title, x_title, range_x, x)


if __name__ == '__main__':
    app.run_server(debug=True)
