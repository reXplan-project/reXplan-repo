import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import resiliencyTool as rt
from dash import dcc, html

TIMEINDEX = 'timestep'
COLOR = 'iteration'

external_stylesheets = [dbc.themes.DARKLY]
app = dash.Dash(__name__, title='Interactive Model Dashboard', external_stylesheets=[external_stylesheets])

simulationName = 'Simulation_1'

df = pd.read_csv(rt.config.path.globalDatabaseFile(simulationName), header = [0,1,2,3,4], index_col = 0)
df = df.dropna(axis = 1) # get rid of scalar values
FIELDS_y = set(df.columns.get_level_values(level = 'field'))
df.index.name = TIMEINDEX #TODO: initialize this while before  saving the dataframe
df = df.stack(level = ['iteration', 'subfield', 'unit','network_element']).reset_index()
FIELDS_x = set(df.columns.get_level_values(level = 'field')) - FIELDS_y 

features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicatessen']

# models = ['PCA', 'UMAP', 'AE', 'VAE']
# df_average = df[features].mean()
# max_val = df.max().max()



app.layout = html.Div([
    html.Div([

        html.Div([

            html.Div([
                html.Label('Model selection'),], style={'font-size': '18px'}),

            dcc.Dropdown(
                id='crossfilter-field_y',
                options = [{'label':x, 'value':x} for x in FIELDS_y],
                value=min(FIELDS_y),
                clearable=False

            )], style={'width': '49%', 'display': 'inline-block'}
        ),

        html.Div([

            html.Div([
                html.Label('Feature selection'),], style={'font-size': '18px', 'width': '40%', 'display': 'inline-block'}),

            html.Div([
                dcc.RadioItems(
                    id='gradient-scheme',
                    options=[
                        {'label': 'Orange to Red', 'value': 'OrRd'},
                        {'label': 'Viridis', 'value': 'Viridis'},
                        {'label': 'Plasma', 'value': 'Plasma'}
                    ],
                    value='Plasma',
                    labelStyle={'float': 'right', 'display': 'inline-block', 'margin-right': 10}
                ),
            ], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),

            dcc.Dropdown(
                id='crossfilter-field_x',
                options = [{'label':x, 'value':x} for x in FIELDS_x],
                value=min(FIELDS_x),
                clearable=False
            )], style={'width': '49%', 'float': 'right', 'display': 'inline-block'}

        )], style={'backgroundColor': 'rgb(17, 17, 17)', 'padding': '10px 5px'}
    ),

    html.Div([

        dcc.Graph(
            id='scatter-plot',
            hoverData={'points': [{'customdata': 0}]}
        )

    ], style={'width': '100%', 'height':'90%', 'display': 'inline-block', 'padding': '0 20'}),

    html.Div([
        dcc.Graph(id='point-plot'),
    ], style={'display': 'inline-block', 'width': '100%'}),

    ], style={'backgroundColor': 'rgb(17, 17, 17)'},
)


@app.callback(
    dash.dependencies.Output('scatter-plot', 'figure'),
    [
        dash.dependencies.Input('crossfilter-field_y', 'value'),
        dash.dependencies.Input('crossfilter-field_x', 'value'),
        dash.dependencies.Input('gradient-scheme', 'value')
    ]
)
def update_grap(field_y, field_x, gradient):
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
        x = field_x, 
        y = field_y,
        color = COLOR,
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
        legend_title_text='to update',
        height=650, margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
        hovermode='closest',
        template='plotly_dark'
    )

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return fig


# def create_point_plot(df, title):

#     fig = go.Figure(
#         data=[
#             go.Bar(name='Average', x=features, y=df_average.values, marker_color='#c178f6'),
#             go.Bar(name=title, x=features, y=df.values, marker_color='#89efbd')
#         ]
#     )

#     fig.update_layout(
#         barmode='group',
#         height=225,
#         margin={'l': 20, 'b': 30, 'r': 10, 't': 10},
#         template='plotly_dark'
#     )

#     fig.update_xaxes(showgrid=False)

#     fig.update_yaxes(type="log", range=[0,5])

#     return fig


# @app.callback(
#     dash.dependencies.Output('point-plot', 'figure'),
#     [
#         dash.dependencies.Input('scatter-plot', 'hoverData')
#     ]
# )
# def update_point_plot(hoverData):
#     index = hoverData['points'][0]['customdata']
#     title = f'Customer {index}'
#     return create_point_plot(df[features].iloc[index], title)


if __name__ == '__main__':
    app.run_server(debug=True)
