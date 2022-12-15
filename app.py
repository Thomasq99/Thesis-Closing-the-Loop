import dash.exceptions
from dash import Dash, html, dcc, DiskcacheManager
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from Dash_helper import run_ACE
import os
import diskcache
import plotly.graph_objects as go
from Concepts.ConceptBank import ConceptBank
import time

# TODO add support for numpy array

cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # my GPU is too small to save enough images in its V-RAM to get the gradients


def blank_fig() -> go.Figure:
    """ function that returns a blank figure

    @return blank figure."""
    fig = go.Figure(go.Scatter(x=[], y=[]))
    fig.update_layout(template=None)
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)

    return fig


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# define ace initialization input menu
ace_inputs = [
    dbc.Label('Model selection'),
    dbc.Input(value='InceptionV3', id='model_selection', type='text'),

    html.Br(),

    dbc.Label('Data source directory'),
    dbc.Input(value='./data/ImageNet', id='data_path', placeholder='Path to source_dir or np.array',
              type='text'),

    html.Br(),

    dbc.Label('Output directory'),
    dbc.Input(value='./ACE_output/ImageNet', id='working_dir', placeholder='working_dir', type='text'),

    html.Br(),

    dbc.Label('Name of target class'),
    dbc.Input(id='target_class', placeholder='target class', type='text'),

    html.Br(),

    dbc.Label('Bottlenecks'),
    dbc.Input(id='bottlenecks', placeholder='comma separated bottlenecks',
              type='text'),

    html.Br(),

    dbc.Button('Start ACE', id='start_ACE')
]

app.layout = dbc.Container(children=
[
    dcc.Store(id='concept_bank'),

    html.H1(children='Creating the Concepts', style={'textAlign': 'center'}),

    html.Div(children='''
        Creating the Concept Bank, Using ACE to automatically generate concepts.
        Using CAVs to represent concepts.
    ''', style={'textAlign': 'center'}),

    html.Br(),

    html.Div(children=
    [
        dbc.Row(
            [
                dbc.Col([html.H5("ACE initialization parameters"), html.Br()] + ace_inputs +
                        [dbc.Spinner(html.Div(id='ace_output_text'))], width=4),
                dbc.Col([html.H5('Visualization of Concepts', style={'textAlign': 'center'}),
                         dbc.Spinner(dcc.Graph(figure=blank_fig(), id='cav_images',
                                   style={'overflowY': 'scroll', 'overflowX': 'scroll', 'width': '63vw',
                                         'height': '70vh'})),
                         dcc.Dropdown(id='bottleneck_dropdown_cav_images', value="Not initialized",
                                      disabled=True)],
                        width=8)
            ]
        )
    ])
])


@app.callback([Output('ace_output_text', 'children'), Output('concept_bank', 'data'),
               Output('bottleneck_dropdown_cav_images', 'disabled'),
               Output('bottleneck_dropdown_cav_images', 'options'),
               Output('bottleneck_dropdown_cav_images', 'value')],
              [Input('start_ACE', 'n_clicks'),
              State('model_selection', 'value'),
              State('data_path', 'value'),
              State('working_dir', 'value'),
              State('target_class', 'value'),
              State('bottlenecks', 'value')],
              running=[(Output('start_ACE', 'disabled'), True, False)],
              background=True,
              manager=background_callback_manager)
def start_ace(n_clicks, model_name, path_to_source, path_to_working_dir, target_class, bottlenecks):
    if n_clicks is None:  # prevent from initializing on start
        raise dash.exceptions.PreventUpdate("Button has not been clicked yet")
    else:
        bottlenecks = [bn.strip() for bn in bottlenecks.split(',')]
        concept_bank_dct = run_ACE(model_name, path_to_source, path_to_working_dir, target_class, bottlenecks)

        # sort concept banks:
        print('sorting concept bank')
        for bottleneck in concept_bank_dct:
            concept_bank_dct[bottleneck].sort_concepts()

        concept_bank_dct = {bn: concept_bank_dct[bn].to_dict() for bn in concept_bank_dct.keys()}
    # TODO allow for loading existing concept_bank and adding concepts to it.
    return 'Done', concept_bank_dct, False, bottlenecks, bottlenecks[0]


@app.callback(Output('cav_images', 'figure'),
              [Input('bottleneck_dropdown_cav_images', 'value'),
               State('concept_bank', 'data')],
              running=[(Output('bottleneck_dropdown_cav_images', 'disabled'), True, False)])
def create_figure(bottleneck, concept_bank_dct):
    if bottleneck == 'Not initialized':
        raise dash.exceptions.PreventUpdate("ACE has not been run yet")
    else:
        concept_bank = ConceptBank(concept_bank_dct[bottleneck])
        concept_bank.load_in_memory()
        fig = concept_bank.plot_concepts(num_images=10)
        fig.show()
        print('returning figure')
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
