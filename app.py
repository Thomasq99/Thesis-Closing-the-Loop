import dash.exceptions
from dash import Dash, html, dcc, DiskcacheManager
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from Concepts.ACE_helper import save_images, save_concepts
from Dash_helper import prepare_ACE, plot_concepts
import os
import numpy as np
import diskcache
import plotly.graph_objects as go


#TODO add support for numpy array

cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # my GPU is too small to save enough images in its V-RAM to get the gradients


def blank_fig():
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

app.layout = html.Div(children=
[
    dcc.Store(id='ace'),

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
                        [dbc.Spinner(html.Div(id='ace_output_text'))], width={'offset': 1, 'size': 3}),
                dbc.Col([html.H5('Graph'),
                         dcc.Graph(figure=blank_fig(), id='cav_images')], width=8)
            ]
        )
    ])
])


@app.callback([Output('ace_output_text', 'children'), Output('ace', 'data'), Output('cav_images', 'figure')],
              [Input('start_ACE', 'n_clicks'),
              State('model_selection', 'value'),
              State('data_path', 'value'),
              State('working_dir', 'value'),
              State('target_class', 'value'),
              State('bottlenecks', 'value')],
              running=[(Output('start_ACE', 'disabled'), True, False)],
              background=True,
              manager=background_callback_manager
              )
def start_ace(n_clicks, model_name, path_to_source, path_to_working_dir, target_class, bottlenecks):
    if n_clicks is None:  # prevent from initializing on start
        raise dash.exceptions.PreventUpdate("Button has not been clicked yet")
    else:
        discovered_concepts_dir, ace = prepare_ACE(model_name, path_to_source, path_to_working_dir, target_class, bottlenecks)
        print('Find patches')
        ace.create_patches_for_data()
        image_dir = os.path.join(discovered_concepts_dir, 'images')
        os.makedirs(image_dir)
        save_images(image_dir, (ace.discovery_images * 255).astype(np.uint8))  # save images used for creating patches

        print('Discover concepts')
        ace.discover_concepts(method='KM', param_dicts={'n_clusters': 25})
        del ace.dataset  # Free memory
        del ace.image_numbers
        del ace.patches

        # Save discovered concept images (resized and original sized)
        save_concepts(ace, discovered_concepts_dir)

        print('Calculating CAVs')
        accuracies = ace.cavs(in_memory=True)

        print('Computing TCAVs')
        scores = ace.tcavs(test=False)

        fig = plot_concepts(bottlenecks, ace)

        # only keep concept names in dictionary
        ace.concept_dict = {bn: {'concepts': ace.concept_dict[bn]['concepts']} for bn in ace.concept_dict.keys()}

    return repr(ace), ace.concept_dict, fig


if __name__ == '__main__':
    app.run_server(debug=True)
