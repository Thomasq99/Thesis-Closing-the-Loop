import shutil
import dash.exceptions
from dash import Dash, html, dcc, DiskcacheManager
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from Concepts.ACE_helper import save_images, save_concepts, load_images_from_files
from Dash_helper import prepare_ACE
import os
import numpy as np
import diskcache
import plotly.graph_objects as go
import pickle as p
import tensorflow as tf
from Concepts.ConceptBank import ConceptBank


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
                        [dbc.Spinner(html.Div(id='ace_output_text'))], width=4),
                dbc.Col([html.H5('Visualization of Concepts', style={'textAlign': 'center'}),
                         dcc.Graph(figure=blank_fig(), id='cav_images',
                                   style={'overflowY': 'scroll', 'overflowX': 'scroll', 'width': '63vw',
                                          'height': '75vh'})
                         ],
                        width=8)
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
        bottlenecks = [bn.strip() for bn in bottlenecks.split(',')]
        mode = 'max'
        overwrite = False
        discovered_concepts_dir, ace = prepare_ACE(model_name, path_to_source, path_to_working_dir, target_class,
                                                   bottlenecks, overwrite=overwrite)
        # find if patches are already created once
        image_dir = os.path.join(discovered_concepts_dir, target_class, 'images')
        concept_bank_dct = {bn: ConceptBank(bn, working_dir=path_to_working_dir) for bn in bottlenecks}

        # check which bottlenecks are not yet created
        bn_to_find_concepts_for = [bn for bn in bottlenecks if not os.listdir(
            os.path.join(discovered_concepts_dir, target_class, bn))]
        bn_precomputed_concepts = list(set(bottlenecks) - set(bn_to_find_concepts_for))

        if bn_precomputed_concepts:
            for bn in bn_precomputed_concepts:
                print(f'loading in concepts for {bn}')
                concepts = os.listdir(os.path.join(discovered_concepts_dir, target_class, bn))
                concepts = [concept for concept in concepts if not concept.endswith('patches')]
                concept_bank_dct[bn] = ConceptBank(bn, path_to_working_dir, concepts)

        if bn_to_find_concepts_for:  # if not empty discover concepts for bottlenecks
            print(f'discovering concepts for {bn_to_find_concepts_for}')
            print('Creating patches')
            ace.bottlenecks = bn_to_find_concepts_for
            if os.path.exists(image_dir):
                ace.create_patches_for_data(discovery_images=load_images_from_files(
                    [os.path.join(image_dir, file) for file in os.listdir(image_dir)]))
            else:
                os.makedirs(image_dir)
                ace.create_patches_for_data()
                save_images(image_dir,
                            (ace.discovery_images * 255).astype(np.uint8))  # save images used for creating patches

            print('Discover concepts')
            ace.discover_concepts(method='KM', param_dicts={'n_clusters': 25})
            del ace.dataset  # Free memory
            del ace.image_numbers
            del ace.patches

            # Save discovered concept images (resized and original sized)
            save_concepts(ace, os.path.join(discovered_concepts_dir, target_class))

            print('Calculating CAVs')
            accuracies = ace.cavs(in_memory=True, ow=overwrite)

            print('combining CAVs')
            ace.save_cavs(accuracies, mode=mode)

            concept_dict = {bn: ace.concept_dict[bn]['concepts'] for bn in ace.concept_dict.keys()}

            for bn in concept_dict.keys():
                concept_bank_dct[bn] = ConceptBank(bn, path_to_working_dir, concept_dict[bn])

        del ace
        print('plotting')
        #TODO add selection of bn
        fig = concept_bank_dct[bottlenecks[0]].plot_concepts(num_images=10)
        concept_bank_dct = {bn: concept_bank_dct[bn].to_dict() for bn in concept_bank_dct.keys()}
    return 'Done', concept_bank_dct, fig


if __name__ == '__main__':
    app.run_server(debug=True)
