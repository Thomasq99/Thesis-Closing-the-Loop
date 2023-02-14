import dash.exceptions
from dash import Dash, html, dcc, DiskcacheManager, ctx
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from Dash_helper import run_ACE, create_new_concept, get_sessions_concepts, get_sessions_bottlenecks, blank_fig, \
    get_class_labels

from Concepts.helper import ace_create_source_dir_imagenet
import os
import diskcache
from Concepts.ConceptBank import ConceptBank
import tensorflow as tf
import pickle as p
import base64
import io

# initialize background caching
cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # my GPU is too small to save enough images in its V-RAM to get the gradients
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# define layout components of the concept bank menu
concept_bank_menu = [
    html.H5('Concept Bank', style={'text-align': 'center', 'margin-bottom': '11px'}),
    dbc.Col([
        html.Div(dbc.Button([dbc.Spinner([html.Div('Automatically extract concepts', id='spinner')])],
                            id='start_ACE', outline=True, color='primary'),
                 className="d-grid gap-2"),
        dbc.InputGroup([
            dbc.Button("Remove concept", id='remove_concept_button', outline=True, color='primary', n_clicks=0,
                       disabled=True),
            dbc.Select(id='remove_concept_select', disabled=True)
        ], style={'margin-top': '15px', 'margin-bottom': '15px'}),

        dcc.Upload(dbc.Button('Upload images to add concept', outline=True, color="primary", style={'width': '100%'}),
                   multiple=True, id='upload_concept'),

        dcc.Upload(dbc.Button('Import Concept Bank', outline=True, color='primary', style={'width': '100%'}),
                   style={'margin-top': '15px', 'margin-bottom': '15px'}, id='import_cb_button'),

        dbc.Row([
            dbc.Col(html.Div(dbc.Button('Clear Concept Bank', outline=True, color='primary', id='clear_concept_bank',
                                        n_clicks=0, disabled=True),
                             className="d-grid gap-2"), width=6),

            dbc.Col(html.Div(dbc.Button('Save current Concept Bank', id='save_button', outline=True, color='primary'),
                             className="d-grid gap-2"), width=6)
        ]),
        html.Div(id='save_output', style={'display': 'none'}),
    ])
]

# define settings of the concept bank
settings_menu = [
    html.H5("Settings", style={'text-align': 'center', 'margin-bottom': '0px', 'margin-top': '15px'}),
    dbc.Label('Session directory:', style={'margin-bottom': '0px'}),

    dbc.Col([
        dbc.Input(value='./ACE_output/ImageNet', id='working_dir', placeholder='working_dir', type='text',
                  required=True)
    ], width=12),

    dbc.Col(
        [
            dbc.Label('Model selection:'),
            dbc.Input(value='InceptionV3', id='model_selection', type='text', required=True),
            dbc.Label('Data directory:'),
            dbc.Input(value='./data/ImageNet', id='data_path', placeholder='Path to source_dir or np.array',
                      type='text')
        ], width=6),

    dbc.Col(
        [
            dbc.Label('Target Class:'),
            dbc.Input(id='target_class', placeholder='e.g. toucan', type='text'),
            dbc.Label('Bottleneck layers:'),
            dbc.Input(id='bottlenecks', placeholder='e.g. mixed8, mixed3',
                      type='text'),

        ], width=6)
]


# define Concept_bank tab
Concept_bank_tab_content = dbc.Container(
    [
        dcc.Store(id='concept_bank'),
        dcc.Store(id='model_layers'),

        # html.H3('Creating the Concepts', style={'textAlign': 'center'}),

        # html.Div('''
        # Creating the Concept Bank, Using ACE to automatically generate concepts.
        # Using CAVs to represent concepts.
        # ''', style={'textAlign': 'center'}),

        html.Br(),

        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [  # Settings
                                dbc.Row(settings_menu, style={'height': "50%"}),

                                # Concept bank menu definition
                                dbc.Row(concept_bank_menu, style={'height': "50%"})

                            ],
                            width=4),

                        # Visualization of the concepts
                        dbc.Col(
                            [
                                html.H5('Visualization of Concepts', style={'textAlign': 'center'}),
                                dbc.Spinner(dcc.Graph(figure=blank_fig(), id='cav_images',
                                                      style={'overflowY': 'scroll', 'overflowX': 'scroll',
                                                             'width': '64vw',
                                                             'height': '70vh'})),
                                dbc.InputGroup([
                                    dbc.Button('Update Concept Visualization', id='create_fig_button', outline=True,
                                               color='primary', n_clicks=0, disabled=True),
                                    dbc.Select(id='bottleneck_dropdown_cav_images', disabled=True)]),
                            ],
                            width=8)
                    ]
                )
            ], fluid=True)
    ], fluid=True)

exploring_concept_space_tab_content = html.Div('hoi')

tabs = dbc.Tabs(
    [
        dbc.Tab(Concept_bank_tab_content, label='Concept Bank'),
        dbc.Tab(exploring_concept_space_tab_content, label='Exploring the Concept Space', disabled=True)
    ]
)

app.layout = dbc.Container([html.H2('Closing the Concept Loop', style={'textAlign': 'center'}), tabs], fluid=True)

@app.callback([Output('spinner', 'children'),
               Output('concept_bank', 'data'),
               Output('bottleneck_dropdown_cav_images', 'disabled'),
               Output('bottleneck_dropdown_cav_images', 'options'),
               Output('bottleneck_dropdown_cav_images', 'value'),
               Output('remove_concept_select', 'options'),
               Output('create_fig_button', 'disabled'),
               Output('remove_concept_button', 'disabled'),
               Output('clear_concept_bank', 'disabled'),
               Output('remove_concept_select', 'disabled')],
              [Input('start_ACE', 'n_clicks'),
               Input('clear_concept_bank', 'n_clicks'),
               Input('remove_concept_button', 'n_clicks'),
               Input('import_cb_button', 'contents'),
               Input('upload_concept', 'contents'),
               State('upload_concept', 'filename'),
               State('model_selection', 'value'),
               State('data_path', 'value'),
               State('working_dir', 'value'),
               State('target_class', 'value'),
               State('bottlenecks', 'value'),
               State('concept_bank', 'data'),
               State('remove_concept_select', 'value')],
              running=[(Output('start_ACE', 'disabled'), True, False),
                       (Output('remove_concept_button', 'disabled'), True, False)],
              background=True,
              prevent_initial_call=True,
              manager=background_callback_manager)
def update_concept_bank(b1, b2, b3, uploaded_cb, list_of_concept_images, image_filenames, model_name, path_to_source,
                        path_to_working_dir, target_class, bottlenecks, stored_info, remove_concept_name):

    # get the id of the button that triggered the callback
    triggered_id = ctx.triggered_id

    # get the stored data from the dcc.Storage
    if stored_info is None:
        concept_bank_dct = {}
        classes = []
    else:
        concept_bank_dct, classes = stored_info['concept_bank_dct'], stored_info['classes']

    if triggered_id == 'start_ACE':  # automatically extract concepts
        bottlenecks = [bn.strip() for bn in bottlenecks.split(',')]
        # TODO add loading in current concept bank
        concept_bank_dct, classes, found_new = run_ACE(model_name, path_to_source, path_to_working_dir, target_class,
                                                       bottlenecks, concept_bank_dct, classes)

        # sort concept banks:
        if found_new:
            print('sorting concept bank')
            for bottleneck in concept_bank_dct:
                if isinstance(concept_bank_dct[bottleneck], dict):
                    concept_bank_dct[bottleneck] = ConceptBank(concept_bank_dct[bottleneck])
                concept_bank_dct[bottleneck].sort_concepts()

        # convert ConceptBank objects to dicts for storing in dash
        concept_bank_dct = {bn: concept_bank_dct[bn].to_dict() if not isinstance(concept_bank_dct[bn], dict) else
        concept_bank_dct[bn] for bn in concept_bank_dct.keys()}

        remove_options = get_sessions_concepts(concept_bank_dct)
        bottleneck_options = get_sessions_bottlenecks(concept_bank_dct)
        stored_info = {'concept_bank_dct': concept_bank_dct, 'classes': classes}

        return 'Automatically extract concepts', stored_info, False, bottleneck_options, bottlenecks[0], \
               remove_options, False, False, False, False

    elif triggered_id == 'clear_concept_bank':  # clear current concept bank
        return 'Automatically extract concepts', None, False, None, None, None, True, True, True, True

    elif triggered_id == 'remove_concept_button':
        bottlenecks = [bn.strip() for bn in bottlenecks.split(',')]
        # TODO Failsafe if all concepts are removed
        bottleneck, concept_name = remove_concept_name.split(', ')

        # load concept bank
        concept_bank_dct = {bn: ConceptBank(concept_bank_dct[bn]) for bn in concept_bank_dct.keys()}
        # remove concept
        concept_bank_dct[bottleneck].remove_concept(concept_name)
        # extract concept bank
        concept_bank_dct = {bn: concept_bank_dct[bn].to_dict() for bn in concept_bank_dct.keys()}

        remove_options = get_sessions_concepts(concept_bank_dct)
        bottleneck_options = get_sessions_bottlenecks(concept_bank_dct)
        stored_info = {'concept_bank_dct': concept_bank_dct, 'classes': classes}

        return 'Automatically extract concepts', stored_info, False, bottleneck_options, bottlenecks[0], \
               remove_options, False, False, False, False

    elif triggered_id == 'upload_concept':
        bottlenecks = [bn.strip() for bn in bottlenecks.split(',')]
        if model_name == 'InceptionV3':
            model = tf.keras.applications.inception_v3.InceptionV3()
        elif os.path.exists(model_name):
            model = tf.keras.models.load_model(model_name)
        else:
            raise ValueError(f'{model_name} is not a directory to a model nor the InceptionV3model')

        # load concept bank
        concept_bank_dct = {bn: ConceptBank(concept_bank_dct[bn]) for bn in concept_bank_dct.keys()}
        for bottleneck in bottlenecks:
            concept_name = create_new_concept(list_of_concept_images, image_filenames, path_to_working_dir, bottleneck, model)

            if bottleneck in concept_bank_dct:
                concept_bank_dct[bottleneck].add_concept([concept_name])
            else:
                # TODO refactor class_to_id
                class_to_id = ace_create_source_dir_imagenet('./data/ImageNet', path_to_source, 'toucan',
                                                             num_random_concepts=20, ow=False)
                concept_bank_dct[bottleneck] = ConceptBank(dict(bottleneck=bottleneck, working_dir=path_to_working_dir,
                                                                concept_names=[concept_name], class_id_dct=class_to_id,
                                                                model_name=model_name))
            concept_bank_dct[bottleneck].sort_concepts()
        # extract concept bank
        concept_bank_dct = {bn: concept_bank_dct[bn].to_dict() for bn in concept_bank_dct.keys()}

        remove_options = get_sessions_concepts(concept_bank_dct)
        bottleneck_options = get_sessions_bottlenecks(concept_bank_dct)
        stored_info = {'concept_bank_dct': concept_bank_dct, 'classes': classes}

        return 'Automatically extract concepts', stored_info, False, bottleneck_options, bottlenecks[0], \
               remove_options, False, False, False, False

    elif triggered_id == 'import_cb_button':
        content_type, content_string = uploaded_cb.split(',')
        decoded = base64.b64decode(content_string)
        p_file = io.BytesIO(decoded)
        stored_info = p.loads(p_file.read())
        concept_bank_dct = stored_info['concept_bank_dct']
        bottleneck_options = get_sessions_bottlenecks(concept_bank_dct)
        bottleneck = bottleneck_options[0]['value']
        remove_options = get_sessions_concepts(concept_bank_dct)
        return 'Automatically extract concepts', stored_info, False, bottleneck_options, bottleneck, \
               remove_options, False, False, False, False


@app.callback(Output('save_output', 'children'),
              [Input('save_button', 'n_clicks'),
               State('concept_bank', 'data')],
              prevent_initial_call=True)
def save_concept_bank(b1, stored_info):
    with open('./saved_concept_bank.pkl', 'wb') as file:
        p.dump(stored_info, file, protocol=-1)
    return None


@app.callback([Output('cav_images', 'figure'),
               Output('create_fig_button', 'n_clicks')],
              [Input('create_fig_button', 'n_clicks'),
               Input('bottleneck_dropdown_cav_images', 'value'),
               State('concept_bank', 'data')],
              running=[(Output('bottleneck_dropdown_cav_images', 'disabled'), True, False)],
              prevent_initial_call=True)
def create_figure(n_clicks, bottleneck, stored_info):
    if (bottleneck is None) and not n_clicks:
        return blank_fig(), 0
    elif n_clicks:
        concept_bank_dct = stored_info['concept_bank_dct']
        concept_bank = ConceptBank(concept_bank_dct[bottleneck])
        fig = concept_bank.plot_concepts(num_images=10)
        return fig, 0
    else:
        raise dash.exceptions.PreventUpdate()


@app.callback(Output('model_layers', 'data'),
              [Input('model_selection', 'valid'),
              State('model_selection', 'value')])
def store_model_layers(is_model_valid, model_name):
    if is_model_valid:
        if model_name == 'InceptionV3':
            model = tf.keras.applications.inception_v3.InceptionV3()
            model_layers = [layer.name for layer in model.layers]
            return {'model_layers': model_layers}
        else:
            return None  # TODO IMPLEMENT
    else:
        return None


# input validation
@app.callback([Output('working_dir', 'valid'), Output('working_dir', 'invalid')],
              [Input('working_dir', 'value')])
def validate_session_dir(session_path):
    is_valid = os.path.exists(session_path)
    return is_valid, not is_valid


@app.callback([Output('data_path', 'valid'), Output('data_path', 'invalid')],
              [Input('data_path', 'value')])
def validate_data_dir(data_path):
    is_valid = os.path.exists(data_path)
    return is_valid, not is_valid


@app.callback([Output('model_selection', 'valid'), Output('model_selection', 'invalid')],
              [Input('model_selection', 'value')])
def validate_model_selection(model_name):
    if model_name == 'InceptionV3':
        return True, False
    else:
        return False, True

    #TODO Implement model selection for own path


@app.callback([Output('target_class', 'valid'), Output('target_class', 'invalid')],
              [Input('target_class', 'value'),
               State('data_path', 'value'),
               Input('data_path', 'valid')])
def validate_class(target_class, data_path, data_path_validity):
    if not data_path_validity:
        return False, True #TODO add error message?
    else:
        target_class_lst = list(get_class_labels(data_path).keys())
        if target_class in target_class_lst:
            return True, False
        else:
            return False, True


@app.callback([Output('bottlenecks', 'valid'), Output('bottlenecks', 'invalid')],
              [Input('bottlenecks', 'value'),
               Input('model_layers', 'data')])
def validate_bottlenecks(bottlenecks, stored_model_layers):
    if (stored_model_layers is not None) and (bottlenecks is not None):
        bottlenecks = [bn.strip() for bn in bottlenecks.split(',')]
        model_layers = stored_model_layers['model_layers']
        bottleneck_valid = set(bottlenecks).issubset(model_layers)
        return bottleneck_valid, not bottleneck_valid
    else:
        return False, True


if __name__ == '__main__':
    app.run_server(debug=True)
