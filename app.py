import time
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import SGDClassifier, LogisticRegressionCV
import dash.exceptions
from dash import Dash, html, dcc, DiskcacheManager, ctx
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from Dash_helper import run_ACE, create_new_concept, get_sessions_concepts, get_sessions_bottlenecks, blank_fig, \
    get_class_labels, load_model
import os
import diskcache
from Concepts.ConceptBank import ConceptBank
from Concepts.helper import load_images_from_files
import tensorflow as tf
import pickle as p
import base64
import io
import numpy as np
import plotly.express as px
import json
import plotly.figure_factory as ff

MAX_ROWS_CONCEPT_VIS = 60
SHAPE_IMGS_CONCEPT_VIS = (60, 60)
NUM_IMGS_PER_CONCEPT_VIS = 8

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
            dcc.Dropdown(id='remove_concept_select', disabled=True, multi=True, className='dash-bootstrap',
                         style={'width': '324px'})
        ], style={'margin-top': '15px', 'margin-bottom': '15px'}),

        dbc.InputGroup([
            dcc.Upload(dbc.Button('Upload images/CAV to add concept', outline=True, color="primary",
                                  style={'width': '100%'}), multiple=True, id='upload_concept'),
            dbc.Input(id='target_class_add_concept', type='text', placeholder='Target class')
        ], style={'margin-top': '15px', 'margin-bottom': '15px'}),

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
            dbc.Input(value='../../data/ImageNet', id='data_path', placeholder='Path to source_dir or np.array',
                      type='text')
        ], width=6),

    dbc.Col(
        [
            dbc.Label('Target classes:'),
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
                                                             'width': '64vw', 'height': '70vh'})),
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

ph_CBM_tab_content = dbc.Container(
    [
        dcc.Store(id='linear_model'),
        html.Div(id='save_output_lm', style={'display': 'none'}),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Row([
                            html.H5('post-hoc CBM settings', style={'text-align': 'center', 'margin-bottom': '15px',
                                                              'margin-top': '15px'}),
                            dbc.Label('Classes to include for classification:'),
                            dcc.Dropdown(id='choose_class', multi=True, className="dash-bootstrap",
                                         placeholder='Classes to classify'),
                            dbc.Label('Run post-hoc CBM:'),
                            dbc.InputGroup([dbc.Button(dbc.Spinner(html.Div("Start post-hoc CBM", id='spinner_ph_CBM')),
                                                       id='start_ph_CBM', outline=True, color='primary', n_clicks=0),
                                            dbc.Select(id='bottleneck_phCBM', placeholder='bottleneck')])
                             ], class_name='h-50'),

                        dbc.Row([
                            html.H5('post-hoc CBM results', style={'text-align': 'center', 'margin-bottom': '15px',
                                                             'margin-top': '15px'}),
                            html.Div('Train data size = NaN', id='train_size'),
                            html.Div('Test data size = NaN', id='test_size'),
                            html.Div("Train Accuracy = NaN", id='train_accuracy'),
                            html.Div("Test Accuracy = NaN", id='test_accuracy'),
                            html.Div('Train AUC = NaN', id='train_auc'),
                            html.Div('Test AUC = NaN', id='test_auc'),
                            dbc.Button("Save linear model", id='save_linear_model_button', outline=True,
                                       color='primary', n_clicks=0),


                        ], class_name='h-50')
                    ], width=3),

                dbc.Col([
                    dbc.Row(html.H5('Visualizations', style={'textAlign': 'center'})),
                    dbc.Row([dbc.Col([dcc.Graph(figure=blank_fig(), id='weight_vis_CBM',
                                               style={'overflowY': 'scroll', 'overflowX': 'scroll', 'height': '70vh'}),
                                      dcc.Dropdown(id='weight_dropdown', className='dash-bootstrap')],
                                     width=6),
                             dbc.Col(dcc.Graph(figure=blank_fig(), id='confusion_matrix',
                                               style={'overflowY': 'scroll', 'overflowX': 'scroll', 'height': '70vh'}),
                                     width=6)])



                ], width=9)
            ]
        )

    ], fluid=True)


tabs = dbc.Tabs(
    [
        dbc.Tab(Concept_bank_tab_content, label='Concept Bank'),
        dbc.Tab(ph_CBM_tab_content, label='post-hoc CBM')
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
               State('target_class_add_concept', 'value'),
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
def update_concept_bank(b1, b2, b3, uploaded_cb, list_of_concept_images, target_class_add_concept, image_filenames,
                        model_name, path_to_source, path_to_working_dir, target_classes, bottlenecks, stored_info,
                        remove_concept_list):

    # get the id of the button that triggered the callback
    triggered_id = ctx.triggered_id

    # get the stored data from the dcc.Storage
    if stored_info is None:
        concept_bank_dct = {}
        classes = []
    else:
        concept_bank_dct, classes = stored_info['concept_bank_dct'], stored_info['classes']

    if triggered_id == 'start_ACE':  # automatically extract concepts
        found_new = []
        bottlenecks = [bn.strip() for bn in bottlenecks.split(',')]
        target_classes = [target_class.strip() for target_class in target_classes.split(',')]
        for target_class in target_classes:
            bottlenecks_to_go_to_dct = [bn for bn in concept_bank_dct.keys() if isinstance(concept_bank_dct[bn],
                                                                                           ConceptBank)]
            if bottlenecks_to_go_to_dct:
                concept_bank_dct = {bn: concept_bank_dct[bn].to_dict() for bn in bottlenecks_to_go_to_dct}
            concept_bank_dct, classes, found_new_ = run_ACE(model_name, path_to_source, path_to_working_dir,
                                                           target_class, bottlenecks, concept_bank_dct, classes)
            found_new.append(found_new_)

        # sort concept banks:
        if True in found_new:
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
        return 'Automatically extract concepts', None, False, dash.no_update, None, dash.no_update, True, \
               True, True, True

    elif triggered_id == 'remove_concept_button':
        if remove_concept_list is None:
            raise dash.exceptions.PreventUpdate
        bottlenecks = [bn.strip() for bn in bottlenecks.split(',')]

        # load concept bank
        concept_bank_dct = {bn: ConceptBank(concept_bank_dct[bn]) for bn in concept_bank_dct.keys()}

        # remove concept
        for i in range(len(remove_concept_list)):
            bottleneck, concept_name = remove_concept_list[i].split(', ')
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
        model = load_model(model_name)

        # load concept bank
        concept_bank_dct = {bn: ConceptBank(concept_bank_dct[bn]) for bn in concept_bank_dct.keys()}
        for bottleneck in bottlenecks:
            concept_name = create_new_concept(list_of_concept_images, image_filenames, path_to_working_dir, bottleneck,
                                              model, target_class_add_concept)

            if bottleneck in concept_bank_dct:
                concept_bank_dct[bottleneck].add_concept([concept_name])
            else:
                class_to_id = get_class_labels(path_to_source)
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
               State('concept_bank', 'data'),
               State('working_dir', 'value')],
              prevent_initial_call=True)
def save_concept_bank(b1, stored_info, session_dir):
    with open(os.path.join(session_dir, 'saved_concept_bank.pkl'), 'wb') as file:
        p.dump(stored_info, file, protocol=-1)
    return None


@app.callback([Output('cav_images', 'figure'),
               Output('create_fig_button', 'n_clicks')],
              [Input('create_fig_button', 'n_clicks'),
               State('bottleneck_dropdown_cav_images', 'value'),
               State('concept_bank', 'data')],
              running=[(Output('bottleneck_dropdown_cav_images', 'disabled'), True, False)],
              prevent_initial_call=True)
def create_figure(n_clicks, bottleneck, stored_info):
    if (bottleneck is None) and not n_clicks:
        return blank_fig(), 0
    elif n_clicks:
        concept_bank_dct = stored_info['concept_bank_dct']
        concept_bank = ConceptBank(concept_bank_dct[bottleneck])
        fig = concept_bank.plot_concepts(num_images=NUM_IMGS_PER_CONCEPT_VIS, max_rows=MAX_ROWS_CONCEPT_VIS,
                                         shape=SHAPE_IMGS_CONCEPT_VIS)
        return fig, 0
    else:
        raise dash.exceptions.PreventUpdate()


@app.callback([Output('start_ph_CBM', 'disabled'),
               Output('choose_class', 'options'),
               Output('bottleneck_phCBM', 'options'),
               Output('bottleneck_phCBM', 'value')],
              [Input('concept_bank', 'data'),
               State('data_path', 'value')], prevent_initial_call=True)
def get_vis_phCBM_options(stored_info, data_path):
    if stored_info is None:
        return True, dash.no_update, dash.no_update, dash.no_update
    else:
        concept_bank_dct = stored_info['concept_bank_dct']
        bottleneck_options = get_sessions_bottlenecks(concept_bank_dct)
        classes = list(get_class_labels(data_path).keys())
        classes.append('ALL')
        options_classes = [{'label': class_, 'value': class_} for class_ in classes]
        return False, options_classes, bottleneck_options, bottleneck_options[0]['value']


@app.callback([Output('linear_model', 'data'),
               Output('spinner_ph_CBM', 'children'),
               Output('weight_dropdown', 'value'),
               Output('confusion_matrix', 'figure'),
               Output('train_accuracy', 'children'),
               Output('train_auc', 'children'),
               Output('test_accuracy', 'children'),
               Output('test_auc', 'children'),
               Output('train_size', 'children'),
               Output('test_size', 'children')],
              [Input('start_ph_CBM', 'n_clicks'),
               State('choose_class', 'value'),
               State('concept_bank', 'data'),
               State('bottleneck_phCBM', 'value'),
               State('data_path', 'value')],
              prevent_initial_call=True)
def run_phCBM(b1, classes, stored_info, bottleneck, data_path):

    # get class_to_id and id_to_folder dictionaries
    with open(os.path.join(data_path, 'class_index.json')) as file:
        dct = json.load(file)

    class_to_id = {}
    id_to_folder = {}
    for key, value in dct.items():
        id_to_folder[int(key)] = value[0]
        class_to_id[value[1]] = int(key)

    if 'ALL' in classes:
        classes = list(class_to_id.keys())

    print('Projecting images onto concept subspace')
    concept_bank_dct = stored_info['concept_bank_dct']
    concept_bank = ConceptBank(concept_bank_dct[bottleneck])

    images = []
    labels = []
    for idx, class__ in enumerate(classes):
        filenames = [os.path.join(data_path, id_to_folder[class_to_id[class__]], filename) for filename in
                     os.listdir(os.path.join(data_path, id_to_folder[class_to_id[class__]]))]
        imgs = load_images_from_files(filenames, max_imgs=1000)
        images.append(imgs)
        labels.extend([idx]*imgs.shape[0])
    images_arr = np.concatenate(images, axis=0)

    projected_imgs = concept_bank.project_onto_conceptspace(images_arr)
    labels = np.array(labels)

    print('training linear model to classify images based on concepts')

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1234)
    train_index, test_index = next(sss.split(images_arr, labels))

    X_train_proj, X_test_proj = projected_imgs[train_index], projected_imgs[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    classifier = LogisticRegressionCV(penalty='l1', solver='saga', max_iter=10000, multi_class='multinomial')

    classifier.fit(X_train_proj, y_train)

    # accuracy and AUC
    train_predictions = classifier.predict(X_train_proj)
    train_accuracy = np.mean((y_train == train_predictions)) * 100.
    predictions = classifier.predict(X_test_proj)
    test_accuracy = np.mean((y_test == predictions)) * 100.

    y_score_test = classifier.predict_proba(X_test_proj)
    test_auc = metrics.roc_auc_score(y_test, y_score_test, multi_class='ovr')
    y_score_train = classifier.predict_proba(X_train_proj)
    train_auc = metrics.roc_auc_score(y_train, y_score_train, multi_class='ovr')

    # confusion matrix fig
    confusion_matrix = metrics.confusion_matrix(y_test, predictions)
    z = confusion_matrix.astype(int)
    x = classes
    y = classes

    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in z]

    # set up figure
    confusion_matrix_fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')

    # add title
    confusion_matrix_fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                      # xaxis = dict(title='x'),
                      # yaxis = dict(title='x')
                      )

    # add custom xaxis title
    confusion_matrix_fig.add_annotation(dict(font=dict(color="black", size=14),
                                             x=0.5, y=-0.15, showarrow=False, text="Predicted value",
                                             xref="paper", yref="paper"))

    # add custom yaxis title
    confusion_matrix_fig.add_annotation(dict(font=dict(color="black", size=14), x=-0.35, y=0.5, showarrow=False,
                                             text="Real value", textangle=-90, xref="paper", yref="paper"))

    # adjust margins to make room for yaxis title
    confusion_matrix_fig.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    confusion_matrix_fig['data'][0]['showscale'] = True

    model_bytes = p.dumps(classifier)
    model_str = base64.b64encode(model_bytes).decode()

    return model_str, 'Start ph_CBM', classes[0], confusion_matrix_fig,\
           f'Train Accuracy = {train_accuracy:.2f}%', f'Test Accuracy = {test_accuracy:.2f}%', \
           f'Train AUC = {train_auc:.2f}', f'Test AUC = {test_auc:.2f}', f'Train data size = {X_train_proj.shape[0]}',\
           f'Test data size = {X_test_proj.shape[0]}'


@app.callback(Output('save_output_lm', 'children'),
              [Input('save_linear_model_button', 'n_clicks'),
               State('linear_model', 'data'),
               State('working_dir', 'value')],
              prevent_initial_call=True)
def save_linear_model(b1, model_str, session_dir):
    # Decode the base64 string back to a bytes object
    model_bytes = base64.b64decode(model_str.encode())
    # Convert the bytes object back to a model object using pickle
    model = p.loads(model_bytes)
    with open(os.path.join(session_dir, 'saved_linear_model.pkl'), 'wb') as file:
        p.dump(model, file, protocol=-1)
    return None


@app.callback(Output('model_layers', 'data'),
              [Input('model_selection', 'valid'),
              State('model_selection', 'value')])
def store_model_layers(is_model_valid, model_name):
    if is_model_valid:
        if model_name == 'InceptionV3':
            model = tf.keras.applications.inception_v3.InceptionV3()
        else:
            model = tf.keras.models.load_model(model_name)
        model_layers = [layer.name for layer in model.layers]
        return {'model_layers': model_layers}
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
        try:
            start = time.time()
            tf.keras.models.load_model(model_name)
            print(time.time() - start)
            return True, False
        except:
            print(f'{model_name} is not a valid model')
            return False, True


@app.callback([Output('target_class', 'valid'), Output('target_class', 'invalid')],
              [Input('target_class', 'value'),
               State('data_path', 'value'),
               Input('data_path', 'valid')],
              prevent_initial_call=True)
def validate_class(target_classes, data_path, data_path_validity):
    if not data_path_validity or target_classes is None:
        return False, True
    else:
        target_class_lst = list(get_class_labels(data_path).keys())
        target_classes = [target_class.strip() for target_class in target_classes.split(',')]

        for target_class in target_classes:
            if target_class not in target_class_lst:
                return False, True

        return True, False


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


@app.callback([Output('target_class_add_concept', 'valid'), Output('target_class_add_concept', 'invalid')],
              [Input('target_class_add_concept', 'value'),
               State('data_path', 'value'),
               Input('data_path', 'valid')],
              prevent_initial_call=True)
def validate_class_add_concept(target_classes, data_path, data_path_validity):
    if not data_path_validity or target_classes is None:
        return False, True
    else:
        target_class_lst = list(get_class_labels(data_path).keys())
        target_classes = [target_class.strip() for target_class in target_classes.split(',')]

        for target_class in target_classes:
            if target_class not in target_class_lst:
                return False, True

        return True, False


@app.callback([Output('weight_dropdown', 'options'),
               Output('weight_dropdown', 'disabled')],
              [Input('choose_class', 'value'),
               State('data_path', 'value')],
              prevent_initial_call=True)
def update_weight_dropdown(classes, data_path):
    # get class_to_id and id_to_folder dictionaries
    with open(os.path.join(data_path, 'class_index.json')) as file:
        dct = json.load(file)

    class_to_id = {}
    id_to_folder = {}
    for key, value in dct.items():
        id_to_folder[int(key)] = value[0]
        class_to_id[value[1]] = int(key)

    if 'ALL' in classes:
        classes = list(class_to_id.keys())

    if len(classes) == 2:
        disabled = True
    else:
        disabled = False

    return classes, disabled


@app.callback(Output('weight_vis_CBM', 'figure'),
              [Input('weight_dropdown', 'value'),
               State('choose_class', 'value'),
               State('bottleneck_phCBM', 'value'),
               State('concept_bank', 'data'),
               State('linear_model', 'data')],
              prevent_initial_call=True)
def update_weight_vis(class_, classes, bottleneck, stored_info_conceptbank, stored_info_linear_model):
    model_bytes = base64.b64decode(stored_info_linear_model.encode())
    # Convert the bytes object back to a model object using pickle
    classifier = p.loads(model_bytes)
    class_idx = classes.index(class_)

    concept_bank_dct = stored_info_conceptbank['concept_bank_dct']
    concept_bank = ConceptBank(concept_bank_dct[bottleneck])
    if len(classes) == 2:
        colors = [classes[0] if c < 0 else classes[1] for c in classifier.coef_[0]]

        regression_fig = px.bar(
            x=concept_bank.concept_names, y=classifier.coef_[0], color=colors,
            color_discrete_sequence=['red', 'blue'],
            labels=dict(x='Feature', y='Linear coefficient'),
            title='<i><b>Regression Weights</b></i>'
        )

    else:
        colors = ['Negative' if c < 0 else 'Positive' for c in classifier.coef_[class_idx]]

        # regression weight plot
        regression_fig = px.bar(
            x=concept_bank.concept_names, y=classifier.coef_[class_idx], color= colors,
            labels=dict(x='Feature', y='Linear coefficient'),
            title='<i><b>Regression Weights</b></i>'
        )

    return regression_fig


if __name__ == '__main__':
    app.run_server(debug=True)
