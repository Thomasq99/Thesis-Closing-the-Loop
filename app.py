import dash.exceptions
from dash import Dash, html, dcc, DiskcacheManager, ctx
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from Dash_helper import run_ACE
import os
import diskcache
import plotly.graph_objects as go
from Concepts.ConceptBank import ConceptBank


def blank_fig() -> go.Figure:
    """ function that returns a blank figure

    @return blank figure."""
    fig = go.Figure(go.Scatter(x=[], y=[]))
    fig.update_layout(template=None)
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
    return fig


cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # my GPU is too small to save enough images in its V-RAM to get the gradients
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

concept_bank_menu = [
    html.H5('Concept Bank', style={'text-align': 'center', 'margin-bottom': '0px'}),
    dbc.Col([
        dbc.Label('Session directory:', style={'margin-bottom': '0px'}),
        dbc.Input(value='./ACE_output/ImageNet', id='working_dir', placeholder='working_dir', type='text',
                  required=True),
        dbc.Label('Remove Concept:'),
        dbc.InputGroup([
            dbc.Button("Remove concept", id='remove_concept_button', outline=True, color='primary', n_clicks=0,
                       disabled=True),
            dbc.Select(id='remove_concept_select', disabled=True),
        ]),

        html.Br(),
        dbc.Row([
            dbc.Col(html.Div(dbc.Button('Clear Concept Bank', outline=True, color='primary', id='clear_concept_bank',
                                        n_clicks=0, disabled=True),
                             className="d-grid gap-2"), width=6),

            dbc.Col(dcc.Upload(dbc.Button('Upload Zip File of concept', outline=True, color="primary",
                                          style={'width': '100%'})), width=6)
        ]),
        # html.Br(),
        # html.Div(dbc.Button('Load Existing Concept Bank', outline=True, color='primary'), className="d-grid gap-2")

    ])
]

# define ace menu
ace_menu = [
    html.H5("ACE parameters", style={'text-align': 'center', 'margin-bottom': '0px', 'margin-top': '15px'}),
    dbc.Col(
        [
            dbc.Label('Model selection:'),
            dbc.Input(value='InceptionV3', id='model_selection', type='text'),
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

        ], width=6),
    html.Div(dbc.Button([dbc.Spinner([html.Div('Run ACE', id='spinner')])],
                        id='start_ACE', outline=True, color='primary'),
             className="d-grid gap-2")
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

    dbc.Container(children=
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        # Concept bank menu definition
                        dbc.Row(concept_bank_menu, style={'height': "50%"}),

                        # ACE Menu definition
                        dbc.Row(ace_menu, style={'height': "50%"})
                    ],
                    width=4),

                # Visualization of the concepts
                dbc.Col(
                    [
                        html.H5('Visualization of Concepts', style={'textAlign': 'center'}),
                        dbc.Spinner(dcc.Graph(figure=blank_fig(), id='cav_images',
                                              style={'overflowY': 'scroll', 'overflowX': 'scroll', 'width': '64vw',
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
def update_concept_bank(b1, b2, b3, model_name, path_to_source, path_to_working_dir, target_class, bottlenecks,
                        concept_bank_dct, remove_concept_name):
    triggered_id = ctx.triggered_id

    if triggered_id == 'start_ACE':
        #TODO add loading in current concept bank
        bottlenecks = [bn.strip() for bn in bottlenecks.split(',')]
        concept_bank_dct = run_ACE(model_name, path_to_source, path_to_working_dir, target_class, bottlenecks)

        # sort concept banks:
        print('sorting concept bank')
        for bottleneck in concept_bank_dct:
            concept_bank_dct[bottleneck].sort_concepts()

        concept_bank_dct = {bn: concept_bank_dct[bn].to_dict() for bn in concept_bank_dct.keys()}

        remove_options = []
        for bn in concept_bank_dct:
            for concept in concept_bank_dct[bn]['concept_names']:
                label = f'{bn}, {concept}'
                value = f'{bn},{concept}'
                remove_options.append({'value': value, 'label': label})

        bottleneck_options = [{'value': bn, 'label': bn} for bn in bottlenecks]

        return 'Run ACE', concept_bank_dct, False, bottleneck_options, bottlenecks[0], remove_options, False, False, \
               False, False

    elif triggered_id == 'clear_concept_bank':
        return 'Run ACE', None, False, None, None, None, True, True, True, True

    elif triggered_id == 'remove_concept_button':
        #TODO Failsafe if all concepts are removed
        bottleneck, concept_name = remove_concept_name.split(',')

        # load concept bank
        concept_bank_dct = {bn: ConceptBank(concept_bank_dct[bn]) for bn in concept_bank_dct.keys()}
        # remove concept
        concept_bank_dct[bottleneck].remove_concept(concept_name)
        # extract concept bank
        concept_bank_dct = {bn: concept_bank_dct[bn].to_dict() for bn in concept_bank_dct.keys()}

        # update options for removing concept
        remove_options = []
        for bn in concept_bank_dct:
            for concept in concept_bank_dct[bn]['concept_names']:
                label = f'{bn}, {concept}'
                value = f'{bn}_{concept}'
                remove_options.append({'value': value, 'label': label})

        bottlenecks = list(concept_bank_dct.keys())
        bottleneck_options = [{'value': bn, 'label': bn} for bn in bottlenecks]

        return 'Run ACE', concept_bank_dct, False, bottleneck_options, bottlenecks[0], remove_options, False, False, \
               False, False


@app.callback([Output('cav_images', 'figure'),
               Output('create_fig_button', 'n_clicks')],
              [Input('create_fig_button', 'n_clicks'),
               Input('bottleneck_dropdown_cav_images', 'value'),
               State('concept_bank', 'data')],
              running=[(Output('bottleneck_dropdown_cav_images', 'disabled'), True, False)],
              prevent_initial_call=True)
def create_figure(n_clicks, bottleneck, concept_bank_dct):
    if (bottleneck is None) and not n_clicks:
        return blank_fig(), 0
    elif n_clicks:
        concept_bank = ConceptBank(concept_bank_dct[bottleneck])
        fig = concept_bank.plot_concepts(num_images=10)
        return fig, 0
    else:
        raise dash.exceptions.PreventUpdate()


if __name__ == '__main__':
    app.run_server(debug=True)
