import dash
from dash import html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import dash_table

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Dictionary for model filenames
model_files = {'LR': 'ist_south_tower_real_model_2019_LR.csv',
               'RF': 'ist_south_tower_real_model_2019_RF.csv',
               'NN': 'ist_south_tower_real_model_2019_NN.csv'}

feature_selection_results = {
    'kbest': {
        3: ['time_of_day', 'week_day', 'power_m1'],
        5: ['time_of_day', 'week_day', 'hour', 'solarRad_W/m2', 'power_m1'],
        6: ['time_of_day', 'hdh', 'week_day', 'hour', 'solarRad_W/m2', 'power_m1']
    },
    'rfe': {
        3: ['week_day', 'hour', 'hdh'],
        5: ['time_of_day', 'week_day', 'hour', 'hdh', 'kW_solar_rel'],
        6: ['time_of_day', 'week_day', 'hour', 'hdh', 'temp_C', 'kW_solar_rel']
    },
    'rf': ['time_of_day', 'hour', 'week_day', 'hdh', 'temp_C', 'solarRad_W/m2']
}

# Dictionary for month selection
month_mapping = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4}

# Dictionary for metric values
metrics_data = {
    'LR': np.array([0.1949, 0.0151]),
    'RF': np.array([0.1791, 0.0170]),
    'NN': np.array([0.1897, 0.0371])
}

# Load data
data_lr = pd.read_csv('ist_south_tower_real_model_2019_LR.csv')
data_rf = pd.read_csv('ist_south_tower_real_model_2019_RF.csv')
data_nn = pd.read_csv('ist_south_tower_real_model_2019_NN.csv')

# put 'date' into datetime
data_lr['date'] = pd.to_datetime(data_lr['date'])
data_rf['date'] = pd.to_datetime(data_rf['date'])
data_nn['date'] = pd.to_datetime(data_nn['date'])

# data for EDA
raw_data = pd.read_csv('testData_2019_SouthTower.csv')
raw_data['Date'] = pd.to_datetime(raw_data['Date'])
raw_data = raw_data.rename(columns={'South Tower (kWh)': 'Power_kW'})
cleaned_data = pd.read_csv('IST_South_Tower2019_with_features_test.csv')
cleaned_data['date'] = pd.to_datetime(cleaned_data['date'])

# start object of app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    html.H1('IST\'s South Tower Energy Dashboard - 100278 Vasco Morais', style={'textAlign': 'center'}),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Plotting Real vs Predicted', value='tab-1'),
        dcc.Tab(label='Raw data and EDA', value='tab-2'),
        dcc.Tab(label='Feature Selection', value='tab-3'),  # New tab for Feature Selection
    ]),
    html.Div(id='tabs-content')
])


# Create Plot Function
def create_plot(selected_data, model_name, selected_month):
    return {
        'data': [
            {'x': selected_data['date'], 'y': selected_data['Power_kW'], 'type': 'scatter', 'name': 'Real'},
            {'x': selected_data['date'], 'y': selected_data[f'{model_name}_pred'], 'type': 'scatter',
             'name': 'Prediction'},
        ],
        'layout': {
            'title': f'{model_name} Prediction vs. Real for {selected_month}',
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'Energy Consumption (kWh)'}
        }
    }

# Update Plot Function
@app.callback(Output('model-prediction-plot', 'figure'),
              [Input('model-dropdown', 'value'),
               Input('month-dropdown', 'value')])
def update_plot(model_name, selected_month):
    selected_month_num = month_mapping[selected_month]
    selected_data = None

    if model_name == 'LR':
        selected_data = data_lr[data_lr['date'].dt.month == selected_month_num]
    elif model_name == 'RF':
        selected_data = data_rf[data_rf['date'].dt.month == selected_month_num]
    elif model_name == 'NN':
        selected_data = data_nn[data_nn['date'].dt.month == selected_month_num]

    return create_plot(selected_data, model_name, selected_month)
# Update Metrics Function
@app.callback(Output('model-metrics', 'children'),
              [Input('model-dropdown', 'value')])
def update_metrics(model_name):
    cvRMSE, NMBE = metrics_data[model_name]
    return html.Div([
        dash_table.DataTable(
            columns=[{'name': 'Metric', 'id': 'metric'}, {'name': 'Value', 'id': 'value'}],
            data=[
                {'metric': 'cvRMSE', 'value': cvRMSE},
                {'metric': 'NMBE', 'value': NMBE}
            ],
            style_table={'margin': 'auto'},  # Centering the table
            style_cell={'textAlign': 'center'},  # Centering cell content
            style_data_conditional=[
                {'if': {'column_id': 'metric'}, 'fontWeight': 'bold'},
                {'if': {'column_id': 'metric'},
                 'backgroundColor': 'rgba(255, 0, 0, 0.3)',  # Red background color with 50% opacity
                 'color': 'black'},
                {'if': {'column_id': 'value'},
                 'backgroundColor': 'rgba(0, 0, 255, 0.3)',  # Red background color with 50% opacity
                 'color': 'black'}
                # Making metric names bold
            ]
        )
    ], style={'textAlign': 'center'})  # Centering the div containing the table

# Add a new callback function to render content for the Feature Selection tab
@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H4('Real vs Predicted by model, month with metrics display.'),
            html.Div([
                html.Label('Choose the model to predict power consumption.'),
                dcc.Dropdown(
                    id='model-dropdown',
                    options=[{'label': model, 'value': model} for model in model_files.keys()],
                    value='LR',
                    style={'width': '50%'}
                ),
                html.Label('Choose the month to represent.'),
                dcc.Dropdown(
                    id='month-dropdown',
                    options=[{'label': month, 'value': month} for month in month_mapping.keys()],
                    value='Jan',
                    style={'width': '50%'}
                ),
            ]),
            html.Div([
                dcc.Graph(id='model-prediction-plot')
            ]),
            html.Div(id='model-metrics')
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H4('Displaying Raw Data and EDA'),
            html.Div([
                html.Label('Select feature for EDA:'),
                dcc.Dropdown(
                    id='feature-dropdown',
                    options=[{'label': 'Power Consumption', 'value': 'Power_kW'},
                             {'label': 'Temperature', 'value': 'temp_C'},
                             {'label': 'Relative Humidity', 'value': 'HR'},
                             {'label': 'Solar Irradiation', 'value': 'solarRad_W/m2'},
                             ],
                    value='Power_kW'
                ),
                html.Label('Select visualization:'),
                dcc.Dropdown(
                    id='visualization-dropdown',
                    options=[{'label': 'Time Series', 'value': 'time_series'},
                             {'label': 'Box Plot', 'value': 'box_plot'},
                             {'label': 'Histogram', 'value': 'histogram'}],
                    value='time_series'
                ),
            ]),
            html.Div([
                dcc.Graph(id='eda-plot')
            ])
        ])
    elif tab == 'tab-3':  # Add content for the Feature Selection tab
        return html.Div([
            html.H4('Feature Selection'),
            html.Div([
                html.Label('Select feature selection method:'),
                dcc.Dropdown(
                    id='feature-selection-method',
                    options=[
                        {'label': 'KBest', 'value': 'kbest'},
                        {'label': 'Recursive Feature Elimination (RFE)', 'value': 'rfe'},
                        {'label': 'Random Forest (RF)', 'value': 'rf'}
                    ],
                    value='kbest'
                ),
                html.Div(id='feature-selection-options')  # This will be populated based on the selected method
            ]),
            html.Div(id='feature-selection-table')  # This will display the feature selection results
        ])


# Callback to update EDA plot
@app.callback(Output('eda-plot', 'figure'),
              [Input('feature-dropdown', 'value'),
               Input('visualization-dropdown', 'value')])

def update_eda_plot(feature, visualization):
    # Define the dropdown options list
    feature_dropdown_options = [
        {'label': 'Power Consumption', 'value': 'Power_kW'},
        {'label': 'Temperature', 'value': 'temp_C'},
        {'label': 'Relative Humidity', 'value': 'HR'},
        {'label': 'Solar Irradiation', 'value': 'solarRad_W/m2'}
    ]
    # Retrieve the label associated with the selected feature
    feature_label = None
    for option in feature_dropdown_options:
        if option['value'] == feature:
            feature_label = option['label']
            break

    if not feature_label:
        raise ValueError(f"Label not found for feature '{feature}'")

    # Create plot for raw_data
    raw_trace = {}
    cleaned_trace = {}

    # Choose data to plot based on visualization type
    if visualization == 'time_series':
        raw_trace = {'x': raw_data['Date'], 'y': raw_data[feature], 'type': 'scatter', 'name': 'Raw Data'}
        cleaned_trace = {'x': cleaned_data['date'], 'y': cleaned_data[feature], 'type': 'scatter', 'name': 'Cleaned Data'}
        layout = {'title': f'{feature_label} Time Series', 'xaxis': {'title': 'Date'}, 'yaxis': {'title': feature_label}}
    elif visualization == 'box_plot':
        raw_trace = {'y': raw_data[feature], 'type': 'box', 'name': 'Raw Data'}
        cleaned_trace = {'y': cleaned_data[feature], 'type': 'box', 'name': 'Cleaned Data'}
        layout = {'title': f'{feature_label} Box Plot', 'yaxis': {'title': feature_label}}
    else:  # histogram
        raw_trace = {'x': raw_data[feature], 'type': 'histogram', 'name': 'Raw Data'}
        cleaned_trace = {'x': cleaned_data[feature], 'type': 'histogram', 'name': 'Cleaned Data'}
        layout = {'title': f'{feature_label} Histogram', 'xaxis': {'title': feature_label}, 'yaxis': {'title': 'Frequency'}}

    return {
        'data': [raw_trace, cleaned_trace],
        'layout': layout
    }

# Callback to populate options based on selected feature selection method
@app.callback(Output('feature-selection-options', 'children'),
              [Input('feature-selection-method', 'value')])
def update_feature_selection_options(method):
    if method in ['kbest', 'rfe']:
        return html.Div([
            html.Label('Select number of features (k):'),
            dcc.Dropdown(
                id='feature-selection-k',
                options=[{'label': str(i), 'value': i} for i in [3, 5, 6]],  # Options for k
                value=3 if method == 'kbest' else 3,  # Default value
                style={'width': '50%'}
            )
        ])
    else:
        return None  # No additional options needed for Random Forest method

# Callback to generate and display the feature selection table
@app.callback(Output('feature-selection-table', 'children'),
              [Input('feature-selection-method', 'value'),
               Input('feature-selection-k', 'value')])
def update_feature_selection_table(method, k):
    if method in ['kbest', 'rfe']:  # KBest or RFE methods
        features = feature_selection_results[method][k]
    else:  # Random Forest method
        features = feature_selection_results[method]

    # Create a Dash DataTable to display the features
    table = dash_table.DataTable(
        columns=[{'name': 'Rank', 'id': 'rank'},
                 {'name': 'Feature', 'id': 'feature'}],
        data=[{'rank': i + 1, 'feature': features[i]} for i in range(len(features))],
        style_table={'margin': 'auto'},  # Centering the table
        style_cell={'textAlign': 'center'},  # Centering cell content
        style_data_conditional=[
            {'if': {'column_id': 'rank'}, 'fontWeight': 'bold'},
            {'if': {'column_id': 'rank'},
             'backgroundColor': 'rgba(255, 0, 0, 0.3)',  # Red background color with 50% opacity
             'color': 'black'},
            {'if': {'column_id': 'feature'},
             'backgroundColor': 'rgba(0, 0, 255, 0.3)',  # Red background color with 50% opacity
             'color': 'black'}
            # Making rank numbers bold
        ]
    )

    return html.Div([
        html.H3('Selected Features'),
        table
    ])

if __name__ == '__main__':
    app.run_server(debug=False)