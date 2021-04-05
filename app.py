import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import pandas as pd
from os import listdir, remove
import pickle
from time import sleep
from Strategy import *
import matplotlib.pyplot as plt

app = dash.Dash(__name__)

# Define the layout.
app.layout = html.Div([

    # Section title
    html.H1("Section 1: Model preparation"),

    html.Div(
        [
            html.Label('Window Size (Moving average of x days.)', style={'padding': 500}),
            dcc.Slider(
                id='size',
                min=7, max=100,
                marks={days: str(days) for days in range(1, 101, 3)},
                value=30),
            #             "Window Size: ",
            #             dcc.Input(id='input-size', value='30', type='text'),
        ],
        # Style it so that the submit button appears beside the input.
        style={'display': 'inline-block'}
    ),
    dcc.RadioItems(
        options=[
            {'label': 'Decision Tree', 'value': 'Decision Tree'},
            {'label': 'Loglinear', 'value': 'Loglinear'},
            {'label': 'KNN', 'value': 'KNN'}
        ],
        id='md',
        value='Decision Tree',
        labelStyle={'display': 'inline-block'}
    ),
    # Submit button:
    html.Button('Submit', id='submit-button', n_clicks=0),
    # Line break
    html.Br(),
    # Div to hold the initial instructions and the updated info once submit is pressed
    html.Div(id='info', children='Enter a your preferred window size and model then press "submit".'),
    html.Div([
        # Candlestick graph goes here:
        dcc.Graph(id='yield-curve'),
        html.Table(id='blotters')
    ]),
    # Another line break
    html.Br(),
    # Section title
    html.H1("Section 2: Suggest a Trade for next day"),
    # Div to confirm what trade was made
    html.Div(id='info2', children='Here will show trade suggestion for next day.'),
    html.Div(id='caution', children='Action type, Trading Symbol, Amount, Order Type, Price(only for limit order)'),
    # Radio items to select buy or sell
    dcc.Input(id='actn', value='BUY', type='text'),
    # Text input for the currency pair to be traded
    dcc.Input(id='symb', value='IVV', type='text'),
    # Numeric input for the trade amount
    dcc.Input(id='amount', value=100, type='number'),
    dcc.Input(id='type', value='MKT', type='text'),
    dcc.Input(id='price', value=400, type='number'),
    # Submit button for the trade
    html.Button('Confirm', id='submit-trade', n_clicks=0),
])


# Callback for what to do when submit-button is pressed
@app.callback(
    [  # there's more than one output here, so you have to use square brackets to pass it in as an array.
        dash.dependencies.Output('info', 'children'),
        dash.dependencies.Output('yield-curve', 'figure'),
        dash.dependencies.Output('blotters', 'children'),
        dash.dependencies.Output('actn', 'value'),
        dash.dependencies.Output('type', 'value'),
        dash.dependencies.Output('price', 'value')
    ],
    dash.dependencies.Input('submit-button', 'n_clicks'),
    dash.dependencies.State('size', 'value'),
    dash.dependencies.State('md', 'value'),
    prevent_initial_call=True
)
def update_yield_curve(n_clicks, value, md):
    model_data, test_data = strategy(value, md)

    df, test = backtest(model_data, test_data)

    if test['Response'][-1] > 0.5:
        ac = 'BUY'
        tp = 'MKT'
    else:
        ac = 'SELL'
        tp = 'LMT'

    fig = go.Figure(
        data=[
            go.Scatter(
                x=df['Date'],
                y=df['Rate of Return(%)']
            )
        ]
    )
    fig.update_layout(title='Back-Test-Yield', yaxis={'hoverformat': '.2%'})
    data = df.dropna(axis=0)
    #     data = data.drop(labels='Rate of Return(%)', axis=1, inplace=True)
    max_rows = 10
    # Return your updated text to currency-output, and the figure to candlestick-graph outputs
    return ('Successfully trained model with window size ' + str(value), fig, html.Table(
        # Header
        [html.Tr([html.Th(col) for col in data.columns])] +
        # Body
        [html.Tr([
            html.Td(data.iloc[i][col]) for col in data.columns
        ]) for i in range(min(len(data), max_rows))]
    ), ac, tp, test['Close'][-1])


@app.callback(
    dash.dependencies.Output('info2', 'children'),
    dash.dependencies.Input('submit-trade', 'n_clicks'),
    dash.dependencies.State('actn', 'value'),
    dash.dependencies.State('price', 'value'),
    prevent_initial_call=True
)
def trade(n_clicks, ac, tp, pc):  # Still don't use n_clicks, but we need the dependency

    trade_order = {'action': ac, 'trade_amt': 100, 'trade_currency': 'IVV', 'price': pc}
    # Dump trade_order as a pickle object to a file connection opened with write-in-binary ("wb") permission:
    with open('trade_order.p', "wb") as f:
        pickle.dump(trade_order, f)
    return msg


# Run it!
if __name__ == '__main__':
    app.run_server(debug=False)