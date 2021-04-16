import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import pandas as pd
from os import listdir, remove
from dash_table import DataTable, FormatTemplate
import pickle
from time import sleep
import json
from Strategy import *
import matplotlib.pyplot as plt

app = dash.Dash(__name__)

# Define the layout.
app.layout = html.Div([

    html.H1(
        'Trading Strategy Dash',
        style={'display': 'block', 'text-align': 'center'}
    ),
    html.Div([
        html.H2('Strategy'),
        html.P('This app explores a simple strategy that works as follows:'),
        html.Ol([
            html.Li([
                "While the market is closed, retrieve the past N days' " + \
                "worth of data for:",
                html.Ul([
                    html.Li("IVV: daily open, close, yield(calculated)"),
                    html.Li(
                        "US Treasury bond yield for 5,10 and 30 years."
                    ),
                    html.Li("Price change of oil(EIA)."),
                ])
            ]),
            html.Li([
                'Fit different machine learning models using features listed below to predict' + \
                " whether the IVV would have yield greater than 0 next day:",
                html.Ul([
                    html.Li(
                        'the output (y): the yield of IVV next day would be greater than 0(1 for True and 0 for false)'),
                    html.Li(
                        "the input (x): the yield of IVV on previous N days, IVV' moving average, yield of bonds, yield of EIA"),
                    html.Li('the models: KNN, Dicision Tree, Loglinear Classifier')
                ])
            ]),
            html.Li(
                "After the model is being trained, we use the model to predict each day's IVV yield," + \
                'we use the model result of last 30 days to do a back-test. '
            ),
            html.Li(
                'If the predicted output is 1, which means the IVV would have postive return next day. We submit one trade:'),
            html.Ul([
                html.Li(
                    'A market order to BUY 100 shares of IVV, which ' + \
                    'fills at open price the next trading day.'
                )
            ]),
            html.Li(
                'If the predicted output is 0, which means the IVV would have negative return next day. We submit one trade:'),
            html.Ul([
                html.Li(
                    'A limit order to SELL all shares of IVV, which ' + \
                    'fills at the close price of the last trading day.'
                )
            ])
        ])
    ],
        style={'display': 'inline-block', 'width': '50%'}
    ),
    html.Div([
        html.H2('Data Note & Disclaimer'),
        html.P(
            'This Dash app makes use of yahoo finance data to fit the model ' + \
            "using pandas_datareader package to read yahoo finance's stock and bond data." + \
            'The original data contains close, open, low, high price and we can use them to calculate the yield.' + \
            "These are all the work we done in fetching data and preprocessing it."
        ),
        html.H2('Parameters'),
        html.Ol([
            html.Li(
                "N: number of days of the moving average of IVV yield, which would be added as a feature into the model"
            ),
            html.Li(
                "model: Which specific machine learning model would be used in training dataset."
            )
        ]),
        html.H2('Window Size (Moving average of x days.)'),
        html.Br(),
        dcc.Slider(
            id='size',
            min=7, max=100,
            marks={days: str(days) for days in range(1, 101, 3)},
            value=30),
        html.Br(),
        html.H2('Model Selection'),
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
        html.Br(),
        # Submit button:
        html.Button('Submit', id='submit-button', n_clicks=0),
        html.Div(id='info', children='Enter a your preferred window size and model then press "submit".'),
    ],
        style={
            'display': 'inline-block', 'width': '50%', 'vertical-align': 'top'
        }
    ),
    # Line break
    html.Br(),
    html.Div([
        html.H2(
            'Trade Ledger',
            style={
                'display': 'inline-block', 'text-align': 'center',
                'width': '100%'
            }
        ),
        DataTable(
            id='ledger',
            fixed_rows={'headers': True},
            style_cell={'textAlign': 'center'},
            style_table={'height': '300px', 'overflowY': 'auto'}
        )
    ]),
    html.Div([
        html.H2(
            'Blotters',
            style={
                'display': 'inline-block', 'text-align': 'center',
                'width': '100%'
            }
        ),
        DataTable(
            id='blotters',
            fixed_rows={'headers': True},
            style_cell={'textAlign': 'center'},
            style_table={'height': '300px', 'overflowY': 'auto'}
        )
    ]),

    html.Div([
        html.H2('Yield-curve'),
        # Candlestick graph goes here:
        dcc.Graph(id='yield-curve')
    ]),
    # Another line break
    html.Br(),
    # Section title
    html.H1("Suggest a Trade for next day"),
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
        dash.dependencies.Output('blotters', 'columns'),
        dash.dependencies.Output('blotters', 'data'),
        dash.dependencies.Output('ledger', 'columns'),
        dash.dependencies.Output('ledger', 'data'),
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

    blotter, ledger, test, sharp = backtest(model_data, test_data)

    if test['Response'][-1] > 0.5:
        ac = 'BUY'
        tp = 'MKT'
    else:
        ac = 'SELL'
        tp = 'LMT'

    fig = go.Figure(
        data=[
            go.Scatter(
                x=ledger['Date'],
                y=ledger['Revenue'],
                name='Asset Return'),
            go.Scatter(
                x=ledger['Date'],
                y=ledger['IVV Yield'],
                name='IVV Return')
        ]
    )
    fig.update_layout(title='Back-Test-Yield', yaxis={'hoverformat': '.2%'})
    #     data = df.dropna(axis=0)
    blotter.reset_index()
    blotter = blotter.to_dict('records')
    blotter_columns = [
        dict(id='Date', name='Date'),
        dict(id='ID', name='ID'),
        dict(id='Type', name='order type'),
        dict(id='actn', name='Action'),
        dict(
            id='Price', name='Order Price', type='numeric',
            format=FormatTemplate.money(2)
        ),
        dict(id='size', name='Order Amount', type='numeric'
             ),
        dict(id='symb', name='Symb')
    ]
    ledger = ledger.to_dict('records')
    ledger_columns = [
        dict(id='Date', name='Date'),
        dict(id='position', name='position'),
        dict(id='Cash', name='Cash'),
        dict(
            id='Stock Value', name='Stock Value', type='numeric',
            format=FormatTemplate.money(2)
        ),
        dict(
            id='Total Value', name='Total Value', type='numeric',
            format=FormatTemplate.money(2)
        ),
        dict(id='Revenue', name='Revenue', type='numeric',
             format=FormatTemplate.percentage(2)
             ),
        dict(id='IVV Yield', name='IVV Yield', type='numeric',
             format=FormatTemplate.percentage(2)
             )
    ]
    return (
    'Successfully trained model with window size ' + str(value), fig, blotter_columns, blotter, ledger_columns, ledger,
    ac, tp, test['Close'][-1])


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
