import os
import pytz
from pytz import timezone
import pandas as pd
from abc import ABCMeta
from typing import Optional, List, Dict, Tuple
from functools import lru_cache
from datetime import datetime
from datetime import timedelta

import plotly.express as px

from jupyter_dash import JupyterDash
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output


def datetime_from_str(dt: str) -> datetime:
    return datetime.strptime(dt, '%Y-%m-%d %H:%M:%S %z').astimezone(timezone('UTC'))

def datetime_to_str(dt: datetime) -> str:
    return dt.strftime('%Y-%m-%d %H:%M:%S %z')

time_range = (
    datetime_from_str('2021-12-06 12:00:00 +0000'),
    datetime_from_str('2021-12-06 12:30:00 +0000'),
)

store_type = 'file'
arctic_host = '192.168.16.64'

class Store(metaclass=ABCMeta):
    def read(self, symbol: str, time_range: Optional[Tuple[datetime,datetime]] = None) -> pd.DataFrame:
        raise NotImplemented

    def list_symbols(self) -> List[str]:
        raise NotImplemented
        
    def max_time_range(self, symbol: str) -> Tuple[datetime, datetime]:
        raise NotImplemented

class ArcticStore(Store):
    def __init__(self, lib: str, arctic_host: str = '127.0.0.1'):
        try:
            import arctic
        except ImportError:
            raise ImportError("arctic is not installed")
            
        conn = arctic.Arctic(arctic_host)
        self.lib = conn[lib]

    @lru_cache(maxsize=2)
    def read(self, symbol: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> pd.DataFrame:
        if not time_range:
            return self.lib.read(symbol)

        return self.lib.read(symbol, date_range=arctic.date.DateRange(*time_range))

    def list_symbols(self) -> List[str]:
        return self.lib.list_symbols()
    
    def max_time_range(self, symbol: str) -> Tuple[datetime, datetime]:
        return (
            self.lib.min_date(symbol).astimezone(pytz.utc),
            self.lib.max_date(symbol).astimezone(pytz.utc),
        )

class FileStore(Store):
    def __init__(self, files: Dict[str, str]):
        data = {}
        for symbol, path in files.items():
            data[symbol] = pd.read_parquet(path)
        self.data = data

    # @lru_cache(maxsize=2)
    def read(self, symbol: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> pd.DataFrame:
        if not time_range:
            return self.data[symbol]
        return self.data[symbol][time_range[0]:time_range[1]]

    def list_symbols(self) -> List[str]:
        return list(self.data.keys())
                
    def max_time_range(self, symbol: str) -> Tuple[datetime, datetime]:
        return (
            self.data[symbol].head(1).index[0].to_pydatetime(),
            self.data[symbol].tail(1).index[0].to_pydatetime(),
        )

store = None

store_type = store_type or os.environ.get('STORE', 'file')

if store_type == 'arctic':
    try:
        import arctic
    except ImportError:
        raise ImportError("arctic is not installed")

    store = ArcticStore(
        lib='bidask',
        arctic_host=arctic_host or os.environ.get('ARCTIC_HOST', '127.0.0.1'),
    )
elif store_type == 'file':
    store = FileStore(
        files={
            # S3
            # 'ETH-USDT-PERP': 's3://nacre-public-data/ETH-USDT-PERP.parquet',
            # 'FIL-USDT-PERP': 's3://nacre-public-data/FIL-USDT-PERP.parquet',

            # HTTP
            # 'ETH-USDT-PERP': 'https://nacre-public-data.s3.amazonaws.com/ETH-USDT-PERP.parquet',
            # 'FIL-USDT-PERP': 'https://nacre-public-data.s3.amazonaws.com/FIL-USDT-PERP.parquet',

            # Local
            'ETH-USDT-PERP': './ETH-USDT-PERP.parquet',
            'FIL-USDT-PERP': './FIL-USDT-PERP.parquet',
        }
    )

symbols = store.list_symbols()
max_time_range = store.max_time_range(symbols[0])
df = store.read(symbols[0], time_range=time_range)

all_bid = list(filter(lambda c: c.startswith('bids_'), df.columns))
all_bid.remove('bids_DYDX')
all_ask = list(filter(lambda c: c.startswith('asks_'), df.columns))
all_ask.remove('asks_DYDX')
all_side = all_bid + all_ask

app = JupyterDash(__name__)
server = app.server

app.layout = html.Div([
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='symbol',
                options=[{'label': i, 'value': i} for i in symbols],
                value=symbols[0],
            )
        ],
        style={'width': '30%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Input(
                id="start_time",
                type="text",
                value=datetime_to_str(time_range[0]),
                placeholder=datetime_to_str(max_time_range[0]),
                debounce=True,
            )
        ],
        style={'width': '30%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Input(
                id="end_time",
                type="text",
                value=datetime_to_str(time_range[1]),
                placeholder=datetime_to_str(max_time_range[1]),
                debounce=True,
            )
        ],
        style={'width': '30%', 'display': 'inline-block'}),
    ]),

    html.Div([
        dcc.Graph(
            id='all_df',
            figure={
                "layout": {
                    "title": "All Exchange Bid Ask",
                    # "height": 1000,  # px
                },
            }
        )
    ]),


    html.Div([
        html.Div([
            dcc.Dropdown(
                id='bid',
                options=[{'label': i.replace('_', ' '), 'value': i} for i in all_side],
                value=all_bid[0],
            )
        ],
        style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='ask',
                options=[{'label': i.replace('_', ' '), 'value': i} for i in all_side],
                value=all_ask[0],
            )
        ],
        style={'width': '49%', 'float': 'right', 'display': 'inline-block'}),

        html.Div([
            dcc.Graph(
                id='spread',
                figure={
                    "layout": {
                        "title": "Selected Spread",
                        # "height": 1000,  # px
                    },
                }
            ),
        ]),
    ]),

    html.Div([
        html.Div([
            dcc.Dropdown(
                id="max",
                options=[{'label': col.replace('_', ' '), 'value': col} for col in all_side],
                value=all_bid,
                multi=True,
                placeholder="Select max",
            ),
        ], style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id="min",
                options=[{'label': col.replace('_', ' '), 'value': col} for col in all_side],
                value=all_bid,
                multi=True,
                placeholder="Select min",
            ),
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'}),

        html.Div([
            dcc.Graph(
                id='min_max',
                figure={
                    "layout": {
                        "title": "Max Min Spread",
                        # "height": 1000,  # px
                    },
                }
            )
        ]),
    ]),
])

@app.callback(
    Output('all_df', 'figure'),
    [Input('symbol', 'value'),
    Input('start_time', 'value'),
    Input('end_time', 'value'),
    ])
def update_bid_ask(symbol, start_time, end_time):
    dff = store.read(symbol, (datetime_from_str(start_time), datetime_from_str(end_time)))
    return px.line(dff, x=dff.index, y=all_side, title='All Exchange Bid Ask', line_shape="hv")

@app.callback(
    Output('spread', 'figure'),
    [Input('symbol', 'value'),
    Input('start_time', 'value'),
    Input('end_time', 'value'),
    Input('bid', 'value'),
    Input('ask', 'value')])
def update_spread(symbol, start_time, end_time, base, quote):
    dff = store.read(symbol, (datetime_from_str(start_time), datetime_from_str(end_time)))
    dff['spread'] = (dff[base] - dff[quote]) / dff[base] * 10_000
    return px.line(dff, x=dff.index, y=['spread'], title=f'{base}-{quote} Spread BPS')

@app.callback(
    Output('min_max', 'figure'),
    [Input('symbol', 'value'),
    Input('start_time', 'value'),
    Input('end_time', 'value'),
    Input('max', 'value'),
    Input('min', 'value')])
def update_min_max(symbol, start_time, end_time, max, min):
    dff = store.read(symbol, (datetime_from_str(start_time), datetime_from_str(end_time)))
    dff['max'] = dff[max].max(axis=1)
    dff['min'] = dff[min].min(axis=1)
    dff['spread'] = (dff['max'] - dff['min']) / dff['max'] * 10_000
    return px.line(dff, x=dff.index, y=['spread'], title=f'Max {max}- Min{min} Spread BPS')


if __name__ == '__main__':
    app.run_server(debug=True)
