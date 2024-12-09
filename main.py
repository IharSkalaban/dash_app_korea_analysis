import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import io
import base64
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Korea Visualization Dashboard"
server = app.server

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    # Хранилище для данных сессии
    dcc.Store(id='session-data', storage_type='session'),
    html.Div(id='page-content')
])

# Константы для стилей
HEADER_STYLE = {
    'color': '#2a3f5f',
    'fontWeight': 'bold',
    'fontFamily': 'Arial, sans-serif',
    'textAlign': 'center',
    'fontSize': '28px',
    'marginBottom': '20px'
}
CONTAINER_STYLE = {
    'backgroundColor': '#f0f4f8',
    'padding': '15px',
    'borderRadius': '10px',
    'boxShadow': '0px 4px 8px rgba(0, 0, 0, 0.1)',
    'margin': '20px'
}




navigation_menu = html.Div([
    dcc.Link('Data Upload', href='/data-upload', style={
        'marginRight': '20px', 'padding': '10px', 'border': '2px solid orange', 'borderRadius': '5px',
        'textDecoration': 'none'
    }),
    dcc.Link('General Results', href='/general-results', style={'marginRight': '20px'}),
    dcc.Link('Detailed Account Information', href='/detailed-account-info', style={'marginRight': '20px'}),
    dcc.Link('Assessment of Stability of Results', href='/stability-assessment', style={'marginRight': '20px'}),
    dcc.Link('Sessions', href='/sessions')
], style={'textAlign': 'center', 'marginTop': '20px'})

data_upload_layout = html.Div([
    navigation_menu,
    html.H1("Data Upload"),
    dcc.Upload(
        id="upload-data",
        children=html.Div(["Drag and Drop or ", html.A("Select a File  *.csv format ")]),
        style={
            "width": "100%",
            "height": "60px",
            "lineHeight": "60px",
            "borderWidth": "1px",
            "borderStyle": "dashed",
            "borderRadius": "5px",
            "textAlign": "center",
            "margin": "10px",
        },
        multiple=False,
    ),
    html.Div(id="upload-status")
])

general_results_layout = html.Div([
    navigation_menu,
    html.H1("General Results"),
    html.Div(id="output-graphs")
])


# Функция для обработки данных
def preprocess_data(data):
    rename_map = {
        'rb': 'rakeback',
        'daily_rb': 'rb',
        'result_day': 'total',
        'account_name': 'Player',
        'day': 'date',
        'win_loss': 'win/Loss',
        '%RakeBack.1': 'rb',
        'Rake': 'rake',
    }

    # Переименование столбцов
    data.rename(columns=lambda col: rename_map[col] if col in rename_map else col, inplace=True)

    # Очистка данных
    numeric_cols = ['win/Loss', 'rb', 'total', 'rake', 'win/Loss.1', '%RakeBack.1']
    for col in numeric_cols:
        if col in data.columns:
            # Удаление неразрывных пробелов и других символов, затем приведение к числовому типу
            data[col] = data[col].replace(r'[^\d.-]', '', regex=True).astype(float, errors='ignore')

    # Преобразование даты в формат datetime
    if 'date' in data.columns:
        sample_date = data['date'].iloc[0]

        if '/' in sample_date:
            # Если дата разделена символом '/', предполагаем формат MM/DD/YYYY
            fmt = '%m/%d/%Y'
        elif '-' in sample_date:
            # Если дата разделена символом '-', предполагаем ISO формат YYYY-MM-DD
            fmt = '%Y-%m-%d'
        elif '.' in sample_date:
            fmt = '%d/%m/%Y'

        else:
            raise ValueError("Unsupported date format. Ensure date column uses '/' or '-' as separators.")

        try:
            data['date'] = pd.to_datetime(data['date'], format=fmt, errors='coerce')
        except Exception:
            raise ValueError("Could not convert 'date' to datetime format. Check the input data.")

    # Проверка обязательных столбцов
    required_columns = ['rb', 'total', 'Player', 'date', 'win/Loss']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return data






# Генерация графика: Ежедневный и Кумулятивный анализ
def generate_daily_analysis_graph(data):
    # Создание графика с ранее подготовленными данными
    daily_summary = data.groupby('date').agg({
        'win/Loss': 'sum',
        'rb': 'sum'
    }).reset_index()

    # Создаем новую метрику total, которая агрегирует Win/Loss и RakeBack
    daily_summary['total'] = daily_summary['win/Loss'] + daily_summary['rb']

    # Создаем cumulative метрики для анализа
    daily_summary['cumulative_total'] = daily_summary['total'].cumsum()
    daily_summary['cumulative_win_loss'] = daily_summary['win/Loss'].cumsum()
    daily_summary['cumulative_rakeback'] = daily_summary['rb'].cumsum()

    fig = go.Figure()

    # Столбцы для Total (используется для вкладки Total)
    fig.add_trace(go.Bar(
        x=daily_summary['date'],
        y=daily_summary['total'],
        name='Daily Total (Win/Loss + RakeBack)',
        marker_color=daily_summary['total'].apply(lambda x: 'green' if x > 0 else 'red'),
        # Красим в зависимости от значения
        visible=False  # Невидимый по умолчанию, используется для вкладки Total
    ))

    # Столбцы для Win/Loss (используется для вкладки Win/Loss + RakeBack)
    fig.add_trace(go.Bar(
        x=daily_summary['date'],
        y=daily_summary['win/Loss'],
        name='Daily Win/Loss',
        marker_color='red',
        visible=False  # Невидимый по умолчанию
    ))

    # Столбцы для RakeBack (используется для вкладки Win/Loss + RakeBack)
    fig.add_trace(go.Bar(
        x=daily_summary['date'],
        y=daily_summary['rb'],
        name='Daily RakeBack',
        marker_color='orange',
        base=0,  # Строим RakeBack всегда от нуля
        visible=False  # Невидимый по умолчанию
    ))

    # Линии для cumulative данных
    fig.add_trace(go.Scatter(
        x=daily_summary['date'],
        y=daily_summary['cumulative_total'] / 1e6,  # Преобразование в миллионы
        mode='lines',
        name='Cumulative Total',
        line=dict(color='green', width=2),
        visible=False  # Невидимый по умолчанию
    ))

    fig.add_trace(go.Scatter(
        x=daily_summary['date'],
        y=daily_summary['cumulative_win_loss'] / 1e6,  # Преобразование в миллионы
        mode='lines',
        name='Cumulative Win/Loss',
        line=dict(color='red', width=2),
        visible=False  # Невидимый по умолчанию
    ))

    fig.add_trace(go.Scatter(
        x=daily_summary['date'],
        y=daily_summary['cumulative_rakeback'] / 1e6,  # Преобразование в миллионы
        mode='lines',
        name='Cumulative RakeBack',
        line=dict(color='orange', width=2),
        visible=False  # Невидимый по умолчанию
    ))

    # Настройка графика
    fig.update_layout(
        title='Daily and Cumulative Analysis: Total, Win/Loss, and RakeBack',
        xaxis_title='Date',
        yaxis_title='Value',
        xaxis=dict(
            tickangle=45,  # Даты под углом 45 градусов
            tickformat="%Y-%m-%d",
            automargin=True
        ),
        bargap=0.2,  # Пространство между столбцами
        template='plotly_white',
        width=1800,  # Растягиваем график по оси Ox
        height=600,  # Высота графика
        barmode='stack'  # Используем "stack" для суммирования значений столбцов
    )

    # Добавление кнопок для переключения между видимыми значениями и размещение их под легендой
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"visible": [False, False, False, True, True, True]}],
                        label="Cumulative ",
                        method="update"
                    ),

                    dict(
                        args=[{"visible": [True, False, False, False, False, False, False]}],
                        label="Daily Result",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [False, True, True, False, False, False, False]}],
                        label="Win/Loss & RakeBack",
                        method="update"
                    )

                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.15,
                xanchor="left",
                y=1.0,
                yanchor="top"
            ),
        ],
        legend=dict(
            x=1.02,  # Располагаем легенду справа за пределами графика
            y=1,
            traceorder="normal",
            font=dict(
                size=12,
            ),
            bgcolor="LightSteelBlue",
            bordercolor="Black",
            borderwidth=0
        )
    )

    return fig


# Генерация графика: Анализ вклада аккаунтов
def generate_account_analysis_graph(data):
    account_summary = data.groupby('Player').agg(
        total=('total', 'sum'),
        num_sessions=('date', 'count'),
        avg_win_loss=('win/Loss', 'mean'),
        avg_rb=('rb', 'mean'),
        first_session=('date', 'min'),
        last_session=('date', 'max')
    ).reset_index()

    total_sum = account_summary['total'].sum()
    account_summary['percent_total'] = (account_summary['total'] / total_sum) * 100
    account_summary['t_months'] = (account_summary['last_session'] - account_summary['first_session']).dt.days / 30

    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        'Average Account Contribution Analysis', 'Cumulative Account Contribution Over Time'))

    # Бар-чарт вклада аккаунтов
    fig.add_trace(go.Bar(
        x=account_summary['Player'],
        y=account_summary['total'],
        hovertext=[f"Contribution: {row['percent_total']:.2f}%<br>"
                   f"Sessions: {row['num_sessions']}<br>"
                   f"Avg Win/Loss: {row['avg_win_loss']:.0f}<br>"
                   f"Avg RB: {row['avg_rb']:.0f}<br>"
                   f"T (months): {row['t_months']:.1f}"
                   for _, row in account_summary.iterrows()],
        hoverinfo="text",
        marker_color=px.colors.qualitative.Set1[:len(account_summary)]
    ), row=1, col=1)

    # Линии cumulative contribution для каждого аккаунта
    for player in data['Player'].unique():
        player_data = data[data['Player'] == player].sort_values(by='date')
        player_data['cumulative_total'] = player_data['total'].cumsum()
        fig.add_trace(go.Scatter(
            x=player_data['date'],
            y=player_data['cumulative_total'],
            mode='lines',
            name=player,
            line=dict(color=px.colors.qualitative.Set1[data['Player'].unique().tolist().index(player)]),
            hoverinfo='name+y'
        ), row=1, col=2)

    fig.update_layout(
        template='plotly_white',
        width=1800,
        height=600
    )

    return fig


# Генерация графика : Анализ Средних Дневных значений Win/Loss, Total
def generate_average_daily_value_graph(data):
    # Сгруппировать данные по аккаунтам
    account_summary = data.groupby('Player').agg(
        total=('total', 'sum'),
        num_sessions=('date', 'count'),
        avg_win_loss=('win/Loss', 'mean'),
        avg_rb=('rb', 'mean'),
        first_session=('date', 'min'),
        last_session=('date', 'max')
    ).reset_index()

    # Рассчитать % вклада от общего Total
    total_sum = account_summary['total'].sum()
    account_summary['percent_total'] = (account_summary['total'] / total_sum) * 100
    # Рассчитать время работы аккаунта в месяцах
    account_summary['t_months'] = (account_summary['last_session'] - account_summary['first_session']).dt.days / 30

    # Добавить агрегирующую строку для всех аккаунтов
    aggregate_row = pd.DataFrame([{
        'Player': 'Total',
        'total': total_sum,
        'num_sessions': data['date'].count(),
        'avg_win_loss': data['win/Loss'].mean(),
        'avg_rb': data['rb'].mean(),
        'percent_total': 100,
        't_months': (data['date'].max() - data['date'].min()).days / 30
    }])
    account_summary = pd.concat([account_summary, aggregate_row], ignore_index=True)

    # Настройка цветов
    account_summary['color'] = account_summary['total'].apply(lambda x: 'green' if x > 0 else 'red')
    account_summary.loc[account_summary['Player'] == 'Total', 'color'] = 'limegreen'

    # Создание subplot для размещения двух новых графиков рядом
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        'Average Daily Win/Loss for Each Account', 'Average Daily Total for Each Account'))

    # Построение первого графика - Средний ежедневный Win/Loss для каждого игрока
    avg_daily_win_loss = data.groupby(['Player', 'date']).agg(
        daily_win_loss=('win/Loss', 'mean')
    ).reset_index()
    avg_win_loss_summary = avg_daily_win_loss.groupby('Player').agg(
        avg_daily_win_loss=('daily_win_loss', 'mean')
    ).reset_index()

    fig.add_trace(go.Bar(
        x=avg_win_loss_summary['Player'],
        y=avg_win_loss_summary['avg_daily_win_loss'],
        marker_color=px.colors.qualitative.Set1[:len(avg_win_loss_summary)],
        hovertext=[f"Avg Daily Win/Loss: {row['avg_daily_win_loss'] / 1000:.2f}K"
                   for _, row in avg_win_loss_summary.iterrows()],
        hoverinfo="text"
    ), row=1, col=1)

    # Построение второго графика - Средний ежедневный Total для каждого игрока
    avg_daily_total = data.groupby(['Player', 'date']).agg(
        daily_total=('total', 'mean')
    ).reset_index()
    avg_total_summary = avg_daily_total.groupby('Player').agg(
        avg_daily_total=('daily_total', 'mean')
    ).reset_index()

    fig.add_trace(go.Bar(
        x=avg_total_summary['Player'],
        y=avg_total_summary['avg_daily_total'],
        marker_color=px.colors.qualitative.Set1[:len(avg_total_summary)],
        hovertext=[f"Avg Daily Total: {row['avg_daily_total'] / 1000:.2f}K"
                   for _, row in avg_total_summary.iterrows()],
        hoverinfo="text"
    ), row=1, col=2)

    # Настройка общих параметров графика
    fig.update_layout(
        template='plotly_white',
        width=1800,  # Уменьшаем ширину графика для лучшего размещения двух графиков рядом
        height=600,  # Уменьшаем высоту графика
        showlegend=False,

    )

    # Настройка осей
    fig.update_xaxes(title_text='Player', row=1, col=1, tickangle=45, automargin=True)
    fig.update_yaxes(title_text='Avg Daily Win/Loss', row=1, col=1)
    fig.update_xaxes(title_text='Player', row=1, col=2, tickangle=45, automargin=True)
    fig.update_yaxes(title_text='Avg Daily Total', row=1, col=2)

    return fig


# Генерация графика : Cравнение влияния rb и win/loss во времени

def generate_rb_impact_comparison(data):
    # Сгруппировать данные по дате, рассчитывая сумму Win/Loss и RB
    win_loss_rb_data = data.groupby('date').agg(
        win_loss_sum=('win/Loss', 'sum'),
        rb_sum=('rb', 'sum')
    ).reset_index()

    # Рассчитать кумулятивную сумму для Win/Loss и RB
    win_loss_rb_data['win_loss_cumsum'] = win_loss_rb_data['win_loss_sum'].cumsum()
    win_loss_rb_data['rb_cumsum'] = win_loss_rb_data['rb_sum'].cumsum()

    # Рассчитать процентное соотношение кумулятивной суммы Win/Loss и RB для каждой даты
    win_loss_rb_data['win_loss_pct'] = (win_loss_rb_data['win_loss_cumsum'] / (
            win_loss_rb_data['win_loss_cumsum'] + win_loss_rb_data['rb_cumsum'])) * 100
    win_loss_rb_data['rb_pct'] = (win_loss_rb_data['rb_cumsum'] / (
            win_loss_rb_data['win_loss_cumsum'] + win_loss_rb_data['rb_cumsum'])) * 100

    # Построение stacked area chart для сравнения Win/Loss и RB
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=win_loss_rb_data['date'],
        y=win_loss_rb_data['win_loss_pct'],
        mode='lines',
        name='Win/Loss % (Total)',
        stackgroup='one',
        line=dict(width=0.5),
        fillcolor='blue',
        visible=True
    ))
    fig.add_trace(go.Scatter(
        x=win_loss_rb_data['date'],
        y=win_loss_rb_data['rb_pct'],
        mode='lines',
        name='RB % (Total)',
        stackgroup='one',
        line=dict(width=0.5),
        fillcolor='green',
        visible=True
    ))

    # Добавление данных для каждого игрока
    for player in data['Player'].unique():
        player_data = data[data['Player'] == player].copy()

        # Расчет кумулятивной суммы для Win/Loss и RB
        player_data['win_loss_cumsum'] = player_data['win/Loss'].cumsum()
        player_data['rb_cumsum'] = player_data['rb'].cumsum()

        # Рассчитать процентное соотношение кумулятивной суммы Win/Loss и RB для каждой даты
        player_data['win_loss_pct'] = (player_data['win_loss_cumsum'] / (
                player_data['win_loss_cumsum'] + player_data['rb_cumsum'])) * 100
        player_data['rb_pct'] = (player_data['rb_cumsum'] / (
                player_data['win_loss_cumsum'] + player_data['rb_cumsum'])) * 100

        # Добавление stacked area chart для сравнения Win/Loss и RB для каждого игрока
        fig.add_trace(go.Scatter(
            x=player_data['date'],
            y=player_data['win_loss_pct'],
            mode='lines',
            name=f'{player} Win/Loss %',
            stackgroup='one',
            line=dict(width=0.5),
            fillcolor='blue',
            visible=False
        ))
        fig.add_trace(go.Scatter(
            x=player_data['date'],
            y=player_data['rb_pct'],
            mode='lines',
            name=f'{player} RB %',
            stackgroup='one',
            line=dict(width=0.5),
            fillcolor='green',
            visible=False
        ))

    # Настройка кнопок для переключения между общим обзором и каждым игроком
    buttons = [
        dict(
            args=[{"visible": [True, True] + [False] * (2 * len(data['Player'].unique()))}],
            label="Total Overview",
            method="update"
        )
    ]

    # Добавление кнопок для каждого игрока
    for i, player in enumerate(data['Player'].unique()):
        visible_array = [False, False] + [False] * (2 * len(data['Player'].unique()))
        visible_array[2 + 2 * i] = True
        visible_array[3 + 2 * i] = True
        buttons.append(
            dict(
                args=[{"visible": visible_array}],
                label=f"{player} Overview",
                method="update"
            )
        )

    # Настройка графика с добавлением кнопок переключения
    fig.update_layout(
        title='Win/Loss vs RB Over Time (in %)',
        xaxis_title='Date',
        yaxis_title='Percentage',
        yaxis=dict(ticksuffix='%', range=[0, 100]),
        template='plotly_white',
        width=1000,
        height=500,
        annotations=[
            dict(
                x=0.5,
                y=1.15,
                xref='paper',
                yref='paper',
                font=dict(size=12),
                align='center'
            )
        ],
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=buttons,
                pad={"r": 10, "t": 30},
                showactive=True,
                x=0.15,
                xanchor="left",
                y=1.1,
                yanchor="top"
            )
        ]
    )
    return fig






@app.callback(
    Output("session-data", "data"),
    Output("upload-status", "children"),
    Input("upload-data", "contents")
)

def update_data_upload(contents):
    if contents is None:
        return dash.no_update, "Upload a file to proceed."
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        data = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

        # Здесь можно применить preprocess_data если нужно
        data = preprocess_data(data)

        # Сохраняем данные в session
        # Преобразуем DF в словарь, чтобы хранить его в Store
        return data.to_dict('records'), "File uploaded successfully!"
    except ValueError as e:
        return dash.no_update, f"Error processing file: {e}"

@app.callback(
    Output("output-graphs", "children"),
    Input("url", "pathname"),
    State("session-data", "data")  # Получаем данные из сессии
)
def update_general_results_output(pathname, session_data):
    if pathname == '/general-results' and session_data is not None:
        # Преобразуем данные из сессии обратно в DataFrame
        df = pd.DataFrame(session_data)
        print("Columns before second preprocess:", df.columns)

        sample_date = df['date'].iloc[0]
        if '/' in sample_date:
            parts = sample_date.split('/')
            if int(parts[0]) > 12:
                fmt = '%d/%m/%Y'
            else:
                fmt = '%m/%d/%Y'
        else:
            # Предположим формат ISO 'YYYY-MM-DDTHH:MM:SS'
            fmt = '%Y-%m-%dT%H:%M:%S'

        df['date'] = pd.to_datetime(df['date'], format=fmt, errors='coerce')

        # Приведение числовых колонок к float/int:
        numeric_cols = ['rb', 'total', 'win/Loss', 'rake', 'win/Loss.1']
        for col in numeric_cols:
            if col in df.columns:
                df[col] =df[col].astype(str).str.replace(' ','',regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Генерируем все графики для General Results, используя df вместо global_data
        daily_analysis_figure = generate_daily_analysis_graph(df)
        account_analysis_figure = generate_account_analysis_graph(df)
        average_daily_value_figure = generate_average_daily_value_graph(df)
        rb_impact_comparison_figure = generate_rb_impact_comparison(df)

        general_results_content = [
            html.Div([
                html.H2("Кумулятивный Итоговый Результат всех аккаунтов", style=HEADER_STYLE),
                dcc.Graph(figure=daily_analysis_figure)
            ], style=CONTAINER_STYLE),

            html.Div([
                html.H2("Оценка Вклада Аккаунтов в Общий Итог и Средние Показатели", style=HEADER_STYLE),
                dcc.Graph(figure=account_analysis_figure)
            ], style=CONTAINER_STYLE),

            html.Div([
                html.H2("Средние дневные значения Win/Loss & Total", style=HEADER_STYLE),
                dcc.Graph(figure=average_daily_value_figure)
            ], style=CONTAINER_STYLE),

            # Новый блок с графиком и описанием в одной строке
            html.Div([
                html.Div([
                    html.H2("Соотношение Win/Loss и Rb", style=HEADER_STYLE),
                    dcc.Graph(figure=rb_impact_comparison_figure, style={'flex': '3', 'paddingRight': '20px'})
                ], style={'flex': '3', 'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center'}),

                html.Div([
                    html.H4("Что Означает График на Практике?", style={'fontWeight': 'bold'}),
                    html.Ul([
                        html.Li(
                            "Когда преобладает синяя область, это означает, что большую часть дохода на конкретном периоде времени приносит Win/Loss."),
                        html.Li(
                            "Когда преобладает зеленая область, это значит, что основной вклад вносит Rb, то есть ваш доход в этот момент времени больше зависит от рейкбека."),
                        html.Li(
                            "С течением времени можно наблюдать динамику того, как меняется вклад выигрышей и рейкбека в общий результат."),
                    ])
                ], style={
                    'margin': '20px',
                    'padding': '10px',
                    'border': '1px solid black',
                    'backgroundColor': '#f9f9f9',
                    'flex': '1',
                    'display': 'flex',
                    'flexDirection': 'column',
                    'justifyContent': 'center'
                })
            ], style={
                'display': 'flex',
                'flexDirection': 'row',
                'alignItems': 'center',
                'marginBottom': '20px',
                'padding': '10px'
            })
        ]

        return general_results_content

    return html.Div("Please upload a file first on the Data Upload page.")



@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/data-upload' or pathname == '/':
        return data_upload_layout
    elif pathname == '/general-results':
        return general_results_layout
    else:
        return html.Div("Page not found")


if __name__ == "__main__":
    app.run_server(debug=True)