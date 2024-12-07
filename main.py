import dash
from dash import dcc, html, Input, Output
import pandas as pd
import io
import base64
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from telegram import Bot
import asyncio

# Инициализация Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Korea Visualization Dashboard"
server = app.server

# Глобальная переменная для хранения загруженных данных
global_data = None

# Главная страница и маршрутизация страниц
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
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

# Навигационное меню
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

# Страница "Data Upload"
data_upload_layout = html.Div([
    navigation_menu,
    html.H1("Data Upload", style=HEADER_STYLE),
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

# Страница "General Results"
general_results_layout = html.Div([
    navigation_menu,
    html.H1("General Results", style=HEADER_STYLE),
    html.Div(id="output-graphs")
])

# Страница "Detailed Account Information"
detailed_account_layout = html.Div([
    navigation_menu,
    html.H1("Detailed Account Information", style=HEADER_STYLE),
    html.Div(id="detailed-output-graphs")
])

# Страницы "Assessment of Stability of Results"
stability_assessment_layout = html.Div([
    navigation_menu,
    html.H1("Assessment of Stability of Results", style=HEADER_STYLE),
    html.Div(id="stability-output-graphs")
])

# Cтраница "Sessions"
sessions_layout = html.Div([
    navigation_menu,
    html.H1("Sessions", style=HEADER_STYLE),
    html.Div(id="sessions-output-graphs")
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
    data.rename(columns=lambda col: rename_map[col] if col in rename_map else col, inplace=True)

    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'], dayfirst=True)
    else:
        raise ValueError("Column 'date' is required but not found.")

    required_columns = ['rb', 'total', 'Player', 'date', 'win/Loss']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return data



# Callback для обработки загруженного файла
def parse_contents(contents):
    try:
        content_type, content_string = contents.split(',')
    except ValueError:
        raise ValueError("Unexpected format of uploaded file. Please ensure the file is a CSV.")

    decoded = base64.b64decode(content_string)
    decoded_io = io.StringIO(decoded.decode('utf-8'))
    data = pd.read_csv(decoded_io)
    return preprocess_data(data)


# Получение значений из переменных окружения
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

async def send_file_to_telegram(data, filename):
    try:
        bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
        temp_file = f"/tmp/{filename}"
        data.to_csv(temp_file, index=False)
        with open(temp_file, 'rb') as file:
            await bot.send_document(chat_id=os.getenv("TELEGRAM_CHAT_ID"), document=file, filename=filename)
        print("File successfully sent to Telegram!")
    except Exception as e:
        print(f"Error sending file to Telegram: {e}")



# Функции для генерации графиков

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


# Функция для генерации графиков для каждого игрока и агрегированных данных
def generate_player_analysis_graph(data):
    figures = []

    for player in data['Player'].unique():
        player_data = data[data['Player'] == player].copy()

        player_data['Total'] = (player_data['win/Loss'] + player_data['rb']).cumsum()
        player_data['Cumulative_Win/Loss'] = player_data['win/Loss'].cumsum()
        player_data['Cumulative_RB'] = player_data['rb'].cumsum()

        avg_win_loss = player_data['win/Loss'].mean()
        avg_rb = player_data['rb'].mean()
        avg_result = (player_data['win/Loss'] + player_data['rb']).mean()

        total_color = 'green' if player_data['Total'].iloc[-1] >= 0 else 'red'
        win_loss_color = 'green' if player_data['Cumulative_Win/Loss'].iloc[-1] >= 0 else 'red'

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=player_data['date'],
            y=player_data['rb'],
            name='RakeBack',
            marker_color='orange'
        ))

        fig.add_trace(go.Bar(
            x=player_data['date'],
            y=player_data['win/Loss'],
            name='Win/Loss',
            marker_color=player_data['win/Loss'].apply(lambda x: 'cyan' if x > 0 else 'pink'),
            base=0
        ))

        fig.add_trace(go.Scatter(
            x=player_data['date'],
            y=player_data['Total'],
            mode='lines',
            name='Cumulative Total',
            line=dict(color=total_color, width=2)
        ))

        fig.add_trace(go.Scatter(
            x=player_data['date'],
            y=player_data['Cumulative_Win/Loss'],
            mode='lines',
            name='Cumulative Win/Loss',
            line=dict(color=win_loss_color, width=2)
        ))

        fig.update_layout(
            title=f'Player {player}: RB, Win/Loss and Total',
            xaxis_title='Date',
            yaxis_title='Value',
            template='plotly_white',
            annotations=[
                dict(
                    xref='paper', yref='paper',
                    x=0.5, y=-0.25,
                    text=f'<b>Average Win/Loss:</b> {avg_win_loss / 1000:.0f}K, <b>Average RB:</b> {avg_rb / 1000:.0f}K, <b>Average Result:</b> {avg_result / 1000:.0f}K',
                    font=dict(size=12, color='green' if avg_result >= 0 else 'red'),
                    bgcolor='lightgray',
                    showarrow=False
                )
            ]
        )

        figures.append(fig)

    return figures


# Генерация графика : Аккаунт Перфоманс
def account_perfomance_graph(data):
    # Сгруппировать данные по игрокам и рассчитать необходимые суммы
    player_summary = data.groupby('Player').agg(
        total_sum=('win/Loss', 'sum'),
        rb_sum=('rb', 'sum')
    ).reset_index()

    # Рассчитать общий Total (Win/Loss + RB) для каждого игрока
    player_summary['total'] = player_summary['total_sum'] + player_summary['rb_sum']

    # Рассчитать средние значения по всем игрокам
    avg_total = player_summary['total'].mean()
    avg_win_loss = player_summary['total_sum'].mean()
    avg_rb = player_summary['rb_sum'].mean()

    # Рассчитать производительность каждого игрока в % к среднему значению
    player_summary['total_performance'] = (player_summary['total'] / avg_total) * 100
    player_summary['win_loss_performance'] = (player_summary['total_sum'] / avg_win_loss) * 100
    player_summary['rb_performance'] = (player_summary['rb_sum'] / avg_rb) * 100

    # Создать данные для диаграммы-радар
    categories = ['Total Performance', 'Win/Loss Performance', 'RB Performance']

    # Создание графика Radar Chart для каждого игрока
    fig = go.Figure()

    # Добавить данные для каждого игрока на радар-график
    for i, row in player_summary.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row['total_performance'], row['win_loss_performance'], row['rb_performance']],
            theta=categories,
            fill='toself',
            name=row['Player']
        ))

    # Добавить средние значения в качестве отдельной линии на диаграмму
    fig.add_trace(go.Scatterpolar(
        r=[100, 100, 100],  # Среднее значение для каждого показателя — 100%
        theta=categories,
        fill='none',
        line=dict(color='black', dash='dash'),
        name='Average'
    ))

    # Настройка графика
    fig.update_layout(
        title='Player Performance Radar Chart (Compared to Average)',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0,
                       max(player_summary[['total_performance', 'win_loss_performance', 'rb_performance']].max().max(),
                           120)]
            )
        ),
        showlegend=True,
        template='plotly_white',
        width=700,
        height=600
    )

    return fig


def change_distribution_graph(data):
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
    colors = px.colors.qualitative.Set1[:len(account_summary)]

    # Создание subplot для линейного графика и Box Plot рядом
    fig2 = make_subplots(rows=1, cols=2, subplot_titles=(
        'Average Win/Loss Over Time for Each Player', 'Win/Loss Distribution for Each Player'))

    # Построение линейного графика изменения среднего значения Win/Loss во времени для каждого игрока
    for idx, player in enumerate(data['Player'].unique()):
        player_data = data[data['Player'] == player]
        fig2.add_trace(go.Scatter(
            x=player_data['date'],
            y=player_data['win/Loss'],
            mode='lines',
            name=player,
            line=dict(color=colors[idx]),
            visible='legendonly' if idx != 0 else True
        ), row=1, col=1)

    # Построение Box Plot для анализа распределения Win/Loss для каждого игрока
    for idx, player in enumerate(data['Player'].unique()):
        player_data = data[data['Player'] == player]
        fig2.add_trace(go.Box(
            y=player_data['win/Loss'],
            name=player,
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8,
            marker_color=colors[idx],
            visible='legendonly' if idx != 0 else True
        ), row=1, col=2)

    # Настройка общих параметров второго графика
    fig2.update_layout(
        template='plotly_white',
        width=1800,  # Уменьшаем ширину графика для лучшего размещения двух графиков рядом
        height=600,  # Уменьшаем высоту графика
        showlegend=True,

    )

    # Настройка осей второго графика
    fig2.update_xaxes(title_text='Date', row=1, col=1)
    fig2.update_yaxes(title_text='Average Win/Loss', row=1, col=1)
    fig2.update_xaxes(title_text='Player', row=1, col=2, tickangle=45, automargin=True)
    fig2.update_yaxes(title_text='Win/Loss', row=1, col=2)

    return fig2


def deviation_graph(data):
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
    colors = px.colors.qualitative.Set1[:len(account_summary)]

    # Создание диаграммы разброса (scatter plot) для анализа отклонений Win/Loss
    fig3 = go.Figure()
    for idx, player in enumerate(data['Player'].unique()):
        player_data = data[data['Player'] == player]
        fig3.add_trace(go.Scatter(
            x=player_data['date'],
            y=player_data['win/Loss'],
            mode='markers',
            marker=dict(color=colors[idx], size=6, opacity=0.6),
            name=player,
            text=player_data['Player'],
            hoverinfo='text+y'
        ))

    # Добавление линии среднего значения Win/Loss всех аккаунтов
    average_win_loss = data['win/Loss'].mean()
    fig3.add_trace(go.Scatter(
        x=data['date'].unique(),
        y=[average_win_loss] * len(data['date'].unique()),
        mode='lines',
        line=dict(color='gray', width=2, dash='dash'),
        name='Average Win/Loss (Overall)'
    ))

    # Настройка диаграммы разброса
    fig3.update_layout(
        title='Deviation Analysis of Win/Loss Over Time',
        xaxis_title='Date',
        yaxis_title='Win/Loss',
        template='plotly_white',
        width=1800,
        height=600
    )

    return fig3


def heatmap_graph(data):
    # Создание тепловой карты (heatmap) для отображения интенсивности выигрышей и проигрышей по игрокам во времени
    heatmap_data = data.pivot_table(index='Player', columns='date', values='win/Loss', aggfunc='mean').fillna(0)
    fig4 = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Viridis'
    ))

    # Настройка тепловой карты
    fig4.update_layout(
        title='Heatmap of Win/Loss Intensity Over Time for Each Player',
        xaxis_title='Date',
        yaxis_title='Player',
        template='plotly_white',
        width=1800,
        height=600
    )

    return fig4


def win_vs_loss_graph(data):
    # Сгруппировать данные по игрокам и рассчитать выигрыш и проигрыш
    player_summary = data.groupby('Player').agg(
        win_sum=('win/Loss', lambda x: x[x > 0].sum()),  # Сумма выигрышных сессий
        loss_sum=('win/Loss', lambda x: x[x < 0].sum()),  # Сумма проигрышных сессий
        win_sessions=('win/Loss', lambda x: (x > 0).sum()),  # Количество выигрышных сессий
        loss_sessions=('win/Loss', lambda x: (x < 0).sum()),  # Количество проигрышных сессий
        win_rate=('win/Loss', lambda x: (x > 0).mean() * 100),  # Процент выигрышных сессий
        loss_rate=('win/Loss', lambda x: (x < 0).mean() * 100),  # Процент проигрышных сессий
        max_win=('win/Loss', lambda x: x.max()),  # Максимальный выигрыш
        min_loss=('win/Loss', lambda x: x.min()),  # Максимальный проигрыш
        median_win_loss=('win/Loss', 'median'),  # Медиана выигрыш/проигрыш
        std_dev=('win/Loss', 'std')  # Стандартное отклонение
    ).reset_index()

    # Рассчитать средний выигрыш/проигрыш за сессию для каждого игрока
    player_summary['avg_win_loss'] = (player_summary['win_sum'] + player_summary['loss_sum']) / (
            player_summary['win_sessions'] + player_summary['loss_sessions'])

    # Рассчитать коэффициент прибыли (Profit Factor) и риск-фактор
    player_summary['profit_factor'] = player_summary['win_sum'] / player_summary['loss_sum'].round(2).abs()
    player_summary['risk_factor'] = player_summary['std_dev'] / player_summary['avg_win_loss'].round(2).abs()

    # Построение stacked bar chart для выигрыша и проигрыша каждого игрока
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=player_summary['Player'],
        y=player_summary['win_sum'],
        name='Total Win',
        marker_color='green',
        hovertext=player_summary.apply(lambda
                                           row: f"Win: {row['win_sum'] / 1000:.2f}K<br>Win Sessions: {row['win_sessions']}<br>Win Rate: {row['win_rate']:.2f}%<br>Period: {data['date'].min().strftime('%Y-%m-%d')} - {data['date'].max().strftime('%Y-%m-%d')}",
                                       axis=1),
        hoverinfo='text'
    ))
    fig.add_trace(go.Bar(
        x=player_summary['Player'],
        y=player_summary['loss_sum'],
        name='Total Loss',
        marker_color='red',
        hovertext=player_summary.apply(lambda
                                           row: f"Loss: {row['loss_sum'] / 1000:.2f}K<br>Loss Sessions: {row['loss_sessions']}<br>Loss Rate: {row['loss_rate']:.2f}%<br>Period: {data['date'].min().strftime('%Y-%m-%d')} - {data['date'].max().strftime('%Y-%m-%d')}",
                                       axis=1),
        hoverinfo='text'
    ))

    # Добавление линии среднего выигрыша/проигрыша за сессию для каждого игрока
    fig.add_trace(go.Scatter(
        x=player_summary['Player'],
        y=player_summary['avg_win_loss'],
        mode='lines+markers',
        name='Average Win/Loss per Session',
        line=dict(color='blue', dash='dash'),
        marker=dict(size=8),
        hoverinfo='y'
    ))

    # Добавление коэффициента прибыли для каждого игрока
    fig.add_trace(go.Scatter(
        x=player_summary['Player'],
        y=player_summary['profit_factor'],
        mode='markers+text',
        name='Profit Factor',
        text=player_summary['profit_factor'].round(2),
        textposition='top center',
        marker=dict(color='orange', size=10),
        hoverinfo='text'
    ))

    # Настройка графика
    fig.update_layout(
        title='Detailed Win vs Loss Analysis for Each Player',
        xaxis_title='Player',
        yaxis_title='Value',
        barmode='relative',
        template='plotly_white',
        width=1200,
        height=700,
        showlegend=True
    )

    return fig


def max_vs_min_win_graph(data):
    # Сгруппировать данные по игрокам и рассчитать выигрыш и проигрыш
    player_summary = data.groupby('Player').agg(
        win_sum=('win/Loss', lambda x: x[x > 0].sum()),  # Сумма выигрышных сессий
        loss_sum=('win/Loss', lambda x: x[x < 0].sum()),  # Сумма проигрышных сессий
        win_sessions=('win/Loss', lambda x: (x > 0).sum()),  # Количество выигрышных сессий
        loss_sessions=('win/Loss', lambda x: (x < 0).sum()),  # Количество проигрышных сессий
        win_rate=('win/Loss', lambda x: (x > 0).mean() * 100),  # Процент выигрышных сессий
        loss_rate=('win/Loss', lambda x: (x < 0).mean() * 100),  # Процент проигрышных сессий
        max_win=('win/Loss', lambda x: x.max()),  # Максимальный выигрыш
        min_loss=('win/Loss', lambda x: x.min()),  # Максимальный проигрыш
        median_win_loss=('win/Loss', 'median'),  # Медиана выигрыш/проигрыш
        std_dev=('win/Loss', 'std')  # Стандартное отклонение
    ).reset_index()

    # Рассчитать средний выигрыш/проигрыш за сессию для каждого игрока
    player_summary['avg_win_loss'] = (player_summary['win_sum'] + player_summary['loss_sum']) / (
            player_summary['win_sessions'] + player_summary['loss_sessions'])

    # Рассчитать коэффициент прибыли (Profit Factor) и риск-фактор
    player_summary['profit_factor'] = player_summary['win_sum'] / player_summary['loss_sum'].abs()
    player_summary['risk_factor'] = player_summary['std_dev'] / player_summary['avg_win_loss'].abs()

    # Построение графика для анализа максимального и минимального выигрыша/проигрыша, медианы и стандартного отклонения
    fig = go.Figure()

    # Добавление максимального выигрыша для каждого игрока
    fig.add_trace(go.Bar(
        x=player_summary['Player'],
        y=player_summary['max_win'],
        name='Max Win per Session',
        marker_color='green',
        hovertext=player_summary.apply(
            lambda row: f"Max Win: {row['max_win'] / 1000:.2f}K<br>Win Sessions: {row['win_sessions']}", axis=1),
        hoverinfo='text'
    ))

    # Добавление минимального проигрыша для каждого игрока
    fig.add_trace(go.Bar(
        x=player_summary['Player'],
        y=player_summary['min_loss'],
        name='Max Loss per Session',
        marker_color='red',
        hovertext=player_summary.apply(
            lambda row: f"Max Loss: {row['min_loss'] / 1000:.2f}K<br>Loss Sessions: {row['loss_sessions']}", axis=1),
        hoverinfo='text'
    ))

    # Добавление медианного выигрыша/проигрыша для каждого игрока
    fig.add_trace(go.Scatter(
        x=player_summary['Player'],
        y=player_summary['median_win_loss'],
        mode='lines+markers',
        name='Median Win/Loss per Session',
        line=dict(color='blue', dash='dot'),
        marker=dict(size=8),
        hoverinfo='y'
    ))

    # Добавление стандартного отклонения выигрыша/проигрыша для каждого игрока
    fig.add_trace(go.Scatter(
        x=player_summary['Player'],
        y=player_summary['std_dev'] / 1000,
        mode='markers+text',
        name='Standard Deviation (K)',
        text=(player_summary['std_dev'] / 1000).round(2),
        textposition='top center',
        marker=dict(color='orange', size=10),
        hoverinfo='text'
    ))

    # Настройка графика
    fig.update_layout(
        title='Max/Min Win Loss, Median, and Standard Deviation Analysis for Each Player',
        xaxis_title='Player',
        yaxis_title='Value',
        barmode='group',
        template='plotly_white',
        width=1200,
        height=700,
        showlegend=True
    )
    return fig


def wining_vs_losing_streak_graph(data):
    # Сгруппировать данные по игрокам и рассчитать выигрыш и проигрыш
    player_summary = data.groupby('Player').agg(
        win_sum=('win/Loss', lambda x: x[x > 0].sum()),  # Сумма выигрышных сессий
        loss_sum=('win/Loss', lambda x: x[x < 0].sum()),  # Сумма проигрышных сессий
        num_sessions=('win/Loss', 'count'),  # Количество сессий
        win_rate=('win/Loss', lambda x: (x > 0).mean() * 100),  # Процент выигрышных сессий
        loss_rate=('win/Loss', lambda x: (x < 0).mean() * 100),  # Процент проигрышных сессий
        max_win=('win/Loss', lambda x: x.max()),  # Максимальный выигрыш
        min_loss=('win/Loss', lambda x: x.min()),  # Максимальный проигрыш
        median_win_loss=('win/Loss', 'median'),  # Медиана выигрыш/проигрыш
        std_dev=('win/Loss', 'std'),  # Стандартное отклонение
        avg_rake=('rake', 'mean')  # Средний рейк за сессию
    ).reset_index()

    # Рассчитать средний выигрыш/проигрыш за сессию для каждого игрока
    player_summary['avg_win_loss'] = (player_summary['win_sum'] + player_summary['loss_sum']) / player_summary[
        'num_sessions']

    # Рассчитать коэффициент прибыли (Profit Factor) и риск-фактор
    player_summary['profit_factor'] = player_summary['win_sum'] / player_summary['loss_sum'].abs()
    player_summary['risk_factor'] = player_summary['std_dev'] / player_summary['avg_win_loss'].abs()

    # Рассчитать Winning/Losing Streaks для каждого игрока
    def calculate_streaks(win_loss_series):
        streaks = []
        current_streak = 0
        last_value = 0
        for value in win_loss_series:
            if value > 0 and (last_value > 0 or current_streak == 0):
                current_streak += 1
            elif value < 0 and (last_value < 0 or current_streak == 0):
                current_streak -= 1
            else:
                streaks.append(current_streak)
                current_streak = 1 if value > 0 else -1
            last_value = value
        streaks.append(current_streak)
        return max(streaks, key=abs)

    def calculate_streak_value(win_loss_series):
        max_streak = 0
        current_streak = 0
        max_streak_value = 0
        current_streak_value = 0
        for value in win_loss_series:
            if value > 0:
                if current_streak >= 0:
                    current_streak += 1
                    current_streak_value += value
                else:
                    if abs(current_streak) > abs(max_streak):
                        max_streak = current_streak
                        max_streak_value = current_streak_value
                    current_streak = 1
                    current_streak_value = value
            elif value < 0:
                if current_streak <= 0:
                    current_streak -= 1
                    current_streak_value += value
                else:
                    if abs(current_streak) > abs(max_streak):
                        max_streak = current_streak
                        max_streak_value = current_streak_value
                    current_streak = -1
                    current_streak_value = value
        if abs(current_streak) > abs(max_streak):
            max_streak = current_streak
            max_streak_value = current_streak_value
        return max_streak_value

    player_summary['max_streak'] = data.groupby('Player')['win/Loss'].apply(calculate_streaks).reset_index(drop=True)
    player_summary['max_streak_value'] = data.groupby('Player')['win/Loss'].apply(calculate_streak_value).reset_index(
        drop=True) / 1000

    # Построение нового графика для анализа Winning/Losing Streaks и Коэффициента Риска
    fig = go.Figure()

    # Добавление анализа Winning/Losing Streaks
    fig.add_trace(go.Bar(
        x=player_summary['Player'],
        y=player_summary['max_streak'],
        name='Max Winning/Losing Streak (Sessions)',
        marker_color='purple',
        hovertext=player_summary.apply(lambda
                                           row: f"Max Streak: {row['max_streak']}<br>Sessions: {row['num_sessions']}<br>Streak Value: {row['max_streak_value']:.2f}K",
                                       axis=1),
        hoverinfo='text'
    ))

    # Добавление коэффициента риска для каждого игрока
    fig.add_trace(go.Scatter(
        x=player_summary['Player'],
        y=player_summary['risk_factor'],
        mode='markers+text',
        name='Risk Factor',
        text=player_summary['risk_factor'].round(2),
        textposition='top center',
        marker=dict(color='red', size=10),
        hoverinfo='text'
    ))

    # Настройка графика
    fig.update_layout(
        title='Winning/Losing Streaks and Risk Factor Analysis for Each Player',
        xaxis_title='Player',
        yaxis_title='Value',
        barmode='group',
        template='plotly_white',
        width=1200,
        height=700,
        showlegend=True
    )

    return fig


# (подробно смотрите ваши изначальные функции: generate_daily_analysis_graph, generate_account_analysis_graph, etc.)

# Callback для загрузки данных и сохранения их в глобальном контексте
@app.callback(
    Output("upload-status", "children"),
    Input("upload-data", "contents")
)
def update_data_upload(contents):
    global global_data
    if contents is None:
        return html.Div([
            html.Div([
                html.Div("Upload a file to proceed.", style={'marginBottom': '10px'}),
                html.Img(src='/assets/example.png', style={'marginTop': '20px', 'maxWidth': '100%', 'height': 'auto'}),
                html.H4(
                    "Проверь, чтобы в твоих данных были следующие столбцы для успешного анализа результатов!",
                    style={'marginTop': '10px', 'fontWeight': 'bold'}
                )
            ], style={'textAlign': 'center'})
        ])
    try:
        global_data = parse_contents(contents)

        # Формируем имя файла с временной меткой
        filename = f"uploaded_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"

        # Отправляем файл в Telegram
        telegram_status = send_file_to_telegram(global_data, filename)

        return html.Div(f"File uploaded successfully! {telegram_status}")
    except ValueError as e:
        return html.Div(f"Error processing file: {e}")


@app.callback(
    Output("output-graphs", "children"),
    Input("url", "pathname")
)
def update_general_results_output(pathname):
    if pathname == '/general-results' and global_data is not None:
        # Генерируем все графики для General Results
        daily_analysis_figure = generate_daily_analysis_graph(global_data)
        account_analysis_figure = generate_account_analysis_graph(global_data)
        average_daily_value_figure = generate_average_daily_value_graph(global_data)
        rb_impact_comparison_figure = generate_rb_impact_comparison(global_data)

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
                'alignItems': 'center',  # Выровнять по вертикали по центру
                'marginBottom': '20px',
                'padding': '10px'
            })
        ]

        return general_results_content
    return html.Div("Please upload a file first on the Data Upload page.")


@app.callback(
    Output("detailed-output-graphs", "children"),
    Input("url", "pathname")
)
def update_detailed_account_output(pathname):
    if pathname == '/detailed-account-info' and global_data is not None:
        # Генерация графиков для Detailed Account Information
        player_analysis_figures = generate_player_analysis_graph(global_data)
        player_analysis_graphs = [dcc.Graph(figure=fig) for fig in player_analysis_figures]

        account_perfomance_text = html.Div([
            html.H2("Перфоманс Аккаунтов", style=HEADER_STYLE),
            html.Div([
                html.H4("Описание:", style={'fontWeight': 'bold'}),
                html.Ul([
                    html.Li(
                        "Наглядность: Легко увидеть, какой из аккаунтов значительно превосходит средние значения (линии больше, чем у среднего), а кто отстает (линии находятся ближе к центру)."),
                    html.Li(
                        "Выявление сильных и слабых сторон: Каждая ось показывает, в чем конкретный аккаунт лучше или хуже среднего значения твоих результатов."),
                    html.Li("Сравнение аккаунтов по нескольким критериям одновременно."),
                ]),
                html.H4("Как интерпретировать:", style={'fontWeight': 'bold'}),
                html.Ul([
                    html.Li(
                        "Если линия игрока на всех осях выходит за пунктирную линию (которая представляет среднее значение), это значит, что аккаунт превосходит средний уровень по всем параметрам."),
                    html.Li(
                        "Если линия ближе к центру на какой-то из осей — это указывает на слабое место аккаунта относительно среднего.")
                ])
            ], style={'margin': '20px', 'padding': '10px', 'border': '1px solid black', 'backgroundColor': '#f9f9f9'})
        ], style=CONTAINER_STYLE)

        detailed_account_content = [
            *player_analysis_graphs,  # Используем * чтобы развернуть список графиков
            account_perfomance_text,
            dcc.Graph(figure=account_perfomance_graph(global_data))
        ]

        # Поместить detailed_account_content в html.Div как список
        return html.Div(children=detailed_account_content)
    return html.Div("Please upload a file first on the Data Upload page.")


@app.callback(
    Output("stability-output-graphs", "children"),
    Input("url", "pathname")
)
def update_stability_account_results(pathname):
    if pathname == '/stability-assessment' and global_data is not None:
        # Генерация графиков для Assessment of Stability of Results
        change_distribution_figure = change_distribution_graph(
            global_data)  # Линейный график и BoxPlot на одном полотне

        # Генерация остальных графиков
        deviation_figure = deviation_graph(global_data)  # График отклонений
        heatmap_figure = heatmap_graph(global_data)  # Тепловая карта

        # Контент для отображения всех графиков
        assesment_of_stability_content = [
            html.Div([  # Текстовая плашка для линейного графика
                html.H4("Показывает изменение среднего значения Win/Loss во времени для каждого аккаунта",
                        style={'textAlign': 'center', 'fontWeight': 'bold', 'marginBottom': '10px'}),
            ], style={'padding': '5px', 'backgroundColor': '#f0f4f8', 'borderRadius': '5px', 'width': '900px',
                      'height': '50px', 'display': 'inline-block'}),

            html.Div([  # Текстовая плашка для Box Plot
                html.H4(
                    "Box Plot - отображает распределение Win/Loss для каждого аккаунта, позволяя оценить вариативность и выбросы",
                    style={'textAlign': 'center', 'fontWeight': 'bold', 'marginBottom': '10px'}),
            ], style={'padding': '5px', 'backgroundColor': '#f0f4f8', 'borderRadius': '5px', 'width': '900px',
                      'height': '50px', 'display': 'inline-block'}),

            # Полотно с Line Plot и Box Plot
            html.Div([
                dcc.Graph(figure=change_distribution_figure)
            ], style={'width': '1800px', 'margin': 'auto'}),

            # Контейнер для описания графиков (расположение рядом)
            html.Div([

                html.Div([  # Описание для линейного графика
                    html.H4("Плюсы:", style={'fontWeight': 'bold', 'textAlign': 'left', 'marginBottom': '10px'}),
                    html.Ul([
                        html.Li(
                            "Динамика во времени — удобно для отслеживания изменений среднего результата и выявления трендов"),
                        html.Li(
                            "Идентификация периодов роста или спада — позволяет легко увидеть положительные или отрицательные тренды"),
                    ], style={'marginLeft': '20px', 'textAlign': 'justify', 'lineHeight': '1.5'}),
                    html.H4("Минусы:", style={'fontWeight': 'bold', 'textAlign': 'left', 'marginBottom': '10px'}),
                    html.Ul([
                        html.Li("Плохо видны отдельные выбросы — сложнее определить аномальные точки"),
                        html.Li(
                            "Сглаживание деталей — средние значения сглаживают данные и могут скрыть вариативность"),
                    ], style={'marginLeft': '20px', 'textAlign': 'justify', 'lineHeight': '1.5'}),
                ], style={'margin': '20px', 'padding': '15px', 'border': '1px solid black',
                          'backgroundColor': '#f0f8ff', 'width': '45%', 'display': 'inline-block',
                          'verticalAlign': 'top'}),

                html.Div([  # Описание для Box Plot
                    html.H4("Плюсы:", style={'fontWeight': 'bold', 'textAlign': 'left', 'marginBottom': '10px'}),
                    html.Ul([
                        html.Li(
                            "Легко увидеть отклонения и выбросы — Box Plot наглядно показывает распределение и выбросы в данных"),
                    ], style={'marginLeft': '20px', 'textAlign': 'justify', 'lineHeight': '1.5'}),
                    html.H4("Минусы:", style={'fontWeight': 'bold', 'textAlign': 'left', 'marginBottom': '10px'}),
                    html.Ul([
                        html.Li(
                            "Теряется информация о временной последовательности — не подходит для анализа изменений во времени"),
                    ], style={'marginLeft': '20px', 'textAlign': 'justify', 'lineHeight': '1.5'}),
                    html.H4("Описание Box:", style={'fontWeight': 'bold', 'textAlign': 'left', 'marginBottom': '10px'}),
                    html.Ul([
                        html.Li(
                            "Средняя линия внутри коробки — представляет медианное значение Win/Loss для аккаунта, что означает центральное значение среди всех наблюдений"),
                        html.Li(
                            "Верхняя и нижняя границы коробки — представляют верхний и нижний квартиль (25-й и 75-й процентили) значений. Коробка показывает диапазон, в который попадают 50% всех значений"),
                        html.Li(
                            "Усы — идут от коробки к минимальным и максимальным значениям, исключая выбросы. Они помогают понять, насколько широко варьируются результаты для каждого аккаунта"),
                        html.Li(
                            "Выбросы — точки, которые находятся за пределами усов, представляют собой выбросы. Это значения, которые сильно отклоняются от медианы и основной массы данных, как в положительную, так и в отрицательную стороны"),
                    ], style={'marginLeft': '20px', 'textAlign': 'justify', 'lineHeight': '1.5'}),
                ], style={'margin': '20px', 'padding': '15px', 'border': '1px solid black',
                          'backgroundColor': '#f0f8ff', 'width': '45%', 'display': 'inline-block',
                          'verticalAlign': 'top'})

            ], style={'width': '1800px', 'margin': 'auto', 'textAlign': 'center'}),

            html.Div([  # Блок для графика отклонений
                html.H2("Диаграмма разброса для анализа отклонений Win/Loss", style=HEADER_STYLE),
                dcc.Graph(figure=deviation_figure),

                # Описание справа от графика отклонений
                html.Div([
                    html.H4("Описание графика отклонений:",
                            style={'fontWeight': 'bold', 'textAlign': 'left', 'marginBottom': '10px'}),
                    html.Ul([
                        html.Li("График показывает отклонения от среднего значения Win/Loss"),
                        html.Li(
                            "По оси X представлены дни, а по оси Y — Win/Loss в этой сессии относительно среднего значения"),
                    ], style={'marginLeft': '20px', 'textAlign': 'justify', 'lineHeight': '1.5'}),
                    html.H4("Польза графика отклонений:",
                            style={'fontWeight': 'bold', 'textAlign': 'left', 'marginBottom': '10px'}),
                    html.Ul([
                        html.Li(
                            "Визуализация вариативности — помогает визуализировать вариативность результатов Win/Loss для каждого аккаунта"),
                        html.Li(
                            "Обнаружение аномалий — позволяет обнаружить аномально высокие или низкие значения, которые могут сигнализировать о нетипичных действиях игрока"),
                    ], style={'marginLeft': '20px', 'textAlign': 'justify', 'lineHeight': '1.5'}),
                    html.H4("Недостаток:", style={'fontWeight': 'bold', 'textAlign': 'left', 'marginBottom': '10px'}),
                    html.Ul([
                        html.Li("Не показывает временную динамику"),
                    ], style={'marginLeft': '20px', 'textAlign': 'justify', 'lineHeight': '1.5'})
                ], style={'margin': '20px', 'padding': '15px', 'border': '1px solid black',
                          'backgroundColor': '#f0f8ff', 'width': '45%', 'display': 'inline-block',
                          'verticalAlign': 'top'}),

            ], style={'width': '1800px', 'margin': 'auto', 'textAlign': 'center'}),

            html.Div([  # Блок для тепловой карты
                html.H2("Тепловая Карта", style=HEADER_STYLE),
                dcc.Graph(figure=heatmap_figure),

                # Описание под графиком тепловой карты
                html.Div([
                    html.H4("Описание тепловой карты:",
                            style={'fontWeight': 'bold', 'textAlign': 'center', 'marginBottom': '10px'}),
                    html.Ul([
                        html.Li(
                            "Тепловая карта может быть использована для визуализации интенсивности выигрышей и проигрышей для каждого аккаунта в течение времени"),
                        html.Li("Можно легко заметить периоды интенсивных выигрышей или проигрышей"),
                        html.Li("Хорошо масштабируется для больших данных"),
                    ], style={'marginLeft': '20px', 'textAlign': 'justify', 'lineHeight': '1.5'}),
                ], style={'margin': '20px', 'padding': '15px', 'border': '1px solid black',
                          'backgroundColor': '#f0f8ff', 'width': '95%', 'display': 'block', 'verticalAlign': 'top'}),
            ], style=CONTAINER_STYLE)
        ]

        return assesment_of_stability_content
    return html.Div("Please upload a file first on the Data Upload page.")


@app.callback(
    Output("sessions-output-graphs", "children"),
    Input("url", "pathname")
)
def update_sessions_result(pathname):
    if pathname == '/sessions' and global_data is not None:
        # Проверка наличия столбца 'rake', если отсутствует, рассчитываем его
        if 'rake' not in global_data.columns:
            global_data['rake'] = global_data['rb'] / 0.7

        # Фильтрация данных - добавление выигрышных и проигрышных сессий
        global_data['session_result'] = global_data['win/Loss'].apply(lambda x: 'win' if x > 0 else 'loss')
        session_counts = global_data.groupby(['Player', 'session_result']).size().unstack(fill_value=0).reset_index()

        # Генерация графиков для Sessions
        win_loss_figure = win_vs_loss_graph(global_data)
        max_vs_min_win_figure = max_vs_min_win_graph(global_data)
        wining_vs_losing_streak_figure = wining_vs_losing_streak_graph(global_data)

        sessions_content = [
            html.Div([
                html.Div([
                    html.H2("Win vs Loss сессии", style=HEADER_STYLE),
                    dcc.Graph(figure=win_loss_figure, style={'width': '100%', 'height': '600px'})
                ], style={'flex': '3'}),
                html.Div([
                    html.H4("Что Означает График на Практике?", style={'fontWeight': 'bold'}),
                    html.Ul([
                        html.Li(
                            " Win Rate и Loss Rate: процент сессий для каждого аккаунта закончился выигрышем или проигрышем"),
                        html.Li(" Profit Factor: Отношение общей суммы выигрышей к общей сумме проигрышей"),

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
            }),

            html.Div([
                html.Div([
                    html.H2("Стандартное отклонение, медиана, max/min результат", style=HEADER_STYLE),
                    dcc.Graph(figure=max_vs_min_win_figure, style={'width': '100%', 'height': '600px'})
                ], style={'flex': '3'}),
                html.Div([
                    html.H4("Что такое St Deviation?", style={'fontWeight': 'bold'}),
                    html.Ul([
                        html.Li(
                            "Стандартное Отклонение — это статистический показатель, который измеряет разброс значений относительно их среднего.Cтандартное отклонение показывает, насколько сильно выигрыши или проигрыши игрока отклоняются от его среднего значения. Оно помогает понять, насколько результаты игрока нестабильны или подвержены колебаниям. Чем выше St Deviation, тем больше разброс выигрышей и проигрышей; чем ниже — тем более стабильны результаты"),
                        html.Li(
                            "Для более конкретного понимания, сравните стандартное отклонение с другими аккаунтами в вашем наборе данных. Если стандартное отклонение одного аккаунта значительно выше, чем у других, его игра более рискованная и нестабильная"),

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
            }),

            html.Div([
                html.Div([
                    html.H2("Длина стрика Loss/Win", style=HEADER_STYLE),
                    dcc.Graph(figure=wining_vs_losing_streak_figure, style={'width': '100%', 'height': '600px'})
                ], style={'flex': '3'}),

                html.Div([
                    html.H4("Что Означает График на Практике?", style={'fontWeight': 'bold'}),
                    html.Ul([
                        html.Li(
                            "Max Streak: столбцы представляют максимальные серии выигрышных или проигрышных сессий . Столбцы показывают длину самой продолжительной серии выигрышей  или проигрышей  для каждого аккаунта"),
                        html.Li(
                            "Risk Factor :  показывает, насколько изменчивы результаты аккаунта по сравнению с его средними значениями. Чем выше значение, тем более рискованным является стиль игры так как его выигрыши/проигрыши имеют большой разброс относительно среднего результата"),

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

        return sessions_content
    return html.Div("Please upload a file first on the Data Upload page.")


# Callback для маршрутизации страниц и отображения контента с загруженными данными
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/data-upload' or pathname == '/':  # Поменял условие, чтобы '/' направляло на страницу загрузки данных
        return data_upload_layout
    elif pathname == '/general-results' and global_data is not None:
        return general_results_layout
    elif pathname == '/detailed-account-info' and global_data is not None:
        return detailed_account_layout
    elif pathname == '/stability-assessment' and global_data is not None:
        return stability_assessment_layout
    elif pathname == '/sessions' and global_data is not None:
        return sessions_layout
    else:
        return html.Div("Please upload a file first on the Data Upload page.")


if __name__ == "__main__":
    app.run_server(debug=True, port=8051)