from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
import sys
import io


from schwarz import get_rect, f_source, max_iter_schwarz, max_iter_ssor, schwarz_step, EPS, Rectangle


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


# === Инициализируем Dash ===
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Динамическое поле: Метод Шварца"),
    dcc.Graph(id='field-graph'),
    dcc.Interval(
        id='interval-component',
        interval=5000,  # обновление каждые 0.5 секунды
        n_intervals=0,
        disabled=False
    ),
    html.Pre(id="console-log",
             style={
                 "whiteSpace": "pre-wrap",
                 "height": "300px",
                 "overflowY": "scroll",
                 "backgroundColor": "#f0f0f0",
                 "padding": "10px"
             })

])
diff1, diff2 = None, None
global_log_output = None
# rects: list[Rectangle] = []
# scale = 10
# rect_num = 2
# for i in range(0, rect_num):
#     rect = Rectangle(i * 2 * scale, i * 2 * scale, 3 * scale, 3 * scale, grid_step).set_boundary(boundary)
#     rects.append(rect)
# rect1, rect2 = rects

# === Callback для обновления графика ===

@app.callback(
    [Output('field-graph', 'figure'),
     Output('interval-component', 'disabled'),
     Output('console-log', 'children')],
    Input('interval-component', 'n_intervals'),
    prevent_initial_call=True
)
def update_graph(n):
    global diff1, diff2, global_log_output
    stop_updates = False

    log_buffer = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = Tee(sys_stdout, log_buffer)  # теперь пишет и туда, и туда

    print(f"\n\nИтерация {n + 1}")

    # Выполняем шаг метода Шварца, если ещё не сошлись
    if (diff1 is None and diff2 is None) or (diff1 > EPS and diff2 > EPS):
        diff1, diff2 = schwarz_step(rect1, rect2, f_source, max_iter_ssor)

    max_diff, l2_norm = rect1.compute_norm_in_intersection(rect2)
    print(f"Макс. отклонение: {max_diff:.2e}, L2-норма: {l2_norm:.2e}")

    if max_diff < EPS:
        print("Было достигнуто условие сходимости по max_diff.")
        stop_updates = True  # Больше не обновляем

    fig = go.Figure()

    for rect in [rect1, rect2]:
        x_vals = np.linspace(rect.x, rect.x + rect.width, rect.grid_width)
        y_vals = np.linspace(rect.y, rect.y + rect.height, rect.grid_height)
        X, Y = np.meshgrid(x_vals, y_vals)

        fig.add_trace(go.Surface(
            x=X, y=Y, z=rect.matrix,
            colorscale='Jet',
            showscale=True,
            colorbar=dict(title="u(x, y)", x=1.1)
        ))

    # Общая область графика
    all_x = [r.x for r in [rect1, rect2]] + [r.x + r.width for r in [rect1, rect2]]
    all_y = [r.y for r in [rect1, rect2]] + [r.y + r.height for r in [rect1, rect2]]

    fig.update_layout(
        title=f"Итерация {n + 1}, разность: {max_diff:.2e}, L2-норма: {l2_norm:.2e}",
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='u(x, y)',
            xaxis_range=[min(all_x), max(all_x)],
            yaxis_range=[min(all_y), max(all_y)]
        ),
        height=800,
        margin=dict(l=60, r=60, t=60, b=60),
        showlegend=False
    )

    # Получаем вывод
    log_output = log_buffer.getvalue()
    global_log_output = log_output if global_log_output is None else global_log_output + log_output

    sys.stdout = sys_stdout
    return fig, stop_updates, global_log_output


# === Запуск сервера ===
if __name__ == '__main__':
    rect1, rect2 = get_rect()
    app.run(debug=True)
