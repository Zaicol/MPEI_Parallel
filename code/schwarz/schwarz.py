import numpy as np
from matplotlib import pyplot as plt

# Параметры сетки
r1x, r1y = 0, 30
r1w, r1h = 50, 20
r2x, r2y = 20, 0
r2w, r2h = 40, 70
grid_step = 1
max_iter_ssor = 100
max_iter_schwarz = 100


OMEGA = 1.8
EPS = 1e-12
plt.ion()


class Coords:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Rectangle:
    def __init__(self, x, y, width, height, step):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.step = step

        self.grid_width = int(width // step + 1)
        self.grid_height = int(height // step + 1)

        print(f"grid_width: {self.grid_width}, grid_height: {self.grid_height}")
        self.matrix = np.zeros((self.grid_height, self.grid_width), dtype=float)
        self.cross_matrix: np.ndarray | None = None

    def set_boundary(self, func):
        # Верхняя и нижняя границы (по строкам)
        i_vals = np.arange(self.grid_width)
        global_x_vals = self.x + self.step * i_vals
        global_y_top = self.y + self.height
        global_y_bottom = self.y

        self.matrix[0, :] = func(global_x_vals, global_y_bottom)
        self.matrix[-1, :] = func(global_x_vals, global_y_top)

        # Левая и правая границы (по столбцам)
        j_vals = np.arange(self.grid_height)
        global_y_vals = self.y + self.step * j_vals
        global_x_left = self.x
        global_x_right = self.x + self.width

        self.matrix[:, 0] = np.vectorize(func)(global_x_left, global_y_vals)
        self.matrix[:, -1] = np.vectorize(func)(global_x_right, global_y_vals)

    def print_matrix(self):
        for row in self.matrix:
            print(' '.join(f"{val:5.2f}" for val in row))

    def compute_norm_in_intersection(self, other) -> [float, float]:
        """Сливает матрицы в области пересечения, беря значения из other."""
        # Границы пересечения
        x_start = max(self.x, other.x)
        x_end = min(self.x + self.width, other.x + other.width)
        y_start = max(self.y, other.y)
        y_end = min(self.y + self.height, other.y + other.height)

        if x_end <= x_start or y_end <= y_start:
            return

        # Индексы в self
        i_start_self = round((x_start - self.x) / self.step)
        i_end_self = round((x_end - self.x) / self.step) + 1
        j_start_self = round((y_start - self.y) / self.step)
        j_end_self = round((y_end - self.y) / self.step) + 1

        # Индексы в other
        i_start_other = int(round((x_start - other.x) / other.step))
        i_end_other = int(round((x_end - other.x) / other.step)) + 1
        j_start_other = int(round((y_start - other.y) / other.step))
        j_end_other = int(round((y_end - other.y) / other.step)) + 1

        # Обрезаем до допустимых размеров (защита от выхода за границы)
        i_start_self = int(np.clip(i_start_self, 0, self.grid_width - 1))
        i_end_self = int(np.clip(i_end_self, 0, self.grid_width))
        j_start_self = int(np.clip(j_start_self, 0, self.grid_height - 1))
        j_end_self = int(np.clip(j_end_self, 0, self.grid_height))

        i_start_other = int(np.clip(i_start_other, 0, other.grid_width - 1))
        i_end_other = int(np.clip(i_end_other, 0, other.grid_width))
        j_start_other = int(np.clip(j_start_other, 0, other.grid_height - 1))
        j_end_other = int(np.clip(j_end_other, 0, other.grid_height))

        # Извлечение нужных частей матриц
        roi_g1 = self.matrix[j_start_self:j_end_self + 1, i_start_self:i_end_self + 1]
        roi_g2 = other.matrix[j_start_other:j_end_other, i_start_other:i_end_other]

        # Проверка совпадения форм
        assert roi_g1.shape == roi_g2.shape, "Размерности областей не совпадают"

        # Вычисление разностей
        diff = np.abs(roi_g1 - roi_g2)

        # Максимальная разность
        max_diff = np.max(diff)

        # Сумма квадратов
        sum_sq = np.sum(diff ** 2)

        # Число точек
        n_points = roi_g1.size

        # L2 норма
        l2_norm = np.sqrt(sum_sq / n_points)

        return max_diff, l2_norm

    def merge_from_intersection_perimeter(self, other):
        """Копирует в self только периметр области other, лежащий в области пересечения."""

        # Границы пересечения
        x_start = max(self.x, other.x)
        x_end = min(self.x + self.width, other.x + other.width)
        y_start = max(self.y, other.y)
        y_end = min(self.y + self.height, other.y + other.height)

        if x_end <= x_start or y_end <= y_start:
            return

        # Общая функция для пересчёта координат в индексы
        def to_index_x(rect, x):
            return int(round((x - rect.x) / rect.step))

        def to_index_y(rect, y):
            return int(round((y - rect.y) / rect.step))

        # Индексы по x и y в обоих прямоугольниках
        i_start_self = np.clip(to_index_x(self, x_start), 0, self.grid_width - 1)
        i_end_self = np.clip(to_index_x(self, x_end), 0, self.grid_width - 1)
        j_start_self = np.clip(to_index_y(self, y_start), 0, self.grid_height - 1)
        j_end_self = np.clip(to_index_y(self, y_end), 0, self.grid_height - 1)

        i_start_other = np.clip(to_index_x(other, x_start), 0, other.grid_width - 1)
        i_end_other = np.clip(to_index_x(other, x_end), 0, other.grid_width - 1)
        j_start_other = np.clip(to_index_y(other, y_start), 0, other.grid_height - 1)
        j_end_other = np.clip(to_index_y(other, y_end), 0, other.grid_height - 1)

        # --- TOP row (y = y_start) ---
        js_self = j_start_self
        js_other = j_start_other
        for i_s, i_o in zip(range(i_start_self, i_end_self + 1),
                            range(i_start_other, i_end_other + 1)):
            self.matrix[js_self][i_s] = other.matrix[js_other][i_o]

        # --- BOTTOM row (y = y_end) ---
        je_self = j_end_self
        je_other = j_end_other
        for i_s, i_o in zip(range(i_start_self, i_end_self + 1),
                            range(i_start_other, i_end_other + 1)):
            self.matrix[je_self][i_s] = other.matrix[je_other][i_o]

        # --- LEFT column (x = x_start) ---
        is_self = i_start_self
        is_other = i_start_other
        for j_s, j_o in zip(range(j_start_self + 1, j_end_self),  # без углов, чтобы не дублировать
                            range(j_start_other + 1, j_end_other)):
            self.matrix[j_s][is_self] = other.matrix[j_o][is_other]

        # --- RIGHT column (x = x_end) ---
        ie_self = i_end_self
        ie_other = i_end_other
        for j_s, j_o in zip(range(j_start_self + 1, j_end_self),
                            range(j_start_other + 1, j_end_other)):
            self.matrix[j_s][ie_self] = other.matrix[j_o][ie_other]


# Функция f(x, y)
def f_source(x, y):
    return 1


# Граничное условие
def boundary(x, y):
    return 1


def draw_field(rects: list[Rectangle], clear=True, name="Шварц"):
    if clear:
        plt.clf()  # Очистка текущего графика
    else:
        plt.figure()  # Новое окно

    all_x = []
    all_y = []

    for rect in rects:
        all_x += [rect.x, rect.x + rect.width]
        all_y += [rect.y, rect.y + rect.height]

        plt.imshow(rect.matrix,
                   origin='lower',
                   interpolation='none',
                   extent=(rect.x, rect.x + rect.width, rect.y, rect.y + rect.height),
                   cmap='jet',
                   alpha=1)

    # Настройки графика
    plt.colorbar(label='u(x, y)')
    plt.title(name)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.xlim(min(all_x), max(all_x))
    plt.ylim(min(all_y), max(all_y))
    plt.tight_layout()
    plt.autoscale(enable=True)
    plt.draw()
    plt.pause(0.5)


def schwarz_step(rectangle1: Rectangle, rectangle2: Rectangle, func, max_ssor_it=10000):
    print("Solving G2...")
    diff1 = SSOR(rectangle2, func, max_ssor_it)

    # Копируем данные из G2 в G1
    rectangle1.merge_from_intersection_perimeter(rectangle2)

    print("Solving G1...")
    diff2 = SSOR(rectangle1, func, max_ssor_it)

    # Копируем данные из G1 в G2
    rectangle2.merge_from_intersection_perimeter(rectangle1)
    return diff1, diff2


def schwartz_method(rectangle1: Rectangle, rectangle2: Rectangle, func, max_it=10000, max_ssor_it=10000):
    draw_field([rectangle1, rectangle2], name="Начальное состояние")
    iter_num = 0

    for iter_num in range(max_it):
        print(f"\n\nИтерация {iter_num + 1}")
        diff1, diff2 = schwarz_step(rectangle1, rectangle2, func, max_ssor_it)
        max_diff, l2_norm = rectangle1.compute_norm_in_intersection(rectangle2)
        print(f"Макс. отклонение: {max_diff:.2e}, L2-норма: {l2_norm:.2e}")

        if max_diff < EPS:
            draw_field([rectangle1, rectangle2], name=f"Итерация {iter_num + 1} (финальная), отклонение: {max_diff:.2e}, L2: {l2_norm:.2e}")
            print("Было достигнуто условие сходимости по max_diff.")
            break

        if iter_num < 5 or (iter_num + 1) % 10 == 0:
            draw_field([rectangle1, rectangle2], name=f"Итерация {iter_num + 1}, отклонение: {max_diff:.2e}, L2: {l2_norm:.2e}")

    print(f"Метод Шварца завершён. Выполненных итераций: {iter_num + 1}.")


def SSOR(rect: Rectangle, func, max_it=100000):
    hx = rect.step
    hy = rect.step
    hx2 = hx * hx
    hy2 = hy * hy
    denom = 2.0 * (1.0 / hx2 + 1.0 / hy2)

    iter_count = 0
    max_diff = 0.0

    while iter_count < max_it:

        # Прямой проход
        for j in range(1, rect.grid_height - 1):
            yj = rect.y + j * hy
            for i in range(1, rect.grid_width - 1):
                xi = rect.x + i * hx
                rhs = func(xi, yj)
                sum_val = (
                    (rect.matrix[j][i + 1] + rect.matrix[j][i - 1]) / hx2 +
                    (rect.matrix[j + 1][i] + rect.matrix[j - 1][i]) / hy2 +
                    rhs
                )
                unew = (1 - OMEGA) * rect.matrix[j][i] + OMEGA * sum_val / denom
                max_diff = max(max_diff, abs(unew - rect.matrix[j][i]))
                rect.matrix[j][i] = unew

        # Обратный проход
        for j in reversed(range(1, rect.grid_height - 1)):
            yj = rect.y + j * hy
            for i in reversed(range(1, rect.grid_width - 1)):
                xi = rect.x + i * hx
                rhs = func(xi, yj)
                sum_val = (
                    (rect.matrix[j][i + 1] + rect.matrix[j][i - 1]) / hx2 +
                    (rect.matrix[j + 1][i] + rect.matrix[j - 1][i]) / hy2 +
                    rhs
                )
                unew = (1 - OMEGA) * rect.matrix[j][i] + OMEGA * sum_val / denom
                max_diff = max(max_diff, abs(unew - rect.matrix[j][i]))
                rect.matrix[j][i] = unew

        iter_count += 1
        if max_diff <= EPS:
            break

    print(f"SSOR завершён за {iter_count} итераций, норма: {max_diff:.2e}")
    return max_diff


def get_rect():
    rectangle1 = Rectangle(r1x, r1y, r1w, r1h, grid_step)
    rectangle1.set_boundary(boundary)

    rectangle2 = Rectangle(r2x, r2y, r2w, r2h, grid_step)
    rectangle2.set_boundary(boundary)

    return rectangle1, rectangle2


if __name__ == '__main__':
    rect1, rect2 = get_rect()
    schwartz_method(rect1, rect2, f_source, max_iter_schwarz, max_iter_ssor)

