import numpy as np
import matplotlib.pyplot as plt

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


class Rectangle:
    def __init__(self, x, y, width, height, step):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.step = step

        self.grid_width = width // step + 1
        self.grid_height = height // step + 1

        print(f"grid_width: {self.grid_width}, grid_height: {self.grid_height}")
        self.matrix = np.zeros((self.grid_height, self.grid_width), dtype=float)
        self.cross_matrix: np.ndarray | None = None

    def set_boundary(self, func):
        # Верхняя и нижняя границы (по строкам)
        i_vals = np.arange(self.grid_width)
        global_x_vals = self.x + self.step * i_vals
        global_y_top = self.y + self.height
        global_y_bottom = self.y

        self.matrix[0, :] = func(global_x_vals, global_y_top)
        self.matrix[-1, :] = func(global_x_vals, global_y_bottom)

        # Левая и правая границы (по столбцам)
        j_vals = np.arange(self.grid_height)
        global_y_vals = self.y + self.height - self.step * j_vals
        global_x_left = self.x
        global_x_right = self.x + self.width

        self.matrix[:, 0] = np.vectorize(func)(global_x_left, global_y_vals)
        self.matrix[:, -1] = np.vectorize(func)(global_x_right, global_y_vals)

    def print_matrix(self):
        for row in self.matrix:
            print(' '.join(f"{val:5.2f}" for val in row))

    def merge_from_intersection(self, other):
        """Сливает матрицы в области пересечения, беря значения из other."""
        # Границы пересечения
        x_start = max(self.x, other.x)
        x_end = min(self.x + self.width, other.x + other.width)
        y_start = max(self.y, other.y)
        y_end = min(self.y + self.height, other.y + other.height)

        if x_end <= x_start or y_end <= y_start:
            print("Пересечение отсутствует")
            return

        print(f"Область пересечения: x ∈ [{x_start}, {x_end}], y ∈ [{y_start}, {y_end}]")

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

        # Копируем данные из other в self
        self.matrix[j_start_self:j_end_self, i_start_self:i_end_self] = \
            other.matrix[j_start_other:j_end_other, i_start_other:i_end_other]


# Функция f(x, y)
def f_source(x, y):
    return x*x - y


# Граничное условие
def boundary(x, y):
    return 1


def draw_field(rects: list[Rectangle]):
    all_x = [rect1.x, rect1.x + rect1.width, rect2.x, rect2.x + rect2.width]
    all_y = [rect1.y, rect1.y + rect1.height, rect2.y, rect2.y + rect2.height]

    for rect in rects:
        plt.imshow(rect.matrix,
                   origin='lower',
                   extent=(rect.x, rect.x + rect.width, rect.y, rect.y + rect.height),
                   cmap='jet',
                   alpha=1)

    # Настройки графика
    plt.colorbar(label='u(x, y)')
    plt.title('Два прямоугольника на одном графике')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.xlim(min(all_x), max(all_x))
    plt.ylim(min(all_y), max(all_y))
    plt.tight_layout()
    plt.autoscale(enable=True)
    plt.show()


def swartz_method(rectangle1: Rectangle, rectangle2: Rectangle, func, max_it=10000, max_ssor_it=10000):

    for iter_num in range(max_it):
        print(f"Iteration {iter_num + 1}")
        print("Solving G2...")
        iters_g2 = SSOR(rectangle2, func, max_ssor_it)
        print(f"Solved G2 in {iters_g2} iterations")

        # Копируем данные из G2 в G1
        rectangle1.merge_from_intersection(rectangle2)

        print("Solving G1...")
        iters_g1 = SSOR(rectangle1, func, max_ssor_it)
        print(f"Solved G1 in {iters_g1} iterations")

        # Копируем данные из G1 в G2
        rectangle2.merge_from_intersection(rectangle1)

        if iter_num % 10 == 0:
            draw_field([rectangle1, rectangle2])

    print("Метод Шварца завершён")


def SSOR(rect: Rectangle, func, max_it=100000):
    hx2 = rect.step ** 2
    hy2 = rect.step ** 2
    denom = 2 * (hx2 + hy2)

    iter_count = 0
    max_diff = float('inf')

    while max_diff > EPS and iter_count < max_it:
        max_diff = 0.0

        # Прямой проход: от внутренних точек
        for i in range(1, rect.grid_width - 1):
            x = rect.x + i * rect.step
            for j in range(1, rect.grid_height - 1):
                y = rect.y + j * rect.step

                oval = rect.matrix[j][i]
                sum_val = (
                    hy2 * (rect.matrix[j][i + 1] + rect.matrix[j][i - 1]) +
                    hx2 * (rect.matrix[j + 1][i] + rect.matrix[j - 1][i]) +
                    hx2 * hy2 * func(x, y)
                )
                rect.matrix[j][i] = (1 - OMEGA) * oval + OMEGA * sum_val / denom
                max_diff = max(max_diff, abs(rect.matrix[j][i] - oval))

        # Обратный проход
        for i in reversed(range(1, rect.grid_width - 1)):
            x = rect.x + i * rect.step
            for j in reversed(range(1, rect.grid_height - 1)):
                y = rect.y + j * rect.step

                oval = rect.matrix[j][i]
                sum_val = (
                    hy2 * (rect.matrix[j][i + 1] + rect.matrix[j][i - 1]) +
                    hx2 * (rect.matrix[j + 1][i] + rect.matrix[j - 1][i]) +
                    hx2 * hy2 * func(x, y)
                )
                rect.matrix[j][i] = (1 - OMEGA) * oval + OMEGA * sum_val / denom
                max_diff = max(max_diff, abs(rect.matrix[j][i] - oval))

        iter_count += 1

    print(f"SSOR завершён за {iter_count} итераций")
    return iter_count


rect1 = Rectangle(r1x, r1y, r1w, r1h, grid_step)
rect1.set_boundary(boundary)

rect2 = Rectangle(r2x, r2y, r2w, r2h, grid_step)
rect2.set_boundary(boundary)

swartz_method(rect1, rect2, f_source, max_iter_schwarz, max_iter_ssor)
draw_field([rect1, rect2])
