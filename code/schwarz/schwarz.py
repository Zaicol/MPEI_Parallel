import numpy as np
from matplotlib import pyplot as plt

# Параметры сетки
grid_step = 1
max_iter_ssor = 100
max_iter_schwarz = 100
total_schwarz_iterations = 0

OMEGA = 1.8
EPS = 1e-12
plt.ion()


class RectCoords:
    def __init__(self, x_start, y_start, x_end, y_end):
        self.x_start = x_start
        self.y_start = y_start
        self.x_end = x_end
        self.y_end = y_end


class PolarSector:
    def __init__(self, cx, cy, r1, r2, theta1, theta2, r_step, theta_step):
        self.cx = cx  # Центр x
        self.cy = cy  # Центр y
        self.r1 = r1
        self.r2 = r2
        self.theta1 = theta1  # В радианах
        self.theta2 = theta2
        self.r_step = r_step
        self.theta_step = theta_step

        self.nr = int((r2 - r1) // r_step) + 1
        self.ntheta = int((theta2 - theta1) // theta_step) + 1

        print(f"nr (radial points): {self.nr}, ntheta (angular points): {self.ntheta}")
        self.matrix = np.zeros((self.nr, self.ntheta), dtype=float)
        self.cross_matrix: np.ndarray | None = None

    def polar_to_cartesian(self, r_idx, theta_idx):
        """Преобразует индексы сетки в декартовы координаты."""
        r = self.r1 + r_idx * self.r_step
        theta = self.theta1 + theta_idx * self.theta_step
        x = self.cx + r * np.cos(theta)
        y = self.cy + r * np.sin(theta)
        return x, y

    def set_boundary(self, func):
        """Устанавливает граничные условия по внешнему и внутреннему радиусу и по краям дуги."""
        # Внутренний радиус (r = r1)
        for j in range(self.ntheta):
            x, y = self.polar_to_cartesian(0, j)
            self.matrix[0, j] = func(x, y)

        # Внешний радиус (r = r2)
        for j in range(self.ntheta):
            x, y = self.polar_to_cartesian(self.nr - 1, j)
            self.matrix[-1, j] = func(x, y)

        # Левая граница дуги (theta = theta1)
        for i in range(self.nr):
            x, y = self.polar_to_cartesian(i, 0)
            self.matrix[i, 0] = func(x, y)

        # Правая граница дуги (theta = theta2)
        for i in range(self.nr):
            x, y = self.polar_to_cartesian(i, self.ntheta - 1)
            self.matrix[i, -1] = func(x, y)

        return self


class Rectangle:
    def __init__(self, x, y, width, height, step):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.step = step

        self.grid_width = int(width // step)
        self.grid_height = int(height // step)

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

        return self

    def print_matrix(self):
        for row in self.matrix:
            print(' '.join(f"{val:5.2f}" for val in row))

    def get_intersection_coords_rect(self, other: "Rectangle") -> RectCoords | None:
        """
        Возвращает объект RectCoords с границами пересечения двух областей.
        Если пересечения нет — возвращает None.
        """
        x_start = max(self.x, other.x)
        x_end = min(self.x + self.width, other.x + other.width)
        y_start = max(self.y, other.y)
        y_end = min(self.y + self.height, other.y + other.height)

        if x_end <= x_start or y_end <= y_start:
            return None
        return RectCoords(x_start, y_start, x_end, y_end)

    def get_intersection_coords_sector(self, other):
        raise NotImplementedError

    def get_intersection_coords(self, other):
        if isinstance(other, Rectangle):
            return self.get_intersection_coords_rect(other)
        elif isinstance(other, PolarSector):
            return self.get_intersection_coords_sector(other)

    def get_index_clip_rect(self, rect: RectCoords):
        """
        По глобальным координатам rect возвращает кортеж индексов (i_start, i_end, j_start, j_end)
        для среза матрицы: [j_start:j_end+1, i_start:i_end+1]
        """
        i_start = int(round((rect.x_start - self.x) / self.step))
        i_end = int(round((rect.x_end - self.x) / self.step))
        j_start = int(round((rect.y_start - self.y) / self.step))
        j_end = int(round((rect.y_end - self.y) / self.step))

        i_start = np.clip(i_start, 0, self.grid_width)
        i_end = np.clip(i_end, 0, self.grid_width)
        j_start = np.clip(j_start, 0, self.grid_height)
        j_end = np.clip(j_end, 0, self.grid_height)

        return i_start, i_end, j_start, j_end

    def compute_norm_in_intersection_rect(self, other: "Rectangle"):
        rect = self.get_intersection_coords_rect(other)
        if rect is None:
            return

        i_start_self, i_end_self, j_start_self, j_end_self = self.get_index_clip_rect(rect)
        i_start_other, i_end_other, j_start_other, j_end_other = other.get_index_clip_rect(rect)

        roi_g1 = self.matrix[j_start_self:j_end_self, i_start_self:i_end_self]
        roi_g2 = other.matrix[j_start_other:j_end_other, i_start_other:i_end_other]

        assert roi_g1.shape == roi_g2.shape, "Размерности областей не совпадают"

        diff = np.abs(roi_g1 - roi_g2)
        max_diff = np.max(diff)
        sum_sq = np.sum(diff ** 2)
        n_points = roi_g1.size
        l2_norm = np.sqrt(sum_sq / n_points)

        return max_diff, l2_norm

    def compute_norm_in_intersection_sector(self, other: PolarSector):
        raise NotImplementedError

    def compute_norm_in_intersection(self, other):
        if isinstance(other, Rectangle):
            return self.compute_norm_in_intersection_rect(other)
        elif isinstance(other, PolarSector):
            return self.compute_norm_in_intersection_sector(other)

    def merge_from_intersection_perimeter_rectangle(self, other: "Rectangle"):
        rect = self.get_intersection_coords_rect(other)
        if rect is None:
            return

        i_start_self, i_end_self, j_start_self, j_end_self = self.get_index_clip_rect(rect)
        i_start_other, i_end_other, j_start_other, j_end_other = other.get_index_clip_rect(rect)

        # Выделяем пересекающиеся области
        self_roi = self.matrix[j_start_self:j_end_self, i_start_self:i_end_self]
        other_roi = other.matrix.copy()[j_start_other:j_end_other, i_start_other:i_end_other]

        # Создаем маску для периметра (остальные значения станут np.nan)
        mask = np.full_like(self.matrix, False, dtype=bool)
        mask[0, :] = True  # верхняя строка
        mask[-1, :] = True  # нижняя строка
        mask[:, 0] = True  # левая колонка
        mask[:, -1] = True  # правая колонка

        self_masked = np.where(mask, self.matrix.copy(), np.nan)[j_start_self:j_end_self, i_start_self:i_end_self]

        # Копируем только не-nan значения из other_masked в self_roi_part
        np.copyto(self_roi, other_roi, where=~np.isnan(self_masked))

        return self

    def merge_from_intersection_perimeter_sector(self, other: PolarSector):
        raise NotImplementedError

    def merge_from_intersection_perimeter(self, other):
        if isinstance(other, Rectangle):
            return self.merge_from_intersection_perimeter_rectangle(other)
        elif isinstance(other, PolarSector):
            return self.merge_from_intersection_perimeter_sector(other)


# Функция f(x, y)
def f_source(x, y):
    return 1


# Граничное условие
def boundary(x, y):
    return 1


def draw_field(rects: list[Rectangle], sectors: list[PolarSector], clear=True, name="Шварц"):
    if clear:
        plt.clf()
    else:
        plt.figure()

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

    for obj in sectors:
        # Шаг 1: Вычисляем границы ячеек
        r_edges = np.linspace(obj.r1 - obj.r_step / 2, obj.r2 + obj.r_step / 2, obj.nr + 1)
        theta_edges = np.linspace(obj.theta1 - obj.theta_step / 2, obj.theta2 + obj.theta_step / 2, obj.ntheta + 1)

        R_edges, THETA_edges = np.meshgrid(r_edges, theta_edges, indexing='ij')
        X_edges = obj.cx + R_edges * np.cos(THETA_edges)
        Y_edges = obj.cy + R_edges * np.sin(THETA_edges)

        all_x += [np.min(X_edges), np.max(X_edges)]
        all_y += [np.min(Y_edges), np.max(Y_edges)]

        # Шаг 2: Используем границы ячеек
        plt.pcolormesh(X_edges, Y_edges, obj.matrix, shading='flat', cmap='jet', alpha=1)

    plt.colorbar(label='u(x, y)')
    plt.title(name)
    plt.xlabel('x')
    plt.ylabel('y')

    if all_x and all_y:
        plt.xlim(min(all_x), max(all_x))
        plt.ylim(min(all_y), max(all_y))

    plt.tight_layout()
    plt.draw()
    plt.pause(0.5)


def schwarz_step(rectangle1: Rectangle, rectangle2: Rectangle, func, max_ssor_it=10000):
    max_diff, l2_norm = None, None

    intersection = rectangle1.get_intersection_coords(rectangle2)
    print("Solving G2...")
    SSOR(rectangle2, func, max_ssor_it)

    if intersection is not None:
        # Копируем данные из G2 в G1
        rectangle1.merge_from_intersection_perimeter(rectangle2)

    print("Solving G1...")
    SSOR(rectangle1, func, max_ssor_it)

    if intersection is not None:
        # Копируем данные из G1 в G2
        rectangle2.merge_from_intersection_perimeter(rectangle1)

        # Вычисляем норму разности в области пересечения
        max_diff, l2_norm = rectangle1.compute_norm_in_intersection(rectangle2)

        print(f"Макс. отклонение: {max_diff:.2e}, L2-норма: {l2_norm:.2e}")

    return max_diff, l2_norm


def schwartz_method(rectangle1: Rectangle, rectangle2: Rectangle, rect_array: list[Rectangle],
                    sect_array: list[PolarSector], func, max_it=10000, max_ssor_it=10000):
    draw_field(rect_array, sect_array, name="Начальное состояние")
    if rectangle1.get_intersection_coords(rectangle2) is None:
        max_it = 1

    for iter_num in range(max_it):
        print(f"\n\nИтерация {iter_num + 1}")
        max_diff, l2_norm = schwarz_step(rectangle1, rectangle2, func, max_ssor_it)

        if max_diff is not None and max_diff < EPS:
            draw_field(rect_array, sect_array,
                       name=f"Итерация {iter_num + 1} (финальная), отклонение: {max_diff:.2e}, L2: {l2_norm:.2e}")
            print("Было достигнуто условие сходимости по max_diff.")
            break

        if iter_num < 5 or (iter_num + 1) % 10 == 0:
            add_max_diff = f", отклонение: {max_diff:.2e}, L2: {l2_norm:.2e}" if max_diff is not None else ""
            draw_field(rect_array, sect_array,
                       name=f"Итерация {iter_num + 1}{add_max_diff}")

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


def get_two_rects():
    r1x, r1y = 0, 30
    r1w, r1h = 50, 20
    r2x, r2y = 20, 0
    r2w, r2h = 40, 70
    rectangle1 = Rectangle(r1x, r1y, r1w, r1h, grid_step).set_boundary(boundary)
    rectangle2 = Rectangle(r2x, r2y, r2w, r2h, grid_step).set_boundary(boundary)
    return rectangle1, rectangle2


def get_diagonal_rects(scale, rect_num):
    return_rects: list[Rectangle] = []
    for i in range(0, rect_num):
        rect = Rectangle(i * 2 * scale, i * 2 * scale, 3 * scale, 3 * scale, grid_step).set_boundary(boundary)
        return_rects.append(rect)
    return return_rects


if __name__ == '__main__':
    initial_rects: list[Rectangle] = get_diagonal_rects(10, 4)  # get_two_rects()
    initial_sectors: list[PolarSector] = [PolarSector(40, 40, 10, 20, 0, np.pi * 2, 0.5, 0.1).set_boundary(boundary)]
    draw_field([], initial_sectors)
    for r1_index in range(len(initial_rects)):
        for r2_index in range(len(initial_rects)):
            if r1_index == r2_index:
                continue
            rect1 = initial_rects[r1_index]
            rect2 = initial_rects[r2_index]
            print("=" * 50)
            print(f"Рассматриваемые прямоугольники: {r1_index + 1}, {r2_index + 1}")
            schwartz_method(rect1, rect2, initial_rects, initial_sectors, f_source, max_iter_schwarz, max_iter_ssor)
            print("=" * 50)
    input("Ожидание ввода...")  # Чтобы matplotlib не закрывался
