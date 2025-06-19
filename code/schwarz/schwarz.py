import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from dataclasses import dataclass

# Параметры сетки
grid_step = 1
max_iter_ssor = 100
max_iter_schwarz = 100
total_schwarz_iterations = 0

OMEGA = 1.8
EPS = 1e-12
plt.ion()


@dataclass
class PolarCoords:
    r_start: float
    r_end: float
    theta_start: float
    theta_end: float


@dataclass
class RectCoords:
    x_start: float
    y_start: float
    x_end: float
    y_end: float


class PolarSector:
    def __init__(self, cx, cy, r1, r2, theta1, theta2, r_step, theta_step):
        self.cx = cx  # Центр x
        self.cy = cy  # Центр y
        self.r1 = r1
        self.r2 = r2
        self.theta1 = theta1  # В радианах
        self.theta2 = theta2
        self.is_ring = theta2 % (np.pi * 2) == theta1
        if self.theta2 - self.theta1 > np.pi * 2:
            raise ValueError(f"theta2 - theta1 > 2 * np.pi: {self.theta2 - self.theta1}")

        self.r_step = r_step
        self.theta_step = theta_step

        self.nr = int((r2 - r1) // r_step) + 1
        self.ntheta = int((theta2 - theta1) // theta_step) + 1

        print(f"nr (radial points): {self.nr}, ntheta (angular points): {self.ntheta}")
        self.matrix = np.zeros((self.nr, self.ntheta), dtype=float)
        self.cross_matrix: np.ndarray | None = None

    def __str__(self):
        return f"PolarSector(cx={self.cx}, cy={self.cy}, r1={self.r1}, r2={self.r2}, theta1={self.theta1}, theta2={self.theta2}, r_step={self.r_step}, theta_step={self.theta_step})"

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

        if self.is_ring:
            return self

        # Левая граница дуги (theta = theta1)
        for i in range(self.nr):
            x, y = self.polar_to_cartesian(i, 0)
            self.matrix[i, 0] = func(x, y)

        # Правая граница дуги (theta = theta2)
        for i in range(self.nr):
            x, y = self.polar_to_cartesian(i, self.ntheta - 1)
            self.matrix[i, -1] = func(x, y)

        return self

    def merge_from_intersection_perimeter_rectangle(self, other: "Rectangle"):
        """
        Обновляет значения на периметре полярного сектора, если они находятся внутри прямоугольника.
        Для каждой точки периметра:
          1. Переводит её в декартовы координаты (x, y).
          2. Проверяет, находится ли точка внутри прямоугольника.
          3. Если да — делает билинейную интерполяцию по 4 соседним узлам other.matrix.
          4. Записывает результат в self.matrix.
        """

        # Создаём регулярный интерполятор по сетке прямоугольника
        x_vals = other.x + np.arange(other.grid_width) * other.step
        y_vals = other.y + np.arange(other.grid_height) * other.step
        interpolator = RegularGridInterpolator((y_vals, x_vals), other.matrix,
                                               bounds_error=False, fill_value=np.nan)

        # Проходим по периметру полярного сектора
        for j in [0, self.nr - 1]:  # Внутренний и внешний радиус
            for i in range(self.ntheta):
                r = self.r1 + j * self.r_step
                theta = self.theta1 + i * self.theta_step
                x = self.cx + r * np.cos(theta)
                y = self.cy + r * np.sin(theta)

                # Проверяем, находится ли точка внутри прямоугольника
                if not (other.x <= x <= other.x + other.width and
                        other.y <= y <= other.y + other.height):
                    continue

                # Интерполируем значение
                value = interpolator((y, x))
                if not np.isnan(value):
                    self.matrix[j, i] = value

        if self.is_ring:
            return self

        for i in [0, self.ntheta - 1]:  # Левая и правая дуги
            for j in range(1, self.nr - 1):  # Пропускаем уже обработанные углы
                r = self.r1 + j * self.r_step
                theta = self.theta1 + i * self.theta_step
                x = self.cx + r * np.cos(theta)
                y = self.cy + r * np.sin(theta)

                if not (other.x <= x <= other.x + other.width and
                        other.y <= y <= other.y + other.height):
                    continue

                value = interpolator((y, x))
                if not np.isnan(value):
                    self.matrix[j, i] = value

        return self

    def merge_from_intersection_perimeter_sector(self, other: "PolarSector"):
        rect = self.get_intersection_coords_sector(other)
        if rect is None:
            return

        r_vals = other.r1 + np.arange(other.nr) * other.r_step
        theta_vals = other.theta1 + np.arange(other.ntheta) * other.theta_step
        interpolator = RegularGridInterpolator(
            (r_vals, theta_vals), other.matrix,
            bounds_error=False, fill_value=np.nan
        )

        # Обрабатываем периметр текущего сектора (self)

        # Внутренний и внешний радиус (по всем углам)
        for i in [0, self.nr - 1]:
            for j in range(self.ntheta):
                r = self.r1 + i * self.r_step
                theta = self.theta1 + j * self.theta_step

                value = interpolator((r, theta))
                if not np.isnan(value):
                    self.matrix[i, j] = value

        # Левая и правая границы дуги (по всем радиусам)
        for j in [0, self.ntheta - 1]:
            for i in range(1, self.nr - 1):
                r = self.r1 + i * self.r_step
                theta = self.theta1 + j * self.theta_step

                value = interpolator((r, theta))
                if not np.isnan(value):
                    self.matrix[i, j] = value

        return self

    def merge_from_intersection_perimeter(self, other):
        if isinstance(other, Rectangle):
            return self.merge_from_intersection_perimeter_rectangle(other)
        elif isinstance(other, PolarSector):
            return self.merge_from_intersection_perimeter_sector(other)
        raise TypeError(f"Неподдерживаемый тип объекта: {type(other)}")

    def get_intersection_coords_rect(self, rect: "Rectangle") -> PolarCoords | None:
        """
        Возвращает приблизительные границы пересечения с прямоугольником в полярной системе координат.
        """
        # Вычисляем AABB (границы прямоугольника) в полярной системе
        corners = [
            (rect.x, rect.y),
            (rect.x + rect.width, rect.y),
            (rect.x, rect.y + rect.height),
            (rect.x + rect.width, rect.y + rect.height)
        ]

        r_vals = []
        theta_vals = []
        for x, y in corners:
            dx = x - self.cx
            dy = y - self.cy
            r = np.hypot(dx, dy)
            theta = np.arctan2(dy, dx)
            # Приводим θ в диапазон [0, 2π) для корректности
            if theta < 0:
                theta += 2 * np.pi
            r_vals.append(r)
            theta_vals.append(theta)

        r_start = max(self.r1, min(r_vals))
        r_end = min(self.r2, max(r_vals))
        theta_start = max(self.theta1, min(theta_vals))
        theta_end = min(self.theta2, max(theta_vals))

        if r_end <= r_start or theta_end <= theta_start:
            return None
        return PolarCoords(r_start, r_end, theta_start, theta_end)

    def get_intersection_coords_sector(self, other: "PolarSector") -> PolarCoords | None:
        # Сэмплируем точки по границе другого сектора в декартовой системе
        other_points = []

        for j in [0, other.nr - 1]:  # по r
            for i in range(other.ntheta):
                x, y = other.polar_to_cartesian(j, i)
                other_points.append((x, y))

        for i in [0, other.ntheta - 1]:  # по θ
            for j in range(other.nr):
                x, y = other.polar_to_cartesian(j, i)
                other_points.append((x, y))

        # Переводим все точки other в полярные координаты относительно self
        r_thetas = []
        for x, y in other_points:
            dx = x - self.cx
            dy = y - self.cy
            r = np.hypot(dx, dy)
            theta = np.arctan2(dy, dx) + np.pi
            r_thetas.append((r, theta))

        # Фильтруем те точки, что попадают в self
        inside = [
            (r, theta)
            for r, theta in r_thetas
            if self.r1 <= r <= self.r2 and self.theta1 <= theta <= self.theta2
        ]

        if not inside:
            return None

        # Находим минимальные и максимальные значения r и θ в зоне пересечения
        rs, thetas = zip(*inside)
        r_start = max(min(rs), self.r1)
        r_end = min(max(rs), self.r2)
        theta_start = max(min(thetas), self.theta1)
        theta_end = min(max(thetas), self.theta2)

        if r_end <= r_start or theta_end <= theta_start:
            return None

        return PolarCoords(r_start, r_end, theta_start, theta_end)

    def get_intersection_coords(self, other):
        if isinstance(other, Rectangle):
            return self.get_intersection_coords_rect(other)
        elif isinstance(other, PolarSector):
            return self.get_intersection_coords_sector(other)
        raise TypeError(f"Неподдерживаемый тип объекта: {type(other)}")

    def has_intersection_rect(self, other: "Rectangle"):
        for j in range(other.matrix.shape[0]):
            for i in range(other.matrix.shape[1]):

                # Глобальные декартовы координаты точки
                x = other.x + i * other.step
                y = other.y + j * other.step

                # Переводим в полярные координаты относительно other
                dx = x - self.cx
                dy = y - self.cy
                r = np.hypot(dx, dy)
                theta = np.arctan2(dy, dx) % (2 * np.pi)

                # Проверяем попадание в сектор
                if not (self.r1 <= r <= self.r2):
                    continue
                if not (self.theta1 <= theta <= self.theta2):
                    continue
                return True
        return False

    def has_intersection(self, other) -> bool:
        if isinstance(other, Rectangle):
            return self.has_intersection_rect(other)
        elif isinstance(other, PolarSector):
            return self.get_intersection_coords_sector(other) is not None
        raise TypeError(f"Неподдерживаемый тип объекта: {type(other)}")

    def compute_norm_in_intersection_rect(self, other: "Rectangle"):
        """
        Вычисляет максимальное и среднеквадратичное различие значений между self и other
        в зоне их пересечения.

        1. Создаётся temp_matrix как копия self.matrix, но с nan вне пересечения.
        2. Для каждой точки temp_matrix:
           a. Переводится в декартовы координаты (x, y).
           b. Проверяется принадлежность к other.
           c. Если принадлежит — интерполируется значение из other.
        3. Сравниваются self.matrix и interpolated_values.
        """

        # Шаг 1: Создаём временную матрицу с nan
        temp_matrix = np.full_like(self.matrix, np.nan)

        # Шаг 2: Подготавливаем интерполятор для other
        x_vals = other.x + np.arange(other.grid_width) * other.step
        y_vals = other.y + np.arange(other.grid_height) * other.step
        interpolator = RegularGridInterpolator(
            (y_vals, x_vals), other.matrix,
            bounds_error=False, fill_value=np.nan
        )

        # Шаг 3: Проходим по всем точкам self.matrix
        for i in range(self.nr):
            for j in range(self.ntheta):
                r = self.r1 + i * self.r_step
                theta = self.theta1 + j * self.theta_step

                # Переводим в декартовы координаты
                x = self.cx + r * np.cos(theta)
                y = self.cy + r * np.sin(theta)

                # Проверяем, находится ли точка внутри прямоугольника
                if not (other.x <= x <= other.x + other.width and
                        other.y <= y <= other.y + other.height):
                    continue

                # Интерполируем значение из other
                value = interpolator((y, x))
                if not np.isnan(value):
                    temp_matrix[i, j] = value

        # Шаг 4: Сравниваем только те точки, где оба значения определены
        valid_mask = ~np.isnan(temp_matrix) & ~np.isnan(self.matrix)
        if not np.any(valid_mask):
            return 0.0, 0.0

        diffs = np.abs(self.matrix[valid_mask] - temp_matrix[valid_mask])
        max_diff = np.max(diffs)
        l2_norm = np.sqrt(np.mean(diffs ** 2))

        return max_diff, l2_norm

    def compute_norm_in_intersection_sector(self, other: "PolarSector"):
        # Получаем пересечение в полярных координатах
        polar_clip = self.get_intersection_coords_sector(other)
        if polar_clip is None:
            return 0, 0

        diffs = []

        # Перебираем точки пересечения по сетке в полярных координатах
        r_vals = np.arange(polar_clip.r_start, polar_clip.r_end, self.r_step)
        theta_vals = np.arange(polar_clip.theta_start, polar_clip.theta_end, self.theta_step)

        for r in r_vals:
            for theta in theta_vals:
                # Индексы в self
                ri_self = int((r - self.r1) / self.r_step)
                ti_self = int((theta - self.theta1) / self.theta_step)
                if ri_self < 0 or ri_self >= self.nr or ti_self < 0 or ti_self >= self.ntheta:
                    continue

                # Индексы в other
                ri_other = int((r - other.r1) / other.r_step)
                ti_other = int((theta - other.theta1) / other.theta_step)
                if ri_other < 0 or ri_other >= other.nr or ti_other < 0 or ti_other >= other.ntheta:
                    continue

                v_self = self.matrix[ri_self, ti_self]
                v_other = other.matrix[ri_other, ti_other]
                diffs.append(abs(v_self - v_other))

        if not diffs:
            return 0, 0

        diffs = np.array(diffs)
        max_diff = np.max(diffs)
        l2_norm = np.sqrt(np.mean(diffs ** 2))
        return max_diff, l2_norm

    def compute_norm_in_intersection(self, other):
        if isinstance(other, Rectangle):
            return self.compute_norm_in_intersection_rect(other)
        elif isinstance(other, PolarSector):
            return self.compute_norm_in_intersection_sector(other)
        raise TypeError(f"Неподдерживаемый тип объекта: {type(other)}")

    def ssor(self, rhs_func, max_it=100000):
        dr = self.r_step
        dtheta = self.theta_step
        dr2 = dr * dr
        dtheta2 = dtheta * dtheta
        denom = 2.0 * (1.0 / dr2 + 1.0 / (self.r1 ** 2 * dtheta2))  # стартовая оценка

        iter_count = 0

        while iter_count < max_it:
            max_diff = 0.0

            max_diff = max(max_diff, self._ssor_pass(rhs_func, forward=True, dr2=dr2, dtheta2=dtheta2))
            max_diff = max(max_diff, self._ssor_pass(rhs_func, forward=False, dr2=dr2, dtheta2=dtheta2))

            iter_count += 1
            if max_diff <= EPS:
                break

        print(f"SSOR для PolarSector завершён за {iter_count} итераций, норма: {max_diff:.2e}")
        return max_diff

    def _ssor_pass(self, rhs_func, forward: bool, dr2: float, dtheta2: float) -> float:
        max_diff = 0.0

        j_range = range(1, self.nr - 1)
        i_range = range(self.ntheta)  # Обрабатываем все углы

        if not forward:
            j_range = reversed(j_range)
            i_range = reversed(i_range)

        for j in j_range:
            r = self.r1 + j * self.r_step
            for i in i_range:
                theta = self.theta1 + i * self.theta_step
                x = self.cx + r * np.cos(theta)
                y = self.cy + r * np.sin(theta)

                rhs = rhs_func(x, y)

                # Индексы по theta, с учётом зацикливания, если кольцо
                i_minus = (i - 1) % self.ntheta if self.is_ring else i - 1
                i_plus = (i + 1) % self.ntheta if self.is_ring else i + 1

                # Пропускаем граничные точки, если не кольцо
                if not self.is_ring and (i_minus < 0 or i_plus >= self.ntheta):
                    continue

                # Основная формула
                denom = 2.0 * (1.0 / dr2 + 1.0 / (r * r * dtheta2))
                sum_val = (
                        (self.matrix[j][i_plus] + self.matrix[j][i_minus]) / (r * r * dtheta2) +
                        (self.matrix[j + 1][i] + self.matrix[j - 1][i]) / dr2 +
                        rhs
                )
                unew = (1 - OMEGA) * self.matrix[j][i] + OMEGA * sum_val / denom

                max_diff = max(max_diff, abs(unew - self.matrix[j][i]))
                self.matrix[j][i] = unew

        return max_diff


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


    def __str__(self):
        return f"Rectangle(x={self.x}, y={self.y}, width={self.width}, height={self.height}, step={self.step})"

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

    def get_intersection_coords_sector(self, other: PolarSector) -> RectCoords | None:
        """
        Вычисляет пересечение прямоугольной области (self) с сектором (other) в декартовой системе.
        Возвращает минимальный ограничивающий прямоугольник пересечения, если оно есть.
        """
        # Генерируем точки по сетке сектора в декартовой системе
        r_vals = other.r1 + np.arange(other.nr) * other.r_step
        theta_vals = other.theta1 + np.arange(other.ntheta) * other.theta_step

        xs = []
        ys = []

        for r in r_vals:
            for theta in theta_vals:
                x = other.cx + r * np.cos(theta)
                y = other.cy + r * np.sin(theta)

                # Проверяем, попадает ли точка внутрь прямоугольника
                if self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height:
                    xs.append(x)
                    ys.append(y)

        if not xs or not ys:
            return None

        x_start = max(min(xs), self.x)
        x_end = min(max(xs), self.x + self.width)
        y_start = max(min(ys), self.y)
        y_end = min(max(ys), self.y + self.height)

        return RectCoords(x_start, y_start, x_end, y_end)

    def get_intersection_coords(self, other):
        if isinstance(other, Rectangle):
            return self.get_intersection_coords_rect(other)
        elif isinstance(other, PolarSector):
            return self.get_intersection_coords_sector(other)
        raise TypeError(f"Неподдерживаемый тип объекта: {type(other)}")

    def has_intersection_sector(self, other: PolarSector):
        for j in range(self.matrix.shape[0]):
            for i in range(self.matrix.shape[1]):

                # Глобальные декартовы координаты точки
                x = self.x + i * self.step
                y = self.y + j * self.step

                # Переводим в полярные координаты относительно other
                dx = x - other.cx
                dy = y - other.cy
                r = np.hypot(dx, dy)
                theta = np.arctan2(dy, dx) % (2 * np.pi)

                # Проверяем попадание в сектор
                if not (other.r1 <= r <= other.r2):
                    continue
                if not (other.theta1 <= theta <= other.theta2):
                    continue
                return True
        return False

    def has_intersection(self, other) -> bool:
        if isinstance(other, Rectangle):
            return self.get_intersection_coords_rect(other) is not None
        elif isinstance(other, PolarSector):
            return self.has_intersection_sector(other)
        raise TypeError(f"Неподдерживаемый тип объекта: {type(other)}")

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
        """
        Обновляет значения на периметре прямоугольника, если они находятся внутри полярного сектора.
        Для каждой точки периметра:
          1. Переводит координаты точки в систему other (r, theta).
          2. Проверяет, находится ли точка внутри сектора.
          3. Если да — делает билинейную интерполяцию по 4 соседним узлам other.matrix.
          4. Записывает результат в self.matrix.
        """

        # Создаём маску периметра (все края)
        mask = np.zeros_like(self.matrix, dtype=bool)
        mask[0, :] = True  # верхняя граница
        mask[-1, :] = True  # нижняя граница
        mask[:, 0] = True  # левая граница
        mask[:, -1] = True  # правая граница

        # Шаг сетки прямоугольника
        step = self.step

        # Подготавливаем регулярный интерполятор
        r_vals = other.r1 + np.arange(other.nr) * other.r_step
        theta_vals = other.theta1 + np.arange(other.ntheta) * other.theta_step
        interpolator = RegularGridInterpolator((r_vals, theta_vals), other.matrix,
                                               bounds_error=False, fill_value=np.nan)

        # Проходим по всем точкам периметра
        for j in range(self.matrix.shape[0]):
            for i in range(self.matrix.shape[1]):
                if not mask[j, i]:
                    continue  # Пропускаем внутренние точки

                # Глобальные декартовы координаты точки
                x = self.x + i * step
                y = self.y + j * step

                # Переводим в полярные координаты относительно other
                dx = x - other.cx
                dy = y - other.cy
                r = np.hypot(dx, dy)
                theta = np.arctan2(dy, dx) % (2 * np.pi)

                # Проверяем попадание в сектор
                if r < other.r1 or r > other.r2:
                    continue
                if not (other.theta1 <= theta <= other.theta2):
                    continue

                # Интерполируем значение
                value = interpolator((r, theta))
                if not np.isnan(value):
                    self.matrix[j, i] = value

        return self

    def merge_from_intersection_perimeter(self, other):
        if isinstance(other, Rectangle):
            return self.merge_from_intersection_perimeter_rectangle(other)
        elif isinstance(other, PolarSector):
            return self.merge_from_intersection_perimeter_sector(other)
        raise TypeError(f"Неподдерживаемый тип объекта: {type(other)}")

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
        rect = self.get_intersection_coords_sector(other)
        if rect is None:
            return 0, 0

        x_vals = np.arange(rect.x_start, rect.x_end, self.step)
        y_vals = np.arange(rect.y_start, rect.y_end, self.step)

        diffs = []
        for y in y_vals:
            for x in x_vals:
                i = int((x - self.x) / self.step)
                j = int((y - self.y) / self.step)

                if i < 0 or i >= self.grid_width or j < 0 or j >= self.grid_height:
                    continue

                v_rect = self.matrix[j, i]

                # Переводим (x, y) в полярные координаты
                dx = x - other.cx
                dy = y - other.cy
                r = np.hypot(dx, dy)
                theta = np.arctan2(dy, dx)
                if theta < 0:
                    theta += 2 * np.pi

                # Проверка попадания в сектор
                if not (other.r1 <= r <= other.r2 and other.theta1 <= theta <= other.theta2):
                    continue

                # Индексы в полярной сетке
                ri = int((r - other.r1) / other.r_step)
                ti = int((theta - other.theta1) / other.theta_step)

                if ri < 0 or ri >= other.nr or ti < 0 or ti >= other.ntheta:
                    continue

                v_sector = other.matrix[ri, ti]
                diff = abs(v_rect - v_sector)
                diffs.append(diff)

        if not diffs:
            return 0, 0

        diffs = np.array(diffs)
        max_diff = np.max(diffs)
        l2_norm = np.sqrt(np.mean(diffs ** 2))
        return max_diff, l2_norm

    def compute_norm_in_intersection(self, other):
        if isinstance(other, Rectangle):
            return self.compute_norm_in_intersection_rect(other)
        elif isinstance(other, PolarSector):
            return self.compute_norm_in_intersection_sector(other)
        raise TypeError(f"Неподдерживаемый тип объекта: {type(other)}")

    def ssor(self, rhs_func, max_it=100000):
        hx = self.step
        hy = self.step
        hx2 = hx * hx
        hy2 = hy * hy
        denom = 2.0 * (1.0 / hx2 + 1.0 / hy2)

        iter_count = 0

        while iter_count < max_it:
            max_diff = 0.0

            max_diff = max(max_diff, self._ssor_pass(rhs_func, reverse=False, hx2=hx2, hy2=hy2, denom=denom))
            max_diff = max(max_diff, self._ssor_pass(rhs_func, reverse=True, hx2=hx2, hy2=hy2, denom=denom))

            iter_count += 1
            if max_diff <= EPS:
                break

        print(f"SSOR завершён за {iter_count} итераций, норма: {max_diff:.2e}")
        return max_diff

    def _ssor_pass(self, rhs_func, reverse: bool, hx2: float, hy2: float, denom: float) -> float:
        hx = self.step
        hy = self.step
        max_diff = 0.0

        j_range = range(1, self.grid_height - 1)
        i_range = range(1, self.grid_width - 1)
        if reverse:
            j_range = reversed(j_range)
            i_range = reversed(i_range)

        for j in j_range:
            yj = self.y + j * hy
            for i in i_range:
                xi = self.x + i * hx
                rhs = rhs_func(xi, yj)
                sum_val = (
                        (self.matrix[j][i + 1] + self.matrix[j][i - 1]) / hx2 +
                        (self.matrix[j + 1][i] + self.matrix[j - 1][i]) / hy2 +
                        rhs
                )
                unew = (1 - OMEGA) * self.matrix[j][i] + OMEGA * sum_val / denom
                max_diff = max(max_diff, abs(unew - self.matrix[j][i]))
                self.matrix[j][i] = unew

        return max_diff


# Функция f(x, y)
def f_source(x, y):
    return x % 10 + np.cos(y)


# Граничное условие
def boundary(x, y):
    return 1


def draw_field(objects_array: list[Rectangle | PolarSector], clear=True, name="Шварц"):
    if clear:
        plt.clf()
    else:
        plt.figure()

    all_x = []
    all_y = []

    for obj in objects_array:
        if isinstance(obj, Rectangle):
            all_x += [obj.x, obj.x + obj.width]
            all_y += [obj.y, obj.y + obj.height]

            plt.imshow(obj.matrix,
                       origin='lower',
                       interpolation='none',
                       extent=(obj.x, obj.x + obj.width, obj.y, obj.y + obj.height),
                       cmap='jet',
                       alpha=1)

        elif isinstance(obj, PolarSector):
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


def schwarz_step(obj_1: Rectangle | PolarSector, obj_2: Rectangle | PolarSector, func, max_ssor_it=10000):
    max_diff, l2_norm = None, None

    print("Solving G2...")
    obj_2.ssor(func, max_ssor_it)

    intersection = obj_1.has_intersection(obj_2)
    print(f"Intersection: {intersection}")
    if intersection is not None:
        # Копируем данные из G2 в G1
        obj_1.merge_from_intersection_perimeter(obj_2)

    print("Solving G1...")
    obj_1.ssor(func, max_ssor_it)

    if intersection is not None:
        # Копируем данные из G1 в G2
        obj_2.merge_from_intersection_perimeter(obj_1)

        # Вычисляем норму разности в области пересечения
        max_diff, l2_norm = obj_1.compute_norm_in_intersection(obj_2)

        print(f"Макс. отклонение: {max_diff:.2e}, L2-норма: {l2_norm:.2e}")

    return max_diff, l2_norm


def schwartz_method(obj_1: Rectangle | PolarSector, obj_2: Rectangle | PolarSector,
                    objects_array: list[Rectangle | PolarSector],
                    func, max_it=10000, max_ssor_it=10000):
    draw_field(objects_array, name="Начальное состояние")
    if obj_1.has_intersection(obj_2) is None:
        max_it = 1

    for iter_num in range(max_it):
        print(f"\n\nИтерация {iter_num + 1}")
        max_diff, l2_norm = schwarz_step(obj_1, obj_2, func, max_ssor_it)

        if max_diff is not None and max_diff < EPS:
            draw_field(objects_array,
                       name=f"Итерация {iter_num + 1} (финальная), отклонение: {max_diff:.2e}, L2: {l2_norm:.2e}")
            print("Было достигнуто условие сходимости по max_diff.")
            break

        if iter_num < 5 or (iter_num + 1) % 10 == 0:
            add_max_diff = f", отклонение: {max_diff:.2e}, L2: {l2_norm:.2e}" if max_diff is not None else ""
            draw_field(objects_array,
                       name=f"Итерация {iter_num + 1}{add_max_diff}")

    print(f"Метод Шварца завершён. Выполненных итераций: {iter_num + 1}.")


def get_my_version(circle=True, rectangle=True, divisions=8, step=1, theta_step=0.05):
    circles = []
    if circle:
        for c in range(divisions):
            base_width = np.pi * 2 / divisions
            circles.append(PolarSector(50, 50, 20, 30,
                                       base_width * c, base_width * (c + 1) + np.pi / np.ceil(divisions * 1.5),
                                       step, theta_step).set_boundary(boundary))
    rectangles = []
    if rectangle:
        for r in range(divisions):
            base_width = 50 / divisions
            additional_width = 50 / np.ceil(divisions * 1.5)
            start_x = 25 + base_width * r
            total_width = base_width
            if r == 0:
                total_width += additional_width
            elif r == divisions - 1:
                start_x -= additional_width
            else:
                start_x -= additional_width / 2
                total_width += additional_width / 2
            rectangles.append(Rectangle(25 + base_width * r, 45, base_width + 50 / np.ceil(divisions * 1.5), 10, step).set_boundary(boundary))
    return circles + rectangles


def get_two_rects():
    r1x, r1y = 0, 30
    r1w, r1h = 50, 20
    r2x, r2y = 20, 0
    r2w, r2h = 40, 70
    rectangle1 = Rectangle(r1x, r1y, r1w, r1h, grid_step).set_boundary(boundary)
    rectangle2 = Rectangle(r2x, r2y, r2w, r2h, grid_step).set_boundary(boundary)
    return [rectangle1, rectangle2]


def get_diagonal_rects(scale, rect_num):
    return_rects: list[Rectangle] = []
    for i in range(0, rect_num):
        rect = Rectangle(i * 2 * scale, i * 2 * scale, 3 * scale, 3 * scale, grid_step).set_boundary(boundary)
        return_rects.append(rect)
    return return_rects


def get_circle_with_rect(scale=1):
    circle = PolarSector(50, 50, 20, 30, 0, np.pi * 2, scale, np.pi / 15 * scale).set_boundary(boundary)
    rect = Rectangle(25, 45, 50, 10, scale).set_boundary(boundary)
    return [circle, rect]


def get_two_circles():
    circle1 = PolarSector(50, 50, 20, 30, np.pi, np.pi * 2, 0.5, 0.1).set_boundary(boundary)
    circle2 = PolarSector(50, 50, 20, 30, 0, np.pi * 1.2, 0.5, 0.1).set_boundary(boundary)
    return [circle1, circle2]


if __name__ == '__main__':
    # initial_rects: list[Rectangle] = get_diagonal_rects(10, 4)  # get_two_rects()
    # initial_sectors: list[PolarSector] = [PolarSector(40, 40, 10, 20, 0, np.pi * 2, 0.5, 0.1).set_boundary(boundary)]
    rects_sectors: list[Rectangle | PolarSector] = get_my_version(circle=True, rectangle=True, divisions=4, step=1, theta_step=0.05)
    for r1_index in range(len(rects_sectors)):
        for r2_index in range(len(rects_sectors)):
            if r1_index == r2_index:
                continue
            obj1 = rects_sectors[r1_index]
            obj2 = rects_sectors[r2_index]
            obj1_name = "прямоугольник" if isinstance(obj1, Rectangle) else "дуга"
            obj2_name = "прямоугольник" if isinstance(obj2, Rectangle) else "дуга"
            print("=" * 50)
            print(f"Рассматриваемые объекты: {r1_index + 1} ({obj1_name}), {r2_index + 1} ({obj2_name})")
            schwartz_method(obj1, obj2, rects_sectors, f_source, max_iter_schwarz, max_iter_ssor)
            print("=" * 50)
    plt.show(block=True)
