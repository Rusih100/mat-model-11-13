from matplotlib import pyplot as plt
from random import randint, choice, random
from math import factorial
from functools import wraps
from tqdm import tqdm

import numpy as np
import itertools
import time
import os

path = r'C:/Users/user/Desktop/mm'  # win
# path = r'/Users/dashapetrova/Desktop/mm'  # mac

EXP = np.e


def timeit(method):
    """Декоратор для измерения времени"""
    @wraps(method)
    def timed(*args, **kw):
        ts = time.monotonic()
        result = method(*args, **kw)
        te = time.monotonic()
        s = te - ts
        print(f'{method.__name__}: {s:2.3f} s')
        return result
    return timed


def create_random_dot():
    """Создает рандомную точку в пределах от 0 до 100"""
    return randint(0, 100), randint(0, 100)


def create_dot_list(n):
    """Генерирует лист с рандомными точками"""
    return [create_random_dot() for _ in range(n)]


def distance_dot(delta_x, delta_y):
    """Расстояние между двумя точками, принимает разность координат"""
    return (delta_x**2 + delta_y**2) ** 0.5


def get_distance_matrix(count_city, dots):
    """Матрица расстояний для городов"""
    data = [[x for x in range(count_city)] for y in range(count_city)]
    for i in range(count_city):
        for j in range(count_city):
            data[i][j] = distance_dot(dots[i][0] - dots[j][0], dots[i][1] - dots[j][1])
    return data


def get_combinations_list(cities):
    """Все возможные комбинации путей, проходящее через все города"""
    result = []
    for item in tqdm(itertools.permutations(cities, len(cities)), total=factorial(len(cities)),
                     desc='Генерация всех возможных путей'):
        item_list = list(item) + [item[0]]
        result.append(item_list)
    return result


def get_distance_path(data, dist_matrix):
    """Вычисляет общую длину пути"""
    result = []
    for index in range(len(data) - 1):
        _ = dist_matrix[data[index]][data[index+1]]
        result.append(_)
    return sum(result)


def search_min_combination(combinations, dist_matrix):
    """Однопроходный перебор всех комбинаций и поиск среди них минимальной.
    Возвращает кортеж (<конфигурация>, <длина>)"""
    min_path = 10e5
    min_comb = None
    for comb in tqdm(combinations, total=factorial(len(dist_matrix)), desc='Поиск минимального пути'):
        len_path = get_distance_path(comb, dist_matrix)
        if len_path < min_path:
            min_path = len_path
            min_comb = comb
    return min_comb, min_path


def combination_list_to_dot(combination, dot_list):
    """Переделывает лист коомбинаций с номерами городов в лист с координатами"""
    return [dot_list[i] for i in combination]


def get_all_path_coord(dot_list):
    """Возвращает список со всеми маршрутами"""
    result = []
    for x, y in itertools.combinations(dot_list, 2):
        result.append(x)
        result.append(y)
    return result


def save_result_to_txt(coord, name):
    """Записывает последовательность точек в файл"""
    file_path = path + r'/' + name + '.txt'

    try:
        os.remove(file_path)
    except FileNotFoundError:
        pass

    with open(file_path, 'w') as file:
        for x, y in coord:
            file.write(f'{x} {y} \n')


def get_random_path(cities):
    """Генерирует начальный случайный путь на основе списка городов"""
    cities_copy = list(cities)
    result = []
    for j in range(len(cities)):
        item = choice(cities_copy)
        cities_copy.remove(item)
        result.append(item)
    return result


def random_pair(n):
    """Генерирует пару случайных не одинаковых чисел"""
    x, y = randint(0, n), randint(0, n)
    if x != y:
        return x, y
    else:
        return random_pair(n)


def monte_carlo(steps, cities, dist_matrix):
    """Жадный алгоритм в методе Монте-Карло"""

    cities = get_random_path(cities)
    start_cities = cities.copy()
    path_len = get_distance_path(cities + [cities[0]], dist_matrix)

    for step in tqdm(range(steps), desc='Жадный алгоритм'):
        i, j = random_pair(len(cities) - 1)
        cities[i], cities[j] = cities[j], cities[i]
        new_path = get_distance_path(cities + [cities[0]], dist_matrix)
        if new_path < path_len:
            path_len = new_path
        else:
            cities[i], cities[j] = cities[j], cities[i]

    start_cities += [start_cities[0]]
    cities += [cities[0]]
    return start_cities, cities, path_len


def metropolis(steps, cities, dist_matrix):
    """Алгоритм имитации отжига / Алгоритм Метрополиса"""
    cities = get_random_path(cities)
    start_cities = cities.copy()
    path_old = get_distance_path(cities + [cities[0]], dist_matrix)

    for temperature in tqdm(np.arange(5, 0, -0.01), desc='Алгоритм Метрополиса'):

        for step in range(steps):

            save_path = cities.copy()

            i, j = random_pair(len(cities) - 1)
            cities[i], cities[j] = cities[j], cities[i]

            path_new = get_distance_path(cities + [cities[0]], dist_matrix)

            delta = path_new - path_old

            if delta > 0:
                f = EXP ** (-delta / temperature)
                r = random()
                if r > f:
                    cities = save_path.copy()
                    path_new = path_old

            path_old = path_new

    start_cities += [start_cities[0]]
    cities += [cities[0]]

    return start_cities, cities, path_old


def coord_list_to_x_y(array):
    list_x, list_y = [], []
    for x, y in array:
        list_x.append(x)
        list_y.append(y)
    return list_x, list_y


def lab_11(n):
    count_city = n  # Колличество городов

    dots_data = create_dot_list(count_city)  # Координаты городов
    cities = list(range(count_city))  # Номера городов от 1 до n

    dist_matrix = get_distance_matrix(count_city, dots_data)  # Список расстояний
    combinations = get_combinations_list(cities)  # Коомбинации прохода по городам / tqdm

    # Поиск минимального маршрута
    min_path, len_min_path = search_min_combination(combinations, dist_matrix)
    print(f'\nМинимальный маршрут:\n'
          f'Длина = {len_min_path}\n'
          f'{min_path}\n')

    min_path_coord = combination_list_to_dot(min_path, dots_data)

    random_path = choice(combinations)
    random_path_coord = combination_list_to_dot(random_path, dots_data)

    all_path_coord = get_all_path_coord(dots_data)

    # MATPLOTLIB

    plt.figure(figsize=[14, 4.2])

    plt.subplot(1, 3, 1)
    plt.plot(*coord_list_to_x_y(all_path_coord), marker='o', mfc='r', ms=10)
    plt.title('Всевозможные пути', size=10)
    plt.xlim([-10, 110])
    plt.ylim([-10, 110])

    plt.subplot(1, 3, 2)
    plt.plot(*coord_list_to_x_y(min_path_coord), marker='o', mfc='r', ms=10)
    plt.title('Минимальный путь', size=10)
    plt.xlim([-10, 110])
    plt.ylim([-10, 110])

    plt.subplot(1, 3, 3)
    plt.plot(*coord_list_to_x_y(random_path_coord), marker='o', mfc='r', ms=10)
    plt.title('Случайный начальный путь', size=10)
    plt.xlim([-10, 110])
    plt.ylim([-10, 110])

    plt.suptitle(f'Алгоритм полного перебора ({len(cities)} городов)\n', y=1, size=16)
    plt.show()

    # GNUPLOT

    # save_result_to_txt(all_path_coord, '11/all_path')
    # save_result_to_txt(min_path_coord, '11/min_path')
    # save_result_to_txt(random_path_coord, '11/random_path')


def lab_12(n, steps):
    count_city = n  # Колличество городов

    dots_data = create_dot_list(count_city)  # Координаты городов
    cities = list(range(count_city))  # Номера городов от 0 до n
    dist_matrix = get_distance_matrix(count_city, dots_data)  # Список расстояний

    start_path, min_path, len_min_path = monte_carlo(steps, cities, dist_matrix)

    print(f'\nМинимальный маршрут по Монте-Карло:\n'
          f'Длина = {len_min_path}\n'
          f'{start_path} - Стартовый \n'
          f'{min_path} - Минимальный \n')

    # combinations = get_combinations_list(cities)  # Коомбинации прохода по городам / tqdm
    # min_path_2, len_min_path_2 = search_min_combination(combinations, dist_matrix)
    #
    # print(f'\nМинимальный маршрут по перебору:\n'
    #       f'Длина = {len_min_path_2}\n'
    #       f'{min_path_2}\n')

    min_path_1_coord = combination_list_to_dot(min_path, dots_data)
    start_coord = combination_list_to_dot(start_path, dots_data)
    # min_path_2_coord = combination_list_to_dot(min_path_2, dots_data)
    all_path = get_all_path_coord(dots_data)

    # MATPLOTLIB
    plt.figure(figsize=[14, 4.2])

    plt.subplot(1, 3, 1)
    plt.plot(*coord_list_to_x_y(all_path), marker='o', mfc='r', ms=10)
    plt.title('Всевозможные пути', size=10)
    plt.xlim([-10, 110])
    plt.ylim([-10, 110])

    plt.subplot(1, 3, 2)
    plt.plot(*coord_list_to_x_y(start_coord), marker='o', mfc='r', ms=10)
    plt.title('Начальный путь', size=10)
    plt.xlim([-10, 110])
    plt.ylim([-10, 110])

    plt.subplot(1, 3, 3)
    plt.plot(*coord_list_to_x_y(min_path_1_coord), marker='o', mfc='r', ms=10)
    plt.title('Минимальный путь', size=10)
    plt.xlim([-10, 110])
    plt.ylim([-10, 110])

    plt.suptitle(f'Алгоритм Монте-Карло ({len(cities)} городов, {steps} шагов)\n', y=1, size=16)
    plt.show()

    # GNUPLOT

    # save_result_to_txt(min_path_1_coord, '12/end')
    # save_result_to_txt(start_coord, '12/start')
    # save_result_to_txt(min_path_2_coord, '12/min')
    # save_result_to_txt(all_path, '12/all')


def lab_13(n, steps):
    count_city = n  # Колличество городов

    dots_data = create_dot_list(count_city)  # Координаты городов
    cities = list(range(count_city))  # Номера городов от 0 до n
    dist_matrix = get_distance_matrix(count_city, dots_data)  # Список расстояний

    start_path, min_path, len_min_path = metropolis(steps, cities, dist_matrix)

    print(f'\nМинимальный маршрут по алгоритму Метрополиса:\n'
          f'Длина = {len_min_path}\n'
          f'{start_path} - Стартовый \n'
          f'{min_path} - Минимальный \n')

    min_path_1_coord = combination_list_to_dot(min_path, dots_data)
    start_coord = combination_list_to_dot(start_path, dots_data)
    all_path = get_all_path_coord(dots_data)

    # MATPLOTLIB

    plt.figure(figsize=[14, 4.2])

    plt.subplot(1, 3, 1)
    plt.plot(*coord_list_to_x_y(all_path), marker='o', mfc='r', ms=10)
    plt.title('Всевозможные пути', size=10)
    plt.xlim([-10, 110])
    plt.ylim([-10, 110])

    plt.subplot(1, 3, 2)
    plt.plot(*coord_list_to_x_y(start_coord), marker='o', mfc='r', ms=10)
    plt.title('Начальный путь', size=10)
    plt.xlim([-10, 110])
    plt.ylim([-10, 110])

    plt.subplot(1, 3, 3)
    plt.plot(*coord_list_to_x_y(min_path_1_coord), marker='o', mfc='r', ms=10)
    plt.title('Минимальный путь', size=10)
    plt.xlim([-10, 110])
    plt.ylim([-10, 110])

    plt.suptitle(f'Алгоритм Метрополиса ({len(cities)} городов, {steps} шагов)\n', y=1, size=16)
    plt.show()

    # GNUPLOT

    # save_result_to_txt(min_path_1_coord, '13/end')
    # save_result_to_txt(start_coord, '13/start')
    # save_result_to_txt(all_path, '13/all')


@timeit
def main():
    # lab_11(10)  # Аргументы: Колличество городов
    # lab_12(30, 30000000)  # Аргументы: Колличество городов, колличество шагов
    lab_13(10, 50000)  # Аргументы: Колличество городов, колличество шагов
    pass


if __name__ == '__main__':
    main()