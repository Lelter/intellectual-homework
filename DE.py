import numpy as np
import random


def fit_fun(X):
    return -np.abs(np.sin(X[0]) * np.cos(X[1]) * np.exp(np.abs(1 - np.sqrt(X[0] ** 2 + X[1] ** 2) / np.pi)))


class Unit:
    def __init__(self, x_min, x_max, dim) -> None:
        self._pos = np.array([x_min+random.random()*(x_max-x_min)
                             for i in range(dim)])  # 初始化位置，范围在x_min和x_max之间

        self._mutation = np.array([0.0 for i in range(dim)])  # 个体突变后的向量
        self._crossover = np.array([0.0 for i in range(dim)])  # 个体交叉后的向量
        self._fitnessValue = fit_fun(self._pos)  # 个体适应度

    def set_pos(self, i, value):
        self._pos[i] = value

    def get_pos(self, i):
        return self._pos[i]

    def set_mutation(self, i, value):
        self._mutation[i] = value

    def get_mutation(self, i):
        return self._mutation[i]

    def set_crossover(self, i, value):
        self._crossover[i] = value

    def get_crossover(self, i):
        return self._crossover[i]

    def set_fitnessValue(self, value):
        self._fitnessValue = value

    def get_fitnessValue(self):
        return self._fitnessValue


class DE:
    def __init__(self, dim, size, iter_num, x_min, x_max, best_fitness_value=flaot('Inf'), F=0.5, CR=0.8) -> None:
        self.F = F  # 变异系数
        self.CR = CR  # 交叉系数
        self.dim = dim  # 维度
        self.size = size  # 总群个数
        self.iter_num = iter_num  # 迭代次数
        self.x_min = x_min
        self.x_max = x_max
        self.best_fitness_value = best_fitness_value
        self.best_position = [0.0 for i in range(dim)]  # 全局最优解
        self.fitness_val_list = []  # 每次迭代最优适应值
        self.unit_init = [Unit(self.x_min, self.x_max, self.dim)
                          for i in range(self.size)]  # 初始化群体
        pass
