import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math


# 局部最优问题
# 超参数选择问题

# def fit_fun(X):
#     return -np.abs(np.sin(X[0]) * np.cos(X[1]) * np.exp(np.abs(1 - np.sqrt(X[0] ** 2 + X[1] ** 2) / np.pi)))
def fit_fun(X, name):
    if name == 'sphere':
        return np.sum(X ** 2)
    if name == 'schwefel':
        y = 418.9829 * len(X) - np.sum(X * np.sin(np.sqrt(np.abs(X))))

    if name == 'rosenbrock2' or name == 'rosenbrock3' or name == 'rosenbrock4' or name == 'rosenbrock5' or name == 'rosenbrock6' or name == 'rosenbrock7':
        y = 0
        for i in range(len(X) - 1):
            y += 100 * (X[i + 1] - X[i] ** 2) ** 2 + (X[i] - 1) ** 2
    if name == 'rastrigrin':
        y = 20+ np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X))
        # y = (100 * (X[1] - X[0] ** 2) ** 2 + (X[0] - 1) ** 2 + 100 * (X[2] - X[1] ** 2) ** 2 + (X[1] - 1) ** 2 + 100 * (
        #         X[3] - X[2] ** 2) ** 2 + (
        #              X[2] - 1) ** 2)  # 四维Rosenbrock函数
    if name == 'gramacy&lee':
        y = math.sin(10 * math.pi * X) / 2 * X + (X - 1) ** 4
    # else:
    #     return 0
    return y


class Unit:
    def __init__(self, x_min, x_max, dim) -> None:
        self._pos = np.array([x_min + random.random() * (x_max - x_min)
                              for i in range(dim)])  # 初始化位置，范围在x_min和x_max之间

        self._mutation = np.array([0.0 for i in range(dim)])  # 个体突变后的向量
        self._crossover = np.array([0.0 for i in range(dim)])  # 个体交叉后的向量
        self._fitnessValue = fit_fun(self._pos, name)  # 个体适应度

    def set_pos(self, i, value):
        self._pos[i] = value

    def get_pos(self):
        return self._pos

    def set_mutation(self, i, value):
        self._mutation[i] = value

    def get_mutation(self):
        return self._mutation

    def set_crossover(self, i, value):
        self._crossover[i] = value

    def get_crossover(self):
        return self._crossover

    def set_fitnessValue(self, value):
        self._fitnessValue = value

    def get_fitnessValue(self):
        return self._fitnessValue


class DE:
    def __init__(self, dim, size, iter_num, x_min, x_max, best_fitness_value=float('Inf'), F=0.5, CR=0.8) -> None:
        self.F = F  # 变异系数
        self.CR = CR  # 交叉系数
        self.dim = dim  # 维度
        self.size = size  # 总群个数
        self.iter_num = iter_num  # 迭代次数
        self.x_min = x_min  # 设置自变量的最小值
        self.x_max = x_max  # 设置自变量的最大值
        self.best_fitness_value = best_fitness_value  # 设置最优适应值
        self.best_position = [0.0 for i in range(dim)]  # 全局最优解
        self.fitness_val_list = []  # 每次迭代最优适应值
        self.unit_init = [Unit(self.x_min, self.x_max, self.dim)
                          for i in range(self.size)]  # 初始化群体

    def get_kth_unit(self, k):
        return self.unit_init[k]  # 返回第k个个体

    def set_bestFitnessValue(self, value):
        self.best_fitness_value = value  # 设置最优适应值

    def get_bestFitnessValue(self):
        return self.best_fitness_value  # 返回最优适应值

    def set_bestPosition(self, i, value):
        self.best_position[i] = value  # 设置最优位置

    def get_bestPosition(self):
        return self.best_position  # 返回最优位置

    def mutation_fun(self):
        for i in range(self.size):
            r1 = r2 = r3 = 0  # 初始化三个随机数
            while r1 == i or r2 == i or r3 == i or r1 == r2 or r2 == r3 or r1 == r3:  # 设置三个随机数不可以是同一个个体，且不能是i
                r1 = random.randint(0, self.size - 1)  #
                r2 = random.randint(0, self.size - 1)
                r3 = random.randint(0, self.size - 1)
            mutation = self.get_kth_unit(r1).get_pos(
            ) + self.F * (self.get_kth_unit(r2).get_pos() - self.get_kth_unit(r3).get_pos())  # 计算突变后的向量，F为突变系数
            for j in range(self.dim):
                if self.x_min <= mutation[j] <= self.x_max:  # 设置自变量的范围
                    self.get_kth_unit(i).set_mutation(j, mutation[j])  # 设置突变后的向量,后面交叉可能用到
                else:
                    rand_value = self.x_min + random.random() * (self.x_max - self.x_min)  # 如果变量超出范围，随机取值
                    self.get_kth_unit(i).set_mutation(j, rand_value)

    def crossover(self):
        for unit in self.unit_init:
            for i in range(self.dim):
                rand_i = random.randint(0, self.dim - 1)
                rand_float = random.random()
                if rand_float <= self.CR or rand_i == i:  # 如果随机量i=i或者随机数小于变量概率
                    unit.set_crossover(i, unit.get_mutation()[i])  # 设置交叉位置为变异后的值
                else:
                    unit.set_crossover(i, unit.get_pos()[i])  # 否则保持不变

    def selection(self):
        for unit in self.unit_init:
            new_fitness_value = fit_fun(unit.get_crossover(), name)  # 计算交叉后的适应值
            if new_fitness_value < unit.get_fitnessValue():  # 如果交叉后的适应值比当前适应值好
                unit.set_fitnessValue(new_fitness_value)  # 设置为交叉后的适应值
                for i in range(self.dim):  # 设置为交叉后的位置
                    unit.set_pos(i, unit.get_crossover()[i])
            if new_fitness_value < self.get_bestFitnessValue():  # 如果比最好的适应值还要好
                self.set_bestFitnessValue(new_fitness_value)  # 设置为最好的适应值
                for i in range(self.dim):
                    self.set_bestPosition(i, unit.get_crossover()[i])  # 设置为最好的位置

    def update(self):
        for i in range(self.iter_num):
            self.mutation_fun()  # 变异
            self.crossover()  # 交叉
            self.selection()  # 选择
            self.fitness_val_list.append(self.get_bestFitnessValue())  # 添加适应值列表
            print('第%d次迭代，最优适应度为：%f' % (i, self.get_bestFitnessValue()))
        return self.fitness_val_list, self.get_bestPosition()


if __name__ == '__main__':
    fun_name = ['sphere', 'schwefel', 'rosenbrock2', 'rosenbrock3', 'rosenbrock4', 'rosenbrock5', 'rosenbrock6',
                'rosenbrock7', 'rastrigrin', 'gramacy&lee']
    x = [[-100, 100], [-500, 500], [-30, 30], [-30, 30], [-30, 30], [-30, 30], [-30, 30], [-30, 30], [0, 5.12],
         [-0.5, 2.5]]
    f_min = [0, 0, 0, 0, 0, 0, 0, 0, 0, -0.68351]
    dim_list = [7, 7, 2, 3, 4, 5, 6, 7, 2, 1]
    fit_val_list_all = []
    best_pos_list = []
    fig, ax = plt.subplots(2, 5, figsize=(10, 8))
    fig.suptitle('DE', fontsize=20)

    for i in range(0, 10):
        dim = dim_list[i]  # 维度
        size = 1000  # 种群大小
        iter_num = 400  # 迭代次数
        x_min = x[i][0]  # 自变量范围
        x_max = x[i][1]  # 自变量范围
        name = fun_name[i]
        de = DE(dim, size, iter_num, x_min, x_max)
        fit_var_list2, best_pos2 = de.update()  # 开始迭代
        print("DE最优位置:" + str(best_pos2))  # 输出最佳位置
        print("DE最优解:" + str(fit_var_list2[-1]))  # 输出最佳解
        fit_val_list_all.append(fit_var_list2)
        best_pos_list.append(best_pos2)
        sns.set_style('whitegrid')

        j = i // 5
        k = i % 5
        ax[j, k].plot(fit_var_list2, 'r-', label='DE')
        ax[j, k].scatter(iter_num, fit_var_list2[-1], c='r', marker='o', )
        ax[j, k].annotate('min', xy=(iter_num, fit_var_list2[-1]), xytext=(iter_num, fit_var_list2[-1]))

        ax[j, k].set_title(name)
        # ax[j,k].set_xlabel('iteration')
        # ax[j,k].set_ylabel('fitness')
        # fig = plt.figure(figsize=(10, 8))
        # ax=fig.add_subplot(111)
        # ax.plot(fit_var_list2, 'b-', label='DE')
        # ax.set_xlabel('iteration')
        # ax.set_ylabel('fitness')
        # ax.set_title('DE')
        # plt.plot(np.linspace(0, iter_num, iter_num), fit_var_list2)
        # plt.title(name + '_DE')
        # sns.regplot(np.linspace(0, iter_num, iter_num), fit_var_list2, order=1)

        # plt.plot(np.linspace(0, iter_num, iter_num), fit_var_list2, alpha=0.5, label="DE")  # 画图
    accuracy = []
    for i in range(len(fit_val_list_all)):
        accuracy.append(abs(f_min[i] - fit_val_list_all[i][-1]))
        # accuracy.append(abs(f_min[i] - i[-1]))
    # accuracy= np.array(accuracy)
    print(accuracy)
    table = pd.DataFrame(columns=['函数名', '实际最小值', '计算最小值', '偏差'])
    for i in range(len(fun_name)):
        table.loc[i] = [fun_name[i], f_min[i], fit_val_list_all[i][-1], accuracy[i]]
    print(table)
    table.to_csv('DE_table.csv', index=False)
    plt.show()
