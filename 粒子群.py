import math
import random
import numpy as np
import matplotlib.pyplot as plt


# QUSETION:
# 1.粒子群算法的种群规模的选取多少合适？
# 2.粒子群算法容易陷入局部最优解，如何解决？
# 3.性能如何改进？
# 4.是否可以自适应调节算法的参数？
# 5.粒子群算法是否可以应用于其他问题?

class PSO:
    def __init__(self, parameters):
        # 初始化
        self.NGEN = parameters[0]  # 迭代的代数
        self.pop_size = parameters[1]  # 种群大小
        self.var_num = len(parameters[2])  # 变量个数
        self.bound = []  # 变量的约束范围
        self.bound.append(parameters[2])  # x的最小值
        self.bound.append(parameters[3])  # x的最大值

        self.pop_x = np.zeros((self.pop_size, self.var_num))  # 所有粒子的位置
        self.pop_v = np.zeros((self.pop_size, self.var_num))  # 所有粒子的速度
        self.p_best = np.zeros((self.pop_size, self.var_num))  # 每个粒子历史最优的位置
        self.g_best = np.zeros((1, self.var_num))  # 全局最优的位置

        # 初始化第0代初始全局最优解
        temp = 1000000000  # 记录最优解的适应值大小
        for i in range(self.pop_size):
            for j in range(self.var_num):
                self.pop_x[i][j] = random.uniform(self.bound[0][j], self.bound[1][j])  # 随机初始化粒子位置，每个变量范围为[xmin,xmax]
                self.pop_v[i][j] = random.uniform(-60, 60)  # 初始化速度范围为[0,1]
            self.p_best[i] = self.pop_x[i]  # 储存最优的个体
            fit = self.fitness(self.p_best[i])  # 计算适应值，公式为y=x1^2+x2^2+x3^3+x4^4
            if fit < temp:
                self.g_best = self.p_best[i]  # 将全局最优粒子的位置设置为该粒子的位置
                temp = fit  # 更新最优解的适应值

    def fitness(self, ind_var):
        """
        个体适应值计算
        """
        x1 = ind_var[0]  # x1
        x2 = ind_var[1]  # x2
        x3 = ind_var[2]  # x3
        x4 = ind_var[3]  # x4
        # y = 0.5 + (math.sin((x1 ** 2 + x2 ** 2) ** 1 / 2) - 0.5) / (1 + 0.001 * (x1 ** 2 + x2 ** 2) ** 2)
        # y = x1 ** 2 + x2 ** 2 + x3 ** 3 + x4 ** 4  # 公式为y=x1^2+x2^2+x3^3+x4^4
        y = (100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2 + 100 * (x3 - x2 ** 2) ** 2 + (x2 - 1) ** 2 + 100 * (
                x4 - x3 ** 2) ** 2 + (
                     x3 - 1) ** 2)
        return y  # 返回适应值

    def update_operator(self, pop_size):
        """
        更新算子：更新下一时刻的位置和速度
        """
        c1 = 2  # 个体因子，一般为2
        c2 = 2  # 全局因子，一般为2
        w = 0.4  # 自身权重因子，惯性权重

        for i in range(pop_size):  # 对每个个体
            # 更新速度
            self.pop_v[i] = w * self.pop_v[i] + c1 * random.uniform(0, 1) * (
                    self.p_best[i] - self.pop_x[i]) + c2 * random.uniform(0, 1) * (self.g_best - self.pop_x[i])

            # 更新位置
            self.pop_x[i] = self.pop_x[i] + self.pop_v[i]  # 公式为x+v
            # 越界保护
            for j in range(self.var_num):
                if self.pop_x[i][j] < self.bound[0][j]:
                    self.pop_x[i][j] = self.bound[0][j]  # 如果个体中的四个变量超过了最小值，重新赋值为最小值
                if self.pop_x[i][j] > self.bound[1][j]:
                    self.pop_x[i][j] = self.bound[1][j]  # 如果个体中四个变量的值超过了最大值，重新赋值为最大值
            # 更新p_best和g_best
            if self.fitness(self.pop_x[i]) < self.fitness(self.p_best[i]):  # 如果某个个体的适应值超过了自身的最优值
                self.p_best[i] = self.pop_x[i]  # 重新赋值自身历史最优
            if self.fitness(self.pop_x[i]) < self.fitness(self.g_best):  # 如果超过了全局最优
                self.g_best = self.pop_x[i]  # 重新赋值全局最优解

    def main(self):
        every_iter_best_fitness = []  # 存储每代的最优解
        # self.ng_best = np.zeros((1, self.var_num))[0]  # 存储最好的位置，初始化为[0,0,0,0]
        self.ng_best = [10, 10, 10, 10]
        for gen in range(self.NGEN):  # 对每一代
            self.update_operator(self.pop_size)  # 更新速度位置，更新最优值
            print('############ Generation {} ############'.format(str(gen + 1)))
            if self.fitness(self.g_best) < self.fitness(self.ng_best):  # 如果这一代的最优解大于所有代的最好位置
                self.ng_best = self.g_best.copy()  # 赋值给ng_best
            every_iter_best_fitness.append(self.fitness(self.ng_best))  # 添加每一代所有个体中的最优解

            print('最好的位置：{}'.format(self.ng_best))  # 打印这一代之前的最好位置
            print('最好的函数值：{}'.format(self.fitness(self.ng_best)))  # 打印这一带之前的最好适应值
        print("---- End of (successful) Searching ----")

        plt.figure()
        plt.title("Figure1")
        plt.xlabel("iterators", size=14)  # 代数
        plt.ylabel("fitness", size=14)  # 适应值
        t = [t for t in range(self.NGEN)]  # t为[1,NGEN,1]
        plt.plot(t, every_iter_best_fitness, color='b', linewidth=2)  # 画图
        plt.show()


if __name__ == '__main__':
    NGEN = 100  # 迭代次数
    popsize = 5  # 种群大小
    low = [-2, -2, -2, -2]  # 变量最小值
    up = [2, 2, 2, 2]  # 变量最大值
    parameters = [NGEN, popsize, low, up]  # 组合成列表
    pso = PSO(parameters)  # 开始迭代
    pso.main()
