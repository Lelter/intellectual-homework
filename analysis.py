import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
#显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set(font="simhei",style="whitegrid")
data = pd.read_csv('./data/compare_table.csv')
sns.lineplot(data=data, x='城市数量', y='迭代次数')
plt.show()
