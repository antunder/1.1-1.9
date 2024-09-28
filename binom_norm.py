import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm
from matplotlib import rcParams

# 设置默认字体路径
rcParams['font.family'] = 'Microsoft YaHei'
rcParams['axes.unicode_minus'] = False
# 参数设置
n = 100    # 样本数量
p = 0.5    # 成功的概率

# 生成二项分布数据
x = np.arange(0, n + 1)
binom_pmf = binom.pmf(x, n, p)

# 计算正态分布的均值和标准差
mu = n * p
sigma = np.sqrt(n * p * (1 - p))

# 生成正态分布数据
x_norm = np.linspace(mu - 3*sigma, mu + 3*sigma, 1000)
norm_pdf = norm.pdf(x_norm, mu, sigma)

# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(x, binom_pmf, 'bo', label='二项分布 PMF', markersize=8)
plt.plot(x_norm, norm_pdf, 'r-', label='正态分布 PDF', linewidth=2)

plt.title('二项分布 vs. 正态分布')
plt.xlabel('成功次数')
plt.ylabel('概率')
plt.legend()
plt.grid(True)
plt.show()
