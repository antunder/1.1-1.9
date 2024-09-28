import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from sympy import symbols, solve, Poly, S,Eq
from sympy.solvers.inequalities import reduce_rational_inequalities
from matplotlib import rcParams

# 设置默认字体路径
rcParams['font.family'] = 'Microsoft YaHei'
rcParams['axes.unicode_minus'] = False
p0 = 0.1

z1 = 1.6425

z2 = -1.282

m_values_1 = []
m_values_2 = []

n_range = np.arange(1,101)
for n in n_range:
    x = symbols('x')
    m1 = symbols('m1')
    m2 = symbols('m2')
    a = 1/n
    b = -0.2
    c1 = n*p0**2-p0*(1-p0)*z1**2
    c2 = n*p0**2-p0*(1-p0)*z2**2
    equation1 = Eq(a*m1**2+b*m1+c1,0)
    roots1 = solve(equation1,m1)
    equation2 = Eq(a*m2**2+b*m2+c2,0)
    roots2 = solve(equation2,m2)
    #95%拒收
    m1 = roots1[1]
    #90%接收
    m2 = roots2[1]
    print(roots1,roots2)
    m_values_1.append(int(float(m1)))
    m_values_2.append(int(float(m2)))
plt.figure(figsize=(12, 6))
plt.plot(n_range, m_values_1, label='95% 信度(拒收) 次品阈值', marker='o')
plt.plot(n_range, m_values_2, label='90% 信度(接收) 次品阈值', marker='x')
plt.xlabel('样本量(n)')
plt.ylabel('次品数量(m)')
plt.title('不同样本数量下所能接受次品数量')
plt.legend()
plt.grid(True)
plt.show()