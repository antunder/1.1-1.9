import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import pandas as pd
from gener import generate
# 设置默认字体路径
def o_001():
    # p = 0.14#正常
    # p = 0.2661#真实值偏小
    p = 0.5296#真实值偏大
    cost = (h_3_11+z[1][2])/(1-p_3_11)+2*h_1_110+f+p*F[1]+p_3_11*F[0]+z[2][0]+p_1_100*(z[1][0]+z[1][1])
    pro = (1-p)*s+p*h_3_11+p_3_11*(y[6]+y[7])
    return pro-cost
def o_101():
    # p = 0.271  # 正常
    # p = 0.1694  # 真实值偏小
    p = 0.3639#真实值偏大
    cost = (h_3_11+z[1][2])/(1-p_3_11)+(h_1_110+z[1][0])/(1-p_1_110)+h_1_110+f+p_3_11*F[0]+p_1_110*F[0]+p*F[1]+p*z[1][1]+z[2][0]
    pro = (1-p)*s+p_3_11*(y[6]+y[7])+p_1_110*(y[0]+y[1]+y[2]*9/19)+p*(h_3_11+h_1_110)
    return pro-cost
def o_111():
    # p = 0.1  # 正常
    # p = 0.06  # 真实值偏小
    p = 0.14#偏大
    cost = (h_3_11+z[1][2])/(1-p_3_11)+(h_1_110+z[1][0])*2/(1-p_1_110)+f+p_3_11*F[0]+2*p_1_110*F[0]+p*F[1]+p*l
    pro = s*(1-p)+p_3_11*(y[6]+y[7])+2*p_1_110*(y[0]+y[1]+y[2]*9/19)+p*(h_3_11+2*h_1_110)
    return pro-cost
def p_001():
    # p = 0.271  # 正常
    # p = 0.1694  # 真实值偏小
    p = 0.3639#真实值偏大
    cost = (h_3_11+z[1][2])/(1-p_3_11)+2*h_1_111+f+p*F[0]+p*(z[1][0]+z[1][1])+z[2][0]
    pro = (1-p)*s+p*h_3_11+p_3_11*(y[6]+y[7])
    return pro-cost
def p_101():
    # p = 0.19  # 正常
    # p = 0.1164  # 真实值偏小
    p = 0.2604 #偏大
    cost = (h_3_11+z[1][2])/(1-p_3_11)+(h_1_111+z[1][0])/(1-p_1_111)+h_1_111+f+p*F[1]+p_3_11*F[0]+p_1_111*F[0]+p*z[1][1]+z[2][0]
    # cost = (h_3_11+z[1][2])/(1-p_3_11)+(h_1_111+z[1][0])/(1-p_1_111)+h_1_111+f+p*F[1]+p_3_11*F[0]+p_1_111*F[0]+p*z[1][1]+p*l
    pro = (1-p)*s+p*(h_3_11+h_1_111)+p_3_11*(y[6]+y[7])+p_1_111*(y[0]+y[1]+y[2])
    return pro-cost
def p_111():
    # p = 0.1  # 正常
    # p = 0.06  # 真实值偏小
    p = 0.14#偏大
    cost = (h_3_11+z[1][2])/(1-p_3_11)+2*(h_1_111+z[1][0])/(1-p_1_111)+f+p*F[1]+p_3_11*F[0]+2*p_1_111*F[0]+p*l
    pro = (1-p)*s+0.271*(h_3_11+h_1_111*2)+p_3_11*(y[6]+y[7])+2*p_1_111*(y[0]+y[1]+y[2])
    return pro-cost

rcParams['font.family'] = 'Microsoft YaHei'
rcParams['axes.unicode_minus'] = False
#次品率
m = 0.1
# m = generate(m)[0]#偏小
# m = generate(m)[1]#偏大
#购买成本
y = np.array([2,8,12,2,8,12,8,12])
#检测成本
z = np.array([[1,1,2,1,1,2,1,2],[4,4,4,0,0,0,0,0],[6,0,0,0,0,0,0,0]])
#装配成本
f = 8
#调换损失
l = 40
#拆解费用
F = np.array([6,10])
#市场售价
s = 200
# 针对半成品的拆解的净成本与丢弃成本对比
# 针对半成品1/2得到次品率
p_1_100 = -3 * m ** 2 + m ** 3 + 3 * m
p_1_110 = 2 * m - m ** 2
p_1_111 = m
# 半成品3次品率
p_3_10 = 2 * m - m ** 2
p_3_11 = m
# 净成本
q_1_100 = y[0] + y[1] * (p_1_100 - m) / p_1_100 + y[2] * (p_1_100 - m) / p_1_100
q_1_110 = y[0] + y[1] + y[2] * (p_1_110 - m) / p_1_110
q_1_111 = y[0] + y[1] + y[2]
q_1_list = np.array([q_1_100, q_1_110, q_1_111])
q_1_label = ['100', '110', '111']
# 针对半成品3
# 净成本
q_3_10 = y[6] + y[7] * (p_3_10 - m) / p_3_10
q_3_11 = y[6] + y[7]
q_3_list = [q_3_10, q_3_11]
q_3_label = ['10', '11']
df = pd.DataFrame(q_1_list, index=q_1_label, columns=['净成本'])
df.to_csv('半成品1净成本_真实值偏大.csv')
df = pd.DataFrame(q_3_list, index=q_3_label, columns=['净成本'])
df.to_csv('半成品3净成本_真实值偏大.csv') #绘制折线图
# plt.figure(figsize=(12, 6))
#
#
# plt.subplot(1, 2, 1)
# plt.plot(q_1_label, q_1_list, marker='o',  label='丢弃情况下成本',linestyle='-')
# plt.plot(q_1_label,np.ones((3,))*F[0],marker='o',label='拆解成本', linestyle='-')
# plt.title('半成品1/2丢弃成本对比')
# plt.xlabel('零配件检测情况')
# plt.ylabel('净成本')
# plt.xticks(rotation=45, ha='right')
# for x, y in zip(q_1_label,q_1_list ):
#     plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points",  ha='center')
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.plot(q_3_label, q_3_list, marker='o', label='丢弃情况下成本',linestyle='-')
# plt.plot(q_3_label,np.ones((2,))*F[0],marker='o',label='拆解成本',  linestyle='-')
# plt.title('半成品3丢弃成本对比')
# plt.xlabel('零配件检测情况')
# plt.ylabel('净成本')
# plt.xticks(rotation=45, ha='right')
# for x, y in zip(q_3_label,q_3_list ):
#     plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points",  ha='center')
# plt.tight_layout()
# plt.legend()
#
# plt.show()


#半成品1/2单价h
h_1_100 = (y[0]+z[0][0])/(1-m)+f+y[1]+y[2]
h_1_110 = (y[0]+y[1]+z[0][0]+z[0][1])/(1-m)+f+y[2]
h_1_111 = (y[0]+y[1]+y[2]+z[0][0]+z[0][1]+z[0][2])/(1-m)+f
#半成品3单价h
h_3_10 = (y[6]+z[0][7])/(1-m)+y[7]+f
h_3_11 = (y[6] + y[7]+z[0][6]+z[0][7])/(1-m)+f
#半成品1/2生产成本
pro_1_100 = h_1_100/(1-p_1_100)
pro_1_110 = h_1_110/(1-p_1_110)
#半成品3生产成本
pro_3_10 = h_3_10/(1-p_3_10)
pro_3_11 = h_3_11/(1-p_3_11)
#相对节省拆解费用
# s = F[0]*(p_3_10-p_3_11)
#相对节省调换费用
ps = 2*np.min([p_1_110,p_1_100,p_1_111])+p_3_11+(p_3_11-1)*np.min([p_1_110,p_1_100,p_1_111])**2-2*p_3_10*np.min([p_1_110,p_1_100,p_1_111])
#净成本
c_1 = (h_1_110+z[1][0])/(1-p_1_110)+h_1_110+h_3_11-p_1_110*l
c_3 = (h_3_11+z[1][2])/(1-p_3_11)+2*h_1_110-p_3_11*l
print(c_1,c_3)
#半成品1/2单价
# data = np.array([[h_1_100, h_1_110, h_1_111,],[p_1_100, p_1_110, p_1_111,]])
# print(data)
# row_labels = ['单价','半成品次品率']
# col_labels = ['100','110','111']
# df = pd.DataFrame(data, index=row_labels, columns=col_labels)
# df.to_csv('半成品1and2单价及次品率_真实值偏大.csv')
# #半成品3单价及次品率
# data = np.array([[h_3_10,h_3_11],[p_3_10,p_3_11]])
# # print(data)
# row_labels = ['单价','半成品次品率']
# col_labels = ['10','11']
# df = pd.DataFrame(data, index=row_labels, columns=col_labels)
# df.to_csv('半成品3单价及次品率_真实值偏大.csv')

#
#
# #最终利润
# data = np.array([[o_001(),o_101(),o_111()],[p_001(),p_101(),p_111()],])
# row_labels = ['零配件检测:半成品1/2:110,半成品3:11','零配件检测:半成品1/2:111,半成品3:11']
# col_labels = ['001','101','111']
# for i in range(data.shape[0]):
#     # 每个策略
#     y_values = data[i,:]
#     plt.plot(col_labels, data[i, :], label=row_labels[i], marker='o')
#     # 标出每个点的数值
#     for x, y in zip(col_labels, y_values):
#         plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 5), ha='center')
# # 添加标题和标签
# plt.title('决策方案净利润:真实值偏大',pad=20)
# plt.xlabel('半成品检测状况')
# plt.ylabel('净利润')
# plt.legend()
# plt.show()
# print(data.shape)
# df = pd.DataFrame(data, index=row_labels, columns=col_labels)
# df.to_csv('最终利润_真实值偏大.csv')
