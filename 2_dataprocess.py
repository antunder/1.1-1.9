import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from gener import generate
import pandas as pd
# 设置默认字体路径
rcParams['font.family'] = 'Microsoft YaHei'
rcParams['axes.unicode_minus'] = False
def netcost(n):
    e = np.ones((6,))
    #计算零件数量
    num = e-x[n]
    for i in range(6):
        num[i] = e[i]/num[i]
    #成本
    if n == 1:
        b = 0
    else:
        b = 1
    cost = num*(y[n]+z[n])+y[b]
    #节省损失
    loss = x[n]*l
    return cost-loss
def neither_1_or_2():
    p_b = x[0]+x[1]+x[2]-x[0]*x[1]-x[0]*x[2]-x[1]*x[2]+x[0]*x[1]*x[2]
    p_1_none = (p_b-x[0])/p_b
    p_2_none = (p_b-x[1])/p_b
    q = y[0]*p_1_none+y[1]*p_2_none
    return q
def check1or2():
    p_b = x[1]+x[2]-x[1]*x[2]
    q = y[0]+y[1]*(p_b-x[1])/p_b
    return q
def check1and2():
    q = y[1]+y[2]
    return q

#不同情形：
def s001():
    cost=y[0]+y[1]+y[2]+z[2]+pb0*(f+z[0]+z[1])
    pro = (e-pb0)*s[2]
    return pro-cost
def s101():
    cost = (y[0]+z[0])/(e-x[0])+y[1]+y[2]+z[2]+pb1*(f+z[1])
    pro = (e-pb1)*s[2]+pb1*y[0]
    return pro-cost
def s111():
    cost = (y[0]+z[0])/(e-x[0])+(y[1]+z[1])/(e-x[1])+y[2]+z[2]+pb2*(f+z[2])
    pro = (e-pb2)*s[2]+pb2*(e-x[2])*s[2]-pb2*y[2]
    return pro-cost
def s000():
    cost = y[0]+y[1]+y[2]+z[2]+pb0*(f+z[0]+z[1]+l)
    pro = (e-pb0)*s[2]
    return pro-cost
def s100():
    cost = (y[0]+z[0])/(e-x[0])+y[1]+y[2]+pb1*(l+f+z[1])
    pro = (e-pb1)*s[2]+pb1*y[0]
    return pro-cost
def s110():
    cost = (y[0]+z[0])/(e-x[0])+(y[1]+z[1])/(e-x[1])+y[2]+pb2*(l+f+z[2])
    pro = (e-pb2)*s[2]+pb2*(e-x[2])*s[2]-pb2*y[2]
    return pro-cost
#针对情形5单独列出
def s011():
    cost = (y[1]+z[1])/(e-x[1])+y[0]+y[2]+z[2]+pb3*(f+z[0])
    pro = (e-pb3)*s[2]+pb3*y[1]
    return pro-cost
def s010():
    cost = (y[1]+z[1])/(e-x[1])+y[0]+y[2]+pb3*(f+z[0]+l)
    pro = (e-pb2)*s[2]+pb3*y[1]
    return pro-cost
#针对情形六单独列出
def sp000():
    return -(y[0]+y[1]+y[2]+l*pb60-(1-pb60)*s[2])
def sp100():
    return -((y[0]+z[0])/(e-x[0])+y[1]+y[2]+pb61*l-(1-pb61)*s[2])
def sp110():
    return -((y[0]+z[0])/(e-x[0])+(y[1]+z[1])/(e-x[1])+y[2]+pb62*l-(1-pb62)*s[2])
#数据部分
#次品率
x = np.array([[0.1,0.2,0.1,0.2,0.1,0.05],
              [0.1,0.2,0.1,0.2,0.2,0.05],
              [0.1,0.2,0.1,0.2,0.1,0.05]])
#购买/装配成本
y = np.array([[4,4,4,4,4,4],
              [18,18,18,18,18,18],
              [6,6,6,6,6,6]])
#检测成本
z = np.array([[2,2,2,1,8,2],
              [3,3,3,1,1,3],
              [3,3,3,2,2,3]])
#成品的市场售价
s = np.array([56,56,56,56,56,56])
#调换损失
l = np.array([6,6,30,30,10,10])
#拆解费用
f = np.array([5,5,5,5,5,40])
# 真实值偏小
# for i in range(x.shape[0]):
#     for j in range(x.shape[1]):
#         x[i][j] = generate(x[i][j])[0]
# #真实值偏大
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        x[i][j] = generate(x[i][j])[1]
print(x)
#
# # #净成本部分
# # print("在检测其中某一个零件的情况下，省去调换成本产生的净成本：")
# data = np.array([netcost(0),netcost(1)])
# # print(data)
#
# row_labels = ['检测零件1','检测零件2']
# # col_labels = ['情形1','情形2','情形3','情形4','情形5','情形6']
# col_labels = [1,2,3,4,5,6]
#
# plt.subplot(1,2,1)
# # 绘制折线图
# for i in range(data.shape[0]):
#     # 每个策略
#     y_values = data[i,:]
#     plt.plot(col_labels, data[i, :], label=row_labels[i], marker='o')
#     # 标出每个点的数值
#     for a, y in zip(col_labels, y_values):
#         plt.annotate(f'{y:.2f}', (a, y), textcoords="offset points", xytext=(0, 5), ha='center')
# plt.xticks(range(min(col_labels), max(col_labels) + 1))
# # 添加标题和标签
# plt.title('真实次品率偏小边界值',pad=20)
# plt.xlabel('情形')
# plt.ylabel('净成本')
#
# # 显示图例
# plt.legend()
# plt.show()
# 显示图形



# #净成本部分
# # print("在检测其中某一个零件的情况下，省去调换成本产生的净成本：")
# data = np.array([netcost(0),netcost(1)])
# # print(data)
#
# row_labels = ['检测零件1','检测零件2']
# col_labels = ['情形1','情形2','情形3','情形4','情形5','情形6']
# col_labels = [1,2,3,4,5,6]
# # 绘制折线图
# plt.subplot(1,2,2)
# for i in range(data.shape[0]):
#     # 每个策略
#     y_values = data[i,:]
#     plt.plot(col_labels, data[i, :], label=row_labels[i], marker='o')
#     # 标出每个点的数值
#     for x, y in zip(col_labels, y_values):
#         plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 5), ha='center')
# plt.xticks(range(min(col_labels), max(col_labels) + 1))
# # 添加标题和标签
# plt.title('真实次品率偏大边界值',pad=20)
# plt.xlabel('情形')
# plt.ylabel('净成本')
#
# # 显示图例
# plt.legend()
#
# # 显示图形
# plt.show()

# df = pd.DataFrame(data, index=row_labels, columns=col_labels)
# df.to_csv('省去调换成本产生净成本_真实值偏大.csv',encoding='utf-8-sig')


#丢弃拆解对比部分
# 将一维矩阵转换为二维矩阵
# data = np.vstack((neither_1_or_2(),check1or2(),check1and2(),f))
# # print(data)
# row_labels = ['均不检测', '检测其一（检测1）', '均检测', '拆解费用']
# col_labels = ['情形1','情形2','情形3','情形4','情形5','情形6']
# df = pd.DataFrame(data, index=row_labels, columns=col_labels)
# df.to_csv('丢弃拆解对比数据_真实值偏小.csv',encoding='utf-8-sig')

# 设置柱状图的参数
# num_groups = data.shape[0]  # 组数据
# num_bars = data.shape[1]    # 每组6个柱
# bar_width = 0.8 / num_groups  # 每组的宽度
# index = np.arange(num_bars)   # 每组的索引位置

# # 绘制柱状图
# fig, ax = plt.subplots()
# # 自定义颜色
# labels = ['均不检测', '检测其一（检测1）', '均检测', '拆解费用']
# colors = ['#ADD8E6', '#87CEEB', '#4682B4', '#00008B']
# # 为每一列的数据绘制柱状图
# for i in range(num_groups):
#     ax.bar(index + i * bar_width, data[i, :], bar_width, label=labels[i],color=colors[i])
# # 添加标签和标题
# ax.set_xlabel('情形')
# ax.set_ylabel('拆解成本')
# ax.set_title('由（1）（2）决策产生的丢弃损失VS拆解成本：真实值偏大')
# # ax.set_xticks(index + bar_width * (num_groups - 1) / 2)
# # ax.set_xticklabels([f'类 {i+1}' for i in range(num_bars)])
# ax.legend()
# ax.set_xticks(np.arange(data.shape[1]) + 0.5, np.arange(1,7))
# # plt.yticks(np.arange(data.shape[0]) + 0.5, ['均不检测', '检测其一（检测1）', '均检测', '拆解费用'])
# # 显示图形
# plt.show()


#不同情形
pb0 = x[0]+x[1]+x[2]-x[0]*x[1]-x[0]*x[2]-x[1]*x[2]+x[0]*x[1]*x[2]
pb1 = x[1]+x[2]-x[1]*x[2]
pb2 = x[2]
pb3 = x[0] + x[2] - x[0] * x[2]
pb60 = pb0[5]
pb61 = pb1[5]
pb62 = pb2[5]
e = np.ones((6,))
"""
#情形1~4策略
stra14 = ['001','101','111','000','100','110']
data = np.vstack((s001(),s101(),s111(),s000(),s100(),s110()))
print(data)
row_labels = ['001','101','111','000','100','110']
col_labels = ['情形1','情形2','情形3','情形4']
df = pd.DataFrame(data[:,:4], index=row_labels, columns=col_labels)
df.to_csv('情形1-4策略_真实值偏大.csv',encoding='utf-8-sig')
situations = [1,2,3,4]
# 绘制折线图
# for i in range(data.shape[0]):
#     # 每个策略
#     y_values = data[i, :4]
#     plt.plot(situations, data[i, :4], label=stra14[i], marker='o')
#     # 标出每个点的数值
#     for x, y in zip(situations, y_values):
#         plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 5), ha='center')
# plt.xticks(range(min(situations), max(situations) + 1))
# # 添加标题和标签
# plt.title('情形1~4在不同策略的净利润:真实值偏小',pad=20)
# plt.xlabel('情形')
# plt.ylabel('净利润')
#
# # 显示图例
# plt.legend()
#
# # 显示图形
# plt.show()
print(data)
"""

# #情形5策略
# stra5 = ['001','011','111','000','010','110']
# data = np.vstack((s001(),s011(),s111(),s000(),s010(),s110()))
# print(data)
# row_labels = ['001','011','111','000','010','110']
# df = pd.DataFrame(data[:,4], index=row_labels)
# df.to_csv('情形5策略_真实值偏小.csv',encoding='utf-8-sig')
# # 绘制折线图
# plt.plot(stra5,data[:,4],marker='o')
# for x, y in zip(stra5,data[:,4] ):
#     plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points",  ha='center')
# # 添加标题和标签
# plt.title('情形5不同策略的净利润:真实值偏小',pad=20)
# plt.xlabel('策略名称')
# plt.ylabel('净利润')
#
# # 显示图例
# plt.legend()

# 显示图形
# plt.show()
# print(data)



# #情形6策略
# stra6 = ['000','100','110']
# data = np.vstack((sp000(),sp100(),sp110()))
# print(data)
# df = pd.DataFrame(data[:,5], index=stra6)
# df.to_csv('情形6策略_真实值偏大.csv',encoding='utf-8-sig')
# # 绘制折线图
# plt.plot(stra6,data[:,5],marker='o')
# for x, y in zip(stra6,data[:,5] ):
#     plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", ha='center')
# # 添加标题和标签
# plt.title('情形6不同策略的净利润',pad=20)
# plt.xlabel('策略名称')
# plt.ylabel('净利润')
# # 设置 y 轴范围
# plt.ylim(13, 23)
#
# # 显示图例
# plt.legend(['净利润'])
#
# # 显示图形
# plt.show()

