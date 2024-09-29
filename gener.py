import numpy as np
def generate(mean_defective_rate):
    std_dev = 0.02  # 标准差

    # 生成一个符合高斯分布的随机值
    random_value = np.random.normal(loc=mean_defective_rate, scale=std_dev)

    # 计算95%置信区间
    confidence_interval = (mean_defective_rate - 1.96 * std_dev, mean_defective_rate + 1.96 * std_dev)

    # 确保次品率在0到1之间
    random_value = np.clip(random_value, 0, 1)

    # 输出结果和置信区间
    print(f'Generated random value (defective rate): {random_value:.4f}')
    print(f'95% confidence interval: ({confidence_interval[0]:.4f}, {confidence_interval[1]:.4f})')
    return  confidence_interval
#for R2

#for R7