import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu  # 统计检验工具
import matplotlib.pyplot as plt

# 设置图片清晰度（DPI）
plt.rcParams['figure.dpi'] = 300

# 设置中文字体，确保中文正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']

# 设置负号正常显示
plt.rcParams['axes.unicode_minus'] = False

# 加载企业多维度数据
data = pd.read_csv('企业多维度数据.csv')

# 根据is_potential_enterprise列划分高增长组和非高增长组
# 1表示高增长组，0表示非高增长组
data['growth_group'] = data['is_potential_enterprise'].apply(lambda x: '高增长组' if x == 1 else '非高增长组')

# 选取需要分析的特征列
columns_to_analyze = ['高科技标签数', '新兴产业标签数', '政府支持标签数', '业务扩张标签数',
                      '人员扩张标签数', '诉讼次数', '近一年案件数', '被告次数占比(%)',
                      '成立年限_年', '融资频率_次每年', '组织形式_编码', '实缴资本完成率_%',
                      '投资方质量评分', '股东数量', '对数注册资本']

# 按增长分组计算各特征的均值和中位数
grouped_mean = data.groupby('growth_group')[columns_to_analyze].mean()    # 均值
grouped_median = data.groupby('growth_group')[columns_to_analyze].median()  # 中位数

# 初始化存储p值的字典（用于记录统计检验结果）
p_values = {}

# 创建画布：每行2个子图（非高增长组和高增长组），行数等于特征数量
fig, axes = plt.subplots(nrows=len(columns_to_analyze), ncols=2, figsize=(10, 5 * len(columns_to_analyze)))

# 遍历每个需要分析的特征
for i, col in enumerate(columns_to_analyze):
    # 获取高增长组和非高增长组的特征数据，并删除缺失值
    high_growth_data = data[data['growth_group'] == '高增长组'][col].dropna()
    non_high_growth_data = data[data['growth_group'] == '非高增长组'][col].dropna()

    # 绘制非高增长组的直方图
    axes[i, 0].hist(non_high_growth_data, bins=20, alpha=0.7, label='非高增长组')
    axes[i, 0].set_title(f'{col} 非高增长组分布')
    axes[i, 0].set_xlabel(col)
    axes[i, 0].set_ylabel('频数')
    axes[i, 0].legend()

    # 绘制高增长组的直方图
    axes[i, 1].hist(high_growth_data, bins=20, alpha=0.7, label='高增长组')
    axes[i, 1].set_title(f'{col} 高增长组分布')
    axes[i, 1].set_xlabel(col)
    axes[i, 1].set_ylabel('频数')
    axes[i, 1].legend()

    # 选择合适的统计检验方法：
    # 1. 样本量足够大（>=30）或存在空组时，使用t检验
    # 2. 样本量较小时，使用非参数的秩和检验（mann-whitney u检验）
    if (len(high_growth_data) >= 30 and len(non_high_growth_data) >= 30) or \
       (high_growth_data.shape[0] == 0) or (non_high_growth_data.shape[0] == 0):
        # 独立样本t检验：检验两组数据的均值是否存在显著差异
        _, p_value = ttest_ind(high_growth_data, non_high_growth_data)
    else:
        # 秩和检验：非参数检验，不要求数据符合正态分布
        _, p_value = mannwhitneyu(high_growth_data, non_high_growth_data)

    # 存储p值（p值越小，两组差异越显著）
    p_values[col] = p_value

# 调整子图布局，避免标签重叠
plt.tight_layout()

# 保存直方图图片
plt.savefig('特征工程/直方图.png')

# 将p值转换为DataFrame，便于查看和保存
p_values_df = pd.DataFrame.from_dict(p_values, orient='index', columns=['p 值'])

# 创建Excel文件，保存各类统计结果
with pd.ExcelWriter('特征工程/对比结果.xlsx') as writer:
    grouped_mean.to_excel(writer, sheet_name='均值对比')       # 均值对比结果
    grouped_median.to_excel(writer, sheet_name='中位数对比')   # 中位数对比结果
    p_values_df.to_excel(writer, sheet_name='p 值结果')        # 统计检验p值结果