import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# 读取企业多维度数据CSV文件
df = pd.read_csv('企业多维度数据.csv')

# 选择需要进行相关性分析的数值型特征列
# 包含各类标签数、诉讼相关指标、增长评分、企业潜力标识等
numeric_columns = ['高科技标签数', '新兴产业标签数', '政府支持标签数', '业务扩张标签数', 
                 '人员扩张标签数', '诉讼次数', '近一年案件数', '被告次数占比(%)',
                 'growth_score', 'is_potential_enterprise', '成立年限_年', 
                 '融资频率_次每年', '实缴资本完成率_%', '投资方质量评分', 
                 '股东数量', '对数注册资本', '组织形式_编码']

# 处理缺失值：使用0填充所有缺失数据
df_numeric = df[numeric_columns].fillna(0)

# 计算各特征与高增长标识(is_potential_enterprise)的相关系数
# 并按相关系数从大到小排序
correlation_with_target = df_numeric.corr()['is_potential_enterprise'].sort_values(ascending=False)

# 打印各特征与高增长标识的相关系数
print("各特征与高增长标识(is_potential_enterprise)的相关系数:")
print("=" * 60)
for feature, corr in correlation_with_target.items():
    # 排除目标变量自身
    if feature != 'is_potential_enterprise':
        print(f"{feature:.<20} {corr:.4f}")

# 构建相关性矩阵和热力图
# 计算所有数值型特征之间的相关系数矩阵
correlation_matrix = df_numeric.corr()

# 设置图形大小
plt.figure(figsize=(16, 14))

# 创建热力图
# 创建上三角掩码，用于只显示下三角部分（避免重复显示）
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, 
           mask=mask,          # 应用掩码
           annot=True,         # 显示相关系数值
           fmt='.2f',          # 保留两位小数
           cmap='RdBu_r',      # 红蓝色系（红色正相关，蓝色负相关）
           center=0,           # 以0为中心
           square=True,        # 正方形单元格
           cbar_kws={'shrink': 0.8})  # 调整颜色条大小

plt.title('企业多维度数据特征相关性热力图', fontsize=16, fontweight='bold')
plt.tight_layout()  # 调整布局
plt.show()  # 显示热力图

# 识别高度相关的特征群组
# 筛选出相关系数绝对值大于0.7的特征对（高度相关）
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_value = correlation_matrix.iloc[i, j]
        # 排除自身与自身的相关（虽然理论上i≠j，但做双重保险）
        if abs(corr_value) > 0.7 and correlation_matrix.columns[i] != correlation_matrix.columns[j]:
            high_corr_pairs.append((
                correlation_matrix.columns[i],
                correlation_matrix.columns[j],
                corr_value
            ))

# 按相关性绝对值从大到小排序
high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

# 打印高度相关的特征对
print("\n高度相关的特征对 (|r| > 0.7):")
print("=" * 50)
for feature1, feature2, corr in high_corr_pairs:
    print(f"{feature1:<20} - {feature2:<20}: {corr:.4f}")

# 定义函数：识别高度相关的特征群组
def find_correlation_clusters(correlation_matrix, threshold=0.7):
    """
    找出高度相关的特征群组（组内特征两两相关系数绝对值>阈值）
    
    参数:
        correlation_matrix: 相关系数矩阵
        threshold: 相关系数阈值，默认0.7
    返回:
        clusters: 特征群组列表
    """
    clusters = []
    features = list(correlation_matrix.columns)
    visited = set()  # 记录已处理的特征
    
    for feature in features:
        if feature in visited:
            continue
            
        # 初始化当前群组，包含当前特征
        cluster = [feature]
        visited.add(feature)
        
        # 寻找与当前特征高度相关的其他特征
        for other_feature in features:
            if other_feature not in visited and abs(correlation_matrix.loc[feature, other_feature]) > threshold:
                cluster.append(other_feature)
                visited.add(other_feature)
        
        # 只保留包含多个特征的群组
        if len(cluster) > 1:
            clusters.append(cluster)
    
    return clusters

# 调用函数获取高度相关的特征群组
correlation_clusters = find_correlation_clusters(correlation_matrix)

# 打印特征群组
print("\n高度相关的特征群组:")
print("=" * 40)
for i, cluster in enumerate(correlation_clusters, 1):
    print(f"群组 {i}: {cluster}")

# 分析与高增长标识相关性最强的前10个特征
top_features = correlation_with_target[1:11]  # 排除目标变量自身

# 创建可视化图形
plt.figure(figsize=(12, 8))
# 为正相关和负相关设置不同颜色（红色正相关，蓝色负相关）
colors = ['red' if x > 0 else 'blue' for x in top_features.values]
bars = plt.barh(range(len(top_features)), top_features.values, color=colors, alpha=0.7)

plt.xlabel('相关系数')
plt.title('与高增长标识最相关的特征 (Top 10)')
plt.yticks(range(len(top_features)), top_features.index)

# 在条形上添加数值标签
for bar, value in zip(bars, top_features.values):
    # 根据正负相关调整标签位置
    plt.text(bar.get_width() + (0.01 if value > 0 else -0.03), 
             bar.get_y() + bar.get_height()/2, 
             f'{value:.3f}', 
             ha='left' if value > 0 else 'right', 
             va='center')

plt.grid(axis='x', alpha=0.3)  # 添加网格线
plt.tight_layout()
plt.show()