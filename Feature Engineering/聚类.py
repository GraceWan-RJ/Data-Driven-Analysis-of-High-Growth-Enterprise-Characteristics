import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler  # 数据标准化工具
from sklearn.cluster import KMeans  # K均值聚类算法
from sklearn.metrics import silhouette_score  # 轮廓系数（评估聚类效果）
from sklearn.decomposition import PCA  # 主成分分析（用于降维可视化）
import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息

# 设置中文字体，确保中文正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']
# 设置负号显示
plt.rcParams['axes.unicode_minus'] = False

# 读取企业多维度数据
df = pd.read_csv('企业多维度数据.csv')

# 查看数据基本信息
print("数据形状:", df.shape)  # 输出数据行数和列数
print("\n数据前5行:")
print(df.head())  # 输出前5行数据

# 选择用于聚类的数值型特征
features = [
    'growth_score', '高科技标签数', '新兴产业标签数', '政府支持标签数',
    '业务扩张标签数', '人员扩张标签数', '诉讼次数', '近一年案件数', 
    '被告次数占比(%)', '成立年限_年', '融资频率_次每年', '实缴资本完成率_%',
    '投资方质量评分', '股东数量', '对数注册资本', '组织形式_编码'
]

# 数据预处理：提取用于聚类的特征列
df_cluster = df[features].copy()

# 填充缺失值：使用各列的中位数填充（相比均值更抗异常值）
df_cluster = df_cluster.fillna(df_cluster.median())

# 输出处理后的数据信息
print(f"\n用于聚类的特征数量: {len(features)}")
print(f"处理后数据形状: {df_cluster.shape}")

# 数据标准化：将各特征缩放到均值为0、标准差为1的范围内
# 消除不同特征量纲差异对聚类结果的影响
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)

# 定义函数：寻找最优K值（聚类数量）
def find_optimal_k(X, max_k=8):
    """
    利用肘部法则和轮廓系数寻找最优聚类数量K
    
    参数:
        X: 标准化后的特征数据
        max_k: 最大尝试的K值，默认8
    返回:
        k_range: K值范围
        inertias: 不同K值对应的簇内平方和
        silhouette_scores: 不同K值对应的轮廓系数
    """
    inertias = []  # 存储簇内平方和（越小越好）
    silhouette_scores = []  # 存储轮廓系数（越接近1越好）
    k_range = range(2, max_k+1)  # K值从2到max_k
    
    for k in k_range:
        # 构建KMeans模型，设置随机种子确保结果可复现
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)  # 记录簇内平方和
        
        # 轮廓系数需要至少2个簇才能计算
        if k > 1:
            score = silhouette_score(X, kmeans.labels_)
            silhouette_scores.append(score)
    
    # 绘制肘部法则和轮廓系数图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 肘部法则图：寻找"肘部"位置对应的K值
    ax1.plot(k_range, inertias, 'bo-')
    ax1.set_xlabel('K值')
    ax1.set_ylabel('簇内平方和(Inertia)')
    ax1.set_title('肘部法则 - 寻找最优K值')
    ax1.grid(True)
    
    # 轮廓系数图：值越高说明聚类效果越好
    ax2.plot(range(2, max_k+1), silhouette_scores, 'ro-')
    ax2.set_xlabel('K值')
    ax2.set_ylabel('轮廓系数')
    ax2.set_title('轮廓系数 - 寻找最优K值')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return k_range, inertias, silhouette_scores

# 调用函数寻找最优K值
k_range, inertias, silhouette_scores = find_optimal_k(X_scaled)

# 选择K=4进行聚类（基于轮廓系数和业务解释性综合判断）
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)  # 获取聚类结果

# 将聚类结果添加到原始数据中
df['cluster'] = clusters

# 输出聚类基本信息
print(f"\n聚类完成! 使用K={optimal_k}")
print("各簇企业数量:")
print(df['cluster'].value_counts().sort_index())  # 统计每个簇的企业数量

# 定义函数：分析每个簇的特征
def analyze_clusters(df, features, clusters):
    """
    分析每个簇的特征均值和相对重要性
    
    参数:
        df: 包含聚类结果的数据框
        features: 用于聚类的特征列表
        clusters: 聚类标签
    返回:
        cluster_analysis: 各簇特征均值
        feature_importance: 各簇特征相对重要性（标准化后）
    """
    # 按簇分组计算各特征的均值
    cluster_analysis = df.groupby('cluster')[features].mean()
    
    # 计算每个特征在各簇中的相对重要性（标准化处理）
    # 公式：(簇均值 - 总体均值) / 总体标准差
    feature_importance = cluster_analysis.copy()
    for feature in features:
        feature_importance[feature] = (cluster_analysis[feature] - cluster_analysis[feature].mean()) / cluster_analysis[feature].std()
    
    return cluster_analysis, feature_importance

# 调用函数分析各簇特征
cluster_analysis, feature_importance = analyze_clusters(df, features, clusters)

# 输出各簇特征均值
print("\n各簇特征均值:")
print(cluster_analysis.round(2))  # 保留两位小数

# 定义函数：可视化聚类结果
def visualize_clusters(df, features, cluster_analysis):
    """
    可视化各簇的特征对比和特征重要性
    
    参数:
        df: 包含聚类结果的数据框
        features: 用于聚类的特征列表
        cluster_analysis: 各簇特征均值
    """
    # 1. 主要特征对比图：选择几个关键特征进行可视化
    key_features = ['growth_score', '高科技标签数', '政府支持标签数', '诉讼次数', '融资频率_次每年']
    
    # 创建2行3列的子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()  # 将子图数组展平
    
    # 为每个关键特征绘制条形图
    for i, feature in enumerate(key_features):
        if i < len(axes):
            cluster_analysis[feature].plot(kind='bar', ax=axes[i], 
                                         color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            axes[i].set_title(f'{feature} - 各簇对比')
            axes[i].set_ylabel(feature)
            axes[i].tick_params(axis='x', rotation=45)  # 旋转x轴标签避免重叠
    
    # 隐藏多余的子图
    for i in range(len(key_features), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # 2. 热力图显示特征重要性
    plt.figure(figsize=(12, 8))
    sns.heatmap(feature_importance.T, annot=True, cmap='RdBu_r', center=0,
               fmt='.2f', linewidths=0.5)
    plt.title('各簇特征相对重要性热力图\n(标准化后的均值)')
    plt.tight_layout()
    plt.show()

# 调用函数可视化聚类结果
visualize_clusters(df, features, cluster_analysis)

# 定义函数：PCA降维可视化聚类结果
def pca_visualization(X_scaled, clusters, df):
    """
    使用PCA将高维数据降维到2维，可视化聚类结果
    
    参数:
        X_scaled: 标准化后的特征数据
        clusters: 聚类标签
        df: 原始数据框
    """
    # 使用PCA将数据降维到2个主成分
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # 绘制散点图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, 
                        cmap='viridis', alpha=0.7, s=50)
    plt.colorbar(scatter, label='簇')
    plt.xlabel(f'主成分1 ({pca.explained_variance_ratio_[0]:.2%})')  # 显示主成分解释方差比例
    plt.ylabel(f'主成分2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.title('PCA - 企业聚类可视化')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 输出PCA解释方差比
    print(f"PCA解释方差比: {pca.explained_variance_ratio_}")

# 调用函数进行PCA降维可视化
pca_visualization(X_scaled, clusters, df)

# 定义函数：生成簇画像
def create_cluster_profiles(df, cluster_analysis, feature_importance):
    """
    为每个簇创建详细的画像，包括企业数量、典型企业和主要特征
    
    参数:
        df: 包含聚类结果的数据框
        cluster_analysis: 各簇特征均值
        feature_importance: 各簇特征相对重要性
    返回:
        cluster_profiles: 包含各簇画像的字典
    """
    cluster_profiles = {}
    
    for cluster_id in range(optimal_k):
        # 提取当前簇的企业数据
        cluster_data = df[df['cluster'] == cluster_id]
        # 构建簇画像
        profile = {
            '企业数量': len(cluster_data),
            # 获取前3个典型企业名称
            '典型企业': cluster_data.head(3)['name(M|2147483647)'].tolist(),
            'growth_score均值': cluster_analysis.loc[cluster_id, 'growth_score'],
            '主要特征': {}
        }
        
        # 找出该簇最显著的5个特征（按相对重要性绝对值排序）
        cluster_imp = feature_importance.loc[cluster_id]
        top_features = cluster_imp.abs().nlargest(5).index.tolist()
        
        # 记录每个主要特征的均值和相对重要性
        for feature in top_features:
            profile['主要特征'][feature] = {
                '均值': cluster_analysis.loc[cluster_id, feature],
                '相对重要性': cluster_imp[feature]
            }
        
        cluster_profiles[f'簇{cluster_id}'] = profile
    
    return cluster_profiles

# 生成各簇详细画像
cluster_profiles = create_cluster_profiles(df, cluster_analysis, feature_importance)

# 打印详细的簇画像
print("\n" + "="*50)
print("各簇详细画像")
print("="*50)

for cluster_id, profile in cluster_profiles.items():
    print(f"\n{cluster_id}:")
    print(f"  企业数量: {profile['企业数量']}")
    print(f"  典型企业: {profile['典型企业']}")
    print(f"  增长分数均值: {profile['growth_score均值']:.2f}")
    print("  主要特征:")
    for feature, stats in profile['主要特征'].items():
        print(f"    - {feature}: {stats['均值']:.2f} (重要性: {stats['相对重要性']:.2f})")

# 保存聚类结果到CSV文件
df.to_csv('企业聚类结果.csv', index=False, encoding='utf-8-sig')
print(f"\n聚类结果已保存到 '企业聚类结果.csv'")

# 定义函数：为每个簇分配业务标签
def assign_business_labels(cluster_profiles):
    """
    基于簇的特征为每个簇分配具有业务意义的标签
    
    参数:
        cluster_profiles: 各簇画像字典
    返回:
        business_labels: 各簇的业务标签
    """
    business_labels = {}
    
    for cluster_id, profile in cluster_profiles.items():
        # 提取关键指标用于标签判断
        growth = profile['growth_score均值']
        tech = profile['主要特征'].get('高科技标签数', {}).get('均值', 0)
        gov_support = profile['主要特征'].get('政府支持标签数', {}).get('均值', 0)
        litigation = profile['主要特征'].get('诉讼次数', {}).get('均值', 0)
        
        # 根据特征规则分配业务标签
        if growth > 30 and tech > 5:
            label = "高增长创新型企业"
        elif growth < 25 and litigation < 10:
            label = "稳健传统型企业"
        elif growth > 28 and litigation > 20:
            label = "高风险高增长型企业"
        elif gov_support > 3:
            label = "政府支持型潜力企业"
        else:
            label = "混合型企业发展中企业"
            
        business_labels[cluster_id] = label
    
    return business_labels

# 为各簇分配业务标签
business_labels = assign_business_labels(cluster_profiles)

# 打印业务标签
print("\n" + "="*50)
print("业务标签分配")
print("="*50)
for cluster_id, label in business_labels.items():
    print(f"{cluster_id}: {label}")

# 输出最终总结报告
print("\n" + "="*60)
print("企业聚类分析总结报告")
print("="*60)
print(f"分析企业总数: {len(df)}")
print(f"使用聚类数: {optimal_k}")
print(f"平均轮廓系数: {silhouette_score(X_scaled, clusters):.3f}")

# 输出各类型企业分布
print("\n各类型企业分布:")
cluster_counts = df['cluster'].value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    label = business_labels.get(f'簇{cluster_id}', '未知类型')
    percentage = (count / len(df)) * 100  # 计算占比
    print(f"  {label}: {count}家企业 ({percentage:.1f}%)")

# 基于聚类结果给出建议
print("\n建议:")
print("1. 高增长创新型企业: 重点投资和支持")
print("2. 政府支持型潜力企业: 加强政策引导和资源对接") 
print("3. 高风险高增长型企业: 加强风险监控和法律支持")
print("4. 稳健传统型企业: 推动数字化转型和业务升级")