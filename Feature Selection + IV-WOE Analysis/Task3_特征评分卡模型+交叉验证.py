import pandas as pd
import numpy as np
from scipy import stats

# 读取数据
df = pd.read_excel('企业聚类结果.xlsx')

# 定义特征分类
good_binning_features = [
    '融资频率_次每年', 
    '投资方质量评分', 
    '对数注册资本'
]   

custom_binning_features = [
    '股东数量', 
    '实缴资本完成率_%', 
    '近一年案件数', 
    '被告次数占比(%)', 
    '高科技标签数', 
    '政府支持标签数'
]

# 添加cluster到特征列表中
categorical_features = ['cluster']

selected_features = good_binning_features + custom_binning_features + categorical_features

# 创建目标变量
df['target'] = (df['growth_score'] > df['growth_score'].median()).astype(int)
print(f"目标变量分布: {df['target'].value_counts().to_dict()}")

# 创建新的DataFrame来存储WOE结果
woe_df = df[['eid']].copy()

def calculate_woe_iv(df, feature, target='target'):
    """
    计算特征的WOE和IV值
    """
    # 处理缺失值
    temp_df = df[[feature, target]].copy()
    temp_df[feature] = temp_df[feature].fillna(-999)  # 用-999表示缺失值
    
    # 分组统计
    total_good = temp_df[target].sum()
    total_bad = len(temp_df) - total_good
    
    woe_dict = {}
    iv_total = 0
    
    # 对每个分组计算WOE和IV
    for value in temp_df[feature].unique():
        group_data = temp_df[temp_df[feature] == value]
        good_count = group_data[target].sum()
        bad_count = len(group_data) - good_count
        
        # 避免除零错误
        if good_count == 0:
            good_count = 0.5
        if bad_count == 0:
            bad_count = 0.5
            
        # 计算分布
        good_dist = good_count / total_good
        bad_dist = bad_count / total_bad
        
        # 计算WOE
        if bad_dist == 0:
            woe = 0
        else:
            woe = np.log(good_dist / bad_dist)
        
        # 计算IV
        iv = (good_dist - bad_dist) * woe
        
        woe_dict[value] = {
            'woe': woe,
            'iv': iv,
            'good_count': good_count,
            'bad_count': bad_count,
            'total_count': len(group_data)
        }
        
        iv_total += iv
    
    return woe_dict, iv_total

def create_woe_binning(df, feature, target='target', n_bins=5, binning_type='quantile'):
    """
    创建WOE分箱
    """
    temp_df = df[[feature, target]].copy()
    temp_df[feature] = temp_df[feature].fillna(-999)
    
    # 对连续变量进行分箱
    if binning_type == 'quantile':
        temp_df[f'{feature}_bin'], bins = pd.qcut(temp_df[feature], q=n_bins, duplicates='drop', retbins=True, labels=False)
    else:
        temp_df[f'{feature}_bin'], bins = pd.cut(temp_df[feature], bins=n_bins, retbins=True, labels=False)
    
    # 计算每个分箱的WOE
    woe_dict = {}
    iv_total = 0
    
    total_good = temp_df[target].sum()
    total_bad = len(temp_df) - total_good
    
    for bin_num in sorted(temp_df[f'{feature}_bin'].unique()):
        bin_data = temp_df[temp_df[f'{feature}_bin'] == bin_num]
        good_count = bin_data[target].sum()
        bad_count = len(bin_data) - good_count
        
        # 避免除零错误
        if good_count == 0:
            good_count = 0.5
        if bad_count == 0:
            bad_count = 0.5
            
        good_dist = good_count / total_good
        bad_dist = bad_count / total_bad
        
        woe = np.log(good_dist / bad_dist)
        iv = (good_dist - bad_dist) * woe
        
        woe_dict[bin_num] = {
            'woe': woe,
            'iv': iv,
            'good_count': good_count,
            'bad_count': bad_count,
            'total_count': len(bin_data),
            'bin_range': f"{bins[bin_num]:.4f}-{bins[bin_num+1]:.4f}" if bin_num < len(bins)-1 else f"{bins[bin_num]:.4f}+"
        }
        
        iv_total += iv
    
    return temp_df[f'{feature}_bin'], woe_dict, iv_total, bins

def improved_bin_financing_frequency(x):
    """改进的融资频率分箱 - 基于业务逻辑"""
    if pd.isna(x) or x == 0:
        return 0  # 无融资活动
    elif 0 < x <= 0.5:
        return 1  # 低频融资（2年1次或更少）
    elif 0.5 < x <= 1:
        return 2  # 年度融资
    elif 1 < x <= 2:
        return 3  # 半年至季度融资
    elif 2 < x <= 4:
        return 4  # 高频融资（季度至月度）
    else:
        return 5  # 超高频融资

# 股东数量使用原来的分箱方式
def bin_shareholder_count(x):
    """股东数量的原始分箱方式"""
    if x == 0:
        return 0  # 未披露/异常
    elif x == 1:
        return 1  # 单一股东
    elif 2 <= x <= 5:
        return 2  # 少数股东
    elif 6 <= x <= 20:
        return 3  # 中型结构
    else:
        return 4  # 复杂结构

# Cluster多级分类函数 - 基于您提供的业务含义
def bin_cluster(x):
    """基于业务含义的cluster分箱"""
    if x == 2:
        return 0  # Cluster 2: 资本驱动的最高增长组 - 最高优先级
    elif x == 0:
        return 1  # Cluster 0: 高增长但高风险的存量资本型
    elif x == 3:
        return 2  # Cluster 3: 政策与高新驱动的稳健增长组
    elif x == 1:
        return 3  # Cluster 1: 低增长、低活跃的过渡组
    else:
        return 4  # 其他/异常值
    
# 近一年案件数多级分类函数
def bin_case_count(x):
    """近一年案件数的分箱"""
    if pd.isna(x) or x == 0:
        return 0  # 近期无风险 (L0)
    elif 1 <= x <= 2:
        return 1  # 近期低风险 (L1)
    elif 3 <= x <= 5:
        return 2  # 近期中风险 (L2)
    else:  # x >= 6
        return 3  # 近期高风险 (L3)

# 被告次数占比多级分类函数
def bin_defendant_ratio(x):
    """被告次数占比的分箱"""
    if pd.isna(x) or x == 0:
        return 0  # 无被告记录 (L0)
    elif 0 < x <= 50:
        return 1  # 原告为主 (L1)
    elif 50 < x <= 100:
        return 2  # 被告为主 (L2)
    else:  # x > 100
        return 3  # 数据异常 (L3)

# 高科技标签数多级分类函数
def bin_high_tech_tags(x):
    """高科技标签数的分箱"""
    if pd.isna(x):
        return 0  # 未披露
    elif x == 0:
        return 1  # 无
    elif 1 <= x <= 3:
        return 2  # 初级
    elif 4 <= x <= 6:
        return 3  # 中级
    else:  # x >= 7
        return 4  # 高级

# 政府支持标签数多级分类函数
def bin_gov_support_tags(x):
    """政府支持标签数的分箱"""
    if pd.isna(x) or x == 0:
        return 0  # 无 (L0)
    elif x == 1:
        return 1  # 入门 (L1)
    elif 2 <= x <= 3:
        return 2  # 初级 (L2)
    elif 4 <= x <= 6:
        return 3  # 中级 (L3)
    else:  # x >= 7
        return 4  # 高级 (L4)

print("开始进行改进的WOE统一分数分配处理...")

# ============================================================================
# 第一步：收集所有特征的WOE值和IV值
# ============================================================================

print("\n" + "="*50)
print("收集所有特征的WOE值和IV值")
print("="*50)

# 存储所有特征的WOE字典和IV值
feature_woe_dicts = {}
feature_iv_values = {}

# 第一部分：对分箱效果好的特征进行WOE分箱
print("\n对分箱效果好的特征进行WOE分箱:")
for feature in good_binning_features:
    print(f"\n处理特征: {feature}")
    
    if feature not in df.columns:
        print(f"警告: 特征 {feature} 在数据集中不存在")
        continue
    
    try:
        # 创建WOE分箱
        bin_series, woe_dict, iv_total, bins = create_woe_binning(df, feature, n_bins=5)
        
        # 存储分箱结果和WOE字典
        woe_df[f'{feature}_分箱'] = bin_series
        feature_woe_dicts[feature] = woe_dict
        feature_iv_values[feature] = iv_total
        
        # 存储分箱信息
        woe_df[f'{feature}_分箱信息'] = bin_series.map(
            {bin_num: f"分箱{bin_num}({woe_dict[bin_num]['bin_range']})" 
             for bin_num in woe_dict.keys()}
        )
        
        print(f"  IV值: {iv_total:.4f}")
        
    except Exception as e:
        print(f"  WOE分箱失败: {e}")

# 第二部分：对自定义分箱特征进行改进的WOE计算
print("\n对自定义分箱特征进行改进的WOE计算:")

# 股东数量多级分类和WOE计算 - 使用原始分箱方式
if '股东数量' in df.columns:
    print(f"\n处理特征: 股东数量 - 使用原始分箱方式")
    shareholder_data = df['股东数量'].copy()
    shareholder_data = shareholder_data.fillna(0)
    
    # 使用原始的分箱函数
    binned_shareholder = shareholder_data.apply(bin_shareholder_count)
    woe_df['股东数量_分箱'] = binned_shareholder
    
    # 计算WOE
    woe_dict, iv_total = calculate_woe_iv(pd.DataFrame({
        '股东数量': binned_shareholder,
        'target': df['target']
    }), '股东数量')
    
    # 存储WOE字典和IV值
    feature_woe_dicts['股东数量'] = woe_dict
    feature_iv_values['股东数量'] = iv_total
    
    bin_labels = {
        0: "未披露/异常 (0)",
        1: "单一股东 (1)",
        2: "少数股东 (2-5)", 
        3: "中型结构 (6-20)",
        4: "复杂结构 (21+)"
    }
    woe_df['股东数量_分箱信息'] = binned_shareholder.map(bin_labels)
    
    print(f"  IV值: {iv_total:.4f}")

# 改进的融资频率多级分类和WOE计算
if '融资频率_次每年' in df.columns:
    print(f"\n处理特征: 融资频率_次每年 - 使用改进的业务逻辑分箱")
    financing_data = df['融资频率_次每年'].copy()
    financing_data = financing_data.fillna(0)
    
    # 使用改进的分箱函数
    binned_financing = financing_data.apply(improved_bin_financing_frequency)
    woe_df['融资频率_次每年_分箱'] = binned_financing
    
    # 计算WOE
    woe_dict, iv_total = calculate_woe_iv(pd.DataFrame({
        '融资频率_次每年': binned_financing,
        'target': df['target']
    }), '融资频率_次每年')
    
    # 存储WOE字典和IV值
    feature_woe_dicts['融资频率_次每年'] = woe_dict
    feature_iv_values['融资频率_次每年'] = iv_total
    
    bin_labels = {
        0: "无融资活动 (0)",
        1: "低频融资 (0-0.5]",
        2: "年度融资 (0.5-1]", 
        3: "半年至季度融资 (1-2]",
        4: "高频融资 (2-4]",
        5: "超高频融资 (4+)"
    }
    woe_df['融资频率_次每年_分箱信息'] = binned_financing.map(bin_labels)
    
    print(f"  IV值: {iv_total:.4f}")

# 实缴资本完成率的多级分类和WOE计算
if '实缴资本完成率_%' in df.columns:
    print(f"\n处理特征: 实缴资本完成率_%")
    capital_data = df['实缴资本完成率_%'].copy()
    capital_data = capital_data.fillna(0)
    
    def bin_capital_completion(x):
        if x == 0:
            return 0  # 未披露
        elif 0 < x <= 50:
            return 1  # 低比例实缴
        elif 50 < x < 100:
            return 2  # 中高比例实缴
        elif x == 100:
            return 3  # 完全实缴
        else:
            return 4  # 异常值
    
    binned_capital = capital_data.apply(bin_capital_completion)
    woe_df['实缴资本完成率_%_分箱'] = binned_capital
    
    # 计算WOE
    woe_dict, iv_total = calculate_woe_iv(pd.DataFrame({
        '实缴资本完成率_%': binned_capital,
        'target': df['target']
    }), '实缴资本完成率_%')
    
    # 存储WOE字典和IV值
    feature_woe_dicts['实缴资本完成率_%'] = woe_dict
    feature_iv_values['实缴资本完成率_%'] = iv_total
    
    bin_labels = {
        0: "未披露 (0)",
        1: "低比例实缴 (0-50]",
        2: "中高比例实缴 (50-100)", 
        3: "完全实缴 (100)",
        4: "异常值 (>100)"
    }
    woe_df['实缴资本完成率_%_分箱信息'] = binned_capital.map(bin_labels)
    
    print(f"  IV值: {iv_total:.4f}")

# 近一年案件数的多级分类和WOE计算
if '近一年案件数' in df.columns:
    print(f"\n处理特征: 近一年案件数")
    case_data = df['近一年案件数'].copy()
    case_data = case_data.fillna(0)
    
    # 使用分箱函数
    binned_cases = case_data.apply(bin_case_count)
    woe_df['近一年案件数_分箱'] = binned_cases
    
    # 计算WOE
    woe_dict, iv_total = calculate_woe_iv(pd.DataFrame({
        '近一年案件数': binned_cases,
        'target': df['target']
    }), '近一年案件数')
    
    # 存储WOE字典和IV值
    feature_woe_dicts['近一年案件数'] = woe_dict
    feature_iv_values['近一年案件数'] = iv_total
    
    bin_labels = {
        0: "近期无风险 (0)",
        1: "近期低风险 (1-2)", 
        2: "近期中风险 (3-5)",
        3: "近期高风险 (6+)"
    }
    woe_df['近一年案件数_分箱信息'] = binned_cases.map(bin_labels)
    
    print(f"  IV值: {iv_total:.4f}")
    print(f"  分箱WOE信息:")
    for bin_num, info in sorted(woe_dict.items()):
        print(f"    {bin_labels[bin_num]}: WOE={info['woe']:.4f}, "
              f"好样本={info['good_count']}, 坏样本={info['bad_count']}")

# 被告次数占比的多级分类和WOE计算
if '被告次数占比(%)' in df.columns:
    print(f"\n处理特征: 被告次数占比(%)")
    defendant_data = df['被告次数占比(%)'].copy()
    defendant_data = defendant_data.fillna(0)
    
    # 使用分箱函数
    binned_defendant = defendant_data.apply(bin_defendant_ratio)
    woe_df['被告次数占比(%)_分箱'] = binned_defendant
    
    # 计算WOE
    woe_dict, iv_total = calculate_woe_iv(pd.DataFrame({
        '被告次数占比(%)': binned_defendant,
        'target': df['target']
    }), '被告次数占比(%)')
    
    # 存储WOE字典和IV值
    feature_woe_dicts['被告次数占比(%)'] = woe_dict
    feature_iv_values['被告次数占比(%)'] = iv_total
    
    bin_labels = {
        0: "无被告记录 (0)",
        1: "原告为主 (0-50]", 
        2: "被告为主 (50-100]",
        3: "数据异常 (>100)"
    }
    woe_df['被告次数占比(%)_分箱信息'] = binned_defendant.map(bin_labels)
    
    print(f"  IV值: {iv_total:.4f}")
    print(f"  分箱WOE信息:")
    for bin_num, info in sorted(woe_dict.items()):
        print(f"    {bin_labels[bin_num]}: WOE={info['woe']:.4f}, "
              f"好样本={info['good_count']}, 坏样本={info['bad_count']}")

# 高科技标签数的多级分类和WOE计算
if '高科技标签数' in df.columns:
    print(f"\n处理特征: 高科技标签数")
    tech_data = df['高科技标签数'].copy()
    tech_data = tech_data.fillna(0)  # 将未披露视为0
    
    # 使用分箱函数
    binned_tech = tech_data.apply(bin_high_tech_tags)
    woe_df['高科技标签数_分箱'] = binned_tech
    
    # 计算WOE
    woe_dict, iv_total = calculate_woe_iv(pd.DataFrame({
        '高科技标签数': binned_tech,
        'target': df['target']
    }), '高科技标签数')
    
    # 存储WOE字典和IV值
    feature_woe_dicts['高科技标签数'] = woe_dict
    feature_iv_values['高科技标签数'] = iv_total
    
    bin_labels = {
        0: "未披露",
        1: "无 (0)", 
        2: "初级 (1-3)",
        3: "中级 (4-6)",
        4: "高级 (7+)"
    }
    woe_df['高科技标签数_分箱信息'] = binned_tech.map(bin_labels)
    
    print(f"  IV值: {iv_total:.4f}")
    print(f"  分箱WOE信息:")
    for bin_num, info in sorted(woe_dict.items()):
        print(f"    {bin_labels[bin_num]}: WOE={info['woe']:.4f}, "
              f"好样本={info['good_count']}, 坏样本={info['bad_count']}")

# 政府支持标签数的多级分类和WOE计算
if '政府支持标签数' in df.columns:
    print(f"\n处理特征: 政府支持标签数")
    gov_data = df['政府支持标签数'].copy()
    gov_data = gov_data.fillna(0)
    
    # 使用分箱函数
    binned_gov = gov_data.apply(bin_gov_support_tags)
    woe_df['政府支持标签数_分箱'] = binned_gov
    
    # 计算WOE
    woe_dict, iv_total = calculate_woe_iv(pd.DataFrame({
        '政府支持标签数': binned_gov,
        'target': df['target']
    }), '政府支持标签数')
    
    # 存储WOE字典和IV值
    feature_woe_dicts['政府支持标签数'] = woe_dict
    feature_iv_values['政府支持标签数'] = iv_total
    
    bin_labels = {
        0: "无 (0)",
        1: "入门 (1)", 
        2: "初级 (2-3)",
        3: "中级 (4-6)",
        4: "高级 (7+)"
    }
    woe_df['政府支持标签数_分箱信息'] = binned_gov.map(bin_labels)
    
    print(f"  IV值: {iv_total:.4f}")
    print(f"  分箱WOE信息:")
    for bin_num, info in sorted(woe_dict.items()):
        print(f"    {bin_labels[bin_num]}: WOE={info['woe']:.4f}, "
              f"好样本={info['good_count']}, 坏样本={info['bad_count']}")

# 第三部分：对分类特征（cluster）进行WOE计算
print("\n对分类特征cluster进行WOE计算:")
if 'cluster' in df.columns:
    print(f"\n处理特征: cluster - 基于业务逻辑分箱")
    cluster_data = df['cluster'].copy()
    cluster_data = cluster_data.fillna(-1)  # 处理缺失值
    
    # 使用基于业务含义的分箱函数
    binned_cluster = cluster_data.apply(bin_cluster)
    woe_df['cluster_分箱'] = binned_cluster
    
    # 计算WOE
    woe_dict, iv_total = calculate_woe_iv(pd.DataFrame({
        'cluster': binned_cluster,
        'target': df['target']
    }), 'cluster')
    
    # 存储WOE字典和IV值
    feature_woe_dicts['cluster'] = woe_dict
    feature_iv_values['cluster'] = iv_total
    
    # 基于业务含义的标签
    bin_labels = {
        0: "资本驱动最高增长组 (Cluster 2)",
        1: "高增长高风险资本型 (Cluster 0)", 
        2: "政策高新稳健增长组 (Cluster 3)",
        3: "低增长低活跃过渡组 (Cluster 1)",
        4: "其他/异常值"
    }
    woe_df['cluster_分箱信息'] = binned_cluster.map(bin_labels)
    
    print(f"  IV值: {iv_total:.4f}")

# ============================================================================
# 第二步：基于WOE值的线性转换计算评分卡分数（行业标准方法）
# ============================================================================

print("\n" + "="*50)
print("基于WOE值的线性转换计算评分卡分数")
print("="*50)

def calculate_scorecard_points(woe_value, factor=20, offset=600, pdo=20):
    """
    使用行业标准方法计算评分卡分数
    公式: Score = offset + factor * ln(odds)
    其中 odds = good/bad, 而WOE = ln(good_dist/bad_dist)
    由于我们无法直接得到odds，通常使用WOE作为ln(odds)的代理
    或者使用: Score = A - B * WOE (如果WOE与风险负相关)
    
    这里我们使用: Score = base_score - factor * WOE
    """
    # 基础分数设为600，每20分odds翻倍/减半(PDO)
    base_score = 600
    # 使用负号是因为WOE值越大表示风险越小，应该给更高分数
    score = base_score - factor * woe_value
    return score

# 为每个特征计算评分卡分数
print("\n计算各特征的评分卡分数:")

for feature in feature_woe_dicts.keys():
    print(f"\n处理特征: {feature}")
    
    # 获取该特征的分箱列名
    bin_column = f'{feature}_分箱'
    if bin_column not in woe_df.columns:
        print(f"  警告: 找不到分箱列 {bin_column}")
        continue
    
    # 创建评分卡分数列
    score_column = f'{feature}_评分卡分数'
    
    # 根据分箱映射WOE值，然后计算评分卡分数
    woe_mapping = {bin_num: info['woe'] for bin_num, info in feature_woe_dicts[feature].items()}
    woe_values = woe_df[bin_column].map(woe_mapping)
    
    # 计算评分卡分数
    scorecard_scores = woe_values.apply(lambda woe: calculate_scorecard_points(woe))
    woe_df[score_column] = scorecard_scores
    
    print(f"  评分卡分数范围: {scorecard_scores.min():.2f} - {scorecard_scores.max():.2f}")
    print(f"  各分箱WOE和分数:")
    for bin_num, info in sorted(feature_woe_dicts[feature].items()):
        score = calculate_scorecard_points(info['woe'])
        print(f"    分箱{bin_num}: WOE={info['woe']:.4f}, 分数={score:.2f}")

# ============================================================================
# 第三步：计算总评分卡分数
# ============================================================================

print("\n" + "="*50)
print("计算总评分卡分数")
print("="*50)

# 收集所有评分卡分数列
scorecard_columns = [col for col in woe_df.columns if col.endswith('_评分卡分数')]
print(f"用于计算总分的评分卡特征: {scorecard_columns}")

if scorecard_columns:
    # 计算初始总评分卡分数
    woe_df['初始总评分卡分数'] = woe_df[scorecard_columns].sum(axis=1)
    
    # 由于初始总评分可能为负，我们将其线性变换到0-1000分范围（行业常用范围）
    min_score = woe_df['初始总评分卡分数'].min()
    max_score = woe_df['初始总评分卡分数'].max()
    
    print(f"初始总评分卡分数范围: {min_score:.2f} - {max_score:.2f}")
    
    # 线性变换到0-1000分
    if max_score != min_score:
        woe_df['标准化总评分卡分数'] = ((woe_df['初始总评分卡分数'] - min_score) / (max_score - min_score) * 1000).round(2)
    else:
        woe_df['标准化总评分卡分数'] = 500  # 如果所有分数相同，给中间分
    
    # 同时计算百分制分数（0-100分）
    if max_score != min_score:
        woe_df['百分制评分卡分数'] = ((woe_df['初始总评分卡分数'] - min_score) / (max_score - min_score) * 100).round(2)
    else:
        woe_df['百分制评分卡分数'] = 50
    
    print(f"标准化总评分卡分数范围: {woe_df['标准化总评分卡分数'].min():.2f} - {woe_df['标准化总评分卡分数'].max():.2f}")
    print(f"百分制评分卡分数范围: {woe_df['百分制评分卡分数'].min():.2f} - {woe_df['百分制评分卡分数'].max():.2f}")
else:
    woe_df['初始总评分卡分数'] = 0
    woe_df['标准化总评分卡分数'] = 0
    woe_df['百分制评分卡分数'] = 0

# ============================================================================
# 第四部分：创建详细的评分卡结果
# ============================================================================

print("\n" + "="*50)
print("创建详细的评分卡结果")
print("="*50)

# 合并原始数据和WOE结果
final_df = pd.concat([df, woe_df.drop('eid', axis=1)], axis=1)

# 创建评分卡详情表
scorecard_details = []

for feature in feature_woe_dicts.keys():
    woe_dict = feature_woe_dicts[feature]
    iv_value = feature_iv_values[feature]
    
    for bin_num, info in sorted(woe_dict.items()):
        score = calculate_scorecard_points(info['woe'])
        
        scorecard_details.append({
            '特征': feature,
            '分箱编号': bin_num,
            'WOE值': info['woe'],
            'IV贡献': info['iv'],
            '评分卡分数': score,
            '好样本数': info['good_count'],
            '坏样本数': info['bad_count'],
            '总样本数': info['total_count'],
            '分箱范围': info.get('bin_range', f'分箱{bin_num}'),
            '特征IV值': iv_value
        })

# 创建评分卡详情DataFrame
scorecard_detail_df = pd.DataFrame(scorecard_details)

# 重新排列列顺序
scorecard_detail_df = scorecard_detail_df[[
    '特征', '分箱编号', '分箱范围', 'WOE值', '评分卡分数', 
    '好样本数', '坏样本数', '总样本数', 'IV贡献', '特征IV值'
]]

# 保存详细评分卡结果到新文件
output_filename = '企业特征评分卡结果_详细.xlsx'

with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
    # 保存主要结果
    final_df.to_excel(writer, sheet_name='企业评分卡结果', index=False)
    
    # 保存评分卡详情
    scorecard_detail_df.to_excel(writer, sheet_name='评分卡详情', index=False)
    
    # 创建特征重要性汇总表
    feature_importance = []
    for feature, iv in feature_iv_values.items():
        feature_importance.append({
            '特征': feature,
            'IV值': iv,
            '预测能力': '强' if iv >= 0.3 else '中等' if iv >= 0.1 else '弱' if iv >= 0.02 else '无'
        })
    
    feature_importance_df = pd.DataFrame(feature_importance).sort_values('IV值', ascending=False)
    feature_importance_df.to_excel(writer, sheet_name='特征重要性', index=False)
    
    # 创建分数分布统计表
    score_stats = []
    for col in ['标准化总评分卡分数', '百分制评分卡分数']:
        if col in final_df.columns:
            score_stats.append({
                '分数类型': col,
                '最小值': final_df[col].min(),
                '最大值': final_df[col].max(),
                '平均值': final_df[col].mean(),
                '中位数': final_df[col].median(),
                '标准差': final_df[col].std(),
                '企业数量': len(final_df)
            })
    
    score_stats_df = pd.DataFrame(score_stats)
    score_stats_df.to_excel(writer, sheet_name='分数分布统计', index=False)

print(f"\n详细的评分卡结果已保存到: {output_filename}")

# ============================================================================
# 第五部分：显示结果预览和统计信息
# ============================================================================

print("\n" + "="*50)
print("结果预览和统计信息")
print("="*50)

# 显示结果预览
print("\n前5行评分卡结果预览:")
preview_columns = ['eid'] 
for feature in selected_features:
    if f'{feature}_评分卡分数' in final_df.columns:
        preview_columns.append(f'{feature}_评分卡分数')
if '标准化总评分卡分数' in final_df.columns:
    preview_columns.append('标准化总评分卡分数')
if '百分制评分卡分数' in final_df.columns:
    preview_columns.append('百分制评分卡分数')

print(final_df[preview_columns].head())

# 详细的统计信息
print("\n" + "="*50)
print("各特征评分卡分数统计:")
print("="*50)

for feature in selected_features:
    score_col = f'{feature}_评分卡分数'
    if score_col in final_df.columns:
        print(f"\n{feature} 评分卡分数:")
        print(f"  分数范围: {final_df[score_col].min():.2f} - {final_df[score_col].max():.2f}")
        print(f"  平均分数: {final_df[score_col].mean():.2f}")
        print(f"  中位数: {final_df[score_col].median():.2f}")

if '标准化总评分卡分数' in final_df.columns:
    print(f"\n标准化总评分卡分数统计:")
    print(f"  分数范围: {final_df['标准化总评分卡分数'].min():.2f} - {final_df['标准化总评分卡分数'].max():.2f}")
    print(f"  平均分数: {final_df['标准化总评分卡分数'].mean():.2f}")
    print(f"  中位数: {final_df['标准化总评分卡分数'].median():.2f}")

if '百分制评分卡分数' in final_df.columns:
    print(f"\n百分制评分卡分数统计:")
    print(f"  分数范围: {final_df['百分制评分卡分数'].min():.2f} - {final_df['百分制评分卡分数'].max():.2f}")
    print(f"  平均分数: {final_df['百分制评分卡分数'].mean():.2f}")
    print(f"  中位数: {final_df['百分制评分卡分数'].median():.2f}")

# 特征重要性分析
print(f"\n" + "="*50)
print("特征IV值分析:")
print("="*50)

for feature, iv in sorted(feature_iv_values.items(), key=lambda x: x[1], reverse=True):
    predictive_power = '强' if iv >= 0.3 else '中等' if iv >= 0.1 else '弱' if iv >= 0.02 else '无'
    print(f"  {feature}: IV={iv:.4f} ({predictive_power}预测能力)")

print("\n基于WOE值的评分卡模型构建完成！")
print(f"详细结果已保存到: {output_filename}")


# ============================================================================
# 第六部分：多维度交叉验证 - 特征选择和模型鲁棒性验证
# ============================================================================

print("\n" + "="*50)
print("多维度交叉验证 - 特征选择和模型鲁棒性验证")
print("="*50)

from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 准备WOE转换后的特征数据
print("\n准备WOE转换后的特征数据...")

# 收集所有特征的WOE值（不是评分卡分数，而是原始的WOE值）
woe_feature_data = []
feature_names = []
feature_woe_mapping = {}

for feature in selected_features:
    if feature in feature_woe_dicts:
        # 为每个特征创建WOE值列
        woe_column = f'{feature}_WOE'
        bin_column = f'{feature}_分箱'
        
        if bin_column in woe_df.columns:
            # 将分箱映射到WOE值
            woe_mapping = {bin_num: info['woe'] for bin_num, info in feature_woe_dicts[feature].items()}
            woe_df[woe_column] = woe_df[bin_column].map(woe_mapping)
            woe_feature_data.append(woe_column)
            feature_names.append(feature)
            feature_woe_mapping[woe_column] = feature

print(f"可用于建模的WOE特征数量: {len(woe_feature_data)}")
print(f"特征列表: {feature_names}")

if len(woe_feature_data) == 0:
    print("警告: 没有找到可用的WOE特征，跳过交叉验证")
else:
    # 准备X和y
    X = woe_df[woe_feature_data].copy()
    y = df['target'].values
    
    # 处理缺失值
    X = X.fillna(0)  # 用0填充缺失的WOE值
    
    print(f"特征矩阵形状: {X.shape}")
    print(f"目标变量分布: {pd.Series(y).value_counts().to_dict()}")
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 设置交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 存储所有结果
    validation_results = {}
    
    # ============================================================================
    # 方法1: L1正则化逻辑回归 (Lasso)
    # ============================================================================
    
    print("\n" + "-"*50)
    print("方法1: L1正则化逻辑回归 (Lasso)")
    print("-"*50)
    
    try:
        # 使用L1正则化的逻辑回归
        l1_lr = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            C=0.1,  # 较小的C值意味着更强的正则化
            random_state=42,
            max_iter=1000
        )
        
        # 交叉验证
        l1_scores = cross_val_score(l1_lr, X_scaled, y, cv=cv, scoring='roc_auc')
        
        # 在整个数据集上训练以获得系数
        l1_lr.fit(X_scaled, y)
        
        # 获取特征系数
        l1_coef_df = pd.DataFrame({
            '特征': [feature_woe_mapping[col] for col in woe_feature_data],
            'WOE列名': woe_feature_data,
            'L1系数': l1_lr.coef_[0],
            '系数绝对值': np.abs(l1_lr.coef_[0])
        }).sort_values('系数绝对值', ascending=False)
        
        print("\nL1正则化特征系数排序:")
        for idx, row in l1_coef_df.iterrows():
            print(f"  {row['特征']}: {row['L1系数']:.4f} (绝对值: {row['系数绝对值']:.4f})")
        
        # 选择非零系数的特征
        selected_by_l1 = l1_coef_df[l1_coef_df['系数绝对值'] > 0.001]
        print(f"\nL1正则化选择的特征数量: {len(selected_by_l1)}")
        print("选中的特征:", selected_by_l1['特征'].tolist())
        
        # 存储L1结果
        validation_results['L1正则化'] = {
            'auc_mean': l1_scores.mean(),
            'auc_std': l1_scores.std(),
            'selected_features': selected_by_l1['特征'].tolist(),
            'n_features': len(selected_by_l1),
            'coefficients': l1_coef_df.set_index('特征')['L1系数'].to_dict()
        }
        
        print(f"\nL1正则化模型性能:")
        print(f"  AUC均值: {l1_scores.mean():.4f} (+/- {l1_scores.std() * 2:.4f})")
        print(f"  各折AUC: {[f'{score:.4f}' for score in l1_scores]}")
        
    except Exception as e:
        print(f"L1正则化失败: {e}")
    
    # ============================================================================
    # 方法2: 递归特征消除 (RFE)
    # ============================================================================
    
    print("\n" + "-"*50)
    print("方法2: 递归特征消除 (RFE)")
    print("-"*50)
    
    try:
        # 使用逻辑回归作为基模型
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        
        # 使用RFE进行特征选择
        rfe = RFE(
            estimator=lr_model,
            n_features_to_select=min(8, len(woe_feature_data)),  # 选择最多8个特征
            step=1  # 每次迭代消除一个特征
        )
        
        rfe.fit(X_scaled, y)
        
        # 获取特征选择结果
        rfe_ranking_df = pd.DataFrame({
            '特征': [feature_woe_mapping[col] for col in woe_feature_data],
            'WOE列名': woe_feature_data,
            'RFE排名': rfe.ranking_,
            '被选择': rfe.support_
        }).sort_values('RFE排名')
        
        print("\nRFE特征排名:")
        for idx, row in rfe_ranking_df.iterrows():
            status = "选中" if row['被选择'] else "未选中"
            print(f"  {row['特征']}: 排名{row['RFE排名']} ({status})")
        
        selected_by_rfe = rfe_ranking_df[rfe_ranking_df['被选择']]
        print(f"\nRFE选择的特征数量: {len(selected_by_rfe)}")
        print("选中的特征:", selected_by_rfe['特征'].tolist())
        
        # 使用RFE选择的特征进行交叉验证
        if len(selected_by_rfe) > 0:
            selected_columns = selected_by_rfe['WOE列名'].tolist()
            X_rfe = X[selected_columns]
            X_rfe_scaled = scaler.fit_transform(X_rfe)
            
            # 逻辑回归模型
            lr_rfe = LogisticRegression(random_state=42, max_iter=1000)
            rfe_scores = cross_val_score(lr_rfe, X_rfe_scaled, y, cv=cv, scoring='roc_auc')
            
            # 存储RFE结果
            validation_results['RFE'] = {
                'auc_mean': rfe_scores.mean(),
                'auc_std': rfe_scores.std(),
                'selected_features': selected_by_rfe['特征'].tolist(),
                'n_features': len(selected_by_rfe),
                'ranking': rfe_ranking_df.set_index('特征')['RFE排名'].to_dict()
            }
            
            print(f"\nRFE特征选择后的模型性能:")
            print(f"  AUC均值: {rfe_scores.mean():.4f} (+/- {rfe_scores.std() * 2:.4f})")
            print(f"  各折AUC: {[f'{score:.4f}' for score in rfe_scores]}")
        
    except Exception as e:
        print(f"RFE特征选择失败: {e}")
    
    # ============================================================================
    # 方法3: 基于IV值的特征选择 (基准)
    # ============================================================================
    
    print("\n" + "-"*50)
    print("方法3: 基于IV值的特征选择 (基准)")
    print("-"*50)
    
    try:
        # 基于IV值选择特征
        iv_threshold = 0.02  # IV值阈值
        selected_by_iv = []
        
        for feature in selected_features:
            if feature in feature_iv_values and feature_iv_values[feature] >= iv_threshold:
                woe_column = f'{feature}_WOE'
                if woe_column in woe_feature_data:
                    selected_by_iv.append(woe_column)
        
        print(f"IV值阈值 ({iv_threshold}) 选择的特征数量: {len(selected_by_iv)}")
        print("选中的特征:", [feature_woe_mapping[col] for col in selected_by_iv])
        
        # 显示各特征IV值
        print("\n各特征IV值:")
        iv_results = []
        for feature in selected_features:
            if feature in feature_iv_values:
                iv_val = feature_iv_values[feature]
                status = "选中" if iv_val >= iv_threshold else "未选中"
                iv_results.append({
                    '特征': feature,
                    'IV值': iv_val,
                    '状态': status
                })
                print(f"  {feature}: {iv_val:.4f} ({status})")
        
        # 使用IV选择的特征进行交叉验证
        if len(selected_by_iv) > 0:
            X_iv = X[selected_by_iv]
            X_iv_scaled = scaler.fit_transform(X_iv)
            
            # 逻辑回归模型
            lr_iv = LogisticRegression(random_state=42, max_iter=1000)
            iv_scores = cross_val_score(lr_iv, X_iv_scaled, y, cv=cv, scoring='roc_auc')
            
            # 存储IV结果
            validation_results['IV值选择'] = {
                'auc_mean': iv_scores.mean(),
                'auc_std': iv_scores.std(),
                'selected_features': [feature_woe_mapping[col] for col in selected_by_iv],
                'n_features': len(selected_by_iv),
                'iv_values': {item['特征']: item['IV值'] for item in iv_results}
            }
            
            print(f"\nIV特征选择后的模型性能:")
            print(f"  AUC均值: {iv_scores.mean():.4f} (+/- {iv_scores.std() * 2:.4f})")
            print(f"  各折AUC: {[f'{score:.4f}' for score in iv_scores]}")
    
    except Exception as e:
        print(f"IV特征选择失败: {e}")
    
    # ============================================================================
    # 方法4: 所有特征的基准模型
    # ============================================================================
    
    print("\n" + "-"*50)
    print("方法4: 所有特征的基准模型")
    print("-"*50)
    
    try:
        # 使用所有特征
        lr_baseline = LogisticRegression(random_state=42, max_iter=1000)
        baseline_scores = cross_val_score(lr_baseline, X_scaled, y, cv=cv, scoring='roc_auc')
        
        # 在整个数据集上训练以获得系数
        lr_baseline.fit(X_scaled, y)
        
        # 获取特征系数
        baseline_coef_df = pd.DataFrame({
            '特征': [feature_woe_mapping[col] for col in woe_feature_data],
            'WOE列名': woe_feature_data,
            '基准系数': lr_baseline.coef_[0],
            '系数绝对值': np.abs(lr_baseline.coef_[0])
        }).sort_values('系数绝对值', ascending=False)
        
        # 存储基准结果
        validation_results['基准模型(所有特征)'] = {
            'auc_mean': baseline_scores.mean(),
            'auc_std': baseline_scores.std(),
            'selected_features': [feature_woe_mapping[col] for col in woe_feature_data],
            'n_features': len(woe_feature_data),
            'coefficients': baseline_coef_df.set_index('特征')['基准系数'].to_dict()
        }
        
        print(f"基准模型(所有特征)性能:")
        print(f"  AUC均值: {baseline_scores.mean():.4f} (+/- {baseline_scores.std() * 2:.4f})")
        print(f"  各折AUC: {[f'{score:.4f}' for score in baseline_scores]}")
        
        print("\n基准模型特征系数排序:")
        for idx, row in baseline_coef_df.head(10).iterrows():
            print(f"  {row['特征']}: {row['基准系数']:.4f}")
    
    except Exception as e:
        print(f"基准模型验证失败: {e}")
    
    # ============================================================================
    # 结果汇总和对比分析
    # ============================================================================
    
    print("\n" + "="*50)
    print("多维度交叉验证结果汇总")
    print("="*50)
    
    # 创建结果汇总表
    summary_data = []
    
    for method, results in validation_results.items():
        summary_data.append({
            '方法': method,
            'AUC均值': results['auc_mean'],
            'AUC标准差': results['auc_std'],
            '特征数量': results['n_features'],
            '选中的特征': ', '.join(results['selected_features'])
        })
    
    summary_df = pd.DataFrame(summary_data).sort_values('AUC均值', ascending=False)
    
    print("\n各方法性能比较:")
    print(summary_df.to_string(index=False))
    
    # 创建特征选择一致性分析
    print("\n" + "-"*50)
    print("特征选择一致性分析")
    print("-"*50)
    
    # 收集所有特征的选择情况
    feature_selection_analysis = []
    
    for feature in selected_features:
        if feature in feature_iv_values:
            selection_info = {
                '特征': feature,
                'IV值': feature_iv_values[feature]
            }
            
            # 检查每个方法是否选择了该特征
            for method in ['L1正则化', 'RFE', 'IV值选择', '基准模型(所有特征)']:
                if method in validation_results:
                    is_selected = feature in validation_results[method]['selected_features']
                    selection_info[f'{method}_选中'] = '是' if is_selected else '否'
                    
                    # 添加系数或排名信息
                    if method in ['L1正则化', '基准模型(所有特征)'] and 'coefficients' in validation_results[method]:
                        selection_info[f'{method}_系数'] = validation_results[method]['coefficients'].get(feature, 0)
                    elif method == 'RFE' and 'ranking' in validation_results[method]:
                        selection_info[f'{method}_排名'] = validation_results[method]['ranking'].get(feature, 999)
            
            feature_selection_analysis.append(selection_info)
    
    feature_selection_df = pd.DataFrame(feature_selection_analysis)
    
    # 计算被选中的次数
    selection_columns = [col for col in feature_selection_df.columns if col.endswith('_选中')]
    feature_selection_df['总选中次数'] = feature_selection_df[selection_columns].apply(
        lambda x: (x == '是').sum(), axis=1
    )
    
    # 按IV值排序
    feature_selection_df = feature_selection_df.sort_values(['总选中次数', 'IV值'], ascending=[False, False])
    
    print("\n特征选择一致性详情:")
    print(feature_selection_df.to_string(index=False))
    
    # ============================================================================
    # 保存交叉验证结果到Excel
    # ============================================================================
    
    print("\n" + "="*50)
    print("保存交叉验证结果到Excel")
    print("="*50)
    
    cross_validation_filename = '评分卡模型_多维度交叉验证结果.xlsx'
    
    with pd.ExcelWriter(cross_validation_filename, engine='openpyxl') as writer:
        # 1. 方法性能汇总
        summary_df.to_excel(writer, sheet_name='方法性能汇总', index=False)
        
        # 2. 特征选择一致性分析
        feature_selection_df.to_excel(writer, sheet_name='特征选择一致性', index=False)
        
        # 3. L1正则化详细结果
        if 'L1正则化' in validation_results:
            l1_details = []
            for feature in selected_features:
                if feature in feature_iv_values:
                    l1_details.append({
                        '特征': feature,
                        'IV值': feature_iv_values[feature],
                        'L1系数': validation_results['L1正则化']['coefficients'].get(feature, 0),
                        'L1系数绝对值': abs(validation_results['L1正则化']['coefficients'].get(feature, 0)),
                        '是否被L1选择': '是' if feature in validation_results['L1正则化']['selected_features'] else '否'
                    })
            l1_details_df = pd.DataFrame(l1_details).sort_values('L1系数绝对值', ascending=False)
            l1_details_df.to_excel(writer, sheet_name='L1正则化详情', index=False)
        
        # 4. RFE详细结果
        if 'RFE' in validation_results:
            rfe_details = []
            for feature in selected_features:
                if feature in feature_iv_values:
                    rfe_details.append({
                        '特征': feature,
                        'IV值': feature_iv_values[feature],
                        'RFE排名': validation_results['RFE']['ranking'].get(feature, 999),
                        '是否被RFE选择': '是' if feature in validation_results['RFE']['selected_features'] else '否'
                    })
            rfe_details_df = pd.DataFrame(rfe_details).sort_values('RFE排名')
            rfe_details_df.to_excel(writer, sheet_name='RFE详情', index=False)
        
        # 5. 基准模型系数
        if '基准模型(所有特征)' in validation_results:
            baseline_details = []
            for feature in selected_features:
                if feature in feature_iv_values:
                    baseline_details.append({
                        '特征': feature,
                        'IV值': feature_iv_values[feature],
                        '基准系数': validation_results['基准模型(所有特征)']['coefficients'].get(feature, 0),
                        '基准系数绝对值': abs(validation_results['基准模型(所有特征)']['coefficients'].get(feature, 0))
                    })
            baseline_details_df = pd.DataFrame(baseline_details).sort_values('基准系数绝对值', ascending=False)
            baseline_details_df.to_excel(writer, sheet_name='基准模型系数', index=False)
        
        # 6. 交叉验证详细分数
        cv_scores_data = []
        for method, results in validation_results.items():
            cv_scores_data.append({
                '方法': method,
                '折1_AUC': results.get('cv_scores', [0, 0, 0, 0, 0])[0] if 'cv_scores' in results else results['auc_mean'],
                '折2_AUC': results.get('cv_scores', [0, 0, 0, 0, 0])[1] if 'cv_scores' in results else results['auc_mean'],
                '折3_AUC': results.get('cv_scores', [0, 0, 0, 0, 0])[2] if 'cv_scores' in results else results['auc_mean'],
                '折4_AUC': results.get('cv_scores', [0, 0, 0, 0, 0])[3] if 'cv_scores' in results else results['auc_mean'],
                '折5_AUC': results.get('cv_scores', [0, 0, 0, 0, 0])[4] if 'cv_scores' in results else results['auc_mean'],
                'AUC均值': results['auc_mean'],
                'AUC标准差': results['auc_std']
            })
        cv_scores_df = pd.DataFrame(cv_scores_data)
        cv_scores_df.to_excel(writer, sheet_name='交叉验证分数', index=False)
    
    print(f"\n多维度交叉验证结果已保存到: {cross_validation_filename}")
    
    # ============================================================================
    # 关键发现和建议
    # ============================================================================
    
    print("\n" + "="*50)
    print("关键发现和建议")
    print("="*50)
    
    # 找出表现最好的方法
    best_method = summary_df.iloc[0]
    print(f"推荐方法: {best_method['方法']}")
    print(f"  - AUC: {best_method['AUC均值']:.4f}")
    print(f"  - 特征数量: {best_method['特征数量']}")
    print(f"  - 稳定性: {best_method['AUC标准差']:.4f} (标准差)")
    
    # 找出被多个方法共同选择的重要特征
    consistently_selected_features = feature_selection_df[feature_selection_df['总选中次数'] >= 2]
    if len(consistently_selected_features) > 0:
        print(f"\n被多个方法共同选择的重要特征 ({len(consistently_selected_features)}个):")
        for idx, row in consistently_selected_features.iterrows():
            print(f"  - {row['特征']} (IV值: {row['IV值']:.4f}, 被{row['总选中次数']}个方法选中)")
    
    # 找出IV值高但未被选择的重要特征
    high_iv_not_selected = feature_selection_df[
        (feature_selection_df['IV值'] >= 0.1) & 
        (feature_selection_df['总选中次数'] == 0)
    ]
    if len(high_iv_not_selected) > 0:
        print(f"\n高IV值但未被选择的特征 (建议进一步分析):")
        for idx, row in high_iv_not_selected.iterrows():
            print(f"  - {row['特征']} (IV值: {row['IV值']:.4f})")
    
    print(f"\n详细的分析结果请查看: {cross_validation_filename}")

print("\n多维度交叉验证完成！")

