import pandas as pd
import numpy as np

# 读取各表数据
legal_risk = pd.read_csv('法律风险指标结果.CSV')
industry = pd.read_csv('行业定位指标结果.CSV')
capital = pd.read_csv('资本活跃度指标结果.CSV')
organization = pd.read_csv('组织成熟度指标结果.CSV')  # 规模扩张对应组织成熟度

# 定义max-min归一化函数
def normalize_series(series):
    """对整个系列进行归一化处理"""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:  # 处理所有值相同的情况，避免除零错误
        return 0.5  # 全部返回0.5
    else:
        return (series - min_val) / (max_val - min_val)

# 对各表进行归一化处理
if 'legal_risk_score' in legal_risk.columns:
    # 直接对整个列应用归一化函数
    legal_risk['normalized_legal_risk'] = normalize_series(legal_risk['legal_risk_score'])
else:
    print("法律风险表中未找到得分列")

if 'industry_score' in industry.columns:
    industry['normalized_industry'] = normalize_series(industry['industry_score'])
else:
    print("行业定位表中未找到得分列")

if 'capital_score' in capital.columns:
    capital['normalized_capital'] = normalize_series(capital['capital_score'])
else:
    print("资本活跃度表中未找到得分列")

if 'organization_score' in organization.columns:
    organization['normalized_organization'] = normalize_series(organization['organization_score'])
else:
    print("组织成熟度表中未找到得分列")

# 合并四张表（使用outer join保留所有eid）
merged_data = legal_risk[['eid', 'normalized_legal_risk']].merge(
    industry[['eid', 'normalized_industry']], on='eid', how='outer'
).merge(
    capital[['eid', 'normalized_capital']], on='eid', how='outer'
).merge(
    organization[['eid', 'normalized_organization']], on='eid', how='outer'
)

# 将缺失值替换为0
merged_data = merged_data.fillna(0)

# 计算综合得分
merged_data['comprehensive_score'] = (
    merged_data['normalized_capital'] * 0.25 +       # 资本活跃度×0.25
    merged_data['normalized_organization'] * 0.2 +   # 规模扩张×0.2
    merged_data['normalized_industry'] * 0.35 +      # 行业定位×0.35
    merged_data['normalized_legal_risk'] * 0.2       # 法律风险×0.2
)

# 保存结果
merged_data.to_csv('企业综合得分结果.CSV', index=False)

# 查看最终结果
print("\n综合得分计算结果前5行:\n", merged_data[['eid', 'comprehensive_score']].head())
print("\n数据处理完成，结果已保存至'企业综合得分结果.CSV'")
