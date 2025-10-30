import pandas as pd
df = pd.read_csv('企业增长潜力综合评分.csv')  # 请替换为你的CSV文件路径

# 计算growth_score列的70%分位数（即前30%的阈值）
threshold = df['growth_score'].quantile(0.7)

# 创建新列：前30%为1，其余为0
df['is_potential_enterprise'] = (df['growth_score'] >= threshold).astype(int)

# 保存结果到新的CSV文件
df.to_csv('result.csv', index=False)

print("处理完成，已添加'is_potential_enterprise'列并保存到result.csv")