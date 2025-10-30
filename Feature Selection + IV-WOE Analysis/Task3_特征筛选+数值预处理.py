import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. 读取数据
df = pd.read_excel('企业聚类结果.xlsx')

# 2. 需要保留的字段
feature_cols = [
    "融资频率_次每年",
    "实缴资本完成率_%",
    "投资方质量评分",
    "股东数量",
    "对数注册资本",
    "近一年案件数",
    "被告次数占比(%)",
    "高科技标签数",
    "政府支持标签数"
]

other_cols = ["eid","name(M|2147483647)" ,"is_potential_enterprise", "growth_score", "cluster"]
keep_cols = feature_cols + [col for col in other_cols if col in df.columns]

# 只保留这些字段
df = df[keep_cols]

# 3. 缺失值填充（使用中位数）
df[feature_cols] = df[feature_cols].apply(lambda x: x.fillna(x.median()))

# 4. 数值标准化（9个特征）
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[feature_cols] = scaler.fit_transform(df_scaled[feature_cols])

# 5. 输出结果为CSV
current_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir, "特征_建模前预处理.csv")
# 输出CSV到当前文件夹
df_scaled.to_csv(output_path, index=False, encoding="utf-8-sig")

print("特征预处理完成，结果已输出至：", output_path)
