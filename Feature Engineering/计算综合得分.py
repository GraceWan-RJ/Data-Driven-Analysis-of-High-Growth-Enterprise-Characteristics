import pandas as pd

# === 1. 读取数据（统一编码） ===
capital_df = pd.read_csv("资本活跃度指标结果.CSV", encoding="utf-8-sig")
org_df = pd.read_csv("组织成熟度指标结果.CSV", encoding="utf-8-sig")
industry_df = pd.read_csv("行业定位指标结果.CSV", encoding="utf-8-sig")
legal_df = pd.read_csv("法律风险指标结果.CSV", encoding="utf-8-sig")
base_df = pd.read_csv("企业基本信息.CSV", encoding="utf-8-sig")

# === 2. 清理列名（去除空格和隐藏字符） ===
for df in [capital_df, org_df, industry_df, legal_df, base_df]:
    df.columns = df.columns.str.strip().str.replace('﻿', '', regex=False)

# === 3. 统一主键字段名 ===
base_df.rename(columns={"#eid(M|2147483647)": "eid"}, inplace=True)

# === 4. 提取关键字段 ===
capital_df = capital_df[["eid", "capital_score"]]
org_df = org_df[["eid", "organization_score"]]
legal_df = legal_df[["eid", "legal_risk_score"]]
industry_df = industry_df[["eid", "industry_score"]]
base_df = base_df[["eid", "name(M|2147483647)"]]

# === 5. 合并所有表 ===
merged_df = (
    base_df.merge(capital_df, on="eid", how="left")
           .merge(org_df, on="eid", how="left")
           .merge(legal_df, on="eid", how="left")
           .merge(industry_df, on="eid", how="left")
)

# === 6. 缺失值补0 ===
merged_df.fillna({
    "capital_score": 0,
    "organization_score": 0,
    "legal_risk_score": 0,
    "industry_score": 0
}, inplace=True)

# === 7. 权重设置 ===
w_capital, w_org, w_legal, w_industry = 0.25, 0.30, 0.25, 0.20

# === 8. 计算综合增长得分 ===
merged_df["growth_score"] = round((
    w_capital * merged_df["capital_score"] +
    w_org * merged_df["organization_score"] +
    w_legal * merged_df["legal_risk_score"] +
    w_industry * merged_df["industry_score"]
),2)

# === 9. 排序输出 ===
top10 = merged_df.sort_values("growth_score", ascending=False).head(10)
print(top10[["eid", "name(M|2147483647)", "growth_score"]])

# === 10. 导出结果 ===
merged_df.to_csv("企业增长潜力综合评分.csv", index=False, encoding="utf-8-sig")
print("已生成文件：企业增长潜力综合评分.csv")
