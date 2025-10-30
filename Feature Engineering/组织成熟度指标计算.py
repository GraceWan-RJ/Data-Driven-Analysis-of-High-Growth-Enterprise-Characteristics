import pandas as pd
import numpy as np


basicinfo_path = "企业基本信息.CSV"

df_b = pd.read_csv(basicinfo_path, sep=',') 

# === 关键列 ===
eid_col = "#eid(M|2147483647)"  
econ_kind_col = "econ_kind(M|2147483647)"
econ_kind_code_col = "econ_kind_code(M|2147483647)"
regist_capi_col = "regist_capi_new(M|2147483647)"   # 注册资本
actual_capi_col = "actual_capi(M|2147483647)"       # 实收资金
scope_col = "scope(M|2147483647)"

# === 工具函数 ===
def minmax(series):
    """
    对序列进行最小-最大归一化（将数据缩放到0-1之间）
    特殊处理：
    - 若序列全为空值，返回全0序列
    - 若序列为常数列（最大值=最小值），返回全0序列
    """
    if series.isna().all():
        return pd.Series(0, index=series.index)
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(0, index=series.index)
    return (s - mn) / (mx - mn)

def count_scope_keywords(x):
    """
    统计企业经营范围中的关键词数量
    逻辑：以中文分号“；”为分隔符，分割经营范围文本，统计非空且长度>1的片段数量
    """
    if pd.isna(x): 
        return 0
    x = str(x).strip()
    parts = [w.strip() for w in x.split("；") if len(w.strip()) > 1]
    return len(parts)

# === (1) 组织形式等级 ===
org_mapping = {
    "1212": 10, "1219": 10, "1210": 10, "5220": 10, "6240": 10,
    "1213": 10, "6220": 10, "1200": 10, "1222": 5,
    "3200": 3, "1100": 1
}

def map_org_level(row):
    """
    根据经济类型代码和描述映射组织形式等级
    优先级：先匹配代码，代码不匹配则根据描述关键词判断
    等级划分：上市/新三板企业(10) > 股份有限公司(5) > 有限责任公司(3) > 个体(1) > 其他(2)
    """
    code = str(row[econ_kind_code_col]).strip() if pd.notna(row[econ_kind_code_col]) else ""
    desc = str(row[econ_kind_col]) if pd.notna(row[econ_kind_col]) else ""
    if code in org_mapping:
        return org_mapping[code]
    if "个体" in desc:
        return 1
    elif "有限责任" in desc:
        return 3
    elif "股份有限" in desc and "上市" not in desc:
        return 5
    elif "上市" in desc or "新三板" in desc:
        return 10
    else:
        return 2

df_b["组织形式等级"] = df_b.apply(map_org_level, axis=1)

# === (2) 注册资本等级 ===
# 对注册资本取对数（log1p = log(1+x)，避免0值导致log无意义）
df_b["注册资本等级"] = np.log1p(pd.to_numeric(df_b[regist_capi_col], errors="coerce").fillna(0))

# === (3) 实缴资本率（实收资金 / 注册资本）===
df_b["实缴资本率"] = np.where(
    # 避免注册资本为0导致除零错误：仅当注册资本>0时计算比例，否则为0
    pd.to_numeric(df_b[regist_capi_col], errors="coerce") > 0, 
    pd.to_numeric(df_b[actual_capi_col], errors="coerce") / pd.to_numeric(df_b[regist_capi_col], errors="coerce"),
    0
)
# 处理计算中可能出现的无穷大或空值，统一转为0
df_b["实缴资本率"] = df_b["实缴资本率"].replace([np.inf, -np.inf], 0).fillna(0)

# === (4) 经营范围广度 ===
df_b["经营范围广度"] = df_b[scope_col].apply(count_scope_keywords)

# === (5) 指标归一化 ===
df_b["组织形式_norm"] = minmax(df_b["组织形式等级"])
df_b["注册资本_norm"] = minmax(df_b["注册资本等级"])
df_b["实缴资本率_norm"] = minmax(df_b["实缴资本率"])
df_b["经营范围_norm"] = minmax(df_b["经营范围广度"])

# === (6) 加权综合得分 ===
# 权重分配：组织形式(30%)、注册资本(25%)、实缴资本率(25%)、经营范围(20%)
# 乘以100将得分缩放至0-100区间，保留2位小数
df_b["organization_score"] = (
    0.3 * df_b["组织形式_norm"] +
    0.25 * df_b["注册资本_norm"] +
    0.25 * df_b["实缴资本率_norm"] +
    0.2 * df_b["经营范围_norm"]
) * 100
df_b["organization_score"] = df_b["organization_score"].round(2)

# === (7) 输出结果 ===
output_path = "组织成熟度指标结果.CSV"
# 关键修改：将eid_col重命名为"eid"
result = df_b[[eid_col, "组织形式等级", "注册资本等级", "实缴资本率", "经营范围广度", "organization_score"]]
result = result.rename(columns={eid_col: "eid"})  # 重命名列名

result.to_csv(output_path, index=False, encoding='utf-8-sig')  # 使用utf-8-sig编码避免中文乱码

print("已生成：组织成熟度指标结果.CSV")
print(result.head(10))

