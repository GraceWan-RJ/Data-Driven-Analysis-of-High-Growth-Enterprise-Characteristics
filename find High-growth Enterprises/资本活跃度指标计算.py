import pandas as pd
import numpy as np
from datetime import datetime

# === 路径 ===
# 读取CSV文件
shareholder_path = "工商公示股东信息表.CSV"
basicinfo_path = "企业基本信息.CSV"

# === 读表 ===
df_s = pd.read_csv(shareholder_path)
df_b = pd.read_csv(basicinfo_path)

# === 关键列 ===
eid_col_basic = "#eid(M|2147483647)"  # 企业基本信息表的eid列
eid_col_share = "eid(M|2147483647)"   # 股东信息表的eid列

start_hold_col = "start_date(M|2147483647)"   # 股东开始参股时间（股东表）
end_hold_col   = "end_date(M|2147483647)"     # 股东退出时间（股东表）
estiblish_col  = "start_date(M|2147483647)"   # 企业成立日期（企业基本信息表）
stock_type_col = "stock_type(M|2147483647)"
real_capi_col  = "total_real_capi_new(M|2147483647)"
should_capi_col= "total_should_capi_new(M|2147483647)"

# === 实用函数 ===
def parse_date(s):
    """
    解析多种格式的日期字符串为pandas datetime类型
    支持格式：%Y-%m-%d, %Y/%m/%d, %Y.%m.%d, %Y年%m月%d日
    """
    if pd.isna(s): return pd.NaT
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y年%m月%d日"):
        try: return pd.to_datetime(s, format=fmt)
        except: pass
    return pd.to_datetime(s, errors="coerce")

def score_investor_v2(stock_type):
    """
    根据股东类型计算投资方质量评分
    评分规则：
      - 外商/外国/港澳台 → 10分
      - 企业法人/法人股东/事业法人/社团法人/机构法人 → 8分
      - 合伙/合伙企业 → 7分
      - 自然人/公民 → 5分
      - 其他投资者/其他地区/独资 → 4分
      - 空白/未知 → 3分
    """
    if pd.isna(stock_type) or str(stock_type).strip() == "" or stock_type == "空白":
        return 3
    elif "外商" in stock_type or "外国" in stock_type or "港澳台" in stock_type:
        return 10
    elif any(k in stock_type for k in ["企业法人", "法人股东", "事业法人", "社团法人", "机构法人"]):
        return 8
    elif any(k in stock_type for k in ["合伙", "合伙企业"]):
        return 7
    elif any(k in stock_type for k in ["自然人", "公民"]):
        return 5
    elif any(k in stock_type for k in ["其他投资者", "其他地区", "独资"]):
        return 4
    else:
        return 3

def minmax(series):
    """
    对序列进行min-max归一化（缩放至0-1范围）
    """
    if series.isna().all(): return pd.Series(0, index=series.index)
    s = series.fillna(0)
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(0, index=series.index)
    return (s - mn) / (mx - mn)

# === 数据预处理 ===
# 处理股东表中的日期列（开始参股时间、退出时间）
for c in [start_hold_col, end_hold_col]:
    if c in df_s.columns:
        df_s[c] = df_s[c].apply(parse_date)

# 处理股东表中的资本列
for c in [real_capi_col, should_capi_col]:
    if c in df_s.columns:
        df_s[c] = pd.to_numeric(df_s[c], errors="coerce").fillna(0)

# 处理企业表中的成立日期列
df_b[estiblish_col] = df_b[estiblish_col].apply(parse_date)

# === 合并 （两表都有 start_date → 用后缀避免冲突）===
df = df_s.merge(
    df_b[[eid_col_basic, estiblish_col]],
    left_on=eid_col_share,
    right_on=eid_col_basic,
    how="left", 
    suffixes=("_s", "_b")
)
est_col = estiblish_col + "_b"
start_hold_col = start_hold_col + "_s"

# === (1) 融资频率：变动次数 / 成立年限 ===
# 成立年限（年）
today = pd.Timestamp(datetime.now().date())

establish = (
df[[eid_col_share, est_col]]
    .groupby(eid_col_share, as_index=False)
    .first()
)
establish["成立年限_年"] = ((today - establish[est_col]).dt.days / 365).fillna(0)
establish["成立年限_年"] = establish["成立年限_年"].clip(lower=0)

# 进入/退出计数
entry_counts = df.groupby(eid_col_share)[start_hold_col].apply(lambda s: s.notna().sum()).reindex(establish[eid_col_share]).fillna(0)
exit_counts  = df.groupby(eid_col_share)[end_hold_col  ].apply(lambda s: s.notna().sum()).reindex(establish[eid_col_share]).fillna(0)

#计算融资频率
freq = establish[[eid_col_share, "成立年限_年"]].copy()
freq["股东变更次数"]  = entry_counts.values + exit_counts.values
freq["融资频率_次每年"] = np.where(freq["成立年限_年"]>0,
                           freq["股东变更次数"] / freq["成立年限_年"], 0)

# === (2) 实缴资本完成率：实缴/应缴 ===
cap_last = df.groupby(eid_col_share)[[real_capi_col, should_capi_col]].last().reindex(establish[eid_col_share]).fillna(0)
cap = cap_last.reset_index()
# 计算完成率
cap["实缴资本完成率_%"] = np.where(cap[should_capi_col] > 0,
                            (cap[real_capi_col] / cap[should_capi_col]) * 100, 0)

# === (3) 投资方质量评分 ===
# 按企业分组计算平均评分
df["_investor_score_"] = df[stock_type_col].apply(score_investor_v2) if stock_type_col in df.columns else 3
score = df.groupby(eid_col_share)["_investor_score_"].mean().reindex(establish[eid_col_share]).fillna(0).reset_index(name="投资方质量评分")

# === 合并结果 ===
result = (
    establish[[eid_col_share]]
    .merge(freq[[eid_col_share, "融资频率_次每年"]], on=eid_col_share, how="left")
    .merge(cap[[eid_col_share, "实缴资本完成率_%"]], on=eid_col_share, how="left")
    .merge(score, on=eid_col_share, how="left")
).fillna(0)

# === (4) 综合资本活跃度得分 ===
# 对各指标进行归一化（0-1范围）
result["融资频率_norm"]   = minmax(result["融资频率_次每年"])
result["实缴资本_norm"]   = minmax(result["实缴资本完成率_%"])
result["投资方质量_norm"] = minmax(result["投资方质量评分"])

result["capital_score"] = (
    0.4 * result["融资频率_norm"] +
    0.3 * result["实缴资本_norm"] +
    0.3 * result["投资方质量_norm"]
) * 100
result["capital_score"] = result["capital_score"].round(2)

# 清理中间列
result = result.drop(columns=["融资频率_norm", "实缴资本_norm", "投资方质量_norm"], errors="ignore")
result = result.rename(columns={eid_col_share: "eid"})

# === 导出为CSV文件 ===
result.to_csv("资本活跃度指标结果.CSV", index=False, encoding="utf-8-sig")
print("已生成：资本活跃度指标结果.CSV")
print(result.head(10))
    
