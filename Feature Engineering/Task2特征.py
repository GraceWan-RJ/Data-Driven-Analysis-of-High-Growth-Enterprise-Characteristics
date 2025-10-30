import pandas as pd
import numpy as np
import re
import os
from datetime import datetime

# ====================== 企业标签统计相关 ======================
class EnterpriseTagCounter:
    def __init__(self):
        # 新兴产业关键词模式
        self.emerging_industry_patterns = [
            r'智能|AI|人工智能', r'大数?据|数据', r'云|云计算', r'物联', r'芯片|集成电路|半导体',
            r'生物|基因|医药', r'新能?源|光伏|风电|太阳能', r'机器?人|自动化', r'VR|AR|虚拟|增强',
            r'区块?链|数字.?货币', r'5G|通信', r'无人.?驾驶|自动.?驾驶', r'量子', r'元宇?宙',
            r'工业.?互联网', r'金融.?科技|FinTech', r'数字.?经济', r'新材料', r'节能.?环保|绿色'
        ]
        # 高科技标签类型
        self.high_tech_types = ['电子信息类认证', '技术领先', '科技奖项', '科技认定']
    
    def is_emerging_industry(self, tag_name):
        """判断是否属于新兴产业"""
        if pd.isna(tag_name):
            return False
        tag_str = str(tag_name)
        for pattern in self.emerging_industry_patterns:
            if re.search(pattern, tag_str, re.IGNORECASE):
                return True
        return False
    
    def identify_emerging_industries(self, df):
        """识别所有新兴产业标签"""
        business_tags = df[df['type_name(M|2147483647)'] == '业务概念']['tag_name(M|2147483647)'].unique()
        return [tag for tag in business_tags if self.is_emerging_industry(tag)]
    
    def count_tags(self, df):
        """统计各企业的标签数量"""
        emerging_industries = self.identify_emerging_industries(df)
        results = []
        
        for company_id, company_data in df.groupby('eid(M|2147483647)'):
            # 1. 高科技标签数
            high_tech_count = 0
            for category in self.high_tech_types:
                high_tech_count += len(company_data[company_data['type_name(M|2147483647)'] == category])
            
            # 2. 新兴产业标签数
            business_tags = company_data[company_data['type_name(M|2147483647)'] == '业务概念']['tag_name(M|2147483647)']
            emerging_count = business_tags.isin(emerging_industries).sum()
            
            # 3. 政府支持标签数
            government_support_count = len(company_data[company_data['type_name(M|2147483647)'] == '政府扶持和奖励'])
            
            # 4. 业务扩张标签数
            business_expansion_count = len(company_data[company_data['type_name(M|2147483647)'] == '业务扩张'])
            
            # 5. 人员扩张标签数
            personnel_expansion_count = len(company_data[company_data['type_name(M|2147483647)'] == '人员扩张'])
            
            results.append({
                'eid': company_id,
                '高科技标签数': high_tech_count,
                '新兴产业标签数': emerging_count,
                '政府支持标签数': government_support_count,
                '业务扩张标签数': business_expansion_count,
                '人员扩张标签数': personnel_expansion_count
            })
        
        return pd.DataFrame(results)


# ====================== 法律风险数据提取相关 ======================
class LegalRiskDataExtractor:
    def __init__(self):
        self.current_year = datetime.now().year

    def batch_extract_enterprise_data(self, enterprise_list, filing_df, hearing_df, judgment_df):
        """批量提取企业核心数据：eid、诉讼次数、近一年案件数、被告次数占比"""
        current_year = self.current_year
        
        # 数据预处理
        for df in [filing_df, hearing_df, judgment_df]:
            if not df.empty:
                # 填充角色字段
                role_fields = ['role(M|2147483647)', 'clean_role(M|2147483647)', 'pure_role(M|2147483647)']
                for field in role_fields:
                    if field in df.columns:
                        df[field] = df[field].fillna('(空白)')
                # 填充时间字段
                time_fields = ['year_date(M|2147483647)', 'pub_date(M|2147483647)', 'start_date(M|2147483647)', 
                              'hearing_date(M|2147483647)', 'date(M|2147483647)']
                for field in time_fields:
                    if field in df.columns:
                        df[field] = df[field].fillna(str(current_year))
        
        results = []
        for eid in enterprise_list:
            try:
                # 筛选该企业在三个表中的所有案件
                filing_cases = filing_df[filing_df['eid(M|2147483647)'] == eid] if not filing_df.empty else pd.DataFrame()
                hearing_cases = hearing_df[hearing_df['eid(M|2147483647)'] == eid] if not hearing_df.empty else pd.DataFrame()
                judgment_cases = judgment_df[judgment_df['eid(M|2147483647)'] == eid] if not judgment_df.empty else pd.DataFrame()
                
                # 合并所有案件数据
                all_cases = pd.concat([filing_cases, hearing_cases, judgment_cases], ignore_index=True)
                total_cases = len(all_cases)  # 诉讼次数
                
                # 统计被告次数
                defendant_cases = 0
                if total_cases > 0:
                    defendant_roles = ['被告', '被上诉人', '被申请人']
                    role_fields = ['role(M|2147483647)', 'clean_role(M|2147483647)', 'pure_role(M|2147483647)']
                    for field in role_fields:
                        if field in all_cases.columns:
                            for role in defendant_roles:
                                defendant_cases += len(all_cases[all_cases[field] == role])
                
                # 统计近一年案件数
                recent_cases = 0
                if total_cases > 0:
                    time_fields = ['year_date(M|2147483647)', 'pub_date(M|2147483647)', 'start_date(M|2147483647)', 
                                  'hearing_date(M|2147483647)', 'date(M|2147483647)']
                    for field in time_fields:
                        if field in all_cases.columns:
                            # 筛选包含当前年或上一年的案件
                            mask = (all_cases[field].astype(str).str.contains(str(current_year), na=False) | 
                                   all_cases[field].astype(str).str.contains(str(current_year - 1), na=False))
                            recent_cases = max(recent_cases, len(all_cases[mask]))
                    # 无时间字段时，按总案件数30%估算
                    if recent_cases == 0:
                        recent_cases = int(total_cases * 0.3)
                
                # 计算被告次数占比（保留2位小数）
                defendant_ratio = round((defendant_cases / total_cases * 100) if total_cases > 0 else 0, 2)
                
                results.append({
                    'eid': eid,
                    '诉讼次数': total_cases,
                    '近一年案件数': recent_cases,
                    '被告次数占比(%)': defendant_ratio
                })
            except Exception as e:
                print(f"提取企业 {eid} 数据时出错: {e}")
                results.append({
                    'eid': eid,
                    '诉讼次数': 0,
                    '近一年案件数': 0,
                    '被告次数占比(%)': 0.0
                })
        
        return pd.DataFrame(results)

    def load_legal_data(self):
        """检查文件存在性并读取法律相关数据"""
        file_paths = {
            'filing': '立案信息关系表.CSV',
            'hearing': '开庭公告关系表.CSV',
            'judgment': '裁判文书新关系表.CSV'
        }
        
        # 检查所有文件是否存在
        missing_files = [path for path in file_paths.values() if not os.path.exists(path)]
        if missing_files:
            for file in missing_files:
                print(f"错误: 文件 {file} 不存在")
            return None, None, None
        
        # 读取CSV文件
        try:
            filing_df = pd.read_csv(file_paths['filing'])
            hearing_df = pd.read_csv(file_paths['hearing'])
            judgment_df = pd.read_csv(file_paths['judgment'])
            print(f"成功加载法律数据:")
            print(f"  立案信息: {len(filing_df)} 条")
            print(f"  开庭公告: {len(hearing_df)} 条")
            print(f"  裁判文书: {len(judgment_df)} 条")
            return filing_df, hearing_df, judgment_df
        except Exception as e:
            print(f"读取法律数据文件出错: {e}")
            return None, None, None

    def check_legal_columns(self, filing_df, hearing_df, judgment_df):
        """检查法律数据必需列"""
        required_col = 'eid(M|2147483647)'
        missing = []
        if not filing_df.empty and required_col not in filing_df.columns:
            missing.append(f"立案信息表缺少 {required_col}")
        if not hearing_df.empty and required_col not in hearing_df.columns:
            missing.append(f"开庭公告表缺少 {required_col}")
        if not judgment_df.empty and required_col not in judgment_df.columns:
            missing.append(f"裁判文书表缺少 {required_col}")
        
        if missing:
            for msg in missing:
                print(msg)
            return False
        return True


# ====================== 特征工程相关 ======================
class FeatureEngineer:
    def __init__(self):
        # 路径配置
        self.shareholder_path = "工商公示股东信息表.CSV"
        self.basicinfo_path = "企业基本信息.CSV"
        self.result_path = "result.csv"
        
        # 关键列定义
        self.eid_col_basic = "#eid(M|2147483647)"  # 企业基本信息表的eid列
        self.eid_col_share = "eid(M|2147483647)"   # 股东信息表的eid列
        self.start_hold_col_original = "start_date(M|2147483647)"   # 股东开始参股时间
        self.end_hold_col_original = "end_date(M|2147483647)"     # 股东退出时间
        self.estiblish_col_original = "start_date(M|2147483647)"   # 企业成立日期
        self.econ_kind_col = "econ_kind(M|2147483647)"              # 企业类型/组织形式列
        self.stock_type_col = "stock_type(M|2147483647)"
        self.real_capi_col = "total_real_capi_new(M|2147483647)"
        self.should_capi_col = "total_should_capi_new(M|2147483647)"
        self.regist_capi_col = "regist_capi_new(M|2147483647)"  # 注册资本列名
        self.stock_name_col = "stock_name(M|2147483647)"  # 股东名称列

    def parse_date(self, s):
        """解析多种格式的日期字符串为pandas datetime类型"""
        if pd.isna(s): return pd.NaT
        s = str(s).strip()
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y年%m月%d日"):
            try: return pd.to_datetime(s, format=fmt)
            except: pass
        return pd.to_datetime(s, errors="coerce")

    def score_investor_v2(self, stock_type):
        """根据股东类型计算投资方质量评分"""
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

    def safe_log(self, x):
        """安全对数变换，处理0和负值"""
        if pd.isna(x) or x <= 0:
            return 0
        return np.log(x)

    def categorize_econ_kind(self, econ_kind):
        """根据企业类型/组织形式进行分类"""
        if pd.isna(econ_kind) or econ_kind == "":
            return "其他"
        
        econ_kind_str = str(econ_kind)
        
        # 使用正则表达式进行分类
        if re.search(r'港澳台', econ_kind_str):
            return "港澳台"
        elif re.search(r'国有', econ_kind_str):
            return "国有"
        elif re.search(r'自然人', econ_kind_str):
            return "自然人"
        elif re.search(r'外商|外国|中外', econ_kind_str):
            return "外商"
        elif re.search(r'合资', econ_kind_str):
            return "合资"
        else:
            return "其他"

    def load_basic_data(self):
        """加载企业基本信息和股东信息"""
        try:
            df_s = pd.read_csv(self.shareholder_path)
            df_b = pd.read_csv(self.basicinfo_path)
            df_result = pd.read_csv(self.result_path)
            print(f"成功加载基础数据:")
            print(f"  股东信息: {len(df_s)} 条")
            print(f"  企业基本信息: {len(df_b)} 条")
            print(f"  结果数据: {len(df_result)} 条")
            return df_s, df_b, df_result
        except Exception as e:
            print(f"读取基础数据文件出错: {e}")
            return None, None, None

    def extract_features(self, df_s, df_b, df_result):
        """提取企业特征"""
        # 数据预处理
        # 预处理股东信息表的日期列
        for c in [self.start_hold_col_original, self.end_hold_col_original]:
            if c in df_s.columns:
                df_s[c] = df_s[c].apply(self.parse_date)

        for c in [self.real_capi_col, self.should_capi_col]:
            if c in df_s.columns:
                df_s[c] = pd.to_numeric(df_s[c], errors="coerce").fillna(0)

        # 预处理企业基本信息表的日期列
        if self.estiblish_col_original in df_b.columns:
            df_b[self.estiblish_col_original] = df_b[self.estiblish_col_original].apply(self.parse_date)

        # 合并数据
        df = df_s.merge(
            df_b[[self.eid_col_basic, self.estiblish_col_original, self.regist_capi_col, self.econ_kind_col]],
            left_on=self.eid_col_share,
            right_on=self.eid_col_basic,
            how="left", 
            suffixes=("_s", "_b")
        )

        # 调整列名
        start_hold_col = self.start_hold_col_original + "_s"   # 合并后股东开始参股时间列
        end_hold_col = self.end_hold_col_original            
        est_col = self.estiblish_col_original + "_b"           # 合并后企业成立日期列

        # (1) 融资频率
        today = pd.Timestamp(datetime.now().date())

        establish = (
            df[[self.eid_col_share, est_col, self.econ_kind_col]]
            .groupby(self.eid_col_share, as_index=False)
            .first()
        )
        establish["成立年限_年"] = ((today - establish[est_col]).dt.days / 365).fillna(0).clip(lower=0)

        # 添加组织形式分类
        establish["组织形式"] = establish[self.econ_kind_col].apply(self.categorize_econ_kind)

        # 计算股东变更次数 - 使用调整后的列名
        freq = establish[[self.eid_col_share, "成立年限_年", "组织形式"]].copy()

        start_counts = df.groupby(self.eid_col_share)[start_hold_col].apply(lambda s: s.notna().sum())
        end_counts = df.groupby(self.eid_col_share)[end_hold_col].apply(lambda s: s.notna().sum())

        # 确保索引对齐
        start_counts = start_counts.reindex(establish[self.eid_col_share]).fillna(0)
        end_counts = end_counts.reindex(establish[self.eid_col_share]).fillna(0)

        freq["股东变更次数"] = start_counts.values + end_counts.values

        # 计算融资频率，避免除零错误
        freq["融资频率_次每年"] = np.where(
            freq["成立年限_年"] > 0,
            freq["股东变更次数"] / freq["成立年限_年"],
            0
        )

        # (2) 实缴资本完成率
        cap_last = df.groupby(self.eid_col_share)[[self.real_capi_col, self.should_capi_col]].last().reindex(establish[self.eid_col_share]).fillna(0)
        cap = cap_last.reset_index()
        cap["实缴资本完成率_%"] = np.where(
            cap[self.should_capi_col] > 0,
            (cap[self.real_capi_col] / cap[self.should_capi_col]) * 100,
            0
        )

        # (3) 投资方质量评分
        df["_investor_score_"] = df[self.stock_type_col].apply(self.score_investor_v2) if self.stock_type_col in df.columns else 3
        score = df.groupby(self.eid_col_share)["_investor_score_"].mean().reindex(establish[self.eid_col_share]).fillna(0).reset_index(name="投资方质量评分")

        # (4) 股东数量统计
        if self.stock_name_col in df_s.columns:
            # 按企业eid分组，统计股东数量
            shareholder_count = df_s.groupby(self.eid_col_share)[self.stock_name_col].apply(
                lambda x: x.notna().sum() 
            ).reset_index(name="股东数量")
        else:
            print(f"警告：股东信息表中未找到股东名称列 '{self.stock_name_col}'")
            # 创建空的股东数量列
            shareholder_count = establish[[self.eid_col_share]].copy()
            shareholder_count["股东数量"] = 0

        # (5) 注册资本 - 只保留对数变换
        if self.regist_capi_col in df_b.columns:
            # 提取注册资本列并转换为数值型
            regist_capi = df_b[[self.eid_col_basic, self.regist_capi_col]].copy()
            regist_capi[self.regist_capi_col] = pd.to_numeric(regist_capi[self.regist_capi_col], errors="coerce").fillna(0)
            
            # 对注册资本进行对数变换，不保留原始值
            regist_capi["对数注册资本"] = regist_capi[self.regist_capi_col].apply(self.safe_log)
        else:
            print(f"警告：企业基本信息表中未找到注册资本列 '{self.regist_capi_col}'")
            # 创建空的注册资本对数变换列
            regist_capi = establish[[self.eid_col_share]].copy()
            regist_capi["对数注册资本"] = 0

        # 合并所有特征
        features = establish.merge(freq[[self.eid_col_share, "融资频率_次每年"]], on=self.eid_col_share, how="left")
        features = features.merge(cap[[self.eid_col_share, "实缴资本完成率_%"]], on=self.eid_col_share, how="left")
        features = features.merge(score, on=self.eid_col_share, how="left")
        features = features.merge(shareholder_count[[self.eid_col_share, "股东数量"]], on=self.eid_col_share, how="left")
        features = features.merge(regist_capi[[self.eid_col_basic, "对数注册资本"]], 
                                 left_on=self.eid_col_share, right_on=self.eid_col_basic, how="left")

        # 选择需要的列并标准化eid列名
        final_features = features[[self.eid_col_share, "成立年限_年", "组织形式", "融资频率_次每年", 
                                  "实缴资本完成率_%", "投资方质量评分", "股东数量", 
                                  "对数注册资本"]].rename(columns={self.eid_col_share: "eid"})

        # 与result.csv合并
        result_columns = ["eid", "name(M|2147483647)", "growth_score", "is_potential_enterprise"]
        if all(col in df_result.columns for col in result_columns):
            result_extracted = df_result[result_columns]
            final_result = result_extracted.merge(final_features, on="eid", how="left")
            
            # 将所有数值列保留三位小数
            numeric_columns = ["成立年限_年", "融资频率_次每年", "实缴资本完成率_%", 
                              "投资方质量评分", "股东数量", "对数注册资本"]
            
            for col in numeric_columns:
                if col in final_result.columns:
                    final_result[col] = final_result[col].round(3)
            
            return final_result
        else:
            missing_cols = [col for col in result_columns if col not in df_result.columns]
            print(f"错误：result.csv中缺少以下列: {missing_cols}")
            return None


# ====================== 主程序 ======================
def main():
    print("===== 开始企业多维度数据合并 =====")
    
    # 1. 加载并处理标签数据
    print("\n----- 处理标签数据 -----")
    try:
        tag_df = pd.read_csv('概念标签.CSV')
        tag_counter = EnterpriseTagCounter()
        tag_results = tag_counter.count_tags(tag_df)
        print(f"标签数据处理完成，包含 {len(tag_results)} 家企业")
    except Exception as e:
        print(f"处理标签数据出错: {e}")
        return

    # 2. 加载并处理法律风险数据
    print("\n----- 处理法律风险数据 -----")
    legal_extractor = LegalRiskDataExtractor()
    filing_df, hearing_df, judgment_df = legal_extractor.load_legal_data()
    if filing_df is None:
        print("法律数据加载失败，无法继续")
        return
    
    if not legal_extractor.check_legal_columns(filing_df, hearing_df, judgment_df):
        print("法律数据缺少必要列，无法继续")
        return
    
    # 获取所有企业ID（去重）
    all_eids = set()
    for df in [filing_df, hearing_df, judgment_df]:
        if not df.empty:
            all_eids.update(df['eid(M|2147483647)'].dropna().unique())
    # 补充标签数据中的企业ID
    all_eids.update(tag_results['eid'].dropna().unique())
    enterprise_list = list(all_eids)
    print(f"共发现 {len(enterprise_list)} 家企业")
    
    legal_results = legal_extractor.batch_extract_enterprise_data(enterprise_list, filing_df, hearing_df, judgment_df)
    print(f"法律风险数据处理完成，包含 {len(legal_results)} 家企业")

    # 3. 加载并处理特征工程数据
    print("\n----- 处理特征工程数据 -----")
    feature_engineer = FeatureEngineer()
    df_s, df_b, df_result = feature_engineer.load_basic_data()
    if df_s is None:
        print("基础数据加载失败，无法继续")
        return
    
    feature_results = feature_engineer.extract_features(df_s, df_b, df_result)
    if feature_results is None:
        print("特征工程处理失败，无法继续")
        return
    print(f"特征工程处理完成，包含 {len(feature_results)} 家企业")

    # 4. 合并所有结果
    print("\n----- 合并所有数据 -----")
    # 首先合并标签数据和法律数据
    combined = tag_results.merge(legal_results, on='eid', how='outer')
    # 再合并特征工程数据
    final_combined = combined.merge(feature_results, on='eid', how='outer')
    
    # 5. 保存最终结果
    output_file = '企业多维度数据.csv'
    final_combined.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n所有数据合并完成！共包含 {len(final_combined)} 家企业，{len(final_combined.columns)} 个特征")
    print(f"结果已保存至 {output_file}")
    print(f"包含的列: {list(final_combined.columns)}")
    
    # 打印前5行预览
    print("\n前5行数据预览:")
    print(final_combined.head())

if __name__ == "__main__":
    main()