import pandas as pd
import numpy as np
from datetime import datetime
import re
import os

class LegalRiskAssessment:
    def __init__(self, data_years=3):
        self.data_years = data_years
        self.current_year = datetime.now().year
        
    def calculate_case_frequency(self, cases_df):
        """计算诉讼频率指标"""
        total_cases = len(cases_df)
        annual_cases = total_cases / self.data_years if self.data_years > 0 else total_cases
        
        # 被告比例 - 检查列是否存在
        defendant_cases = 0
        role_columns = [
            'role(M|2147483647)', 
            'clean_role(M|2147483647)', 
            'pure_role(M|2147483647)'
        ]
        
        # 根据提供的角色分布，定义被告方角色
        defendant_roles = ['被告', '被上诉人', '被申请人']
        
        for role_col in role_columns:
            if role_col in cases_df.columns:
                for role in defendant_roles:
                    defendant_cases += len(cases_df[cases_df[role_col] == role])
        
        defendant_ratio = defendant_cases / total_cases if total_cases > 0 else 0
        
        # 归一化得分
        total_cases_score = max(0, 100 - min(np.log1p(total_cases) * 20, 100))
        annual_cases_score = max(0, 100 - min(np.log1p(annual_cases) * 25, 100))
        defendant_ratio_score = max(0, 100 - (defendant_ratio * 100))
        
        return {
            'total_cases': total_cases,
            'annual_cases': annual_cases,
            'defendant_ratio': defendant_ratio,
            'scores': {
                'total_cases_score': total_cases_score,
                'annual_cases_score': annual_cases_score,
                'defendant_ratio_score': defendant_ratio_score
            }
        }
    
    def calculate_court_severity(self, cases_df):
        """计算法院层级严重性指标"""
        def classify_court(court_name):
            if not isinstance(court_name, str):
                return '其他法院'
            
            court_name = court_name.strip()
            if '最高' in court_name or '最高法院' in court_name:
                return '最高人民法院'
            elif re.search(r'高级|省.*院', court_name):
                return '高级人民法院'
            elif re.search(r'中级|市.*院', court_name):
                return '中级人民法院'
            else:
                return '其他法院'
        
        # 法院权重
        court_weights = {
            '最高人民法院': 1.0, '高级人民法院': 0.8, 
            '中级人民法院': 0.6, '其他法院': 0.3
        }
        
        # 检查法院列是否存在
        if 'court(M|2147483647)' not in cases_df.columns:
            return {
                'avg_court_weight': 0,
                'court_distribution': {},
                'score': 100
            }
        
        cases_df['court_level'] = cases_df['court(M|2147483647)'].apply(classify_court)
        
        weighted_cases = 0
        for level, weight in court_weights.items():
            level_cases = len(cases_df[cases_df['court_level'] == level])
            weighted_cases += level_cases * weight
        
        total_cases = len(cases_df)
        avg_weight = weighted_cases / total_cases if total_cases > 0 else 0
        court_severity_score = max(0, 100 - (avg_weight * 100))
        
        return {
            'avg_court_weight': avg_weight,
            'court_distribution': cases_df['court_level'].value_counts().to_dict(),
            'score': court_severity_score
        }
    
    def calculate_recent_trend(self, cases_df):
        """计算时间趋势指标"""
        current_year = self.current_year
        total_cases = len(cases_df)
        
        # 近一年案件数
        recent_cases = 0
        time_fields = [
            'year_date(M|2147483647)', 
            'pub_date(M|2147483647)', 
            'start_date(M|2147483647)', 
            'hearing_date(M|2147483647)', 
            'date(M|2147483647)'
        ]
        
        for field in time_fields:
            if field in cases_df.columns:
                mask = (
                    cases_df[field].astype(str).str.contains(str(current_year), na=False) | 
                    cases_df[field].astype(str).str.contains(str(current_year - 1), na=False)
                )
                recent_cases = max(recent_cases, len(cases_df[mask]))
        
        # 如果时间字段缺失，使用保守估计
        if recent_cases == 0 and total_cases > 0:
            recent_cases = int(total_cases * 0.3)
        
        recent_ratio = recent_cases / total_cases if total_cases > 0 else 0
        recent_trend_score = max(0, 100 - (recent_ratio * 100))
        
        return {
            'recent_cases': recent_cases,
            'recent_ratio': recent_ratio,
            'score': recent_trend_score
        }
    
    def calculate_comprehensive_score(self, enterprise_data):
        """计算综合法律风险得分"""
        # 合并所有案件数据
        all_cases_list = []
        for case_type in ['filing_cases', 'hearing_cases', 'judgment_cases']:
            if case_type in enterprise_data and not enterprise_data[case_type].empty:
                all_cases_list.append(enterprise_data[case_type])
        
        if not all_cases_list:
            return {
                'comprehensive_score': 100,
                'risk_level': "低风险",
                'note': '无案件数据'
            }
        
        all_cases = pd.concat(all_cases_list, ignore_index=True)
        
        # 计算各项指标
        frequency_metrics = self.calculate_case_frequency(all_cases)
        severity_metrics = self.calculate_court_severity(all_cases)
        trend_metrics = self.calculate_recent_trend(all_cases)
        
        # 权重配置
        weights = {
            'total_cases': 0.20, 'annual_cases': 0.20, 'defendant_ratio': 0.15,
            'court_severity': 0.30, 'recent_trend': 0.15
        }
        
        # 加权计算
        comprehensive_score = (
            frequency_metrics['scores']['total_cases_score'] * weights['total_cases'] +
            frequency_metrics['scores']['annual_cases_score'] * weights['annual_cases'] +
            frequency_metrics['scores']['defendant_ratio_score'] * weights['defendant_ratio'] +
            severity_metrics['score'] * weights['court_severity'] +
            trend_metrics['score'] * weights['recent_trend']
        )
        
        # 风险等级
        risk_level = "低风险" if comprehensive_score >= 80 else "中风险" if comprehensive_score >= 60 else "高风险"
        
        return {
            'comprehensive_score': round(comprehensive_score, 2),
            'risk_level': risk_level,
            'detailed_scores': {
                'frequency_metrics': frequency_metrics,
                'severity_metrics': severity_metrics,
                'trend_metrics': trend_metrics
            }
        }

def preprocess_data(filing_df, hearing_df, judgment_df):
    """数据预处理"""
    # 填充缺失值
    for df in [filing_df, hearing_df, judgment_df]:
        if not df.empty:
            # 角色字段
            role_fields = [
                'role(M|2147483647)', 
                'clean_role(M|2147483647)', 
                'pure_role(M|2147483647)'
            ]
            for field in role_fields:
                if field in df.columns:
                    df[field] = df[field].fillna('(空白)')
            
            # 法院字段
            if 'court(M|2147483647)' in df.columns:
                df['court(M|2147483647)'] = df['court(M|2147483647)'].fillna('未知法院')
            
            # 时间字段
            current_year = datetime.now().year
            time_fields = [
                'year_date(M|2147483647)', 
                'pub_date(M|2147483647)', 
                'start_date(M|2147483647)', 
                'hearing_date(M|2147483647)', 
                'date(M|2147483647)'
            ]
            for field in time_fields:
                if field in df.columns:
                    df[field] = df[field].fillna(str(current_year))
    
    return filing_df, hearing_df, judgment_df

def batch_assess_enterprises(enterprise_list, filing_df, hearing_df, judgment_df):
    """批量评估企业法律风险"""
    assessor = LegalRiskAssessment()
    filing_df, hearing_df, judgment_df = preprocess_data(filing_df, hearing_df, judgment_df)
    
    results = []
    for eid in enterprise_list:
        try:
            # 筛选企业数据
            enterprise_data = {}
            if not filing_df.empty:
                enterprise_data['filing_cases'] = filing_df[filing_df['eid(M|2147483647)'] == eid]
            if not hearing_df.empty:
                enterprise_data['hearing_cases'] = hearing_df[hearing_df['eid(M|2147483647)'] == eid]
            if not judgment_df.empty:
                enterprise_data['judgment_cases'] = judgment_df[judgment_df['eid(M|2147483647)'] == eid]
            
            # 计算评分
            assessment = assessor.calculate_comprehensive_score(enterprise_data)
            
            results.append({
                'eid': eid,
                'legal_risk_score': assessment['comprehensive_score'],
                'risk_level': assessment['risk_level'],
                'total_cases': assessment['detailed_scores']['frequency_metrics']['total_cases'],
                'annual_cases': round(assessment['detailed_scores']['frequency_metrics']['annual_cases'], 2),
                'defendant_ratio': round(assessment['detailed_scores']['frequency_metrics']['defendant_ratio'] * 100, 2),
            })
        except Exception as e:
            print(f"评估企业 {eid} 时出错: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'eid': eid, 
                'legal_risk_score': None, 
                'risk_level': '评估失败',
                'total_cases': 0, 
                'annual_cases': 0, 
                'defendant_ratio': 0
            })
    
    return pd.DataFrame(results)

def load_data_from_csv():
    """从CSV文件加载数据"""
    # 定义CSV文件路径
    filing_file = '立案信息关系表.CSV'
    hearing_file = '开庭公告关系表.CSV'
    judgment_file = '裁判文书新关系表.CSV'
    
    # 检查文件是否存在
    if not os.path.exists(filing_file):
        print(f"错误: 文件 {filing_file} 不存在")
        return None, None, None
    
    if not os.path.exists(hearing_file):
        print(f"错误: 文件 {hearing_file} 不存在")
        return None, None, None
    
    if not os.path.exists(judgment_file):
        print(f"错误: 文件 {judgment_file} 不存在")
        return None, None, None
    
    try:
        # 读取CSV文件
        filing_df = pd.read_csv(filing_file)
        hearing_df = pd.read_csv(hearing_file)
        judgment_df = pd.read_csv(judgment_file)
        
        print(f"成功加载数据:")
        print(f"  立案信息: {len(filing_df)} 条记录")
        print(f"  开庭公告: {len(hearing_df)} 条记录") 
        print(f"  裁判文书: {len(judgment_df)} 条记录")
        
        return filing_df, hearing_df, judgment_df
        
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return None, None, None

def check_csv_columns(filing_df, hearing_df, judgment_df):
    """检查CSV文件的列名"""
    print("检查CSV文件列名...")
    
    # 必需列 - 现在只需要企业ID
    required_columns = {
        'filing': ['eid(M|2147483647)'],
        'hearing': ['eid(M|2147483647)'],
        'judgment': ['eid(M|2147483647)']
    }
    
    # 检查立案信息表
    missing_filing = [col for col in required_columns['filing'] if col not in filing_df.columns]
    if missing_filing:
        print(f"立案信息表缺少必需列: {missing_filing}")
        print(f"立案信息表实际列: {list(filing_df.columns)}")
    
    # 检查开庭公告表
    missing_hearing = [col for col in required_columns['hearing'] if col not in hearing_df.columns]
    if missing_hearing:
        print(f"开庭公告表缺少必需列: {missing_hearing}")
        print(f"开庭公告表实际列: {list(hearing_df.columns)}")
    
    # 检查裁判文书表
    missing_judgment = [col for col in required_columns['judgment'] if col not in judgment_df.columns]
    if missing_judgment:
        print(f"裁判文书表缺少必需列: {missing_judgment}")
        print(f"裁判文书表实际列: {list(judgment_df.columns)}")
    
    if not missing_filing and not missing_hearing and not missing_judgment:
        print("所有必需列都存在，可以继续评估")
        return True
    else:
        print("缺少必需列，无法进行评估")
        return False

# 使用示例
if __name__ == "__main__":
    print("开始法律风险评估...")
    
    # 从CSV文件加载数据
    filing_df, hearing_df, judgment_df = load_data_from_csv()
    
    if filing_df is None or hearing_df is None or judgment_df is None:
        print("数据加载失败，请检查CSV文件是否存在且格式正确")
        exit(1)
    
    # 检查CSV文件列名
    if not check_csv_columns(filing_df, hearing_df, judgment_df):
        exit(1)
    
    # 获取所有企业ID
    all_eids = set()
    for df in [filing_df, hearing_df, judgment_df]:
        if 'eid(M|2147483647)' in df.columns:
            all_eids.update(df['eid(M|2147483647)'].dropna().unique())
    
    enterprise_list = list(all_eids)
    print(f"\n发现 {len(enterprise_list)} 家企业")
    
    # 批量评估企业法律风险
    print("开始批量评估...")
    results_df = batch_assess_enterprises(enterprise_list, filing_df, hearing_df, judgment_df)
    
    # 输出结果
    print("\n评估结果:")
    print(results_df.to_string(index=False))
    
    # 保存结果到CSV文件
    output_file = '法律风险指标结果.CSV'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存至 {output_file}")
    
    # 检查结果DataFrame是否包含risk_level列
    if 'risk_level' in results_df.columns:
        # 统计风险分布
        risk_distribution = results_df['risk_level'].value_counts()
        print(f"\n风险分布:")
        for level, count in risk_distribution.items():
            percentage = count / len(results_df) * 100
            print(f"  {level}: {count} 家 ({percentage:.1f}%)")
    else:
        print("\n警告: 结果中没有找到'risk_level'列")
        print("结果DataFrame的列:", results_df.columns.tolist())
        
        # 尝试查看前几行数据以诊断问题
        print("\n前几行数据:")
        print(results_df.head())