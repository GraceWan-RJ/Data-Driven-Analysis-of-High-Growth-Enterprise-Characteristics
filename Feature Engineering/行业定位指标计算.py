import pandas as pd
import numpy as np
import re

class EnterprisePotentialScorer:
    def __init__(self):
        # 新兴产业关键词模式
        self.emerging_industry_patterns = [
            r'智能|AI|人工智能', r'大数?据|数据', r'云|云计算', r'物联', r'芯片|集成电路|半导体',
            r'生物|基因|医药', r'新能?源|光伏|风电|太阳能', r'机器?人|自动化', r'VR|AR|虚拟|增强',
            r'区块?链|数字.?货币', r'5G|通信', r'无人.?驾驶|自动.?驾驶', r'量子', r'元宇?宙',
            r'工业.?互联网', r'金融.?科技|FinTech', r'数字.?经济', r'新材料', r'节能.?环保|绿色'
        ]
        # 技术相关分类和扣分项
        self.tech_categories = ['电子信息类认证', '技术领先', '科技奖项', '科技认定']
        self.penalty_patterns = ['撤销', '失效']
    
    def is_emerging_industry(self, tag_name):
        """判断是否属于新兴产业"""
        if pd.isna(tag_name):
            return False
        tag_str = str(tag_name)
        for pattern in self.emerging_industry_patterns:
            if re.search(pattern, tag_str, re.IGNORECASE):
                return True
        return False
    
    def smart_emerging_industry_detection(self, df):
        """智能识别新兴产业"""
        business_tags = df[df['type_name(M|2147483647)'] == '业务概念']['tag_name(M|2147483647)'].unique()
        
        # 关键词识别
        emerging_industries = []
        for tag in business_tags:
            if self.is_emerging_industry(tag):
                emerging_industries.append(tag)
        
        print(f"识别到 {len(emerging_industries)} 个新兴产业")
        return emerging_industries
    
    def calculate_scores(self, df):
        """为所有企业计算三个维度得分和总得分"""
        
        # 识别新兴产业
        emerging_industries = self.smart_emerging_industry_detection(df)
        
        results = []
        for company_id, company_data in df.groupby('eid(M|2147483647)'):
            # 1. 新兴产业匹配度得分
            business_tags = company_data[company_data['type_name(M|2147483647)'] == '业务概念']['tag_name(M|2147483647)']
            emerging_score = business_tags.isin(emerging_industries).sum()
            
            # 2. 技术密集度得分
            tech_score = 0
            for category in self.tech_categories:
                tech_score += len(company_data[company_data['type_name(M|2147483647)'] == category])
            
            # 扣分项
            penalty_tags = company_data[
                (company_data['type_name(M|2147483647)'] == '科技认定') & 
                (company_data['tag_name(M|2147483647)'].str.contains('|'.join(self.penalty_patterns), na=False))
            ]
            tech_score -= len(penalty_tags)
            tech_score = max(0, tech_score)
            
            # 3. 政策支持强度得分
            policy_score = len(company_data[company_data['type_name(M|2147483647)'] == '政府扶持和奖励'])
            
            # 总得分（加权）
            total_score = emerging_score * 1.5 + tech_score + policy_score
            
            results.append({
                'eid': company_id,
                'emerging_industry_score': emerging_score,
                'tech_intensity_score': tech_score,
                'policy_support_score': policy_score,
                'industry_score': total_score
            })
        
        return pd.DataFrame(results), emerging_industries

# 使用示例
def main():
    # 读取数据
    df = pd.read_csv('概念标签.CSV')
    
    # 计算得分
    scorer = EnterprisePotentialScorer()
    scores_df, emerging_industries = scorer.calculate_scores(df)
    
    # 输出结果
    print(f"新兴产业数量: {len(emerging_industries)}")
    print(f"总企业数量: {len(scores_df)}")
    print("\n企业得分统计:")
    print(f"新兴产业匹配度平均分: {scores_df['emerging_industry_score'].mean():.2f}")
    print(f"技术密集度平均分: {scores_df['tech_intensity_score'].mean():.2f}")
    print(f"政策支持强度平均分: {scores_df['policy_support_score'].mean():.2f}")
    print(f"总得分平均分: {scores_df['industry_score'].mean():.2f}")
    
    # 显示得分最高的10个企业
    print("\n总得分最高的10个企业:")
    print(scores_df.nlargest(10, 'industry_score')[['eid', 'industry_score']])
    
    # 保存结果
    scores_df.to_csv('行业定位指标结果.CSV', index=False, encoding='utf-8-sig')
    
    return scores_df, emerging_industries

if __name__ == "__main__":
    scores_df, emerging_industries = main()