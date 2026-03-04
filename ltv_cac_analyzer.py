"""
LTV/CAC 分析模块
客户生命周期价值 (LTV) 与获客成本 (CAC) 分析
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


class LTVCACAnalyzer:
    """LTV/CAC 分析器"""
    
    def __init__(self, customer_data: pd.DataFrame, acquisition_data: pd.DataFrame):
        """
        初始化 LTV/CAC 分析器
        
        Args:
            customer_data: 客户数据，包含 columns:
                          - customer_id: 客户 ID
                          - acquisition_date: 获取日期
                          - acquisition_channel: 获取渠道
                          - transactions: 交易记录（列表或 JSON）
            acquisition_data: 获客成本数据，包含 columns:
                             - date: 日期
                             - channel: 渠道
                             - spend: 支出
                             - new_customers: 新增客户数
        """
        self.customer_data = customer_data.copy()
        self.acquisition_data = acquisition_data.copy()
        self.acquisition_data['date'] = pd.to_datetime(self.acquisition_data['date'])
        
        self.customer_data['acquisition_date'] = pd.to_datetime(self.customer_data['acquisition_date'])
        
        self.ltv_by_channel = {}
        self.cac_by_channel = {}
        self.results = {}
    
    def calculate_cac_by_channel(self, period: str = 'M') -> pd.DataFrame:
        """
        计算各渠道的 CAC (Customer Acquisition Cost)
        
        Args:
            period: 聚合周期 ('D'=天，'W'=周，'M'=月)
        """
        # 按渠道和周期聚合
        self.acquisition_data['period'] = self.acquisition_data['date'].dt.to_period(period)
        
        cac_data = self.acquisition_data.groupby(['channel', 'period']).agg({
            'spend': 'sum',
            'new_customers': 'sum'
        }).reset_index()
        
        cac_data['cac'] = cac_data['spend'] / cac_data['new_customers'].replace(0, np.nan)
        
        # 按渠道汇总
        cac_summary = cac_data.groupby('channel').agg({
            'spend': 'sum',
            'new_customers': 'sum',
            'cac': 'mean'
        }).reset_index()
        
        cac_summary['cac'] = cac_summary['spend'] / cac_summary['new_customers']
        
        self.cac_by_channel = cac_summary.set_index('channel')['cac'].to_dict()
        
        return cac_summary
    
    def calculate_ltv_by_channel(self, observation_window: int = 365, 
                                  discount_rate: float = 0.1) -> pd.DataFrame:
        """
        计算各渠道的 LTV (Lifetime Value)
        
        Args:
            observation_window: 观察窗口（天）
            discount_rate: 折现率
        """
        ltv_results = []
        
        for channel in self.customer_data['acquisition_channel'].unique():
            channel_customers = self.customer_data[
                self.customer_data['acquisition_channel'] == channel
            ]
            
            customer_ltvs = []
            for _, customer in channel_customers.iterrows():
                ltv = self._calculate_customer_ltv(
                    customer, 
                    observation_window, 
                    discount_rate
                )
                customer_ltvs.append(ltv)
            
            avg_ltv = np.mean(customer_ltvs)
            median_ltv = np.median(customer_ltvs)
            std_ltv = np.std(customer_ltvs)
            
            ltv_results.append({
                'channel': channel,
                'n_customers': len(customer_ltvs),
                'avg_ltv': avg_ltv,
                'median_ltv': median_ltv,
                'std_ltv': std_ltv,
                'ltv_25': np.percentile(customer_ltvs, 25),
                'ltv_75': np.percentile(customer_ltvs, 75)
            })
        
        ltv_df = pd.DataFrame(ltv_results)
        self.ltv_by_channel = ltv_df.set_index('channel')['avg_ltv'].to_dict()
        
        return ltv_df
    
    def _calculate_customer_ltv(self, customer: pd.Series, 
                                 observation_window: int,
                                 discount_rate: float) -> float:
        """计算单个客户的 LTV"""
        transactions = customer.get('transactions', [])
        
        if not transactions:
            return 0
        
        # 如果是字典/JSON 格式
        if isinstance(transactions, dict):
            transactions = transactions.get('list', [])
        
        acq_date = customer['acquisition_date']
        total_value = 0
        
        for txn in transactions:
            if isinstance(txn, dict):
                txn_date = pd.to_datetime(txn.get('date', acq_date))
                txn_value = txn.get('value', 0)
                days_since_acq = (txn_date - acq_date).days
            else:
                # 假设是数值列表
                txn_value = txn
                days_since_acq = 0
            
            # 只在观察窗口内
            if 0 <= days_since_acq <= observation_window:
                # 应用折现
                discount_factor = 1 / (1 + discount_rate) ** (days_since_acq / 365)
                total_value += txn_value * discount_factor
        
        return total_value
    
    def predict_ltv_cohort(self, cohort_month: int = 12) -> pd.DataFrame:
        """
        基于队列分析预测 LTV
        
        Args:
            cohort_month: 预测月数
        """
        # 按获取月份分组
        self.customer_data['cohort'] = self.customer_data['acquisition_date'].dt.to_period('M')
        
        cohort_data = []
        for cohort, group in self.customer_data.groupby('cohort'):
            # 计算该队列的平均 LTV
            ltvs = []
            for _, customer in group.iterrows():
                ltv = self._calculate_customer_ltv(customer, 365, 0.1)
                ltvs.append(ltv)
            
            cohort_data.append({
                'cohort': str(cohort),
                'n_customers': len(group),
                'avg_ltv': np.mean(ltvs),
                'cumulative_ltv': np.sum(ltvs)
            })
        
        cohort_df = pd.DataFrame(cohort_data)
        
        # 使用线性回归预测未来 LTV
        if len(cohort_df) >= 3:
            cohort_df['cohort_num'] = range(len(cohort_df))
            model = LinearRegression()
            model.fit(cohort_df[['cohort_num']], cohort_df['avg_ltv'])
            
            # 预测未来
            future_cohorts = pd.DataFrame({
                'cohort_num': range(len(cohort_df), len(cohort_df) + cohort_month)
            })
            future_cohorts['predicted_ltv'] = model.predict(future_cohorts[['cohort_num']])
            
            cohort_df['predictions'] = model.predict(cohort_df[['cohort_num']])
        
        return cohort_df
    
    def calculate_ltv_cac_ratio(self) -> pd.DataFrame:
        """计算 LTV/CAC 比率"""
        if not self.cac_by_channel:
            self.calculate_cac_by_channel()
        if not self.ltv_by_channel:
            self.calculate_ltv_by_channel()
        
        # 合并所有渠道
        all_channels = set(self.cac_by_channel.keys()) | set(self.ltv_by_channel.keys())
        
        ratio_data = []
        for channel in all_channels:
            ltv = self.ltv_by_channel.get(channel, 0)
            cac = self.cac_by_channel.get(channel, 0)
            ratio = ltv / cac if cac > 0 else float('inf')
            
            ratio_data.append({
                'channel': channel,
                'ltv': ltv,
                'cac': cac,
                'ltv_cac_ratio': ratio,
                'health': self._assess_health(ratio)
            })
        
        return pd.DataFrame(ratio_data).sort_values('ltv_cac_ratio', ascending=False)
    
    def _assess_health(self, ratio: float) -> str:
        """评估 LTV/CAC 健康度"""
        if ratio >= 3:
            return '优秀 (Excellent)'
        elif ratio >= 2:
            return '良好 (Good)'
        elif ratio >= 1:
            return '一般 (Fair)'
        else:
            return '危险 (Poor)'
    
    def calculate_payback_period(self) -> pd.DataFrame:
        """计算回本周期（月）"""
        if not self.cac_by_channel:
            self.calculate_cac_by_channel()
        
        # 按渠道计算平均月收入
        payback_data = []
        for channel in self.cac_by_channel.keys():
            channel_customers = self.customer_data[
                self.customer_data['acquisition_channel'] == channel
            ]
            
            # 计算平均月收入
            monthly_revenues = []
            for _, customer in channel_customers.iterrows():
                transactions = customer.get('transactions', [])
                if transactions:
                    total_revenue = sum(
                        t.get('value', 0) if isinstance(t, dict) else t 
                        for t in transactions
                    )
                    # 假设平均客户生命周期
                    months = 12
                    monthly_revenues.append(total_revenue / months)
            
            avg_monthly_revenue = np.mean(monthly_revenues) if monthly_revenues else 1
            cac = self.cac_by_channel[channel]
            
            payback_months = cac / avg_monthly_revenue if avg_monthly_revenue > 0 else float('inf')
            
            payback_data.append({
                'channel': channel,
                'cac': cac,
                'avg_monthly_revenue': avg_monthly_revenue,
                'payback_months': payback_months
            })
        
        return pd.DataFrame(payback_data)
    
    def get_summary_metrics(self) -> Dict:
        """获取汇总指标"""
        ratio_df = self.calculate_ltv_cac_ratio()
        payback_df = self.calculate_payback_period()
        
        return {
            'avg_ltv': ratio_df['ltv'].mean(),
            'avg_cac': ratio_df['cac'].mean(),
            'avg_ltv_cac_ratio': ratio_df['ltv_cac_ratio'].mean(),
            'avg_payback_months': payback_df['payback_months'].mean(),
            'channels_excellent': len(ratio_df[ratio_df['ltv_cac_ratio'] >= 3]),
            'channels_poor': len(ratio_df[ratio_df['ltv_cac_ratio'] < 1])
        }


def create_sample_customer_data(n_customers: int = 1000) -> pd.DataFrame:
    """创建示例客户数据"""
    np.random.seed(42)
    
    channels = ['google', 'facebook', 'email', 'organic', 'referral']
    
    data = []
    for i in range(n_customers):
        acq_date = pd.Timestamp.today() - pd.Timedelta(days=np.random.randint(30, 730))
        channel = np.random.choice(channels, p=[0.3, 0.25, 0.15, 0.2, 0.1])
        
        # 生成交易记录
        n_transactions = np.random.poisson(5)
        transactions = []
        for _ in range(n_transactions):
            txn_date = acq_date + pd.Timedelta(days=np.random.randint(1, 365))
            txn_value = np.random.exponential(100) + 20
            transactions.append({
                'date': txn_date.isoformat(),
                'value': round(txn_value, 2)
            })
        
        data.append({
            'customer_id': f'C{i:05d}',
            'acquisition_date': acq_date,
            'acquisition_channel': channel,
            'transactions': transactions
        })
    
    return pd.DataFrame(data)


def create_sample_acquisition_data(n_days: int = 365) -> pd.DataFrame:
    """创建示例获客成本数据"""
    np.random.seed(42)
    
    channels = ['google', 'facebook', 'email', 'organic', 'referral']
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq='D')
    
    data = []
    for date in dates:
        for channel in channels:
            # 不同渠道的 CAC 不同
            base_cac = {'google': 50, 'facebook': 40, 'email': 10, 'organic': 5, 'referral': 15}[channel]
            
            spend = np.random.exponential(base_cac * 10)
            new_customers = int(spend / base_cac * np.random.uniform(0.8, 1.2))
            
            data.append({
                'date': date,
                'channel': channel,
                'spend': round(spend, 2),
                'new_customers': max(1, new_customers)
            })
    
    return pd.DataFrame(data)


if __name__ == '__main__':
    print("创建示例数据...")
    customer_data = create_sample_customer_data(500)
    acquisition_data = create_sample_acquisition_data(90)
    
    print("\n计算 CAC...")
    analyzer = LTVCACAnalyzer(customer_data, acquisition_data)
    cac_df = analyzer.calculate_cac_by_channel()
    print(cac_df)
    
    print("\n计算 LTV...")
    ltv_df = analyzer.calculate_ltv_by_channel()
    print(ltv_df)
    
    print("\nLTV/CAC 比率:")
    ratio_df = analyzer.calculate_ltv_cac_ratio()
    print(ratio_df)
    
    print("\n汇总指标:")
    summary = analyzer.get_summary_metrics()
    for key, value in summary.items():
        print(f"  {key}: {value:.2f}")
