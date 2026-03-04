"""
营销组合模型 (MMM) 分析模块
简化版 Marketing Mix Modeling，用于评估各营销渠道对转化的影响
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class MMMAnalyzer:
    """营销组合模型分析器"""
    
    def __init__(self, marketing_data: pd.DataFrame):
        """
        初始化 MMM 分析器
        
        Args:
            marketing_data: 营销数据，包含 columns:
                           - date: 日期
                           - 各渠道支出：google_spend, facebook_spend, email_spend 等
                           - 目标变量：conversions, revenue 等
        """
        self.data = marketing_data.copy()
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data = self.data.sort_values('date').reset_index(drop=True)
        
        self.spend_columns = self._identify_spend_columns()
        self.target_column = None
        self.model = None
        self.scaler = None
        self.results = {}
    
    def _identify_spend_columns(self) -> List[str]:
        """识别支出列"""
        spend_cols = [col for col in self.data.columns if col.endswith('_spend') or col.endswith('_cost')]
        return spend_cols
    
    def set_target(self, target_column: str):
        """设置目标变量"""
        if target_column not in self.data.columns:
            raise ValueError(f"列 '{target_column}' 不存在")
        self.target_column = target_column
    
    def add_adstock_effect(self, column: str, decay_rate: float = 0.5, max_lag: int = 4) -> pd.Series:
        """
        添加广告滞后效应 (Adstock)
        
        Args:
            column: 要处理的列名
            decay_rate: 衰减率 (0-1)，表示广告效果的持续程度
            max_lag: 最大滞后周期
        """
        values = self.data[column].values
        adstock = np.zeros(len(values))
        
        for t in range(len(values)):
            for lag in range(min(t + 1, max_lag)):
                adstock[t] += values[t - lag] * (decay_rate ** lag)
        
        return pd.Series(adstock, index=self.data.index, name=f'{column}_adstock')
    
    def add_diminishing_returns(self, column: str, exponent: float = 0.5) -> pd.Series:
        """
        添加边际收益递减效应
        
        Args:
            column: 要处理的列名
            exponent: 指数 (0-1)，越小表示递减越快
        """
        return self.data[column] ** exponent
    
    def fit_model(self, target_column: Optional[str] = None, 
                  use_adstock: bool = True, 
                  adstock_decay: float = 0.5,
                  use_diminishing_returns: bool = True,
                  diminishing_exponent: float = 0.5,
                  regularization: float = 1.0):
        """
        拟合 MMM 模型
        
        Args:
            target_column: 目标变量列名
            use_adstock: 是否使用广告滞后效应
            adstock_decay: 广告滞后衰减率
            use_diminishing_returns: 是否使用边际收益递减
            diminishing_exponent: 边际收益递减指数
            regularization: Ridge 回归正则化强度
        """
        if target_column:
            self.set_target(target_column)
        
        if not self.target_column:
            raise ValueError("请先设置目标变量")
        
        # 准备特征
        X = self.data.copy()
        
        # 应用转换
        features = []
        for col in self.spend_columns:
            if use_adstock:
                X[f'{col}_adstock'] = self.add_adstock_effect(col, adstock_decay)
                features.append(f'{col}_adstock')
            else:
                features.append(col)
            
            if use_diminishing_returns:
                feature_col = f'{col}_adstock' if use_adstock else col
                X[f'{feature_col}_dr'] = self.add_diminishing_returns(feature_col, diminishing_exponent)
                features = [f if f != feature_col else f'{feature_col}_dr' for f in features]
        
        # 添加控制变量
        control_vars = ['dayofweek', 'month', 'is_holiday']
        for var in control_vars:
            if var in X.columns:
                features.append(var)
        
        X_model = X[features].fillna(0)
        y = self.data[self.target_column]
        
        # 标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_model)
        
        # 拟合 Ridge 回归（防止多重共线性）
        self.model = Ridge(alpha=regularization)
        self.model.fit(X_scaled, y)
        
        # 存储结果
        self.results['features'] = features
        self.results['coefficients'] = dict(zip(features, self.model.coef_))
        self.results['intercept'] = self.model.intercept_
        self.results['r2'] = self.model.score(X_scaled, y)
        
        return self.results
    
    def get_channel_contributions(self) -> pd.DataFrame:
        """获取各渠道的贡献度"""
        if not self.model:
            raise ValueError("请先拟合模型")
        
        contributions = []
        total_contribution = 0
        
        for col in self.spend_columns:
            # 找到对应的特征
            feature = None
            for f in self.results['features']:
                if col in f:
                    feature = f
                    break
            
            if feature:
                coef = self.results['coefficients'].get(feature, 0)
                avg_spend = self.data[col].mean()
                contribution = coef * avg_spend
                contributions.append({
                    'channel': col.replace('_spend', '').replace('_cost', ''),
                    'coefficient': coef,
                    'avg_spend': avg_spend,
                    'contribution': contribution
                })
                total_contribution += contribution
        
        # 计算百分比
        for item in contributions:
            item['contribution_pct'] = (item['contribution'] / total_contribution * 100) if total_contribution > 0 else 0
        
        return pd.DataFrame(contributions)
    
    def get_roi_by_channel(self) -> pd.DataFrame:
        """计算各渠道的 ROI"""
        contributions = self.get_channel_contributions()
        
        roi_data = []
        for _, row in contributions.iterrows():
            channel = row['channel']
            spend_col = f"{channel}_spend" if f"{channel}_spend" in self.data.columns else f"{channel}_cost"
            
            if spend_col in self.data.columns:
                avg_spend = self.data[spend_col].mean()
                revenue = row['contribution']
                roi = (revenue - avg_spend) / avg_spend * 100 if avg_spend > 0 else 0
                
                roi_data.append({
                    'channel': channel,
                    'avg_spend': avg_spend,
                    'attributed_revenue': revenue,
                    'roi_percent': roi,
                    'roi_ratio': revenue / avg_spend if avg_spend > 0 else 0
                })
        
        return pd.DataFrame(roi_data)
    
    def predict(self, spend_scenario: Dict[str, float]) -> float:
        """
        预测给定支出场景下的结果
        
        Args:
            spend_scenario: 字典，如 {'google_spend': 1000, 'facebook_spend': 500}
        """
        if not self.model:
            raise ValueError("请先拟合模型")
        
        # 创建特征向量
        feature_values = []
        for feature in self.results['features']:
            value = 0
            for col in self.spend_columns:
                if col in feature:
                    base_col = col
                    value = spend_scenario.get(base_col, 0)
                    
                    # 应用 adstock（简化：假设稳态）
                    if 'adstock' in feature:
                        value = value / (1 - 0.5)  # 简化 adstock 计算
                    
                    # 应用 diminishing returns
                    if 'dr' in feature:
                        value = value ** 0.5
                    
                    break
            
            # 控制变量默认值
            if 'dayofweek' in feature or 'month' in feature or 'holiday' in feature:
                value = 0
            
            feature_values.append(value)
        
        # 标准化并预测
        feature_array = np.array(feature_values).reshape(1, -1)
        feature_scaled = self.scaler.transform(feature_array)
        prediction = self.model.predict(feature_scaled)[0]
        
        return max(0, prediction)  # 确保非负
    
    def get_model_summary(self) -> Dict:
        """获取模型摘要"""
        if not self.model:
            raise ValueError("请先拟合模型")
        
        return {
            'r_squared': self.results['r2'],
            'intercept': self.results['intercept'],
            'n_features': len(self.results['features']),
            'top_channels': self.get_channel_contributions().sort_values('contribution', ascending=False).head(5)
        }
    
    def get_marginal_roi(self, channel: str, delta: float = 100) -> float:
        """
        计算某渠道的边际 ROI（增加单位支出的额外回报）
        
        Args:
            channel: 渠道名称
            delta: 支出增量
        """
        # 当前支出
        current_spend = {col: self.data[col].mean() for col in self.spend_columns}
        
        # 增加支出
        spend_col = f"{channel}_spend" if f"{channel}_spend" in self.data.columns else f"{channel}_cost"
        current_spend[spend_col] += delta
        
        # 预测
        current_prediction = self.predict(current_spend)
        current_spend[spend_col] -= delta
        baseline_prediction = self.predict(current_spend)
        
        marginal_revenue = current_prediction - baseline_prediction
        marginal_roi = (marginal_revenue - delta) / delta * 100
        
        return marginal_roi


def create_sample_marketing_data(n_days: int = 365) -> pd.DataFrame:
    """创建示例营销数据"""
    np.random.seed(42)
    
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq='D')
    
    data = {
        'date': dates,
        'google_spend': np.random.exponential(500, n_days),
        'facebook_spend': np.random.exponential(300, n_days),
        'email_spend': np.random.exponential(100, n_days),
        'paid_search_spend': np.random.exponential(400, n_days),
        'display_spend': np.random.exponential(200, n_days),
    }
    
    # 添加季节性
    day_of_year = dates.dayofyear
    data['seasonality'] = 1 + 0.2 * np.sin(2 * np.pi * day_of_year / 365)
    
    # 添加星期效应
    data['dayofweek'] = dates.dayofweek
    
    # 生成转化（基于支出 + 噪声）
    base_conversions = (
        data['google_spend'] * 0.5 +
        data['facebook_spend'] * 0.3 +
        data['email_spend'] * 0.8 +
        data['paid_search_spend'] * 0.4 +
        data['display_spend'] * 0.2
    )
    data['conversions'] = base_conversions * data['seasonality'] + np.random.normal(0, 50, n_days)
    data['conversions'] = np.maximum(0, data['conversions'])
    
    # 生成收入
    data['revenue'] = data['conversions'] * np.random.uniform(80, 120, n_days)
    
    df = pd.DataFrame(data)
    return df


if __name__ == '__main__':
    print("创建示例营销数据...")
    sample_data = create_sample_marketing_data(100)
    print(f"数据形状：{sample_data.shape}")
    
    print("\n拟合 MMM 模型...")
    mmm = MMMAnalyzer(sample_data)
    results = mmm.fit_model(target_column='conversions', regularization=0.5)
    
    print(f"\n模型 R²: {results['r2']:.3f}")
    
    print("\n渠道贡献:")
    contributions = mmm.get_channel_contributions()
    print(contributions)
    
    print("\n渠道 ROI:")
    roi = mmm.get_roi_by_channel()
    print(roi)
