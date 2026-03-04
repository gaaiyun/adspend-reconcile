"""
预算优化模块
基于 ROI 的营销预算智能分配建议
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize, LinearConstraint, Bounds
import warnings
warnings.filterwarnings('ignore')


class BudgetOptimizer:
    """营销预算优化器"""
    
    def __init__(self, channel_data: pd.DataFrame):
        """
        初始化预算优化器
        
        Args:
            channel_data: 渠道数据，包含 columns:
                         - channel: 渠道名称
                         - spend: 当前支出
                         - conversions: 转化数
                         - revenue: 收入
                         - roi: ROI（可选）
        """
        self.data = channel_data.copy()
        self.optimization_results = {}
        
        # 计算 ROI（如果未提供）
        if 'roi' not in self.data.columns:
            if 'revenue' in self.data.columns and 'spend' in self.data.columns:
                self.data['roi'] = (self.data['revenue'] - self.data['spend']) / self.data['spend']
            elif 'conversions' in self.data.columns:
                # 假设每次转化价值 100
                self.data['revenue'] = self.data['conversions'] * 100
                self.data['roi'] = (self.data['revenue'] - self.data['spend']) / self.data['spend']
    
    def optimize_budget(self, total_budget: float, 
                        min_spend_ratio: float = 0.1,
                        max_spend_ratio: float = 0.5,
                        method: str = 'roi_weighted') -> pd.DataFrame:
        """
        优化预算分配
        
        Args:
            total_budget: 总预算
            min_spend_ratio: 最小支出比例（防止某渠道为 0）
            max_spend_ratio: 最大支出比例（防止过度集中）
            method: 优化方法 ('roi_weighted', 'marginal_roi', 'optimization')
        """
        channels = self.data['channel'].tolist()
        n_channels = len(channels)
        
        if method == 'roi_weighted':
            return self._roi_weighted_optimization(total_budget, min_spend_ratio, max_spend_ratio)
        elif method == 'marginal_roi':
            return self._marginal_roi_optimization(total_budget, min_spend_ratio, max_spend_ratio)
        elif method == 'optimization':
            return self._mathematical_optimization(total_budget, min_spend_ratio, max_spend_ratio)
        else:
            raise ValueError(f"未知优化方法：{method}")
    
    def _roi_weighted_optimization(self, total_budget: float,
                                    min_ratio: float, max_ratio: float) -> pd.DataFrame:
        """基于 ROI 权重的优化"""
        # 计算 ROI 权重（确保非负）
        rois = self.data['roi'].clip(lower=0)
        weights = rois / rois.sum() if rois.sum() > 0 else np.ones(len(rois)) / len(rois)
        
        # 应用约束
        weights = np.clip(weights, min_ratio / len(weights), max_ratio)
        weights = weights / weights.sum()  # 重新归一化
        
        optimized_spend = weights * total_budget
        
        results = self.data.copy()
        results['optimized_spend'] = optimized_spend
        results['spend_change'] = optimized_spend - results['spend']
        results['spend_change_pct'] = results['spend_change'] / results['spend'] * 100
        
        # 预测优化后的效果
        results['predicted_revenue'] = optimized_spend * (1 + results['roi'])
        results['predicted_profit'] = results['predicted_revenue'] - optimized_spend
        
        current_profit = (self.data['revenue'] - self.data['spend']).sum()
        optimized_profit = results['predicted_profit'].sum()
        
        self.optimization_results['roi_weighted'] = {
            'current_profit': current_profit,
            'optimized_profit': optimized_profit,
            'improvement': optimized_profit - current_profit,
            'improvement_pct': (optimized_profit - current_profit) / current_profit * 100 if current_profit > 0 else 0
        }
        
        return results
    
    def _marginal_roi_optimization(self, total_budget: float,
                                    min_ratio: float, max_ratio: float) -> pd.DataFrame:
        """基于边际 ROI 的优化（考虑收益递减）"""
        channels = self.data['channel'].tolist()
        n_channels = len(channels)
        
        # 估计边际 ROI（简化：假设收益递减）
        current_spend = self.data['spend'].values
        current_roi = self.data['roi'].values
        
        # 边际 ROI = 当前 ROI * (1 - 支出弹性)
        elasticity = 0.3  # 支出增加 10%，ROI 下降 3%
        marginal_roi = current_roi * (1 - elasticity)
        
        # 迭代优化
        optimized_spend = current_spend.copy()
        budget_step = total_budget / 100
        
        for _ in range(100):
            # 找到边际 ROI 最高的渠道
            best_channel = np.argmax(marginal_roi)
            
            # 增加该渠道预算
            if optimized_spend.sum() < total_budget:
                optimized_spend[best_channel] += budget_step
                # 更新边际 ROI（收益递减）
                marginal_roi[best_channel] *= 0.99
            else:
                break
        
        # 应用约束
        min_spend = total_budget * min_ratio / n_channels
        max_spend = total_budget * max_ratio
        
        optimized_spend = np.clip(optimized_spend, min_spend, max_spend)
        optimized_spend = optimized_spend / optimized_spend.sum() * total_budget
        
        results = self.data.copy()
        results['optimized_spend'] = optimized_spend
        results['spend_change'] = optimized_spend - results['spend']
        results['spend_change_pct'] = results['spend_change'] / results['spend'] * 100
        
        results['predicted_revenue'] = optimized_spend * (1 + results['roi'] * 0.9)  # 考虑递减
        results['predicted_profit'] = results['predicted_revenue'] - optimized_spend
        
        current_profit = (self.data['revenue'] - self.data['spend']).sum()
        optimized_profit = results['predicted_profit'].sum()
        
        self.optimization_results['marginal_roi'] = {
            'current_profit': current_profit,
            'optimized_profit': optimized_profit,
            'improvement': optimized_profit - current_profit,
            'improvement_pct': (optimized_profit - current_profit) / current_profit * 100 if current_profit > 0 else 0
        }
        
        return results
    
    def _mathematical_optimization(self, total_budget: float,
                                    min_ratio: float, max_ratio: float) -> pd.DataFrame:
        """使用数学优化方法"""
        channels = self.data['channel'].tolist()
        n_channels = len(channels)
        
        # 目标函数：最大化总利润（负值用于最小化）
        def objective(spend):
            roi = self.data['roi'].values
            revenue = spend * (1 + roi * 0.9)  # 考虑收益递减
            profit = revenue - spend
            return -profit.sum()  # 最小化负利润
        
        # 约束：总支出等于预算
        constraints = [
            {'type': 'eq', 'fun': lambda x: x.sum() - total_budget}
        ]
        
        # 边界约束
        min_spend = total_budget * min_ratio / n_channels
        max_spend = total_budget * max_ratio
        bounds = Bounds([min_spend] * n_channels, [max_spend] * n_channels)
        
        # 初始解：按当前比例分配
        current_spend = self.data['spend'].values
        x0 = current_spend / current_spend.sum() * total_budget
        
        # 优化
        result = minimize(
            objective, 
            x0, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimized_spend = result.x
        
        results = self.data.copy()
        results['optimized_spend'] = optimized_spend
        results['spend_change'] = optimized_spend - results['spend']
        results['spend_change_pct'] = results['spend_change'] / results['spend'] * 100
        
        results['predicted_revenue'] = optimized_spend * (1 + results['roi'] * 0.9)
        results['predicted_profit'] = results['predicted_revenue'] - optimized_spend
        
        current_profit = (self.data['revenue'] - self.data['spend']).sum()
        optimized_profit = results['predicted_profit'].sum()
        
        self.optimization_results['mathematical'] = {
            'current_profit': current_profit,
            'optimized_profit': optimized_profit,
            'improvement': optimized_profit - current_profit,
            'improvement_pct': (optimized_profit - current_profit) / current_profit * 100 if current_profit > 0 else 0,
            'optimization_success': result.success
        }
        
        return results
    
    def scenario_analysis(self, budget_range: Tuple[float, float], 
                          n_scenarios: int = 10) -> pd.DataFrame:
        """
        预算场景分析
        
        Args:
            budget_range: 预算范围 (min, max)
            n_scenarios: 场景数量
        """
        budgets = np.linspace(budget_range[0], budget_range[1], n_scenarios)
        
        scenarios = []
        for budget in budgets:
            result = self.optimize_budget(budget, method='optimization')
            scenarios.append({
                'total_budget': budget,
                'predicted_revenue': result['predicted_revenue'].sum(),
                'predicted_profit': result['predicted_profit'].sum(),
                'avg_roi': (result['predicted_revenue'].sum() - budget) / budget
            })
        
        return pd.DataFrame(scenarios)
    
    def get_reallocation_recommendations(self, total_budget: Optional[float] = None) -> pd.DataFrame:
        """获取预算重新分配建议"""
        if total_budget is None:
            total_budget = self.data['spend'].sum()
        
        optimized = self.optimize_budget(total_budget, method='optimization')
        
        recommendations = []
        for _, row in optimized.iterrows():
            if row['spend_change_pct'] > 10:
                action = '增加'
            elif row['spend_change_pct'] < -10:
                action = '减少'
            else:
                action = '保持'
            
            recommendations.append({
                'channel': row['channel'],
                'current_spend': row['spend'],
                'recommended_spend': row['optimized_spend'],
                'change_amount': row['spend_change'],
                'change_percent': row['spend_change_pct'],
                'action': action,
                'current_roi': row['roi'],
                'priority': '高' if abs(row['spend_change_pct']) > 30 else '中' if abs(row['spend_change_pct']) > 10 else '低'
            })
        
        return pd.DataFrame(recommendations)
    
    def get_optimization_summary(self, total_budget: Optional[float] = None) -> Dict:
        """获取优化摘要"""
        if total_budget is None:
            total_budget = self.data['spend'].sum()
        
        optimized = self.optimize_budget(total_budget, method='optimization')
        
        return {
            'current_total_spend': self.data['spend'].sum(),
            'optimized_total_spend': total_budget,
            'current_profit': (self.data['revenue'] - self.data['spend']).sum(),
            'optimized_profit': optimized['predicted_profit'].sum(),
            'profit_improvement': optimized['predicted_profit'].sum() - (self.data['revenue'] - self.data['spend']).sum(),
            'channels_to_increase': len(optimized[optimized['spend_change'] > 0]),
            'channels_to_decrease': len(optimized[optimized['spend_change'] < 0]),
            'top_channel': optimized.loc[optimized['optimized_spend'].idxmax(), 'channel'],
            'optimization_details': self.optimization_results.get('mathematical', {})
        }


def create_sample_channel_data(n_channels: int = 6) -> pd.DataFrame:
    """创建示例渠道数据"""
    np.random.seed(42)
    
    channels = ['google', 'facebook', 'email', 'paid_search', 'display', 'organic']
    
    data = []
    for channel in channels:
        # 不同渠道有不同的 ROI 特征
        base_roi = {
            'google': 2.5,
            'facebook': 1.8,
            'email': 4.0,
            'paid_search': 2.2,
            'display': 1.2,
            'organic': 8.0
        }[channel]
        
        spend = np.random.exponential(5000)
        roi = base_roi * np.random.uniform(0.8, 1.2)
        revenue = spend * (1 + roi)
        conversions = int(revenue / 100)
        
        data.append({
            'channel': channel,
            'spend': round(spend, 2),
            'conversions': conversions,
            'revenue': round(revenue, 2),
            'roi': round(roi, 2)
        })
    
    return pd.DataFrame(data)


if __name__ == '__main__':
    print("创建示例渠道数据...")
    channel_data = create_sample_channel_data()
    print(channel_data)
    
    print("\n优化预算分配...")
    optimizer = BudgetOptimizer(channel_data)
    total_budget = channel_data['spend'].sum()
    print(f"总预算：${total_budget:,.2f}")
    
    optimized = optimizer.optimize_budget(total_budget, method='optimization')
    print("\n优化结果:")
    print(optimized[['channel', 'spend', 'optimized_spend', 'spend_change_pct', 'roi']])
    
    print("\n重新分配建议:")
    recommendations = optimizer.get_reallocation_recommendations()
    print(recommendations[['channel', 'action', 'change_percent', 'priority']])
    
    print("\n优化摘要:")
    summary = optimizer.get_optimization_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: ${value:,.2f}")
        else:
            print(f"  {key}: {value}")
