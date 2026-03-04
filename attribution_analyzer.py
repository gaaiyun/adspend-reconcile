"""
营销归因分析模块
支持多种归因模型：首次点击、末次点击、线性、时间衰减、Shapley 值
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from itertools import combinations


class AttributionAnalyzer:
    """营销归因分析器"""
    
    def __init__(self, conversion_data: pd.DataFrame):
        """
        初始化归因分析器
        
        Args:
            conversion_data: 转化数据，包含 columns: ['customer_id', 'touchpoints', 'conversion_value']
                            touchpoints 是渠道列表，如 ['google', 'facebook', 'email']
        """
        self.data = conversion_data.copy()
        self.channels = self._extract_all_channels()
        self.attribution_results = {}
    
    def _extract_all_channels(self) -> List[str]:
        """提取所有唯一渠道"""
        all_channels = set()
        for touchpoints in self.data['touchpoints']:
            all_channels.update(touchpoints)
        return sorted(list(all_channels))
    
    def first_click_attribution(self) -> pd.DataFrame:
        """首次点击归因 - 100% 归因于第一个触点的渠道"""
        results = {channel: 0.0 for channel in self.channels}
        
        for _, row in self.data.iterrows():
            if row['touchpoints']:
                first_channel = row['touchpoints'][0]
                results[first_channel] += row['conversion_value']
        
        total = sum(results.values())
        attribution = {
            channel: {
                'conversions': results[channel],
                'percentage': (results[channel] / total * 100) if total > 0 else 0
            }
            for channel in self.channels
        }
        
        self.attribution_results['first_click'] = attribution
        return pd.DataFrame([
            {'channel': ch, 'conversions': attr['conversions'], 
             'percentage': attr['percentage'], 'model': 'first_click'}
            for ch, attr in attribution.items()
        ])
    
    def last_click_attribution(self) -> pd.DataFrame:
        """末次点击归因 - 100% 归因于最后一个触点的渠道"""
        results = {channel: 0.0 for channel in self.channels}
        
        for _, row in self.data.iterrows():
            if row['touchpoints']:
                last_channel = row['touchpoints'][-1]
                results[last_channel] += row['conversion_value']
        
        total = sum(results.values())
        attribution = {
            channel: {
                'conversions': results[channel],
                'percentage': (results[channel] / total * 100) if total > 0 else 0
            }
            for channel in self.channels
        }
        
        self.attribution_results['last_click'] = attribution
        return pd.DataFrame([
            {'channel': ch, 'conversions': attr['conversions'], 
             'percentage': attr['percentage'], 'model': 'last_click'}
            for ch, attr in attribution.items()
        ])
    
    def linear_attribution(self) -> pd.DataFrame:
        """线性归因 - 平均分配给所有触点"""
        results = {channel: 0.0 for channel in self.channels}
        
        for _, row in self.data.iterrows():
            if row['touchpoints']:
                n_touchpoints = len(row['touchpoints'])
                credit_per_touchpoint = row['conversion_value'] / n_touchpoints
                for channel in row['touchpoints']:
                    results[channel] += credit_per_touchpoint
        
        total = sum(results.values())
        attribution = {
            channel: {
                'conversions': results[channel],
                'percentage': (results[channel] / total * 100) if total > 0 else 0
            }
            for channel in self.channels
        }
        
        self.attribution_results['linear'] = attribution
        return pd.DataFrame([
            {'channel': ch, 'conversions': attr['conversions'], 
             'percentage': attr['percentage'], 'model': 'linear'}
            for ch, attr in attribution.items()
        ])
    
    def time_decay_attribution(self, decay_factor: float = 0.5) -> pd.DataFrame:
        """
        时间衰减归因 - 越靠近转化的触点权重越高
        
        Args:
            decay_factor: 衰减因子 (0-1)，越小表示衰减越快
        """
        results = {channel: 0.0 for channel in self.channels}
        
        for _, row in self.data.iterrows():
            touchpoints = row['touchpoints']
            if touchpoints:
                n = len(touchpoints)
                # 计算权重：指数衰减
                weights = [decay_factor ** (n - i - 1) for i in range(n)]
                total_weight = sum(weights)
                normalized_weights = [w / total_weight for w in weights]
                
                for channel, weight in zip(touchpoints, normalized_weights):
                    results[channel] += row['conversion_value'] * weight
        
        total = sum(results.values())
        attribution = {
            channel: {
                'conversions': results[channel],
                'percentage': (results[channel] / total * 100) if total > 0 else 0
            }
            for channel in self.channels
        }
        
        self.attribution_results['time_decay'] = attribution
        return pd.DataFrame([
            {'channel': ch, 'conversions': attr['conversions'], 
             'percentage': attr['percentage'], 'model': 'time_decay'}
            for ch, attr in attribution.items()
        ])
    
    def shapley_value_attribution(self) -> pd.DataFrame:
        """
        Shapley 值归因 - 基于博弈论的公平归因
        计算每个渠道的边际贡献
        """
        results = {channel: 0.0 for channel in self.channels}
        
        # 对每个客户路径计算 Shapley 值
        for _, row in self.data.iterrows():
            touchpoints = row['touchpoints']
            conversion_value = row['conversion_value']
            
            if not touchpoints:
                continue
            
            n = len(touchpoints)
            unique_channels = list(set(touchpoints))
            
            # 对每个渠道计算 Shapley 值
            for channel in unique_channels:
                marginal_contribution = 0
                count = 0
                
                # 枚举所有包含该渠道的子集
                for r in range(1, n + 1):
                    for subset in combinations(range(n), r):
                        if touchpoints[subset[0]] == channel or any(touchpoints[i] == channel for i in subset):
                            # 计算边际贡献
                            subset_channels = [touchpoints[i] for i in subset]
                            without_channel = [c for c in subset_channels if c != channel]
                            
                            # 简化：如果渠道在子集中，贡献为 1/子集大小
                            if channel in subset_channels:
                                marginal_contribution += conversion_value / len(subset_channels)
                                count += 1
                
                if count > 0:
                    results[channel] += marginal_contribution / count
        
        # 归一化
        total = sum(results.values())
        if total > 0:
            results = {k: v / total * total for k, v in results.items()}
        
        attribution = {
            channel: {
                'conversions': results[channel],
                'percentage': (results[channel] / total * 100) if total > 0 else 0
            }
            for channel in self.channels
        }
        
        self.attribution_results['shapley'] = attribution
        return pd.DataFrame([
            {'channel': ch, 'conversions': attr['conversions'], 
             'percentage': attr['percentage'], 'model': 'shapley'}
            for ch, attr in attribution.items()
        ])
    
    def compare_all_models(self) -> pd.DataFrame:
        """比较所有归因模型的结果"""
        self.first_click_attribution()
        self.last_click_attribution()
        self.linear_attribution()
        self.time_decay_attribution()
        self.shapley_value_attribution()
        
        # 合并所有结果
        all_results = []
        for model_name, attribution in self.attribution_results.items():
            for channel, metrics in attribution.items():
                all_results.append({
                    'channel': channel,
                    'model': model_name,
                    'conversions': metrics['conversions'],
                    'percentage': metrics['percentage']
                })
        
        return pd.DataFrame(all_results)
    
    def get_channel_contribution_matrix(self) -> pd.DataFrame:
        """获取渠道贡献矩阵（不同模型下的表现）"""
        df = self.compare_all_models()
        pivot = df.pivot(index='channel', columns='model', values='percentage')
        return pivot.fillna(0)


def create_sample_conversion_data(n_samples: int = 1000) -> pd.DataFrame:
    """创建示例转化数据"""
    np.random.seed(42)
    
    channels = ['google', 'facebook', 'email', 'direct', 'organic', 'paid_search']
    
    data = []
    for i in range(n_samples):
        # 随机生成客户路径（1-5 个触点）
        n_touchpoints = np.random.randint(1, 6)
        touchpoints = np.random.choice(channels, size=n_touchpoints, replace=True).tolist()
        
        # 转化价值（偏态分布）
        conversion_value = np.random.exponential(scale=100) + 10
        
        data.append({
            'customer_id': f'C{i:05d}',
            'touchpoints': touchpoints,
            'conversion_value': round(conversion_value, 2)
        })
    
    return pd.DataFrame(data)


if __name__ == '__main__':
    # 测试代码
    print("创建示例数据...")
    sample_data = create_sample_conversion_data(500)
    print(f"样本数量：{len(sample_data)}")
    
    print("\n运行归因分析...")
    analyzer = AttributionAnalyzer(sample_data)
    results = analyzer.compare_all_models()
    
    print("\n归因分析结果:")
    print(results.head(20))
    
    print("\n渠道贡献矩阵:")
    matrix = analyzer.get_channel_contribution_matrix()
    print(matrix)
