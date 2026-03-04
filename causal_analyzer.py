"""
因果分析模块
营销活动因果效应评估
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


class CausalAnalyzer:
    """因果效应分析器"""
    
    def __init__(self, data: pd.DataFrame):
        """
        初始化因果分析器
        
        Args:
            data: 数据，包含 columns:
                 - treated: 是否接受处理 (0/1)
                 - outcome: 结果变量
                 - covariates: 协变量（多个列）
        """
        self.data = data.copy()
        self.results = {}
    
    def naive_comparison(self, treatment_col: str = 'treated', 
                         outcome_col: str = 'outcome') -> Dict:
        """
        简单对比分析（有偏估计）
        
        Args:
            treatment_col: 处理变量列名
            outcome_col: 结果变量列名
        """
        treated = self.data[self.data[treatment_col] == 1][outcome_col]
        control = self.data[self.data[treatment_col] == 0][outcome_col]
        
        effect = treated.mean() - control.mean()
        t_stat, p_value = stats.ttest_ind(treated, control)
        
        result = {
            'treated_mean': treated.mean(),
            'treated_std': treated.std(),
            'treated_n': len(treated),
            'control_mean': control.mean(),
            'control_std': control.std(),
            'control_n': len(control),
            'naive_effect': effect,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        self.results['naive'] = result
        return result
    
    def regression_adjustment(self, treatment_col: str = 'treated',
                               outcome_col: str = 'outcome',
                               covariates: Optional[List[str]] = None) -> Dict:
        """
        回归调整法（控制混杂变量）
        
        Args:
            treatment_col: 处理变量列名
            outcome_col: 结果变量列名
            covariates: 协变量列表
        """
        if covariates is None:
            # 自动选择数值型协变量
            covariates = [col for col in self.data.columns 
                         if col not in [treatment_col, outcome_col] 
                         and self.data[col].dtype in ['int64', 'float64']]
        
        # 准备数据
        X = self.data[[treatment_col] + covariates].fillna(0)
        y = self.data[outcome_col]
        
        # 拟合回归
        model = LinearRegression()
        model.fit(X, y)
        
        # 处理效应 = 处理变量的系数
        treatment_effect = model.coef_[0]
        
        # 计算标准误和置信区间
        n = len(self.data)
        p = len(covariates) + 1
        y_pred = model.predict(X)
        residuals = y - y_pred
        mse = np.sum(residuals ** 2) / (n - p)
        
        # 系数方差
        XtX_inv = np.linalg.inv(X.T @ X)
        treatment_se = np.sqrt(mse * XtX_inv[0, 0])
        
        # t 检验
        t_stat = treatment_effect / treatment_se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - p))
        
        # 置信区间
        ci_lower = treatment_effect - 1.96 * treatment_se
        ci_upper = treatment_effect + 1.96 * treatment_se
        
        result = {
            'treatment_effect': treatment_effect,
            'standard_error': treatment_se,
            't_statistic': t_stat,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': p_value < 0.05,
            'r_squared': model.score(X, y),
            'covariates_used': covariates
        }
        
        self.results['regression'] = result
        return result
    
    def propensity_score_matching(self, treatment_col: str = 'treated',
                                   outcome_col: str = 'outcome',
                                   covariates: Optional[List[str]] = None,
                                   caliper: float = 0.1) -> Dict:
        """
        倾向得分匹配 (PSM)
        
        Args:
            treatment_col: 处理变量列名
            outcome_col: 结果变量列名
            covariates: 协变量列表
            caliper: 匹配容差
        """
        if covariates is None:
            covariates = [col for col in self.data.columns 
                         if col not in [treatment_col, outcome_col]
                         and self.data[col].dtype in ['int64', 'float64']]
        
        # 估计倾向得分
        X = self.data[covariates].fillna(0)
        y_treatment = self.data[treatment_col]
        
        ps_model = LogisticRegression(max_iter=1000)
        ps_model.fit(X, y_treatment)
        propensity_scores = ps_model.predict_proba(X)[:, 1]
        
        # 分离处理组和对照组
        treated_idx = self.data[self.data[treatment_col] == 1].index
        control_idx = self.data[self.data[treatment_col] == 0].index
        
        treated_scores = propensity_scores[self.data[treatment_col] == 1]
        control_scores = propensity_scores[self.data[treatment_col] == 0]
        
        # 最近邻匹配
        matched_pairs = []
        used_control = set()
        
        for i, t_idx in enumerate(treated_idx):
            t_score = treated_scores[i]
            
            # 找到未使用的对照组中倾向得分最近的
            best_match = None
            best_distance = float('inf')
            
            for j, c_idx in enumerate(control_idx):
                if c_idx not in used_control:
                    distance = abs(t_score - control_scores[j])
                    if distance < best_distance and distance < caliper:
                        best_distance = distance
                        best_match = c_idx
            
            if best_match is not None:
                matched_pairs.append((t_idx, best_match))
                used_control.add(best_match)
        
        # 计算处理效应
        if len(matched_pairs) > 0:
            treated_outcomes = self.data.loc[[p[0] for p in matched_pairs], outcome_col]
            control_outcomes = self.data.loc[[p[1] for p in matched_pairs], outcome_col]
            
            effects = treated_outcomes.values - control_outcomes.values
            ate = np.mean(effects)
            se = np.std(effects) / np.sqrt(len(effects))
            
            t_stat = ate / se if se > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(effects) - 1))
            
            result = {
                'n_matched_pairs': len(matched_pairs),
                'n_treated_original': len(treated_idx),
                'n_control_original': len(control_idx),
                'match_rate': len(matched_pairs) / len(treated_idx),
                'ate': ate,
                'standard_error': se,
                't_statistic': t_stat,
                'p_value': p_value,
                'ci_lower': ate - 1.96 * se,
                'ci_upper': ate + 1.96 * se,
                'significant': p_value < 0.05
            }
        else:
            result = {
                'n_matched_pairs': 0,
                'error': 'No matches found within caliper'
            }
        
        self.results['psm'] = result
        return result
    
    def diff_in_diff(self, treatment_col: str = 'treated',
                     outcome_col: str = 'outcome',
                     time_col: str = 'time',
                     pre_period: int = 0,
                     post_period: int = 1) -> Dict:
        """
        双重差分法 (DID)
        
        Args:
            treatment_col: 处理变量列名
            outcome_col: 结果变量列名
            time_col: 时间变量列名
            pre_period: 处理前时期标识
            post_period: 处理后时期标识
        """
        # 分离四组
        treated_pre = self.data[(self.data[treatment_col] == 1) & (self.data[time_col] == pre_period)][outcome_col].mean()
        treated_post = self.data[(self.data[treatment_col] == 1) & (self.data[time_col] == post_period)][outcome_col].mean()
        control_pre = self.data[(self.data[treatment_col] == 0) & (self.data[time_col] == pre_period)][outcome_col].mean()
        control_post = self.data[(self.data[treatment_col] == 0) & (self.data[time_col] == post_period)][outcome_col].mean()
        
        # DID 估计量
        did_effect = (treated_post - treated_pre) - (control_post - control_pre)
        
        # 回归方法验证
        self.data['post'] = (self.data[time_col] == post_period).astype(int)
        self.data['treated_post'] = self.data[treatment_col] * self.data['post']
        
        X = self.data[[treatment_col, 'post', 'treated_post']]
        y = self.data[outcome_col]
        
        model = LinearRegression()
        model.fit(X, y)
        
        did_coef = model.coef_[2]  # treated_post 的系数
        
        result = {
            'treated_pre_mean': treated_pre,
            'treated_post_mean': treated_post,
            'control_pre_mean': control_pre,
            'control_post_mean': control_post,
            'did_effect_simple': did_effect,
            'did_effect_regression': did_coef,
            'treated_trend': treated_post - treated_pre,
            'control_trend': control_post - control_pre,
            'interpretation': f"处理组相比对照组额外变化：{did_effect:.2f}"
        }
        
        self.results['did'] = result
        return result
    
    def heterogeneous_effects(self, treatment_col: str = 'treated',
                               outcome_col: str = 'outcome',
                               subgroup_col: str = None,
                               covariates: Optional[List[str]] = None) -> pd.DataFrame:
        """
        异质性处理效应分析
        
        Args:
            treatment_col: 处理变量列名
            outcome_col: 结果变量列名
            subgroup_col: 分组变量
            covariates: 协变量
        """
        if subgroup_col is None:
            # 使用中位数分组
            if covariates is None:
                covariates = [col for col in self.data.columns 
                             if col not in [treatment_col, outcome_col]
                             and self.data[col].dtype in ['int64', 'float64']]
            
            if not covariates:
                return pd.DataFrame()
            
            # 选择第一个协变量进行分组
            subgroup_col = covariates[0]
            median = self.data[subgroup_col].median()
            self.data['subgroup'] = (self.data[subgroup_col] >= median).astype(int)
            subgroup_col = 'subgroup'
        
        # 按组分别估计
        results = []
        for group in self.data[subgroup_col].unique():
            group_data = self.data[self.data[subgroup_col] == group]
            
            analyzer = CausalAnalyzer(group_data)
            effect = analyzer.regression_adjustment(treatment_col, outcome_col, covariates)
            
            results.append({
                'subgroup': group,
                'n_samples': len(group_data),
                'treatment_effect': effect.get('treatment_effect', 0),
                'standard_error': effect.get('standard_error', 0),
                'p_value': effect.get('p_value', 1),
                'significant': effect.get('significant', False)
            })
        
        return pd.DataFrame(results)
    
    def sensitivity_analysis(self, treatment_col: str = 'treated',
                              outcome_col: str = 'outcome',
                              covariates: Optional[List[str]] = None) -> Dict:
        """
        敏感性分析（评估未观测混杂的影响）
        
        Args:
            treatment_col: 处理变量列名
            outcome_col: 结果变量列名
            covariates: 协变量
        """
        base_result = self.regression_adjustment(treatment_col, outcome_col, covariates)
        base_effect = base_result['treatment_effect']
        
        # Rosenbaum 边界分析（简化版）
        gamma_values = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
        sensitivity = []
        
        for gamma in gamma_values:
            # 简化：假设未观测混杂最多使倾向得分变化 gamma 倍
            adjusted_se = base_result['standard_error'] * gamma
            z_score = base_effect / adjusted_se if adjusted_se > 0 else float('inf')
            p_value_adj = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            sensitivity.append({
                'gamma': gamma,
                'adjusted_se': adjusted_se,
                'z_score': z_score,
                'p_value_adjusted': p_value_adj,
                'still_significant': p_value_adj < 0.05
            })
        
        result = {
            'base_effect': base_effect,
            'base_p_value': base_result['p_value'],
            'sensitivity_analysis': pd.DataFrame(sensitivity),
            'robust_to_gamma': max([s['gamma'] for s in sensitivity if s['still_significant']], default=1.0)
        }
        
        self.results['sensitivity'] = result
        return result
    
    def get_full_analysis(self, treatment_col: str = 'treated',
                          outcome_col: str = 'outcome',
                          covariates: Optional[List[str]] = None) -> Dict:
        """执行完整的因果分析"""
        results = {
            'naive_comparison': self.naive_comparison(treatment_col, outcome_col),
            'regression_adjustment': self.regression_adjustment(treatment_col, outcome_col, covariates),
            'propensity_matching': self.propensity_score_matching(treatment_col, outcome_col, covariates),
        }
        
        # 汇总处理效应估计
        effects = []
        for method, result in results.items():
            if 'treatment_effect' in result or 'ate' in result or 'naive_effect' in result:
                effect = result.get('treatment_effect', result.get('ate', result.get('naive_effect', 0)))
                effects.append({
                    'method': method,
                    'effect': effect,
                    'significant': result.get('significant', False)
                })
        
        results['summary'] = pd.DataFrame(effects)
        return results


def create_sample_causal_data(n_samples: int = 1000, 
                               true_effect: float = 10.0) -> pd.DataFrame:
    """创建示例因果推断数据"""
    np.random.seed(42)
    
    # 协变量
    age = np.random.normal(40, 10, n_samples)
    income = np.random.normal(50000, 15000, n_samples)
    engagement = np.random.uniform(0, 1, n_samples)
    
    # 倾向得分（基于协变量）
    propensity = 1 / (1 + np.exp(-(0.05 * age + 0.0001 * income + 2 * engagement - 3)))
    
    # 处理分配
    treated = (np.random.uniform(0, 1, n_samples) < propensity).astype(int)
    
    # 结果（包含处理效应）
    baseline = 50 + 0.5 * age + 0.001 * income + 20 * engagement
    outcome = baseline + true_effect * treated + np.random.normal(0, 20, n_samples)
    
    data = pd.DataFrame({
        'treated': treated,
        'outcome': outcome,
        'age': age,
        'income': income,
        'engagement': engagement
    })
    
    return data


if __name__ == '__main__':
    print("创建示例因果数据...")
    data = create_sample_causal_data(1000, true_effect=10)
    print(f"数据形状：{data.shape}")
    
    print("\n执行因果分析...")
    analyzer = CausalAnalyzer(data)
    results = analyzer.get_full_analysis(
        treatment_col='treated',
        outcome_col='outcome',
        covariates=['age', 'income', 'engagement']
    )
    
    print("\n=== 简单对比 ===")
    print(f"处理组均值：{results['naive_comparison']['treated_mean']:.2f}")
    print(f"对照组均值：{results['naive_comparison']['control_mean']:.2f}")
    print(f"简单效应：{results['naive_comparison']['naive_effect']:.2f}")
    
    print("\n=== 回归调整 ===")
    print(f"处理效应：{results['regression_adjustment']['treatment_effect']:.2f}")
    print(f"95% CI: [{results['regression_adjustment']['ci_lower']:.2f}, {results['regression_adjustment']['ci_upper']:.2f}]")
    print(f"P 值：{results['regression_adjustment']['p_value']:.4f}")
    
    print("\n=== 倾向得分匹配 ===")
    if 'ate' in results['propensity_matching']:
        print(f"ATE: {results['propensity_matching']['ate']:.2f}")
        print(f"匹配对数：{results['propensity_matching']['n_matched_pairs']}")
    
    print("\n=== 方法对比 ===")
    print(results['summary'])
