"""
ROI 预测模块
基于历史的 ROI 预测
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class ROIPredictor:
    """ROI 预测器"""
    
    def __init__(self, historical_data: pd.DataFrame):
        """
        初始化 ROI 预测器
        
        Args:
            historical_data: 历史数据，包含 columns:
                            - date: 日期
                            - channel: 渠道（可选，用于分渠道预测）
                            - spend: 支出
                            - conversions: 转化数
                            - revenue: 收入
                            - 其他特征变量
        """
        self.data = historical_data.copy()
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.sort_values('date').reset_index(drop=True)
        
        self.models = {}
        self.scalers = {}
        self.results = {}
    
    def prepare_features(self, feature_columns: Optional[List[str]] = None,
                         include_lags: bool = True,
                         n_lags: int = 3,
                         include_rolling: bool = True,
                         rolling_window: int = 7) -> pd.DataFrame:
        """
        准备预测特征
        
        Args:
            feature_columns: 特征列列表
            include_lags: 是否包含滞后特征
            n_lags: 滞后阶数
            include_rolling: 是否包含滚动统计
            rolling_window: 滚动窗口大小
        """
        df = self.data.copy()
        
        # 自动识别特征列
        if feature_columns is None:
            exclude_cols = ['date', 'channel', 'roi', 'target']
            feature_columns = [col for col in df.columns 
                             if col not in exclude_cols 
                             and df[col].dtype in ['int64', 'float64']]
        
        # 添加滞后特征
        if include_lags:
            for col in feature_columns:
                if col in df.columns:
                    for lag in range(1, n_lags + 1):
                        df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            
            # ROI 滞后
            if 'roi' in df.columns:
                for lag in range(1, n_lags + 1):
                    df[f'roi_lag_{lag}'] = df['roi'].shift(lag)
        
        # 添加滚动统计
        if include_rolling:
            for col in ['spend', 'revenue', 'roi']:
                if col in df.columns:
                    df[f'{col}_rolling_mean_{rolling_window}'] = df[col].rolling(rolling_window).mean()
                    df[f'{col}_rolling_std_{rolling_window}'] = df[col].rolling(rolling_window).std()
        
        # 添加时间特征
        if 'date' in df.columns:
            df['dayofweek'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['dayofmonth'] = df['date'].dt.day
            df['quarter'] = df['date'].dt.quarter
            df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
        
        # 删除 NaN 行
        df = df.dropna().reset_index(drop=True)
        
        return df
    
    def create_target(self, target_type: str = 'roi', 
                      forecast_horizon: int = 1) -> pd.DataFrame:
        """
        创建预测目标
        
        Args:
            target_type: 目标类型 ('roi', 'revenue', 'conversions')
            forecast_horizon: 预测提前期
        """
        df = self.data.copy()
        
        if target_type == 'roi':
            if 'roi' not in df.columns:
                if 'revenue' in df.columns and 'spend' in df.columns:
                    df['roi'] = (df['revenue'] - df['spend']) / df['spend']
                else:
                    raise ValueError("无法计算 ROI，需要 revenue 和 spend 列")
        
        # 创建前向目标
        df[f'target_{target_type}'] = df[target_type].shift(-forecast_horizon)
        df = df.dropna()
        
        return df
    
    def train_model(self, model_type: str = 'ridge',
                    feature_columns: Optional[List[str]] = None,
                    target_column: str = 'target_roi',
                    test_size: float = 0.2,
                    **model_params) -> Dict:
        """
        训练预测模型
        
        Args:
            model_type: 模型类型 ('linear', 'ridge', 'lasso', 'random_forest', 'gradient_boosting')
            feature_columns: 特征列
            target_column: 目标列
            test_size: 测试集比例
            **model_params: 模型参数
        """
        # 准备数据
        df = self.prepare_features(feature_columns)
        
        if target_column not in df.columns:
            df = self.create_target()
            target_column = 'target_roi'
        
        # 分离特征和目标
        exclude_cols = ['date', 'channel', 'roi', 'spend', 'revenue', 'conversions', 
                       'target_roi', 'target_revenue', 'target_conversions']
        X_cols = [col for col in df.columns if col not in exclude_cols and col != target_column]
        
        X = df[X_cols].fillna(0)
        y = df[target_column]
        
        # 训练集测试集分割（时间序列）
        split_idx = int(len(df) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 选择模型
        if model_type == 'linear':
            model = LinearRegression(**model_params)
        elif model_type == 'ridge':
            model = Ridge(**model_params)
        elif model_type == 'lasso':
            model = Lasso(**model_params)
        elif model_type == 'random_forest':
            model = RandomForestRegressor(**model_params)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(**model_params)
        else:
            raise ValueError(f"未知模型类型：{model_type}")
        
        # 训练
        model.fit(X_train_scaled, y_train)
        
        # 预测
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # 评估
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'test_r2': r2_score(y_test, y_pred_test),
            'n_train': len(y_train),
            'n_test': len(y_test)
        }
        
        # 特征重要性
        if hasattr(model, 'coef_'):
            feature_importance = pd.DataFrame({
                'feature': X_cols,
                'importance': model.coef_,
                'abs_importance': np.abs(model.coef_)
            }).sort_values('abs_importance', ascending=False)
        elif hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            feature_importance = None
        
        # 存储模型
        model_name = f'{model_type}_{target_column}'
        self.models[model_name] = model
        self.scalers[model_name] = scaler
        self.results[model_name] = {
            'metrics': metrics,
            'feature_importance': feature_importance,
            'feature_columns': X_cols
        }
        
        return self.results[model_name]
    
    def predict(self, future_features: pd.DataFrame, 
                model_type: str = 'ridge',
                target_column: str = 'target_roi') -> pd.DataFrame:
        """
        预测未来 ROI
        
        Args:
            future_features: 未来特征数据
            model_type: 使用的模型
            target_column: 目标列
        """
        model_name = f'{model_type}_{target_column}'
        
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 未训练，请先调用 train_model")
        
        model = self.models[model_name]
        scaler = self.scalers[model_name]
        feature_columns = self.results[model_name]['feature_columns']
        
        # 准备特征
        X = future_features[feature_columns].fillna(0)
        X_scaled = scaler.transform(X)
        
        # 预测
        predictions = model.predict(X_scaled)
        
        result = future_features.copy()
        result['predicted_roi'] = predictions
        
        return result
    
    def forecast_roi(self, n_periods: int = 30,
                     spend_scenario: Optional[Dict[str, float]] = None,
                     model_type: str = 'ridge') -> pd.DataFrame:
        """
        预测未来 ROI
        
        Args:
            n_periods: 预测期数
            spend_scenario: 支出场景（可选）
            model_type: 模型类型
        """
        # 获取最后一条数据作为基础
        last_row = self.data.iloc[-1:].copy()
        
        forecasts = []
        current_data = self.data.copy()
        
        for i in range(n_periods):
            # 创建未来日期
            future_date = last_row['date'].iloc[0] + pd.Timedelta(days=i+1) if 'date' in last_row.columns else i
            
            # 准备特征
            future_features = self.prepare_features()
            
            # 使用最后已知值或场景
            if spend_scenario and 'spend' in spend_scenario:
                future_spend = spend_scenario['spend']
            else:
                future_spend = current_data['spend'].iloc[-1]
            
            # 简单预测（使用最近均值）
            predicted_roi = current_data['roi'].iloc[-min(7, len(current_data)):].mean()
            
            forecasts.append({
                'period': i + 1,
                'date': future_date,
                'predicted_roi': predicted_roi,
                'spend': future_spend,
                'predicted_revenue': future_spend * (1 + predicted_roi)
            })
        
        return pd.DataFrame(forecasts)
    
    def scenario_analysis(self, spend_scenarios: Dict[str, float],
                          model_type: str = 'ridge') -> pd.DataFrame:
        """
        支出场景分析
        
        Args:
            spend_scenarios: 支出场景字典，如 {'low': 1000, 'base': 5000, 'high': 10000}
            model_type: 模型类型
        """
        results = []
        
        for scenario_name, spend_amount in spend_scenarios.items():
            forecast = self.forecast_roi(
                n_periods=30,
                spend_scenario={'spend': spend_amount},
                model_type=model_type
            )
            
            results.append({
                'scenario': scenario_name,
                'spend': spend_amount,
                'avg_predicted_roi': forecast['predicted_roi'].mean(),
                'total_predicted_revenue': forecast['predicted_revenue'].sum(),
                'total_profit': forecast['predicted_revenue'].sum() - spend_amount * 30
            })
        
        return pd.DataFrame(results)
    
    def get_model_comparison(self) -> pd.DataFrame:
        """比较所有训练过的模型"""
        comparisons = []
        
        for model_name, result in self.results.items():
            metrics = result['metrics']
            comparisons.append({
                'model': model_name,
                'train_r2': metrics['train_r2'],
                'test_r2': metrics['test_r2'],
                'train_rmse': metrics['train_rmse'],
                'test_rmse': metrics['test_rmse'],
                'test_mae': metrics['test_mae']
            })
        
        return pd.DataFrame(comparisons)
    
    def get_prediction_intervals(self, future_features: pd.DataFrame,
                                  model_type: str = 'ridge',
                                  confidence: float = 0.95) -> pd.DataFrame:
        """
        计算预测区间
        
        Args:
            future_features: 未来特征
            model_type: 模型类型
            confidence: 置信水平
        """
        predictions = self.predict(future_features, model_type)
        
        # 计算残差标准差
        model_name = f'{model_type}_target_roi'
        if model_name in self.results:
            residuals_std = self.results[model_name]['metrics']['test_rmse']
        else:
            residuals_std = 1.0
        
        # Z 值
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        predictions['roi_lower'] = predictions['predicted_roi'] - z_score * residuals_std
        predictions['roi_upper'] = predictions['predicted_roi'] + z_score * residuals_std
        
        return predictions


def create_sample_historical_data(n_days: int = 365) -> pd.DataFrame:
    """创建示例历史数据"""
    np.random.seed(42)
    
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq='D')
    
    # 基础 ROI 随时间有趋势和季节性
    trend = np.linspace(1.5, 2.0, n_days)
    seasonality = 0.3 * np.sin(2 * np.pi * np.arange(n_days) / 365)
    noise = np.random.normal(0, 0.2, n_days)
    
    roi = trend + seasonality + noise
    
    # 支出
    spend = np.random.exponential(5000, n_days)
    
    # 收入
    revenue = spend * (1 + roi)
    
    # 转化
    conversions = (revenue / 100).astype(int)
    
    data = pd.DataFrame({
        'date': dates,
        'spend': spend,
        'revenue': revenue,
        'conversions': conversions,
        'roi': roi
    })
    
    return data


if __name__ == '__main__':
    print("创建示例历史数据...")
    data = create_sample_historical_data(200)
    print(f"数据形状：{data.shape}")
    
    print("\n训练预测模型...")
    predictor = ROIPredictor(data)
    
    # 训练多个模型
    models = ['linear', 'ridge', 'random_forest']
    for model_type in models:
        print(f"\n训练 {model_type} 模型...")
        result = predictor.train_model(
            model_type=model_type,
            n_estimators=100 if model_type == 'random_forest' else {}
        )
        print(f"测试 R²: {result['metrics']['test_r2']:.3f}")
        print(f"测试 RMSE: {result['metrics']['test_rmse']:.3f}")
    
    print("\n模型比较:")
    comparison = predictor.get_model_comparison()
    print(comparison)
    
    print("\nROI 预测（未来 7 天）:")
    forecast = predictor.forecast_roi(n_periods=7)
    print(forecast)
