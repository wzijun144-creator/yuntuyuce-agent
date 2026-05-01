"""
power_prediction.py
云量对光伏发电功率的影响建模与功率预测模块
专为风云四号卫星云图预警Agent设计
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pvlib
from pvlib import location, clearsky
import warnings

warnings.filterwarnings('ignore')


def calculate_clear_sky_ghi(latitude: float, longitude: float, altitude: float,
                            timestamp, linke_turbidity: float = 3.5):
    """使用Ineichen模型计算晴空全球水平辐照度，timestamp默认为北京时间"""
    if not isinstance(timestamp, pd.Timestamp):
        timestamp = pd.Timestamp(timestamp)
    if timestamp.tz is None:
        # 假设无时区的时间均为北京时间
        timestamp = timestamp.tz_localize('Asia/Shanghai')
    else:
        timestamp = timestamp.tz_convert('Asia/Shanghai')

    site = location.Location(latitude=latitude, longitude=longitude,
                             altitude=altitude, tz='Asia/Shanghai')

    times = pd.DatetimeIndex([timestamp])
    clear_sky = site.get_clearsky(times, model='ineichen',
                                  linke_turbidity=linke_turbidity)
    return float(clear_sky['ghi'].iloc[0])


def calculate_cloud_modification_factor(thick_cloud: float, thin_cloud: float,
                                        cloud_score: float = None):
    """计算云衰减因子（CMF）——推荐加权公式"""
    # 厚云衰减系数更高
    cmf = 1.0 - 0.92 * thick_cloud - 0.55 * thin_cloud
    return np.clip(cmf, 0.05, 1.0)


def predict_pv_power(thick_cloud: float, thin_cloud: float, cloud_score: float,
                     latitude: float, longitude: float, altitude: float,
                     capacity_kwp: float, pr: float = 0.78,
                     timestamp: datetime = None,
                     module_temp: float = 35.0):
    """
    完整光伏功率预测
    返回字典，包含所有中间结果，便于分析和可视化
    """
    if timestamp is None:
        timestamp = datetime.now()

    # 确保时间戳带有时区信息，并转换为北京时间
    ts = pd.Timestamp(timestamp)
    if ts.tz is None:
        ts = ts.tz_localize('Asia/Shanghai')
    else:
        ts = ts.tz_convert('Asia/Shanghai')

    ghi_clear = calculate_clear_sky_ghi(latitude, longitude, altitude, ts)
    cmf = calculate_cloud_modification_factor(thick_cloud, thin_cloud, cloud_score)
    ghi_pred = ghi_clear * cmf

    # 温度修正系数（可选）
    gamma = -0.004  # 典型值 -0.4%/℃
    f_temp = 1 + gamma * (module_temp - 25.0)

    power_pred_kw = ghi_pred * capacity_kwp * pr * f_temp / 1000.0

    return {
        'timestamp': ts,
        'ghi_clear': round(ghi_clear, 2),
        'cmf': round(cmf, 4),
        'ghi_pred': round(ghi_pred, 2),
        'power_pred_kw': round(max(0.0, power_pred_kw), 3),
        'capacity_kwp': capacity_kwp,
        'pr': pr,
        'thick_cloud': round(thick_cloud, 4),
        'thin_cloud': round(thin_cloud, 4),
        'cloud_score': round(cloud_score, 4)
    }


def calculate_ramp_rate(current_power: float, future_power: float, capacity_kwp: float):
    """计算功率变化率（用于突变预警）"""
    if capacity_kwp <= 0:
        return 0.0
    ramp = abs(future_power - current_power) / capacity_kwp
    return ramp