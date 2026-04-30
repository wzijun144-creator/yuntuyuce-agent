#运行：streamlit run "网页端制作（6）.py"
# 网页端制作（6）.py - 集成MLP模型版本
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import h5py
import rasterio
import cv2
from skimage import measure
import torch
import torch.nn as nn
import warnings
import plotly.graph_objs as go
import plotly.express as px

# 忽略警告
warnings.filterwarnings("ignore")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 配置区域 ====================
CONFIG = {
    "target_channel": "NOMChannel01",
    "calibration_path": "CALIBRATION_COEF(SCALE+OFFSET)",
    "geo_transform": {"left_lon": 80.0, "top_lat": 50.0, "resolution": 0.02},
    "cdoc_thresholds": {"thin_cloud": 0.15, "thick_cloud": 0.45},
    "seq_len": 3,
    "future_step": 3,
    "pv_alert_threshold": 0.55,
    "mutation_threshold": 0.15,
    "mutation_alert_threshold": 0.2,
    "model_weight_path": "cloud_feature_mlp_model.pth",
    "feature_cols": [
        "thick_cloud_coverage", "thin_cloud_coverage", "cdoc_cloud_score",
        "cloud_area", "cloud_coverage", "cloud_mean_intensity",
        "cloud_std_intensity", "largest_cloud_area", "largest_cloud_perimeter",
        "cloud_count", "contrast", "homogeneity", "energy"
    ]
}


# ==================== 2. 模型定义 ====================
class CloudFeatureMLP(nn.Module):
    def __init__(self, input_dim):
        super(CloudFeatureMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.mutation_head = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.coverage_head = nn.Linear(32, 1)

    def forward(self, x):
        feat = self.network(x)
        prob = self.mutation_head(feat)
        cov = self.coverage_head(feat)
        return prob, cov


# ==================== 3. 特征提取工具 ====================
def setup_texture_features():
    texture_info = {'available': False, 'method': None}
    try:
        import cv2
        texture_info['available'] = True
        texture_info['method'] = 'opencv_alternative'
    except:
        pass
    return texture_info


TEXTURE_INFO = setup_texture_features()


def extract_features_from_hdf(hdf_path):
    """从HDF文件中提取特征"""
    try:
        # 从文件名提取时间信息
        filename = os.path.basename(hdf_path)
        # 尝试从文件名提取时间戳 (例如: 20241124102000)
        import re
        time_match = re.search(r'(\d{14})', filename)
        if time_match:
            dt = datetime.strptime(time_match.group(1), '%Y%m%d%H%M%S')
        else:
            # 如果无法从文件名提取，使用文件修改时间
            dt = datetime.fromtimestamp(os.path.getmtime(hdf_path))

        with h5py.File(hdf_path, 'r') as f:
            dn = f[CONFIG["target_channel"]][:]
            cal_coef = f[CONFIG["calibration_path"]][:]
            ch_idx = int(CONFIG["target_channel"].replace("NOMChannel", "")) - 1
            gain, bias = cal_coef[ch_idx]
            physical_data = gain * dn + bias

        features = {}
        # ... 原有的特征提取代码保持不变 ...

        # **关键：添加datetime字段！**
        features['datetime'] = dt
        features['filename'] = filename

        return True, features, physical_data
    except Exception as e:
        return False, str(e), None

def extract_features_from_hdf(hdf_path):
    """从HDF文件中提取特征"""
    try:
        with h5py.File(hdf_path, 'r') as f:
            dn = f[CONFIG["target_channel"]][:]
            cal_coef = f[CONFIG["calibration_path"]][:]
            ch_idx = int(CONFIG["target_channel"].replace("NOMChannel", "")) - 1
            gain, bias = cal_coef[ch_idx]
            physical_data = gain * dn + bias

        features = {}
        t_thin, t_thick = CONFIG["cdoc_thresholds"]["thin_cloud"], CONFIG["cdoc_thresholds"]["thick_cloud"]
        thick_mask = (physical_data >= t_thick)
        thin_mask = (physical_data >= t_thin) & (physical_data < t_thick)
        all_cloud_mask = (physical_data >= t_thin)
        total_pixels = physical_data.size

        features['thick_cloud_coverage'] = float(np.sum(thick_mask) / total_pixels)
        features['thin_cloud_coverage'] = float(np.sum(thin_mask) / total_pixels)
        features['cdoc_cloud_score'] = features['thick_cloud_coverage'] * 0.8 + features['thin_cloud_coverage'] * 0.2

        cloud_pixels = physical_data[all_cloud_mask]
        features['cloud_area'] = int(np.sum(all_cloud_mask))
        features['cloud_coverage'] = float(features['cloud_area'] / total_pixels)
        features['cloud_mean_intensity'] = float(np.mean(cloud_pixels) if len(cloud_pixels) > 0 else 0)
        features['cloud_std_intensity'] = float(np.std(cloud_pixels) if len(cloud_pixels) > 0 else 0)

        labeled = measure.label(all_cloud_mask)
        regions = measure.regionprops(labeled)
        if regions:
            largest = max(regions, key=lambda x: x.area)
            features['largest_cloud_area'] = int(largest.area)
            features['largest_cloud_perimeter'] = float(largest.perimeter)
            features['cloud_count'] = len(regions)
        else:
            features.update({'largest_cloud_area': 0, 'largest_cloud_perimeter': 0, 'cloud_count': 0})

        if features['cloud_area'] > 100:
            features.update(extract_texture(physical_data))

        return True, features, physical_data
    except Exception as e:
        return False, str(e), None


# ==================== 4. 加载模型和归一化参数 ====================
@st.cache_resource
def load_model_and_params():
    """加载模型和归一化参数"""
    try:
        # 加载归一化参数
        mean = np.load("train_mean.npy")
        std = np.load("train_std.npy")

        # 加载模型
        input_dim = len(CONFIG["feature_cols"]) * CONFIG["seq_len"]
        model = CloudFeatureMLP(input_dim)
        if os.path.exists(CONFIG["model_weight_path"]):
            model.load_state_dict(torch.load(CONFIG["model_weight_path"], map_location='cpu'))
            model.eval()
            return model, mean, std, True
        else:
            return None, None, None, False
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None, None, None, False


# ==================== 5. 预测函数 ====================
def predict_with_model(features_df, model, mean, std):
    """使用模型进行预测"""
    if len(features_df) < CONFIG["seq_len"] + CONFIG["future_step"]:
        return None, "数据不足，需要至少{}条记录".format(CONFIG["seq_len"] + CONFIG["future_step"])

    # 提取特征并归一化
    raw_features = features_df[CONFIG["feature_cols"]].values
    norm_features = (raw_features - mean) / std

    # 滑动窗口预测
    results = []
    with torch.no_grad():
        for i in range(len(norm_features) - CONFIG["seq_len"] - CONFIG["future_step"]):
            input_seq = norm_features[i: i + CONFIG["seq_len"]].flatten()
            input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)

            prob, cov = model(input_tensor)
            target_idx = i + CONFIG["seq_len"] + CONFIG["future_step"]

            results.append({
                'input_time_start': features_df.iloc[i]['datetime'],
                'input_time_end': features_df.iloc[i + CONFIG["seq_len"] - 1]['datetime'],
                'target_time': features_df.iloc[target_idx]['datetime'],
                'predicted_prob': prob.item(),
                'predicted_coverage': cov.item(),
                'actual_coverage': features_df.iloc[target_idx]['cloud_coverage'],
                'mutation_alert': prob.item() > CONFIG["mutation_alert_threshold"],
                'pv_alert': cov.item() > CONFIG["pv_alert_threshold"]
            })

    return results, "success"


# ==================== 6. 网页主界面 ====================
def main():
    st.set_page_config(
        page_title="风云卫星云图突变预警系统",
        page_icon="🛰️",
        layout="wide"
    )

    st.title("🛰️ 风云卫星云图突变预警系统")
    st.markdown("---")

    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 系统控制")

        # 数据来源选择
        data_source = st.radio(
            "数据来源",
            ["上传HDF文件", "选择文件夹", "模拟数据"]
        )

        if data_source == "上传HDF文件":
            uploaded_files = st.file_uploader(
                "选择HDF文件（可多选）",
                type=['hdf'],
                accept_multiple_files=True
            )
        elif data_source == "选择文件夹":
            folder_path = st.text_input("输入HDF文件夹路径",
                                        value=r"F:\风云卫星\测试环境")

        # 预警阈值设置
        st.subheader("预警阈值设置")
        mutation_threshold = st.slider("突变预警阈值", 0.0, 1.0,
                                       CONFIG["mutation_alert_threshold"], 0.05)
        pv_threshold = st.slider("光伏预警阈值", 0.0, 1.0,
                                 CONFIG["pv_alert_threshold"], 0.05)

        # 模型状态
        st.subheader("模型状态")
        model, mean, std, model_loaded = load_model_and_params()
        if model_loaded:
            st.success("✅ 模型已加载")
        else:
            st.error("❌ 模型加载失败，请检查模型文件")

        # 历史记录
        if st.button("📊 查看历史记录"):
            if os.path.exists("prediction_history.csv"):
                history_df = pd.read_csv("prediction_history.csv")
                st.dataframe(history_df.tail(10))
            else:
                st.info("暂无历史记录")

    # 主内容区
    if data_source == "模拟数据":
        # 生成模拟数据
        st.info("使用模拟数据演示")
        num_samples = st.slider("样本数量", 10, 100, 50)

        # 生成模拟特征数据
        np.random.seed(42)
        dates = pd.date_range(start=datetime.now() - timedelta(hours=num_samples),
                              periods=num_samples, freq='30min')

        sim_data = pd.DataFrame({
            'datetime': dates,
            'cloud_coverage': np.random.uniform(0.1, 0.9, num_samples),
            'thick_cloud_coverage': np.random.uniform(0, 0.5, num_samples),
            'thin_cloud_coverage': np.random.uniform(0, 0.4, num_samples),
            'cdoc_cloud_score': np.random.uniform(0, 0.8, num_samples),
            'cloud_area': np.random.randint(1000, 10000, num_samples),
            'cloud_mean_intensity': np.random.uniform(200, 300, num_samples),
            'cloud_std_intensity': np.random.uniform(5, 20, num_samples),
            'largest_cloud_area': np.random.randint(500, 5000, num_samples),
            'largest_cloud_perimeter': np.random.uniform(100, 1000, num_samples),
            'cloud_count': np.random.randint(1, 50, num_samples),
            'contrast': np.random.uniform(0.001, 0.01, num_samples),
            'homogeneity': np.random.uniform(0.7, 0.8, num_samples),
            'energy': np.random.uniform(0.3, 0.6, num_samples)
        })

        if model_loaded:
            results, msg = predict_with_model(sim_data, model, mean, std)
            if results:
                display_results(results, sim_data, mutation_threshold, pv_threshold)
            else:
                st.warning(msg)

    elif data_source == "上传HDF文件" and uploaded_files:
        # 处理上传的文件
        process_uploaded_files(uploaded_files, model, mean, std)

    elif data_source == "选择文件夹" and folder_path:
        # 处理文件夹
        process_folder(folder_path, model, mean, std)


def process_uploaded_files(uploaded_files, model, mean, std):
    """处理上传的HDF文件"""
    st.subheader("📊 正在处理上传的文件...")

    progress_bar = st.progress(0)
    status_text = st.empty()

    all_features = []
    for i, file in enumerate(uploaded_files):
        status_text.text(f"处理文件 {i + 1}/{len(uploaded_files)}: {file.name}")

        # 保存临时文件
        temp_path = f"temp_{file.name}"
        with open(temp_path, 'wb') as f:
            f.write(file.getbuffer())

        # 提取特征
        success, result, img_data = extract_features_from_hdf(temp_path)
        if success:
            result['filename'] = file.name
            result['datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            all_features.append(result)

        # 删除临时文件
        os.remove(temp_path)

        progress_bar.progress((i + 1) / len(uploaded_files))

    if all_features:
        features_df = pd.DataFrame(all_features)
        features_df['datetime'] = pd.to_datetime(features_df['datetime'])
        features_df = features_df.sort_values('datetime').reset_index(drop=True)

        st.success(f"✅ 成功处理 {len(features_df)} 个文件")

        if model and len(features_df) >= CONFIG["seq_len"] + CONFIG["future_step"]:
            results, msg = predict_with_model(features_df, model, mean, std)
            if results:
                display_results(results, features_df,
                                st.session_state.get('mutation_threshold', CONFIG["mutation_alert_threshold"]),
                                st.session_state.get('pv_threshold', CONFIG["pv_alert_threshold"]))
            else:
                st.warning(msg)
        else:
            st.warning("数据不足，无法进行预测")


def process_folder(folder_path, model, mean, std):
    """处理文件夹中的HDF文件"""
    st.subheader(f"📁 扫描文件夹: {folder_path}")

    if not os.path.exists(folder_path):
        st.error("文件夹不存在")
        return

    hdf_files = [f for f in os.listdir(folder_path) if f.upper().endswith('.HDF')]

    if not hdf_files:
        st.warning("未找到HDF文件")
        return

    st.info(f"找到 {len(hdf_files)} 个HDF文件")

    if st.button("开始处理"):
        process_files_batch(hdf_files, folder_path, model, mean, std)


def process_files_batch(hdf_files, folder_path, model, mean, std):
    """批量处理HDF文件"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    all_features = []
    for i, filename in enumerate(hdf_files[:50]):  # 限制处理数量
        status_text.text(f"处理文件 {i + 1}/{min(50, len(hdf_files))}: {filename}")

        file_path = os.path.join(folder_path, filename)
        success, result, img_data = extract_features_from_hdf(file_path)
        if success:
            result['filename'] = filename
            all_features.append(result)

        progress_bar.progress((i + 1) / min(50, len(hdf_files)))

    if all_features:
        features_df = pd.DataFrame(all_features)
        features_df['datetime'] = pd.to_datetime(features_df['datetime'] or datetime.now())
        features_df = features_df.sort_values('datetime').reset_index(drop=True)

        st.success(f"✅ 成功处理 {len(features_df)} 个文件")

        if model and len(features_df) >= CONFIG["seq_len"] + CONFIG["future_step"]:
            results, msg = predict_with_model(features_df, model, mean, std)
            if results:
                display_results(results, features_df,
                                st.session_state.get('mutation_threshold', CONFIG["mutation_alert_threshold"]),
                                st.session_state.get('pv_threshold', CONFIG["pv_alert_threshold"]))
            else:
                st.warning(msg)


def display_results(results, features_df, mutation_threshold, pv_threshold):
    """显示预测结果"""
    results_df = pd.DataFrame(results)

    # 第一行：关键指标
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        latest = results_df.iloc[-1] if len(results_df) > 0 else None
        if latest is not None:
            st.metric("最新突变概率", f"{latest['predicted_prob']:.2%}")

    with col2:
        alert_count = len(results_df[results_df['mutation_alert']])
        st.metric("预警次数", f"{alert_count}/{len(results_df)}")

    with col3:
        pv_alert_count = len(results_df[results_df['pv_alert']])
        st.metric("光伏预警", f"{pv_alert_count}次")

    with col4:
        avg_prob = results_df['predicted_prob'].mean()
        st.metric("平均概率", f"{avg_prob:.2%}")

    st.markdown("---")

    # 第二行：双图展示
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📈 突变概率时间序列")
        fig, ax = plt.subplots(figsize=(10, 5))

        x_labels = [r['target_time'].strftime('%H:%M') if hasattr(r['target_time'], 'strftime')
                    else str(r['target_time']) for r in results]
        x_range = range(len(results))

        ax.plot(x_range, results_df['predicted_prob'], 'o-',
                label='突变概率', color='#ff7f0e', linewidth=2)
        ax.axhline(y=mutation_threshold, color='red', linestyle='--',
                   label=f'预警阈值({mutation_threshold:.0%})')

        # 标记实际预警点
        alert_idx = [i for i, r in enumerate(results) if r['mutation_alert']]
        if alert_idx:
            ax.scatter(alert_idx, [results_df.iloc[i]['predicted_prob'] for i in alert_idx],
                       color='red', s=100, label='触发预警', zorder=5, marker='^')

        ax.set_xlabel('时间序列')
        ax.set_ylabel('突变概率')
        ax.set_title('云图突变概率实时监控')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 设置x轴标签
        step = max(1, len(x_range) // 10)
        ax.set_xticks(x_range[::step])
        ax.set_xticklabels(x_labels[::step], rotation=45)

        plt.tight_layout()
        st.pyplot(fig)

    with col_right:
        st.subheader("📊 预警级别分布")

        # 创建预警级别
        def get_level(prob):
            if prob >= 0.9:
                return '红色预警'
            elif prob >= 0.75:
                return '橙色预警'
            elif prob >= mutation_threshold:
                return '黄色预警'
            else:
                return '正常'

        results_df['warning_level'] = results_df['predicted_prob'].apply(get_level)
        level_counts = results_df['warning_level'].value_counts()

        colors = {'正常': '#2ecc71', '黄色预警': '#f1c40f',
                  '橙色预警': '#e67e22', '红色预警': '#e74c3c'}

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(level_counts.index, level_counts.values,
                      color=[colors.get(x, '#95a5a6') for x in level_counts.index])

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height)}', ha='center', va='bottom')

        ax.set_ylabel('次数')
        ax.set_title('预警级别分布')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")

    # 第三行：云量预测对比
    st.subheader("☁️ 云量预测与实际对比")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_range, results_df['predicted_coverage'], '--',
            label='预测云量', color='#d62728', linewidth=2)
    ax.plot(x_range, results_df['actual_coverage'], '-',
            label='实际云量', color='#1f77b4', linewidth=2)
    ax.axhline(y=pv_threshold, color='green', linestyle=':',
               label=f'光伏阈值({pv_threshold:.0%})')

    # 标记光伏预警点
    pv_idx = [i for i, r in enumerate(results) if r['pv_alert']]
    if pv_idx:
        ax.scatter(pv_idx, [results_df.iloc[i]['predicted_coverage'] for i in pv_idx],
                   color='gold', s=50, label='光伏预警', zorder=5)

    ax.set_xlabel('时间序列')
    ax.set_ylabel('云覆盖率')
    ax.set_title('云量预测与实际对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    step = max(1, len(x_range) // 10)
    ax.set_xticks(x_range[::step])
    ax.set_xticklabels(x_labels[::step], rotation=45)

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # 第四行：预测记录表格
    st.subheader("📋 详细预测记录")

    display_df = results_df[['target_time', 'predicted_prob', 'predicted_coverage',
                             'actual_coverage', 'warning_level']].copy()
    display_df['target_time'] = display_df['target_time'].apply(
        lambda x: x.strftime('%Y-%m-%d %H:%M') if hasattr(x, 'strftime') else str(x)
    )
    display_df.columns = ['预测时间', '突变概率', '预测云量', '实际云量', '预警级别']

    # 添加颜色标记
    def color_warning(val):
        if val == '红色预警':
            return 'background-color: #ffcccc'
        elif val == '橙色预警':
            return 'background-color: #ffe5cc'
        elif val == '黄色预警':
            return 'background-color: #ffffcc'
        else:
            return ''

    st.dataframe(
        display_df.tail(20).style.applymap(color_warning, subset=['预警级别']),
        use_container_width=True,
        hide_index=True
    )

    # 保存到历史记录
    save_to_history(results_df)


def save_to_history(results_df):
    """保存预测结果到历史记录"""
    try:
        history_file = "prediction_history.csv"

        # 准备数据
        history_data = []
        for _, row in results_df.iterrows():
            history_data.append({
                'timestamp': row['target_time'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(row['target_time'],
                                                                                         'strftime') else str(
                    row['target_time']),
                'predict_model_prob': row['predicted_prob'],
                'true_label': None,  # 需要用户反馈
                'model_version': 'mlp_seq3_v1',
                'diff_value': None,
                'current_cov': row['predicted_coverage'],
                'future_cov': row['actual_coverage'],
                'notes': '网页端预测'
            })

        new_df = pd.DataFrame(history_data)

        if os.path.exists(history_file):
            old_df = pd.read_csv(history_file)
            combined = pd.concat([old_df, new_df], ignore_index=True)
        else:
            combined = new_df

        combined.to_csv(history_file, index=False, encoding='utf-8-sig')
    except Exception as e:
        st.warning(f"历史记录保存失败: {e}")


# ==================== 运行应用 ====================
if __name__ == "__main__":
    main()