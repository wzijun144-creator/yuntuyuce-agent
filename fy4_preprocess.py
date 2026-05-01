import h5py
import numpy as np
import rasterio
from rasterio.transform import from_origin
import cv2
from skimage import measure
import json
import os
import glob
import re
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import warnings

# 忽略无关警告
warnings.filterwarnings("ignore")

# ==================== 1. 配置区域 ====================
CONFIG = {
    "root_folder": r"F:\FYDATA\云图数据",
    "save_dir": r"F:\FYDATA\处理后数据",
    "target_channel": "NOMChannel01",
    "calibration_path": "CALIBRATION_COEF(SCALE+OFFSET)",
    "geo_transform": {
        "left_lon": 80.0,
        "top_lat": 50.0,
        "resolution": 0.02
    },
    "cdoc_thresholds": {
        "thin_cloud": 0.15,
        "thick_cloud": 0.45
    },
    # 删除了筛选逻辑，但保留参数配置供参考
    "low_coverage_threshold": 0.0
}


# ==================== 2. 纹理特征环境检查 ====================
def setup_texture_features():
    texture_info = {'available': False, 'method': None}
    try:
        import cv2
        texture_info['available'] = True
        texture_info['method'] = 'opencv_alternative'
    except ImportError:
        try:
            from skimage.feature import local_binary_pattern
            texture_info['available'] = True
            texture_info['method'] = 'local_binary_pattern'
        except:
            pass
    return texture_info


TEXTURE_INFO = setup_texture_features()


class FY4IntegratedProcessor:
    def __init__(self, config):
        self.config = config
        if not os.path.exists(self.config["save_dir"]):
            os.makedirs(self.config["save_dir"], exist_ok=True)

        ch_num = int(re.search(r'\d+', self.config["target_channel"]).group())
        self.band_type = 'VIS' if ch_num <= 6 else 'IR'

    def convert_to_serializable(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return obj

    def extract_time_info(self, filename, file_path):
        matches = re.findall(r'(\d{14})', filename)
        if matches:
            try:
                dt = datetime.strptime(matches[0], '%Y%m%d%H%M%S')
                return dt.strftime('%Y%m%d_%H%M%S'), dt
            except:
                pass
        dt = datetime.fromtimestamp(os.path.getmtime(file_path))
        return dt.strftime('%Y%m%d_%H%M%S'), dt

    def extract_texture(self, data):
        feats = {'contrast': 0.0, 'homogeneity': 0.0, 'energy': 0.0}
        try:
            norm = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
            img_u8 = (norm * 255).astype(np.uint8)
            if TEXTURE_INFO['method'] == 'opencv_alternative':
                edges = cv2.Canny(img_u8, 50, 150)
                feats['contrast'] = float(np.mean(edges) / 255.0)
                _, std = cv2.meanStdDev(img_u8)
                feats['homogeneity'] = float(1.0 / (1.0 + std[0][0] / 255.0))
                hist = cv2.calcHist([img_u8], [0], None, [256], [0, 256])
                feats['energy'] = float(np.sum((hist / np.sum(hist)) ** 2))
        except:
            pass
        return feats

    def extract_features_from_memory(self, physical_data, file_info):
        features = {}

        t_thin = self.config["cdoc_thresholds"]["thin_cloud"]
        t_thick = self.config["cdoc_thresholds"]["thick_cloud"]

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
            features.update(self.extract_texture(physical_data))

        features.update(file_info)
        return features

    def process_pipeline(self, hdf_path):
        try:
            fname = os.path.basename(hdf_path)
            time_str, dt_obj = self.extract_time_info(fname, hdf_path)

            with h5py.File(hdf_path, 'r') as f:
                dn = f[self.config["target_channel"]][:]
                cal_coef = f[self.config["calibration_path"]][:]
                ch_idx = int(self.config["target_channel"].replace("NOMChannel", "")) - 1
                gain, bias = cal_coef[ch_idx]
                physical_data = gain * dn + bias

            safe_name = f"FY4A_CH{ch_idx + 1:02d}_{time_str}"
            tif_path = os.path.join(self.config["save_dir"], f"{safe_name}.tif")
            geo = self.config["geo_transform"]
            transform = from_origin(geo["left_lon"], geo["top_lat"], geo["resolution"], -geo["resolution"])

            with rasterio.open(
                    tif_path, "w", driver="GTiff",
                    height=physical_data.shape[0], width=physical_data.shape[1],
                    count=1, dtype=physical_data.dtype,
                    crs="EPSG:4326", transform=transform
            ) as dst:
                dst.write(physical_data, 1)

            file_info = {'filename': safe_name, 'datetime': dt_obj.isoformat(), 'original_file': fname}
            features = self.extract_features_from_memory(physical_data, file_info)

            json_path = os.path.join(self.config["save_dir"], f"{safe_name}_features.json")
            with open(json_path, 'w', encoding='utf-8') as jf:
                serializable_feats = {k: self.convert_to_serializable(v) for k, v in features.items()}
                json.dump(serializable_feats, jf, indent=2, ensure_ascii=False)

            return True, serializable_feats
        except Exception as e:
            return False, str(e)


# ==================== 3. 执行逻辑 ====================
def main():
    root_dir = CONFIG["root_folder"]
    files = []

    for root, dirs, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.upper().endswith('.HDF'):
                files.append(os.path.join(root, filename))

    if not files:
        print(f"在 {root_dir} 及其子文件夹中未找到任何 HDF 文件，请检查路径。")
        return

    print(f"共找到 {len(files)} 个 HDF 数据文件，准备开始批量处理...")

    processor = FY4IntegratedProcessor(CONFIG)
    results = []

    for hdf_file in tqdm(files, desc="Processing"):
        success, res = processor.process_pipeline(hdf_file)
        if success:
            results.append(res)
        else:
            print(f"文件处理失败: {hdf_file}, 错误: {res}")

    if results:
        df = pd.DataFrame(results)

        # 确保 datetime 是 datetime 类型，并按时间严格排序以保证序列连贯性
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)

        # --- 修改点：删除了低覆盖率过滤逻辑，保留所有数据 ---
        # 即使云量很少，也会记录在案，以维持时间轴的完整

        csv_path = os.path.join(CONFIG["save_dir"], "cdoc_integrated_summary.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8_sig')

        print(f"\n全部子文件夹处理完成！")
        print(f"完整时间序列汇总表已保存至: {csv_path}")
        print(f"总计保留帧数: {len(df)} 条，已按时间轴升序排列")


if __name__ == "__main__":
    main()