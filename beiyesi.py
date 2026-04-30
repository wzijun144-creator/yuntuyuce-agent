import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class BayesianConfidenceEvaluator:
    """
    进一步优化后的混合贝叶斯置信度库
    本次调整：让高云量时更容易触发预警（适合光伏阻断场景）
    """

    def __init__(self, history_file="prediction_history.csv",
                 confidence_threshold=0.40,  # 【降低】0.45 → 0.40
                 bin_count=20):
        self.history_file = history_file
        self.confidence_threshold = confidence_threshold
        self.bin_count = bin_count
        self.likelihood_table = None
        self.prior_mutation = 0.0
        self.bins = None
        self._build_historical_database()

    def _build_historical_database(self):
        if not os.path.exists(self.history_file):
            print("❌ 历史文件不存在！")
            return

        df = pd.read_csv(self.history_file)
        time_col = 'datetime' if 'datetime' in df.columns else 'timestamp'
        df[time_col] = pd.to_datetime(df[time_col])
        if time_col == 'timestamp':
            df = df.rename(columns={'timestamp': 'datetime'})

        def label_mutation(row):
            if pd.notna(row.get('true_label')):
                return row['true_label']
            return 1 if row.get('diff_value', 0) > 0.15 else 0

        df['true_mutation'] = df.apply(label_mutation, axis=1)

        df['cov_bin'] = pd.cut(df['current_cov'], bins=self.bin_count, labels=False)

        self.prior_mutation = df['true_mutation'].mean()
        print(f"✅ 历史数据库构建完成！总样本 {len(df)} 条，先验 P(突变) = {self.prior_mutation:.4f}")

        self._build_likelihood_table(df)

    def _build_likelihood_table(self, df):
        table = pd.DataFrame()
        for b in range(self.bin_count):
            mask = df['cov_bin'] == b
            if mask.sum() == 0:
                continue
            p_m = df.loc[mask, 'true_mutation'].mean()
            p_nm = 1 - p_m
            table.loc[b, 'P(F|M)'] = p_m if p_m > 0 else 1e-6
            table.loc[b, 'P(F|~M)'] = p_nm if p_nm > 0 else 1e-6
        self.likelihood_table = table
        self.bins = np.linspace(0, 1, self.bin_count)

    def compute_posterior_confidence(self, current_cov: float, model_prob: float) -> float:
        bin_idx = np.digitize(current_cov, self.bins) - 1
        bin_idx = np.clip(bin_idx, 0, len(self.likelihood_table) - 1)
        row = self.likelihood_table.iloc[bin_idx]
        p_fm = row['P(F|M)']
        p_fnm = row['P(F|~M)']
        p_f = p_fm * self.prior_mutation + p_fnm * (1 - self.prior_mutation)
        posterior = (p_fm * self.prior_mutation) / (p_f + 1e-8)
        return min(max(posterior, 0.0), 1.0)

    def compute_hybrid_confidence(self, current_cov: float, predicted_cov: float, model_prob: float) -> float:
        """进一步激进版：更强调高云量风险"""
        # 云量风险分数：从0.45就开始上升，0.58以上接近1.0
        cov_risk = min(max((predicted_cov - 0.45) / 0.20, 0.0), 1.0)

        # 突变贝叶斯后验
        mutation_conf = self.compute_posterior_confidence(current_cov, model_prob)

        # 【提高云量权重到0.80】
        final_conf = 0.80 * cov_risk + 0.20 * mutation_conf
        return min(max(final_conf, 0.0), 1.0)

    def should_trigger_warning(self, current_cov: float, predicted_cov: float, model_prob: float) -> tuple:
        conf = self.compute_hybrid_confidence(current_cov, predicted_cov, model_prob)

        if conf >= self.confidence_threshold:
            decision = True
            note = f"置信度 {conf:.4f} ≥ {self.confidence_threshold} → 触发预警"
        else:
            decision = False
            note = f"置信度 {conf:.4f} < {self.confidence_threshold} → 过滤"

        self._log_error_report(current_cov, predicted_cov, model_prob, conf, decision)
        return decision, conf, note

    def _log_error_report(self, current_cov, predicted_cov, model_prob, conf, decision):
        log_file = "bayesian_error_report.csv"
        row = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_cov': current_cov,
            'predicted_cov': predicted_cov,
            'model_prob': model_prob,
            'bayesian_conf': conf,
            'final_decision': decision,
            'prior_mutation': self.prior_mutation
        }
        if os.path.exists(log_file):
            pd.DataFrame([row]).to_csv(log_file, mode='a', header=False, index=False)
        else:
            pd.DataFrame([row]).to_csv(log_file, index=False)

    def get_false_alarm_rate(self) -> dict:
        if not os.path.exists("bayesian_error_report.csv"):
            return {"msg": "暂无日志"}
        df = pd.read_csv("bayesian_error_report.csv")
        total = len(df)
        fp = ((df['final_decision'] == True) & (df['bayesian_conf'] < 0.40)).sum()
        fn = ((df['final_decision'] == False) & (df['bayesian_conf'] > 0.40)).sum()
        return {
            "总样本": total,
            "假阳性率": round(fp / total * 100, 2) if total else 0,
            "假阴性率": round(fn / total * 100, 2) if total else 0,
            "整体误报率": round((fp + fn) / total * 100, 2) if total else 0
        }
