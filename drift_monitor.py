import numpy as np
import torch
from scipy.stats import ks_2samp

class DriftMonitor:
    def __init__(self, baseline_features):
        # baseline_features: Eğitim sırasındaki düğüm özellikleri [N, num_features]
        self.baseline = baseline_features.numpy() if isinstance(baseline_features, torch.Tensor) else baseline_features
        self.incoming_requests = []

    def log_request(self, feature_vector):
        self.incoming_requests.append(feature_vector)

    def check_drift(self, p_value_threshold=0.05):
        if len(self.incoming_requests) < 10:
            return {"status": "INSUFFICIENT_DATA", "message": "Drift analizi için en az 10 canlı istek gerekli."}

        incoming_array = np.array(self.incoming_requests)
        num_features = self.baseline.shape[1]
        drift_results = {}

        for i in range(num_features):
            stat, p_value = ks_2samp(self.baseline[:, i], incoming_array[:, i])
            drift_detected = p_value < p_value_threshold
            drift_results[f"feature_{i}"] = {
                "ks_stat": round(float(stat), 4),
                "p_value": round(float(p_value), 4),
                "drift_detected": bool(drift_detected)
            }

        has_any_drift = any(res["drift_detected"] for res in drift_results.values())
        return {
            "status": "DRIFT_DETECTED" if has_any_drift else "STABLE",
            "feature_details": drift_results
        }