"""
Simple anomaly detection for financial data using statistical methods.
This is a POC implementation using z-scores and IQR methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime


class SimpleAnomalyDetector:
    """Detect anomalies in financial data using statistical methods"""

    def __init__(self, sensitivity: float = 2.0):
        """
        Initialize detector with sensitivity level.

        Args:
            sensitivity: Z-score threshold for anomaly detection (default 2.0)
                        Lower values = more sensitive (more anomalies detected)
        """
        self.sensitivity = sensitivity

    async def detect_anomalies(
        self, data: List[Dict[str, Any]], metric: str, method: str = "zscore"
    ) -> Dict[str, Any]:
        """
        Detect anomalies in financial data.

        Args:
            data: List of dicts with 'date' and metric values
            metric: The metric to analyze (e.g., 'revenue', 'expenses')
            method: Detection method ('zscore' or 'iqr')

        Returns:
            Dict with anomalies and analysis results
        """
        if not data or len(data) < 3:
            return {
                "error": "Insufficient data. Need at least 3 data points.",
                "anomalies": [],
            }

        # Convert to DataFrame
        df = pd.DataFrame(data)
        if "date" not in df.columns or metric not in df.columns:
            return {
                "error": f"Data must contain 'date' and '{metric}' columns",
                "anomalies": [],
            }

        # Ensure date is datetime and sort
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Apply detection method
        if method == "zscore":
            anomalies = self._detect_zscore(df, metric)
        elif method == "iqr":
            anomalies = self._detect_iqr(df, metric)
        else:
            anomalies = self._detect_zscore(df, metric)  # Default to z-score

        # Calculate statistics
        mean_value = df[metric].mean()
        std_value = df[metric].std()
        median_value = df[metric].median()

        # Find significant changes
        df["pct_change"] = df[metric].pct_change()
        significant_changes = df[df["pct_change"].abs() > 0.3]  # 30% change threshold

        # Prepare results
        anomaly_records = []
        for idx in anomalies:
            row = df.iloc[idx]

            # Calculate why it's anomalous
            z_score = (row[metric] - mean_value) / std_value if std_value > 0 else 0
            deviation_pct = (
                ((row[metric] - mean_value) / mean_value * 100)
                if mean_value != 0
                else 0
            )

            # Generate explanation
            if row[metric] > mean_value + 2 * std_value:
                reason = f"Unusually high - {abs(deviation_pct):.1f}% above average"
            elif row[metric] < mean_value - 2 * std_value:
                reason = f"Unusually low - {abs(deviation_pct):.1f}% below average"
            elif idx > 0 and abs(row["pct_change"]) > 0.3:
                change_dir = "increase" if row["pct_change"] > 0 else "decrease"
                reason = f"Sudden {change_dir} of {abs(row['pct_change']*100):.1f}%"
            else:
                reason = "Statistical outlier"

            anomaly_records.append(
                {
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "value": float(row[metric]),
                    "z_score": float(z_score),
                    "deviation_percent": float(deviation_pct),
                    "reason": reason,
                    "severity": self._calculate_severity(z_score),
                }
            )

        # Overall assessment
        anomaly_rate = len(anomalies) / len(df) * 100
        if anomaly_rate > 20:
            data_quality = "High volatility - many anomalies detected"
        elif anomaly_rate > 10:
            data_quality = "Moderate volatility - some unusual patterns"
        else:
            data_quality = "Normal - few anomalies detected"

        return {
            "metric": metric,
            "total_records": len(df),
            "anomaly_count": len(anomalies),
            "anomaly_rate": f"{anomaly_rate:.1f}%",
            "anomalies": anomaly_records,
            "statistics": {
                "mean": float(mean_value),
                "median": float(median_value),
                "std_dev": float(std_value),
                "min": float(df[metric].min()),
                "max": float(df[metric].max()),
            },
            "data_quality": data_quality,
            "detection_method": method,
            "sensitivity": self.sensitivity,
        }

    def _detect_zscore(self, df: pd.DataFrame, metric: str) -> List[int]:
        """Detect anomalies using z-score method"""
        mean = df[metric].mean()
        std = df[metric].std()

        if std == 0:
            return []

        z_scores = np.abs((df[metric] - mean) / std)
        anomaly_indices = df[z_scores > self.sensitivity].index.tolist()

        # Also check for sudden changes
        if len(df) > 1:
            pct_changes = df[metric].pct_change().abs()
            sudden_changes = df[pct_changes > 0.5].index.tolist()  # 50% change
            anomaly_indices = list(set(anomaly_indices + sudden_changes))

        return sorted(anomaly_indices)

    def _detect_iqr(self, df: pd.DataFrame, metric: str) -> List[int]:
        """Detect anomalies using Interquartile Range method"""
        Q1 = df[metric].quantile(0.25)
        Q3 = df[metric].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        anomaly_indices = df[
            (df[metric] < lower_bound) | (df[metric] > upper_bound)
        ].index.tolist()
        return anomaly_indices

    def _calculate_severity(self, z_score: float) -> str:
        """Calculate anomaly severity based on z-score"""
        abs_z = abs(z_score)
        if abs_z > 3:
            return "high"
        elif abs_z > 2.5:
            return "medium"
        elif abs_z > 2:
            return "low"
        else:
            return "minimal"
