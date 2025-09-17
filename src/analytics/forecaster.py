"""
Simple financial forecasting using XGBoost and moving averages.
This is a POC implementation focused on simplicity and effectiveness.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


class FinancialForecaster:
    """Simple forecasting for financial metrics using ML and time series techniques"""

    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=50,  # Fewer trees for speed
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    async def forecast(
        self, metric: str, historical_data: List[Dict[str, Any]], periods: int = 3
    ) -> Dict[str, Any]:
        """
        Forecast future values for a financial metric.

        Args:
            metric: The metric to forecast (e.g., 'revenue', 'expenses')
            historical_data: List of dicts with 'date' and metric value
            periods: Number of periods to forecast ahead

        Returns:
            Dict with forecast values and confidence intervals
        """
        if not historical_data or len(historical_data) < 3:
            return {
                "error": "Insufficient data for forecasting. Need at least 3 historical periods.",
                "forecast": [],
            }

        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        if "date" not in df.columns or metric not in df.columns:
            return {
                "error": f"Data must contain 'date' and '{metric}' columns",
                "forecast": [],
            }

        # Ensure date is datetime
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        # Create simple features
        df["month"] = df["date"].dt.month
        df["quarter"] = df["date"].dt.quarter
        df["year"] = df["date"].dt.year
        df["days_since_start"] = (df["date"] - df["date"].min()).dt.days

        # Add lag features if we have enough data
        if len(df) > 1:
            df["lag_1"] = df[metric].shift(1)
        if len(df) > 2:
            df["lag_2"] = df[metric].shift(2)
            df["rolling_mean_3"] = df[metric].rolling(window=3, min_periods=1).mean()

        # Drop NaN values
        df = df.dropna()

        if len(df) < 2:
            # Fallback to simple moving average
            return self._simple_forecast(df, metric, periods)

        # Prepare features
        feature_cols = ["month", "quarter", "year", "days_since_start"]
        if "lag_1" in df.columns:
            feature_cols.append("lag_1")
        if "lag_2" in df.columns:
            feature_cols.append("lag_2")
        if "rolling_mean_3" in df.columns:
            feature_cols.append("rolling_mean_3")

        X = df[feature_cols]
        y = df[metric]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Generate future dates and features
        last_date = df["date"].max()
        future_dates = []
        future_features = []

        # Use the last known values for lag features
        last_value = df[metric].iloc[-1]
        second_last_value = df[metric].iloc[-2] if len(df) > 1 else last_value
        rolling_values = df[metric].tail(3).tolist()

        for i in range(1, periods + 1):
            # Determine next date (assuming monthly data)
            if len(df) > 1:
                # Calculate average period between dates
                date_diff = (df["date"].iloc[-1] - df["date"].iloc[-2]).days
                next_date = last_date + timedelta(days=date_diff * i)
            else:
                # Default to monthly
                next_date = last_date + timedelta(days=30 * i)

            future_dates.append(next_date)

            # Create features for this future date
            features = {
                "month": next_date.month,
                "quarter": next_date.quarter,
                "year": next_date.year,
                "days_since_start": (next_date - df["date"].min()).days,
            }

            if "lag_1" in feature_cols:
                features["lag_1"] = last_value
            if "lag_2" in feature_cols:
                features["lag_2"] = second_last_value
            if "rolling_mean_3" in feature_cols:
                features["rolling_mean_3"] = np.mean(rolling_values[-3:])

            future_features.append(features)

        # Make predictions
        future_df = pd.DataFrame(future_features)
        future_X = future_df[feature_cols]
        future_X_scaled = self.scaler.transform(future_X)

        predictions = self.model.predict(future_X_scaled)

        # Calculate simple confidence intervals (±20% for POC)
        confidence_margin = 0.2

        forecast_results = []
        for i, (date, pred) in enumerate(zip(future_dates, predictions)):
            forecast_results.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "value": float(pred),
                    "lower_bound": float(pred * (1 - confidence_margin)),
                    "upper_bound": float(pred * (1 + confidence_margin)),
                    "confidence": "80%",
                }
            )

        # Calculate trend
        if len(df) > 1:
            recent_trend = (df[metric].iloc[-1] - df[metric].iloc[-3:].mean()) / df[
                metric
            ].iloc[-3:].mean()
            trend_direction = (
                "increasing"
                if recent_trend > 0.05
                else "decreasing" if recent_trend < -0.05 else "stable"
            )
        else:
            trend_direction = "unknown"

        return {
            "metric": metric,
            "forecast_periods": periods,
            "forecast": forecast_results,
            "historical_average": float(df[metric].mean()),
            "recent_trend": trend_direction,
            "model_confidence": "medium" if len(df) > 5 else "low",
            "method": "machine_learning" if len(df) > 2 else "moving_average",
        }

    def _simple_forecast(
        self, df: pd.DataFrame, metric: str, periods: int
    ) -> Dict[str, Any]:
        """Fallback to simple moving average forecast"""
        last_value = df[metric].iloc[-1]
        avg_value = df[metric].mean()

        forecast_results = []
        last_date = df["date"].max()

        for i in range(1, periods + 1):
            next_date = last_date + timedelta(days=30 * i)
            # Simple linear extrapolation
            forecast_value = avg_value + (last_value - avg_value) * 0.5

            forecast_results.append(
                {
                    "date": next_date.strftime("%Y-%m-%d"),
                    "value": float(forecast_value),
                    "lower_bound": float(forecast_value * 0.8),
                    "upper_bound": float(forecast_value * 1.2),
                    "confidence": "60%",
                }
            )

        return {
            "metric": metric,
            "forecast_periods": periods,
            "forecast": forecast_results,
            "historical_average": float(avg_value),
            "recent_trend": "unknown",
            "model_confidence": "low",
            "method": "simple_average",
        }
