import pandas as pd
import numpy as np
from typing import Dict, Any
from loguru import logger

class PDFSampler:
    def __init__(self, weights: Dict[str, float]):
        self.weights = weights

    def calculate_complexity(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Calculating complexity scores")
        
        # Normalize features
        features_to_normalize = list(self.weights.keys())
        # Add page_count_scaled
        if "page_count" in df.columns:
            df["page_count_scaled"] = df["page_count"] / df["page_count"].max()
            if "page_count_scaled" not in features_to_normalize:
                # Replace page_count_scaled if it was in weights but not in df
                pass

        score = pd.Series(0.0, index=df.index)
        for feature, weight in self.weights.items():
            if feature in df.columns:
                # Simple min-max normalization for the feature
                f_min = df[feature].min()
                f_max = df[feature].max()
                if f_max > f_min:
                    normalized_f = (df[feature] - f_min) / (f_max - f_min)
                else:
                    normalized_f = df[feature]
                score += normalized_f * weight

        df["complexity_score"] = score
        return df

    def select_top_n(self, df: pd.DataFrame, n_high: int, n_med: int, n_low: int, n_outliers: int) -> pd.DataFrame:
        logger.info(f"Selecting {n_high + n_med + n_low + n_outliers} PDFs")
        
        df = df.sort_values("complexity_score", ascending=False)
        
        # High complexity
        high = df.head(n_high).copy()
        high["selection_category"] = "high"
        
        # Low complexity
        low = df.tail(n_low).copy()
        low["selection_category"] = "low"
        
        # Medium complexity (middle of the remaining)
        remaining = df.drop(high.index).drop(low.index)
        mid_idx = len(remaining) // 2
        med = remaining.iloc[max(0, mid_idx - n_med//2) : min(len(remaining), mid_idx + n_med//2 + n_med%2)].copy()
        med["selection_category"] = "medium"
        
        # Outliers (just as an example, pick by file size or something else)
        remaining = remaining.drop(med.index)
        outliers = remaining.sample(min(n_outliers, len(remaining))).copy()
        outliers["selection_category"] = "outlier"
        
        selected = pd.concat([high, med, low, outliers])
        return selected
