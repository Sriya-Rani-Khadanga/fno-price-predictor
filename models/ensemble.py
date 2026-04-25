"""
Ensemble model combining PCP and persistence predictions.
"""
from __future__ import annotations
from typing import Dict
from models.pcp_model import PCPModel
from models.persistence_model import PersistenceModel

class EnsembleModel:
    """Combines profitability and persistence predictions for decision support."""

    def __init__(self):
        self.pcp_model = PCPModel()
        self.persistence_model = PersistenceModel()

    def predict(self, features: Dict) -> Dict:
        prof_prob = self.pcp_model.predict_profitability(features)
        duration = self.persistence_model.predict_duration(features)
        score = prof_prob * min(1.0, duration / 60.0)
        return {"profitability_prob": round(prof_prob, 3),
                "expected_duration_s": round(duration, 1),
                "composite_score": round(score, 3),
                "recommendation": "enter" if score > 0.4 else "wait"}

    def update(self, features: Dict, pnl_outcome: float, duration: float):
        self.pcp_model.update(features, 1.0 if pnl_outcome > 0 else 0.0)
        self.persistence_model.update(features, duration)
