"""
models/bayesian.py
Bayesian Demand & ETA Prediction Model.

Uses a Naive Bayes-style probabilistic model (built from scratch with numpy)
to predict:
  1. Demand level for a zone at a given hour (Low / Medium / High)
  2. Expected travel time adjustment factor given traffic conditions

This supplements the deterministic search algorithms with uncertainty-aware predictions.
"""
from __future__ import annotations
import math
import random
from typing import Dict, List, Tuple


# ──────────────────────────────────────────────────────────
# Training data generation (simulated historical data)
# ──────────────────────────────────────────────────────────

def _generate_training_data(n: int = 500) -> List[Tuple[int, str, str, float]]:
    """
    Simulate historical carpooling demand data.
    Returns list of (hour, zone_type, weather, demand_level) tuples.
    demand_level: 'low' | 'medium' | 'high'
    """
    random.seed(42)
    zones = ["residential", "commercial", "university", "industrial"]
    weathers = ["clear", "rainy", "foggy"]
    data = []

    for _ in range(n):
        hour = random.randint(0, 23)
        zone = random.choice(zones)
        weather = random.choice(weathers)

        # Prior: rush hours → higher demand
        if hour in range(7, 10) or hour in range(17, 20):
            base = 0.6
        elif hour in range(10, 17):
            base = 0.3
        else:
            base = 0.1

        # Zone modifier
        z_mod = {"residential": 0.1, "commercial": 0.15,
                 "university": 0.2, "industrial": 0.05}.get(zone, 0.0)

        # Weather modifier
        w_mod = {"rainy": 0.15, "foggy": 0.05, "clear": 0.0}.get(weather, 0.0)

        prob_high = min(base + z_mod + w_mod, 0.95)
        prob_med = min((1 - prob_high) * 0.6, 0.9)

        r = random.random()
        if r < prob_high:
            demand = "high"
        elif r < prob_high + prob_med:
            demand = "medium"
        else:
            demand = "low"

        data.append((hour, zone, weather, demand))

    return data


# ──────────────────────────────────────────────────────────
# Naive Bayes Classifier
# ──────────────────────────────────────────────────────────

class NaiveBayesDemandPredictor:
    """
    Gaussian Naive Bayes for demand prediction.
    Features: [hour (continuous), zone (categorical), weather (categorical)]
    """

    def __init__(self):
        self._classes = ["low", "medium", "high"]
        self._priors: Dict[str, float] = {}
        self._hour_stats: Dict[str, Tuple[float, float]] = {}   # class → (mean, std)
        self._zone_probs: Dict[str, Dict[str, float]] = {}      # class → zone → P
        self._weather_probs: Dict[str, Dict[str, float]] = {}   # class → weather → P
        self._trained = False

    def fit(self, data: List[Tuple[int, str, str, str]]) -> None:
        """Fit model on training data [(hour, zone, weather, label)]."""
        from collections import defaultdict

        class_counts: Dict[str, int] = defaultdict(int)
        hours_by_class: Dict[str, List[float]] = defaultdict(list)
        zone_counts: Dict[str, Dict[str, int]] = {c: defaultdict(int) for c in self._classes}
        weather_counts: Dict[str, Dict[str, int]] = {c: defaultdict(int) for c in self._classes}

        for hour, zone, weather, label in data:
            class_counts[label] += 1
            hours_by_class[label].append(float(hour))
            zone_counts[label][zone] += 1
            weather_counts[label][weather] += 1

        n_total = len(data)
        for cls in self._classes:
            n_cls = class_counts[cls]
            self._priors[cls] = n_cls / n_total

            # Gaussian params for hour
            hrs = hours_by_class[cls]
            mu = sum(hrs) / len(hrs) if hrs else 12.0
            variance = sum((h - mu) ** 2 for h in hrs) / max(len(hrs) - 1, 1)
            self._hour_stats[cls] = (mu, math.sqrt(variance) + 1e-6)

            # Categorical likelihoods (Laplace smoothing)
            all_zones = {"residential", "commercial", "university", "industrial"}
            all_weathers = {"clear", "rainy", "foggy"}

            self._zone_probs[cls] = {
                z: (zone_counts[cls][z] + 1) / (n_cls + len(all_zones))
                for z in all_zones
            }
            self._weather_probs[cls] = {
                w: (weather_counts[cls][w] + 1) / (n_cls + len(all_weathers))
                for w in all_weathers
            }

        self._trained = True

    def _gaussian_pdf(self, x: float, mu: float, sigma: float) -> float:
        exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
        return math.exp(exponent) / (sigma * math.sqrt(2 * math.pi))

    def predict_proba(self, hour: int, zone: str, weather: str) -> Dict[str, float]:
        """Return posterior probabilities for each demand class."""
        if not self._trained:
            raise RuntimeError("Model not trained. Call fit() first.")

        posteriors: Dict[str, float] = {}
        for cls in self._classes:
            mu, sigma = self._hour_stats[cls]
            log_p = math.log(self._priors[cls] + 1e-9)
            log_p += math.log(self._gaussian_pdf(float(hour), mu, sigma) + 1e-9)
            log_p += math.log(self._zone_probs[cls].get(zone, 1e-6))
            log_p += math.log(self._weather_probs[cls].get(weather, 1e-6))
            posteriors[cls] = log_p

        # Softmax for normalised probabilities
        max_log = max(posteriors.values())
        exp_vals = {cls: math.exp(v - max_log) for cls, v in posteriors.items()}
        total = sum(exp_vals.values())
        return {cls: v / total for cls, v in exp_vals.items()}

    def predict(self, hour: int, zone: str, weather: str) -> str:
        proba = self.predict_proba(hour, zone, weather)
        return max(proba, key=proba.get)

    def predict_eta_factor(self, hour: int, weather: str) -> float:
        """
        Predict traffic delay multiplier (1.0 = no delay, 2.0 = double time).
        """
        demand = self.predict(hour, "commercial", weather)
        base = {"low": 1.0, "medium": 1.35, "high": 1.75}.get(demand, 1.0)
        w_factor = {"rainy": 1.25, "foggy": 1.15, "clear": 1.0}.get(weather, 1.0)
        return round(base * w_factor, 2)


# ──────────────────────────────────────────────────────────
# Convenience factory
# ──────────────────────────────────────────────────────────

def build_predictor() -> NaiveBayesDemandPredictor:
    """Build and train the predictor on synthetic historical data."""
    predictor = NaiveBayesDemandPredictor()
    training_data = _generate_training_data(1000)
    predictor.fit(training_data)
    return predictor
