"""
ml/ml_engine.py
Machine Learning subsystem for the Carpooling AI system.

Modules
-------
1. EDA           – Exploratory Data Analysis on ride history
2. Classifier    – sklearn RandomForest to predict ride match success
3. KMeans        – Zone clustering of city locations
4. BayesianNet   – Extended Bayesian network (pgmpy-free, pure numpy)
5. RatingPredict – Linear regression for driver rating prediction
"""
from __future__ import annotations
import math, random, json
from typing import Dict, List, Tuple, Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model  import LinearRegression, LogisticRegression
from sklearn.cluster       import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics       import (accuracy_score, classification_report,
                                   confusion_matrix, silhouette_score)
from sklearn.decomposition import PCA


# ══════════════════════════════════════════════
# 1.  Synthetic Ride History Generator
# ══════════════════════════════════════════════

def generate_ride_history(n: int = 800, seed: int = 7) -> List[Dict]:
    """
    Generate synthetic historical ride records for ML training.
    Features: hour, distance, detour, n_passengers, weather_code, time_diff,
              driver_rating, passenger_rating
    Label   : matched (1) or not (0)
    """
    rng = random.Random(seed)
    records = []
    weathers = [0, 1, 2]   # 0=clear, 1=rainy, 2=foggy

    for _ in range(n):
        hour         = rng.randint(0, 23)
        distance     = round(rng.uniform(1.0, 25.0), 2)
        detour       = round(rng.uniform(0.5, 20.0), 2)
        n_pass       = rng.choice([1, 1, 1, 2, 2, 3])
        weather      = rng.choice(weathers)
        time_diff    = round(rng.uniform(0, 30), 1)
        driver_rating= round(rng.uniform(3.0, 5.0), 1)
        pass_rating  = round(rng.uniform(3.0, 5.0), 1)
        capacity     = rng.choice([4, 4, 5, 6])

        # Stochastic label logic
        score = 0.0
        if detour < 10:            score += 0.35
        if time_diff < 15:         score += 0.25
        if n_pass <= capacity - 1: score += 0.2
        if driver_rating >= 4.0:   score += 0.1
        if weather == 0:           score += 0.05
        if hour in range(7,10) or hour in range(17,20):
            score += 0.05
        score += rng.uniform(-0.15, 0.15)   # noise
        matched = 1 if score >= 0.45 else 0

        records.append({
            "hour": hour, "distance": distance, "detour": detour,
            "n_passengers": n_pass, "weather_code": weather,
            "time_diff": time_diff, "driver_rating": driver_rating,
            "passenger_rating": pass_rating, "capacity": capacity,
            "matched": matched,
        })
    return records


# ══════════════════════════════════════════════
# 2.  EDA  (Exploratory Data Analysis)
# ══════════════════════════════════════════════

class EDAEngine:
    """Compute EDA statistics & chart-ready data from ride history."""

    def __init__(self, records: List[Dict]):
        self.records = records
        self.n       = len(records)

    def summary_stats(self) -> Dict:
        fields = ["distance", "detour", "time_diff", "driver_rating", "passenger_rating"]
        stats  = {}
        for f in fields:
            vals = [r[f] for r in self.records]
            arr  = np.array(vals)
            stats[f] = {
                "mean":   round(float(arr.mean()), 3),
                "std":    round(float(arr.std()),  3),
                "min":    round(float(arr.min()),  3),
                "max":    round(float(arr.max()),  3),
                "median": round(float(np.median(arr)), 3),
                "q25":    round(float(np.percentile(arr, 25)), 3),
                "q75":    round(float(np.percentile(arr, 75)), 3),
            }
        return stats

    def match_rate_by_hour(self) -> Dict[int, float]:
        bucket: Dict[int, List[int]] = {h: [] for h in range(24)}
        for r in self.records:
            bucket[r["hour"]].append(r["matched"])
        return {h: round(sum(v)/len(v)*100, 1) if v else 0.0
                for h, v in bucket.items()}

    def match_rate_by_weather(self) -> Dict[str, float]:
        labels = {0: "Clear", 1: "Rainy", 2: "Foggy"}
        bucket: Dict[int, List[int]] = {0: [], 1: [], 2: []}
        for r in self.records:
            bucket[r["weather_code"]].append(r["matched"])
        return {labels[k]: round(sum(v)/len(v)*100, 1) if v else 0.0
                for k, v in bucket.items()}

    def distance_distribution(self, bins: int = 12) -> Dict:
        vals  = [r["distance"] for r in self.records]
        arr   = np.array(vals)
        counts, edges = np.histogram(arr, bins=bins)
        return {
            "counts": counts.tolist(),
            "edges":  [round(float(e), 2) for e in edges.tolist()],
        }

    def correlation_matrix(self) -> Dict:
        fields = ["distance", "detour", "time_diff", "driver_rating",
                  "passenger_rating", "matched"]
        mat = np.array([[r[f] for f in fields] for r in self.records])
        corr = np.corrcoef(mat.T)
        return {
            "fields": fields,
            "matrix": [[round(float(v), 3) for v in row] for row in corr],
        }

    def feature_importance_proxy(self) -> Dict[str, float]:
        """Quick variance-based proxy for feature importance."""
        features = ["distance", "detour", "time_diff", "driver_rating",
                    "passenger_rating", "n_passengers", "weather_code"]
        matched   = [r for r in self.records if r["matched"] == 1]
        unmatched = [r for r in self.records if r["matched"] == 0]
        importance = {}
        for f in features:
            m_mean = np.mean([r[f] for r in matched])
            u_mean = np.mean([r[f] for r in unmatched])
            pool_std = np.std([r[f] for r in self.records]) or 1e-6
            importance[f] = round(abs(m_mean - u_mean) / pool_std, 4)
        total = sum(importance.values()) or 1
        return {k: round(v/total, 4) for k, v in importance.items()}

    def full_eda(self) -> Dict:
        return {
            "n_records":           self.n,
            "match_rate_pct":      round(sum(r["matched"] for r in self.records)/self.n*100, 1),
            "summary_stats":       self.summary_stats(),
            "match_rate_by_hour":  self.match_rate_by_hour(),
            "match_rate_by_weather": self.match_rate_by_weather(),
            "distance_distribution": self.distance_distribution(),
            "correlation_matrix":  self.correlation_matrix(),
            "feature_importance":  self.feature_importance_proxy(),
        }


# ══════════════════════════════════════════════
# 3.  Match Success Classifier
# ══════════════════════════════════════════════

FEATURE_COLS = ["hour", "distance", "detour", "n_passengers",
                "weather_code", "time_diff", "driver_rating", "passenger_rating"]

class MatchClassifier:
    """
    RandomForest + GradientBoosting ensemble to predict ride match success.
    Also exposes feature importances and confusion matrix.
    """

    def __init__(self):
        self.rf  = RandomForestClassifier(n_estimators=120, max_depth=8,
                                           random_state=42, n_jobs=-1)
        self.gb  = GradientBoostingClassifier(n_estimators=80, max_depth=4,
                                               learning_rate=0.1, random_state=42)
        self.scaler   = StandardScaler()
        self.trained  = False
        self.metrics: Dict = {}

    def _to_xy(self, records: List[Dict]):
        X = np.array([[r[f] for f in FEATURE_COLS] for r in records])
        y = np.array([r["matched"] for r in records])
        return X, y

    def fit(self, records: List[Dict]) -> Dict:
        X, y = self._to_xy(records)
        Xs   = self.scaler.fit_transform(X)
        X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.2,
                                                    random_state=42, stratify=y)
        self.rf.fit(X_tr, y_tr)
        self.gb.fit(X_tr, y_tr)

        rf_preds  = self.rf.predict(X_te)
        gb_preds  = self.gb.predict(X_te)
        # Ensemble: majority vote
        ens_preds = ((self.rf.predict_proba(X_te)[:,1] +
                      self.gb.predict_proba(X_te)[:,1]) / 2 >= 0.5).astype(int)

        self.metrics = {
            "rf_accuracy":  round(accuracy_score(y_te, rf_preds),  4),
            "gb_accuracy":  round(accuracy_score(y_te, gb_preds),  4),
            "ens_accuracy": round(accuracy_score(y_te, ens_preds), 4),
            "confusion_matrix": confusion_matrix(y_te, ens_preds).tolist(),
            "feature_importances": {
                f: round(float(v), 4)
                for f, v in zip(FEATURE_COLS, self.rf.feature_importances_)
            },
            "n_train": len(X_tr),
            "n_test":  len(X_te),
        }
        self.trained = True
        return self.metrics

    def predict(self, record: Dict) -> Tuple[int, float]:
        """Return (label, confidence)."""
        if not self.trained:
            return 0, 0.0
        x = np.array([[record.get(f, 0) for f in FEATURE_COLS]])
        xs = self.scaler.transform(x)
        prob = (self.rf.predict_proba(xs)[0,1] + self.gb.predict_proba(xs)[0,1]) / 2
        return int(prob >= 0.5), round(float(prob), 4)

    def pca_2d(self, records: List[Dict]) -> Dict:
        """Return 2D PCA projection for visualisation."""
        X, y = self._to_xy(records)
        Xs   = self.scaler.transform(X)
        pca  = PCA(n_components=2, random_state=42)
        proj = pca.fit_transform(Xs)
        return {
            "x":       [round(float(v), 4) for v in proj[:,0]],
            "y":       [round(float(v), 4) for v in proj[:,1]],
            "labels":  y.tolist(),
            "variance_explained": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        }


# ══════════════════════════════════════════════
# 4.  Zone Clustering (K-Means)
# ══════════════════════════════════════════════

class ZoneClusterer:
    """
    K-Means clustering of city node coordinates → zone labels.
    """

    def __init__(self, k: int = 4):
        self.k  = k
        self.km = KMeans(n_clusters=k, random_state=42, n_init=10)
        self.labels_: List[int] = []
        self.centres_: List[Tuple] = []
        self.silhouette_: float = 0.0
        self.fitted = False

    def fit(self, coords: List[Tuple[float, float]],
            names: List[str]) -> Dict:
        X = np.array(coords)
        self.km.fit(X)
        self.labels_  = self.km.labels_.tolist()
        self.centres_ = [(round(float(c[0]),3), round(float(c[1]),3))
                         for c in self.km.cluster_centers_]
        if len(set(self.labels_)) > 1:
            self.silhouette_ = round(float(silhouette_score(X, self.labels_)), 4)
        self.fitted = True

        zone_names = {0:"North", 1:"South", 2:"East", 3:"West",
                      4:"Central", 5:"Harbor", 6:"Industrial"}
        clusters   = {i: [] for i in range(self.k)}
        for name, lbl in zip(names, self.labels_):
            clusters[lbl].append(name)

        return {
            "k":            self.k,
            "silhouette":   self.silhouette_,
            "centres":      self.centres_,
            "labels":       self.labels_,
            "node_names":   names,
            "clusters":     {zone_names.get(i, f"Zone {i}"): v
                             for i, v in clusters.items()},
        }

    def predict_zone(self, x: float, y: float) -> int:
        if not self.fitted:
            return 0
        return int(self.km.predict([[x, y]])[0])


# ══════════════════════════════════════════════
# 5.  Driver Rating Predictor (Linear Regression)
# ══════════════════════════════════════════════

class RatingPredictor:
    """
    Linear Regression to predict post-ride driver rating
    from route characteristics.
    """
    FEAT = ["distance", "detour", "time_diff", "n_passengers", "weather_code"]

    def __init__(self):
        self.model  = LinearRegression()
        self.scaler = StandardScaler()
        self.trained = False
        self.metrics: Dict = {}

    def _synth_data(self, n: int = 600, seed: int = 11) -> Tuple:
        rng  = random.Random(seed)
        X, y = [], []
        for _ in range(n):
            dist  = rng.uniform(1, 25)
            det   = rng.uniform(0, 20)
            td    = rng.uniform(0, 30)
            npas  = rng.choice([1,2,3])
            wc    = rng.choice([0,1,2])
            # Rating heuristic
            r = 5.0 - det*0.05 - td*0.03 + (3-wc)*0.1
            r = max(1.0, min(5.0, r + rng.gauss(0, 0.3)))
            X.append([dist, det, td, npas, wc])
            y.append(round(r, 2))
        return np.array(X), np.array(y)

    def fit(self) -> Dict:
        X, y = self._synth_data()
        Xs   = self.scaler.fit_transform(X)
        X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.2, random_state=42)
        self.model.fit(X_tr, y_tr)
        preds = self.model.predict(X_te)
        mse  = float(np.mean((preds - y_te)**2))
        mae  = float(np.mean(np.abs(preds - y_te)))
        self.metrics = {
            "mse":  round(mse, 4), "rmse": round(math.sqrt(mse), 4),
            "mae":  round(mae, 4),
            "r2":   round(float(self.model.score(X_te, y_te)), 4),
            "coefficients": {f: round(float(c), 4)
                             for f, c in zip(self.FEAT, self.model.coef_)},
        }
        self.trained = True
        return self.metrics

    def predict(self, distance: float, detour: float, time_diff: float,
                n_pass: int, weather: int) -> float:
        if not self.trained:
            return 4.0
        x  = self.scaler.transform([[distance, detour, time_diff, n_pass, weather]])
        return round(float(np.clip(self.model.predict(x)[0], 1.0, 5.0)), 2)


# ══════════════════════════════════════════════
# 6.  ML Engine  (Facade)
# ══════════════════════════════════════════════

class MLEngine:
    """Central facade that trains and exposes all ML modules."""

    def __init__(self):
        self.records         = generate_ride_history(1000)
        self.eda             = EDAEngine(self.records)
        self.classifier      = MatchClassifier()
        self.clusterer       = ZoneClusterer(k=5)
        self.rating_pred     = RatingPredictor()
        self.ready           = False
        # Results cache
        self.eda_result: Dict      = {}
        self.clf_metrics: Dict     = {}
        self.cluster_result: Dict  = {}
        self.rating_metrics: Dict  = {}
        self.pca_data: Dict        = {}

    def train_all(self, city_coords: List[Tuple], city_names: List[str]) -> Dict:
        self.eda_result     = self.eda.full_eda()
        self.clf_metrics    = self.classifier.fit(self.records)
        self.pca_data       = self.classifier.pca_2d(self.records)
        self.cluster_result = self.clusterer.fit(city_coords, city_names)
        self.rating_metrics = self.rating_pred.fit()
        self.ready          = True
        return {
            "eda":      self.eda_result,
            "clf":      self.clf_metrics,
            "cluster":  self.cluster_result,
            "rating":   self.rating_metrics,
            "pca":      self.pca_data,
        }
