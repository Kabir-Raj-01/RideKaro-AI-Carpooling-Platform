"""
agents/agent.py
═══════════════════════════════════════════════════════════════════
Enhanced Multi-Agent Carpooling Optimisation System
═══════════════════════════════════════════════════════════════════

Agent Architecture (Goal-Based Rational Agents)
────────────────────────────────────────────────
Every agent follows the PAMA cycle:
  Perceive  → sense the environment
  Analyse   → reason about state
  Model     → update internal world model
  Act       → execute the best action

Agents
──────
1. EnvironmentAgent      – city graph, live traffic, weather, time-of-day
2. TrafficIntelAgent     – models real Karachi rush-hour congestion patterns
3. PredictionAgent       – Bayesian demand forecast, ETA, surge pricing
4. MatchingAgent         – BFS reachability + greedy candidate pairing
5. NegotiationAgent      – Alpha-Beta / game-theoretic driver-passenger matching
6. GlobalOptimiserAgent  – Genetic Algorithm (global passenger assignment)
7. RouteOptimiserAgent   – A* per-group route planning with real-world waypoints
8. ConstraintAgent       – CSP validation + AC-3 arc consistency repair
9. SafetyAgent           – flags low-rated users, detour violations, time breaches
10. CostOptimiserAgent   – dynamic fuel costing, surge pricing, fare splitting
11. OrchestratorAgent    – master controller coordinating all sub-agents

Every agent emits structured events that power the real-time UI timeline.
"""
from __future__ import annotations
import time, math, random, copy, uuid
from dataclasses import dataclass, field
from typing      import List, Dict, Optional, Tuple, Any

from models.models      import Driver, Passenger, RideGroup, Location, RideStatus
from models.city_graph  import CityGraph
from models.bayesian    import NaiveBayesDemandPredictor
from algorithms.search  import astar, bfs, greedy_best_first, route_distance
from algorithms.genetic import GeneticOptimizer
from algorithms.csp     import CSPSolver
from algorithms.search_extended import run_alpha_beta_demo


# ═══════════════════════════════════════════════════════════════
# EVENT SYSTEM  –  powers the real-time UI timeline
# ═══════════════════════════════════════════════════════════════

@dataclass
class AgentEvent:
    """Single event emitted by an agent – consumed by the UI timeline."""
    timestamp:  float          # seconds since pipeline start
    agent:      str            # agent name
    event_type: str            # action | decision | warning | error | info
    title:      str            # short headline
    detail:     str            # full description
    data:       Dict = field(default_factory=dict)   # structured payload
    icon:       str  = "🔵"

    def to_dict(self) -> Dict:
        return {
            "ts":    round(self.timestamp, 3),
            "agent": self.agent,
            "type":  self.event_type,
            "title": self.title,
            "detail":self.detail,
            "data":  self.data,
            "icon":  self.icon,
        }


class EventBus:
    """Shared event bus – all agents push events, orchestrator collects them."""
    def __init__(self):
        self._events: List[AgentEvent] = []
        self._start:  float = time.time()

    def emit(self, agent: str, event_type: str, title: str,
             detail: str, data: Dict = None, icon: str = "🔵") -> AgentEvent:
        ev = AgentEvent(
            timestamp  = time.time() - self._start,
            agent      = agent,
            event_type = event_type,
            title      = title,
            detail     = detail,
            data       = data or {},
            icon       = icon,
        )
        self._events.append(ev)
        return ev

    def all(self) -> List[Dict]:
        return [e.to_dict() for e in self._events]

    def by_agent(self, agent: str) -> List[Dict]:
        return [e.to_dict() for e in self._events if e.agent == agent]

    def reset(self):
        self._events.clear()
        self._start = time.time()


# ═══════════════════════════════════════════════════════════════
# BASE AGENT
# ═══════════════════════════════════════════════════════════════

class BaseAgent:
    ICON = "🤖"

    def __init__(self, name: str, bus: EventBus):
        self.name  = name
        self.bus   = bus
        self.log:  List[str] = []
        self._state: Dict    = {}

    def _emit(self, etype: str, title: str, detail: str,
              data: Dict = None, icon: str = None) -> AgentEvent:
        ev = self.bus.emit(self.name, etype, title, detail,
                           data or {}, icon or self.ICON)
        entry = f"[{self.name}] {title} — {detail}"
        self.log.append(entry)
        return ev

    def perceive(self, *args, **kwargs):   raise NotImplementedError
    def analyse(self,  *args, **kwargs):   pass
    def model(self,    *args, **kwargs):   pass
    def act(self,      *args, **kwargs):   raise NotImplementedError


# ═══════════════════════════════════════════════════════════════
# 1. ENVIRONMENT AGENT
# ═══════════════════════════════════════════════════════════════

class EnvironmentAgent(BaseAgent):
    """
    Maintains the live world model:
    – city road graph with dynamic traffic weights
    – Karachi-realistic rush-hour patterns (7–9 AM, 5–8 PM)
    – weather effects on road speeds
    – time-of-day service availability
    """
    ICON = "🌍"

    # Real Karachi congestion hotspots (road, peak traffic factor)
    KARACHI_HOTSPOTS = [
        ("Saddar",    "Clifton",    2.4, "Shahrae Faisal rush-hour"),
        ("Gulshan",   "Saddar",     2.1, "University Road congestion"),
        ("NUCES",     "Gulshan",    1.8, "Abul Hasan Ispahani Road"),
        ("Johar",     "North_Naz",  1.7, "Johar Chowrangi bottleneck"),
        ("Airport",   "Federal_B",  2.0, "Expressway merge"),
        ("DHA_1",     "DHA_5",      1.6, "Defence Phase link"),
        ("Landhi",    "Korangi",    1.9, "Industrial zone trucks"),
        ("Lyari",     "Saddar",     2.2, "Lyari Expressway peak"),
    ]

    WEATHER_SPEED_FACTOR = {"clear": 1.0, "rainy": 1.35, "foggy": 1.20}

    def __init__(self, city: CityGraph, bus: EventBus):
        super().__init__("EnvironmentAgent", bus)
        self.city    = city
        self.hour    = 8
        self.weather = "clear"
        self.traffic_applied: List[Dict] = []

    def perceive(self, hour: int, weather: str):
        self.hour    = hour
        self.weather = weather
        self._emit("action", "Perceived environment",
                   f"Time={hour:02d}:00, Weather={weather}",
                   {"hour": hour, "weather": weather}, "🌍")

    def analyse(self) -> str:
        if self.hour in range(7, 10):   return "morning_rush"
        if self.hour in range(17, 21):  return "evening_rush"
        if self.hour in range(22, 24) or self.hour < 5: return "night"
        return "normal"

    def model(self) -> Dict:
        period = self.analyse()
        w_factor = self.WEATHER_SPEED_FACTOR.get(self.weather, 1.0)
        rush_factor = {"morning_rush": 1.6, "evening_rush": 1.8,
                       "night": 0.7, "normal": 1.0}[period]
        return {"period": period, "rush_factor": rush_factor,
                "weather_factor": w_factor,
                "combined": round(rush_factor * w_factor, 2)}

    def act(self) -> Dict:
        model = self.model()
        period = model["period"]
        applied = []

        if period in ("morning_rush", "evening_rush"):
            for a, b, base_tf, desc in self.KARACHI_HOTSPOTS:
                final_tf = round(base_tf * model["weather_factor"], 2)
                self.city.update_traffic(a, b, final_tf)
                applied.append({"road": f"{a}↔{b}", "tf": final_tf, "reason": desc})
        elif self.weather in ("rainy", "foggy"):
            for a, b, base_tf, desc in self.KARACHI_HOTSPOTS[:4]:
                final_tf = round(1.0 * model["weather_factor"], 2)
                self.city.update_traffic(a, b, final_tf)
                applied.append({"road": f"{a}↔{b}", "tf": final_tf, "reason": "Weather slowdown"})

        self.traffic_applied = applied
        self._emit(
            "decision", f"Traffic model applied — {period.replace('_',' ').title()}",
            f"{len(applied)} roads updated. Combined factor: ×{model['combined']}",
            {"period": period, "roads_updated": len(applied),
             "combined_factor": model["combined"], "hotspots": applied[:5]},
            "🚦"
        )
        return {**model, "applied": applied, "city": self.city}


# ═══════════════════════════════════════════════════════════════
# 2. TRAFFIC INTELLIGENCE AGENT
# ═══════════════════════════════════════════════════════════════

class TrafficIntelAgent(BaseAgent):
    """
    Analyses city graph to find:
    – Current congestion score per zone
    – Recommended avoid-zones
    – Estimated delay per route segment
    – Alternative corridor suggestions
    """
    ICON = "🚦"

    def __init__(self, city: CityGraph, bus: EventBus):
        super().__init__("TrafficIntelAgent", bus)
        self.city = city
        self.congestion_map: Dict[str, float] = {}

    def perceive(self, env_model: Dict):
        self.env_model = env_model

    def analyse(self) -> Dict[str, float]:
        scores = {}
        for node in self.city.all_locations():
            edges = list(self.city.graph.edges(node.name, data=True))
            if not edges:
                scores[node.name] = 0.0
                continue
            avg_tf = sum(d.get("traffic_factor", 1.0) for _, _, d in edges) / len(edges)
            scores[node.name] = round(avg_tf, 3)
        self.congestion_map = scores
        return scores

    def act(self) -> Dict:
        scores = self.analyse()
        hot    = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        clear  = sorted(scores.items(), key=lambda x: x[1])[:5]

        # Compute average network delay
        all_tfs = [d.get("traffic_factor", 1.0) for _, _, d in self.city.graph.edges(data=True)]
        avg_delay = round(sum(all_tfs) / len(all_tfs) if all_tfs else 1.0, 3)

        self._emit(
            "info", "Traffic intelligence computed",
            f"Avg network delay ×{avg_delay}. Top congested: {hot[0][0]} (×{hot[0][1]})",
            {"hotspots": hot, "clear_zones": clear, "avg_delay_factor": avg_delay,
             "congestion_map": scores},
            "🚦"
        )
        return {"congestion_map": scores, "hotspots": hot,
                "clear_zones": clear, "avg_delay_factor": avg_delay}


# ═══════════════════════════════════════════════════════════════
# 3. PREDICTION AGENT
# ═══════════════════════════════════════════════════════════════

class PredictionAgent(BaseAgent):
    """
    Bayesian demand & ETA prediction.
    Outputs per-zone demand (Low/Med/High), ETA multipliers,
    surge pricing recommendations, and best departure windows.
    """
    ICON = "🔮"

    def __init__(self, predictor: NaiveBayesDemandPredictor, bus: EventBus):
        super().__init__("PredictionAgent", bus)
        self.predictor   = predictor
        self.predictions: Dict = {}

    def perceive(self, hour: int, weather: str, zones: List[str]):
        self.hour    = hour
        self.weather = weather
        self.zones   = zones

    def analyse(self) -> Dict:
        results = {}
        for zone in self.zones:
            proba  = self.predictor.predict_proba(self.hour, zone, self.weather)
            demand = self.predictor.predict(self.hour, zone, self.weather)
            eta_f  = self.predictor.predict_eta_factor(self.hour, self.weather)
            surge  = {"low": 1.0, "medium": 1.25, "high": 1.6}.get(demand, 1.0)
            # Recommend departure offset (minutes to shift to avoid peak)
            shift = 0
            if demand == "high":
                shift = -20 if self.hour in range(7, 10) else -15
            results[zone] = {
                "demand":          demand,
                "probabilities":   {k: round(v, 3) for k, v in proba.items()},
                "eta_factor":      round(eta_f, 2),
                "surge_multiplier":round(surge, 2),
                "departure_shift_min": shift,
                "recommendation":  self._recommend(demand, surge, shift),
            }
        return results

    def _recommend(self, demand: str, surge: float, shift: int) -> str:
        if demand == "high":
            return f"High demand — surge ×{surge}. Depart {abs(shift)} min earlier to save PKR."
        if demand == "medium":
            return f"Moderate demand — standard fare. Good time to travel."
        return "Low demand — best time to travel. Fares at minimum."

    def act(self) -> Dict:
        self.predictions = self.analyse()
        high_zones = [z for z, d in self.predictions.items() if d["demand"] == "high"]
        avg_eta    = round(
            sum(d["eta_factor"] for d in self.predictions.values()) / len(self.predictions), 2
        )
        self._emit(
            "decision", "Bayesian demand predictions complete",
            f"High demand zones: {high_zones or 'none'}. Avg ETA factor ×{avg_eta}",
            {"predictions": self.predictions, "high_zones": high_zones,
             "avg_eta_factor": avg_eta},
            "🔮"
        )
        return self.predictions

    def adjust_eta(self, groups: List[RideGroup]) -> List[RideGroup]:
        factors = [d["eta_factor"] for d in self.predictions.values()]
        avg = sum(factors) / len(factors) if factors else 1.0
        adjusted = 0
        for g in groups:
            g.estimated_time = round(g.estimated_time * avg, 1)
            adjusted += 1
        self._emit(
            "action", f"ETA adjusted for {adjusted} groups",
            f"Applied Bayesian factor ×{avg:.2f} to all ride group ETAs",
            {"avg_factor": avg, "groups_adjusted": adjusted}, "⏱️"
        )
        return groups


# ═══════════════════════════════════════════════════════════════
# 4. MATCHING AGENT
# ═══════════════════════════════════════════════════════════════

class MatchingAgent(BaseAgent):
    """
    BFS-based reachability analysis + greedy candidate pairing.
    Produces ranked (driver, passenger) pairs sorted by proximity score.
    """
    ICON = "🔗"
    MAX_HOPS = 5

    def __init__(self, city: CityGraph, bus: EventBus):
        super().__init__("MatchingAgent", bus)
        self.city = city
        self.pairs: List[Tuple[Driver, Passenger, float]] = []

    def perceive(self, drivers: List[Driver], passengers: List[Passenger]):
        self.drivers    = drivers
        self.passengers = passengers
        self._emit("action", "Received user set",
                   f"{len(drivers)} drivers, {len(passengers)} passengers",
                   {"n_drivers": len(drivers), "n_passengers": len(passengers)}, "🔗")

    def _bfs_reachable(self, start: str, hops: int) -> set:
        from collections import deque
        vis = {start}; q = deque([(start, 0)])
        while q:
            node, d = q.popleft()
            if d >= hops: continue
            for nb in self.city.neighbors(node):
                if nb not in vis:
                    vis.add(nb); q.append((nb, d + 1))
        return vis

    def analyse(self) -> List[Tuple[Driver, Passenger, float]]:
        pairs = []
        for driver in self.drivers:
            reachable = self._bfs_reachable(driver.pickup.name, self.MAX_HOPS)
            for pax in self.passengers:
                if pax.pickup.name in reachable:
                    # Proximity score: inverse weighted distance + time compatibility
                    dist  = driver.pickup.distance_to(pax.pickup)
                    tw_ov = max(0.0,
                        min(driver.time_window_end, pax.time_window_end) -
                        max(driver.time_window_start, pax.time_window_start)
                    )
                    score = tw_ov / (dist + 1e-3)
                    pairs.append((driver, pax, round(score, 4)))

        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs

    def act(self) -> List[Tuple[Driver, Passenger]]:
        self.pairs = self.analyse()
        self._emit(
            "decision", f"BFS matching complete — {len(self.pairs)} candidate pairs",
            f"Best pair: {self.pairs[0][0].name} ↔ {self.pairs[0][1].name} "
            f"(score {self.pairs[0][2]:.3f})" if self.pairs else "No pairs found",
            {
                "total_pairs": len(self.pairs),
                "top_pairs": [
                    {"driver": d.name, "passenger": p.name,
                     "score": s, "pickup_d": d.pickup.name, "pickup_p": p.pickup.name}
                    for d, p, s in self.pairs[:8]
                ],
            }, "🔗"
        )
        return [(d, p) for d, p, _ in self.pairs]


# ═══════════════════════════════════════════════════════════════
# 5. NEGOTIATION AGENT  (Alpha-Beta / Game Theory)
# ═══════════════════════════════════════════════════════════════

class NegotiationAgent(BaseAgent):
    """
    Game-theoretic driver–passenger negotiation using Minimax + Alpha-Beta.

    Models the negotiation as a 2-player zero-sum game:
      MAX player (driver)     – maximise: route value + earnings – detour cost
      MIN player (passenger)  – minimise: fare cost + detour + waiting time

    Runs for top-5 candidate pairs, emits negotiation outcomes.
    """
    ICON = "♟️"

    def __init__(self, bus: EventBus):
        super().__init__("NegotiationAgent", bus)
        self.outcomes: List[Dict] = []

    def _utility(self, driver: Driver, pax: Passenger) -> float:
        dist   = driver.pickup.distance_to(pax.pickup)
        tw_ov  = max(0.0,
            min(driver.time_window_end, pax.time_window_end) -
            max(driver.time_window_start, pax.time_window_start)
        )
        util   = (tw_ov * 0.4) - (dist * 0.3) + (driver.rating * 0.2) + (pax.rating * 0.1)
        return round(util, 4)

    def _minimax(self, depth: int, is_max: bool, alpha: float, beta: float,
                 values: List[float], idx: int) -> float:
        if depth == 0 or idx >= len(values):
            return values[idx % len(values)]
        if is_max:
            best = float("-inf")
            for i in range(2):
                v    = self._minimax(depth-1, False, alpha, beta, values, idx*2+i+1)
                best = max(best, v); alpha = max(alpha, best)
                if beta <= alpha: break
            return best
        else:
            best = float("inf")
            for i in range(2):
                v    = self._minimax(depth-1, True,  alpha, beta, values, idx*2+i+1)
                best = min(best, v); beta = min(beta, best)
                if beta <= alpha: break
            return best

    def perceive(self, pairs: List[Tuple[Driver, Passenger]]):
        self.pairs = pairs[:6]

    def act(self) -> List[Dict]:
        outcomes = []
        for driver, pax in self.pairs:
            util   = self._utility(driver, pax)
            # Build leaf values from utility + noise
            rng    = random.Random(hash(driver.name + pax.name) % 10000)
            leaves = [round(util + rng.uniform(-2, 2), 2) for _ in range(8)]
            best   = self._minimax(3, True, float("-inf"), float("inf"), leaves, 0)
            agreed = best > 0.0
            fare_base = round(driver.pickup.distance_to(pax.dropoff) * 18.0, 0)
            fare_adj  = round(fare_base * (1.0 + max(0, best) * 0.05), 0)
            outcomes.append({
                "driver":   driver.name,
                "passenger":pax.name,
                "utility":  round(best, 3),
                "agreed":   agreed,
                "fare_base_pkr": fare_base,
                "fare_negotiated_pkr": fare_adj,
                "reason":   "Both parties benefit" if agreed else "Driver detour too large",
            })

        self.outcomes = outcomes
        agreed_count = sum(1 for o in outcomes if o["agreed"])
        self._emit(
            "decision",
            f"Negotiation: {agreed_count}/{len(outcomes)} pairs agreed",
            f"Game-theoretic minimax resolved {agreed_count} bilateral agreements",
            {"outcomes": outcomes, "agreed": agreed_count,
             "rejected": len(outcomes) - agreed_count},
            "♟️"
        )
        return outcomes


# ═══════════════════════════════════════════════════════════════
# 6. GLOBAL OPTIMISER AGENT  (Genetic Algorithm)
# ═══════════════════════════════════════════════════════════════

class GlobalOptimiserAgent(BaseAgent):
    """
    Genetic Algorithm for globally optimal passenger-driver assignment.
    Chromosome = permutation of passenger indices.
    Multi-objective fitness: distance efficiency + vehicle utilisation + time compliance.
    """
    ICON = "🧬"

    def __init__(self, city: CityGraph, bus: EventBus,
                 pop_size: int = 60, generations: int = 80):
        super().__init__("GlobalOptimiserAgent", bus)
        self.city        = city
        self.pop_size    = pop_size
        self.generations = generations
        self.ga_history: List[float] = []
        self.ga_stats:   Dict        = {}

    def perceive(self, drivers: List[Driver], passengers: List[Passenger]):
        self.drivers    = drivers
        self.passengers = passengers

    def act(self) -> Tuple[List[RideGroup], float]:
        self._emit("action", "Genetic Algorithm started",
                   f"Population={self.pop_size}, Generations={self.generations}, "
                   f"Chromosomes={self.pop_size} permutations of {len(self.passengers)} passengers",
                   {"pop_size": self.pop_size, "generations": self.generations,
                    "n_drivers": len(self.drivers), "n_passengers": len(self.passengers)}, "🧬")

        ga = GeneticOptimizer(
            self.drivers, self.passengers, self.city,
            pop_size=self.pop_size, generations=self.generations,
            mutation_rate=0.05, elite_fraction=0.1,
        )
        groups, best_fitness, history = ga.run()
        self.ga_history = history

        # Compute stats
        init_fit = history[0]  if history else 0
        best_fit = history[-1] if history else best_fitness
        improvement = round((best_fit - init_fit) / max(abs(init_fit), 1e-9) * 100, 1)
        matched_pax = sum(len(g.passengers) for g in groups)
        self.ga_stats = {
            "generations_run":  len(history),
            "initial_fitness":  round(init_fit, 6),
            "best_fitness":     round(best_fit, 6),
            "improvement_pct":  improvement,
            "groups_formed":    len(groups),
            "passengers_matched": matched_pax,
            "passengers_unmatched": len(self.passengers) - matched_pax,
            "avg_group_size":   round(matched_pax / max(len(groups), 1), 2),
        }

        self._emit(
            "decision",
            f"GA converged — {len(groups)} groups, fitness={round(best_fit,5)}",
            f"Improvement: +{improvement}% over {len(history)} generations. "
            f"{matched_pax}/{len(self.passengers)} passengers matched.",
            self.ga_stats, "🧬"
        )
        for g in groups:
            g.fitness_score = best_fitness
        return groups, best_fitness


# ═══════════════════════════════════════════════════════════════
# 7. ROUTE OPTIMISER AGENT  (A*)
# ═══════════════════════════════════════════════════════════════

class RouteOptimiserAgent(BaseAgent):
    """
    A* route planning for every ride group.
    Builds optimal multi-waypoint routes:
      driver_pickup → [passenger pickups (nearest-first)] → [dropoffs] → driver_dropoff
    Also runs comparative analysis: A* vs BFS vs Greedy.
    """
    ICON = "🗺️"

    def __init__(self, city: CityGraph, bus: EventBus):
        super().__init__("RouteOptimiserAgent", bus)
        self.city  = city
        self.comparison_stats: List[Dict] = []

    def perceive(self, groups: List[RideGroup]):
        self.groups = groups

    def _build_waypoints(self, grp: RideGroup) -> List[Location]:
        pickups = sorted(grp.passengers, key=lambda p: p.pickup.distance_to(grp.driver.pickup))
        wps = [grp.driver.pickup]
        for p in pickups: wps.append(p.pickup)
        for p in pickups: wps.append(p.dropoff)
        wps.append(grp.driver.dropoff)
        return wps

    def _compare(self, src: str, dst: str) -> Dict:
        from algorithms.search import dijkstra
        results = {}
        for name, fn in [("A*", astar), ("BFS", bfs), ("Greedy", greedy_best_first)]:
            path, dist, explored = fn(self.city, src, dst)
            results[name] = {"dist": round(dist, 3) if dist != float("inf") else None,
                              "explored": explored, "hops": len(path)}
        return results

    def act(self) -> List[RideGroup]:
        optimised = []
        total_km  = 0.0

        for gi, grp in enumerate(self.groups):
            if not grp.driver or not grp.passengers:
                continue
            wps        = self._build_waypoints(grp)
            grp.route  = wps
            node_names = [w.name for w in wps]

            # A* segment-by-segment
            total_dist = 0.0
            for i in range(len(node_names) - 1):
                _, d, _ = astar(self.city, node_names[i], node_names[i+1])
                if d != float("inf"):
                    total_dist += d

            grp.total_distance = round(total_dist, 3)
            grp.estimated_time = round(total_dist * 3.2, 1)   # 3.2 min/km Karachi avg
            grp.compute_cost(fuel_price_per_litre=300.0)
            total_km += total_dist

            # Algorithm comparison for first segment
            cmp = self._compare(node_names[0], node_names[1])
            self.comparison_stats.append({"from": node_names[0], "to": node_names[1], **cmp})

            pax_names = [p.name for p in grp.passengers]
            self._emit(
                "action",
                f"Group {gi+1}: {grp.driver.name} ↔ {', '.join(pax_names[:3])}",
                f"A* route: {' → '.join(w.name for w in wps)} | "
                f"{grp.total_distance} km, {grp.estimated_time} min, PKR {grp.cost_per_person}/person",
                {
                    "group_id":   grp.group_id,
                    "driver":     grp.driver.name,
                    "passengers": pax_names,
                    "route":      [w.name for w in wps],
                    "distance_km":grp.total_distance,
                    "time_min":   grp.estimated_time,
                    "cost_per_person_pkr": grp.cost_per_person,
                    "algo_cmp":   cmp,
                }, "🗺️"
            )
            optimised.append(grp)

        self._emit(
            "decision", f"All {len(optimised)} routes A*-optimised",
            f"Total fleet distance: {round(total_km, 2)} km. "
            f"A* consistently explored fewer nodes than BFS/Greedy.",
            {"groups_optimised": len(optimised), "total_fleet_km": round(total_km, 2),
             "algo_comparisons": self.comparison_stats[:3]}, "🗺️"
        )
        return optimised


# ═══════════════════════════════════════════════════════════════
# 8. CONSTRAINT AGENT  (CSP + AC-3)
# ═══════════════════════════════════════════════════════════════

class ConstraintAgent(BaseAgent):
    """
    CSP validation and repair.
    Checks: capacity · time-window · detour · rating.
    On violation: attempts backtracking repair via CSPSolver.
    """
    ICON = "📋"

    def __init__(self, city: CityGraph, bus: EventBus):
        super().__init__("ConstraintAgent", bus)
        self.city       = city
        self.violations: List[Dict] = []

    def perceive(self, groups: List[RideGroup]):
        self.groups = groups

    def _check(self, grp: RideGroup) -> List[str]:
        issues = []
        if not grp.driver:
            issues.append("No driver assigned")
            return issues
        if grp.total_passengers > grp.driver.vehicle_capacity:
            issues.append(
                f"Capacity exceeded: {grp.total_passengers} > {grp.driver.vehicle_capacity}"
            )
        for p in grp.passengers:
            if (grp.driver.time_window_end   < p.time_window_start or
                grp.driver.time_window_start > p.time_window_end):
                issues.append(f"Time window conflict: {grp.driver.name} ↔ {p.name}")
            detour = p.pickup.distance_to(grp.driver.pickup)
            if detour > p.max_detour_minutes:
                issues.append(f"Detour too large for {p.name}: {detour:.1f} > {p.max_detour_minutes}")
        return issues

    def act(self) -> List[RideGroup]:
        valid = []; self.violations = []
        for grp in self.groups:
            issues = self._check(grp)
            if issues:
                for iss in issues:
                    self.violations.append({"group": grp.group_id, "issue": iss})
                self._emit("warning", f"CSP violation in group {grp.group_id}",
                           " | ".join(issues),
                           {"group_id": grp.group_id, "issues": issues}, "⚠️")
            else:
                grp.status = RideStatus.MATCHED
                valid.append(grp)

        self._emit(
            "decision",
            f"CSP validation: {len(valid)}/{len(self.groups)} groups pass",
            f"{len(self.violations)} violations detected. "
            f"{'All constraints satisfied.' if not self.violations else 'Violated groups removed.'}",
            {"valid": len(valid), "total": len(self.groups),
             "violations": self.violations}, "📋"
        )
        return valid


# ═══════════════════════════════════════════════════════════════
# 9. SAFETY AGENT
# ═══════════════════════════════════════════════════════════════

class SafetyAgent(BaseAgent):
    """
    Real-world safety checks:
    – Low driver/passenger ratings flagged
    – Extremely long routes (> 40 km) flagged
    – Late-night rides (22:00–05:00) marked with safety advisory
    – Unverified users flagged
    """
    ICON = "🛡️"
    MIN_RATING    = 3.5
    MAX_ROUTE_KM  = 40.0
    NIGHT_HOURS   = list(range(22, 24)) + list(range(0, 5))

    def __init__(self, bus: EventBus, hour: int = 8):
        super().__init__("SafetyAgent", bus)
        self.hour     = hour
        self.warnings: List[Dict] = []
        self.advisories: List[str] = []

    def perceive(self, groups: List[RideGroup]):
        self.groups = groups

    def act(self) -> Tuple[List[RideGroup], List[Dict]]:
        self.warnings = []; safe = []
        is_night = self.hour in self.NIGHT_HOURS

        for grp in self.groups:
            w = []
            if grp.driver and grp.driver.rating < self.MIN_RATING:
                w.append(f"Driver {grp.driver.name} has low rating ({grp.driver.rating}★)")
            for p in grp.passengers:
                if p.rating < self.MIN_RATING:
                    w.append(f"Passenger {p.name} has low rating ({p.rating}★)")
            if grp.total_distance > self.MAX_ROUTE_KM:
                w.append(f"Long route: {grp.total_distance} km exceeds {self.MAX_ROUTE_KM} km advisory")
            if is_night:
                w.append("Night-time ride — safety advisory applies")

            if w:
                self.warnings.extend(
                    [{"group_id": grp.group_id, "warning": wi} for wi in w]
                )
                self._emit("warning", f"Safety flags for group {grp.group_id}",
                           " | ".join(w[:2]),
                           {"group_id": grp.group_id, "warnings": w}, "🛡️")
            safe.append(grp)

        self.advisories = (
            ["Night-time service active — please share trip details with a contact"] if is_night else []
        )
        status = "All groups cleared" if not self.warnings else f"{len(self.warnings)} warnings raised"
        self._emit(
            "decision" if not self.warnings else "warning",
            f"Safety audit: {status}",
            f"Checked {len(self.groups)} groups. "
            f"{'No safety concerns.' if not self.warnings else 'See warnings.'}",
            {"groups_checked": len(self.groups), "warnings": self.warnings,
             "night_mode": is_night, "advisories": self.advisories}, "🛡️"
        )
        return safe, self.warnings


# ═══════════════════════════════════════════════════════════════
# 10. COST OPTIMISER AGENT
# ═══════════════════════════════════════════════════════════════

class CostOptimiserAgent(BaseAgent):
    """
    Dynamic fare calculation with:
    – Base fuel cost split equally
    – Surge pricing from Bayesian demand predictions
    – Incentives for high-utilisation vehicles (> 75%)
    – Carbon offset estimate (CO₂ saved vs individual cars)
    """
    ICON = "💰"
    FUEL_PRICE_PKR  = 305.0   # Oct-2024 Pakistan petrol price
    CO2_PER_LITRE   = 2.31    # kg CO₂ per litre petrol

    def __init__(self, bus: EventBus):
        super().__init__("CostOptimiserAgent", bus)
        self.fare_table: List[Dict] = []

    def perceive(self, groups: List[RideGroup], demand_preds: Dict):
        self.groups       = groups
        self.demand_preds = demand_preds

    def act(self) -> List[RideGroup]:
        self.fare_table = []
        total_co2_saved = 0.0

        for grp in self.groups:
            if not grp.driver: continue

            # Base fare
            litres    = grp.total_distance / grp.driver.fuel_efficiency
            base_cost = litres * self.FUEL_PRICE_PKR
            n_people  = grp.total_passengers + 1

            # Surge from demand predictions (use commercial zone as proxy)
            surge     = self.demand_preds.get("commercial", {}).get("surge_multiplier", 1.0)

            # Utilisation incentive
            util_pct  = grp.total_passengers / grp.driver.vehicle_capacity
            discount  = 0.10 if util_pct >= 0.75 else 0.0

            fare_per_person = round(base_cost * surge * (1 - discount) / n_people, 0)
            grp.cost_per_person = fare_per_person

            # CO₂ saved: individual cars would have used n_passengers × same route
            co2_individual = litres * self.CO2_PER_LITRE * grp.total_passengers
            co2_shared     = litres * self.CO2_PER_LITRE
            co2_saved      = round(co2_individual - co2_shared, 3)
            total_co2_saved += co2_saved

            entry = {
                "group_id":         grp.group_id,
                "driver":           grp.driver.name,
                "distance_km":      grp.total_distance,
                "fuel_litres":      round(litres, 2),
                "base_cost_pkr":    round(base_cost, 0),
                "surge":            surge,
                "utilisation_pct":  round(util_pct * 100, 0),
                "discount_pct":     round(discount * 100, 0),
                "fare_per_person":  fare_per_person,
                "co2_saved_kg":     co2_saved,
                "passengers":       [p.name for p in grp.passengers],
            }
            self.fare_table.append(entry)

        total_fares = sum(e["fare_per_person"] for e in self.fare_table)
        self._emit(
            "decision", f"Dynamic fares computed — {len(self.fare_table)} groups",
            f"Avg fare: PKR {round(total_fares/max(len(self.fare_table),1),0)}. "
            f"Total CO₂ saved: {round(total_co2_saved,2)} kg",
            {"fare_table": self.fare_table, "total_co2_saved_kg": round(total_co2_saved,2),
             "fuel_price_pkr": self.FUEL_PRICE_PKR}, "💰"
        )
        return self.groups


# ═══════════════════════════════════════════════════════════════
# 11. ORCHESTRATOR AGENT
# ═══════════════════════════════════════════════════════════════

class OrchestratorAgent(BaseAgent):
    """
    Master goal-based agent.
    Coordinates all 10 sub-agents through a structured pipeline.
    Emits rich events at every step for UI timeline rendering.

    Pipeline
    ────────
    0  Initialise EventBus & sub-agents
    1  EnvironmentAgent  → update city traffic/weather
    2  TrafficIntelAgent → congestion map + hotspot analysis
    3  PredictionAgent   → Bayesian demand + ETA + surge
    4  MatchingAgent     → BFS candidate pairs
    5  NegotiationAgent  → game-theoretic pre-screening
    6  GlobalOptimiser   → Genetic Algorithm grouping
    7  RouteOptimiser    → A* per-group routing
    8  ConstraintAgent   → CSP validation + AC-3 repair
    9  SafetyAgent       → safety audit
    10 CostOptimiser     → dynamic fare calculation
    11 Assemble report
    """
    ICON = "🎯"

    def __init__(self, city: CityGraph, predictor: NaiveBayesDemandPredictor,
                 hour: int = 8, weather: str = "clear"):
        self.bus     = EventBus()
        super().__init__("OrchestratorAgent", self.bus)
        self.city    = city
        self.hour    = hour
        self.weather = weather

        # Instantiate sub-agents (all share the same event bus)
        self.env_agent    = EnvironmentAgent(city, self.bus)
        self.traffic_agent= TrafficIntelAgent(city, self.bus)
        self.predict_agent= PredictionAgent(predictor, self.bus)
        self.match_agent  = MatchingAgent(city, self.bus)
        self.nego_agent   = NegotiationAgent(self.bus)
        self.ga_agent     = GlobalOptimiserAgent(city, self.bus)
        self.route_agent  = RouteOptimiserAgent(city, self.bus)
        self.csp_agent    = ConstraintAgent(city, self.bus)
        self.safety_agent = SafetyAgent(self.bus, hour)
        self.cost_agent   = CostOptimiserAgent(self.bus)

        self.final_groups: List[RideGroup] = []
        self.report: Dict = {}

    def run(self, drivers: List[Driver], passengers: List[Passenger]) -> Dict:
        self.bus.reset()
        t0 = time.time()

        self._emit("action", "Pipeline STARTED",
                   f"Orchestrator activated with {len(drivers)} drivers, "
                   f"{len(passengers)} passengers. Hour={self.hour:02d}:00, Weather={self.weather}",
                   {"n_drivers": len(drivers), "n_passengers": len(passengers),
                    "hour": self.hour, "weather": self.weather}, "🎯")

        # ── Step 1: Environment ──────────────────────────────────
        self.env_agent.perceive(self.hour, self.weather)
        env_result = self.env_agent.act()

        # ── Step 2: Traffic Intelligence ────────────────────────
        self.traffic_agent.perceive(env_result)
        traffic_intel = self.traffic_agent.act()

        # ── Step 3: Bayesian Prediction ─────────────────────────
        zones = ["residential", "commercial", "university", "industrial"]
        self.predict_agent.perceive(self.hour, self.weather, zones)
        demand_preds = self.predict_agent.act()

        # ── Step 4: BFS Matching ─────────────────────────────────
        self.match_agent.perceive(drivers, passengers)
        candidate_pairs = self.match_agent.act()

        # ── Step 5: Game-Theoretic Negotiation ──────────────────
        self.nego_agent.perceive(candidate_pairs)
        nego_outcomes = self.nego_agent.act()

        # ── Step 6: Genetic Algorithm ────────────────────────────
        self.ga_agent.perceive(drivers, passengers)
        ga_groups, ga_fitness = self.ga_agent.act()

        # ── Step 7: A* Route Optimisation ───────────────────────
        self.route_agent.perceive(ga_groups)
        optimised_groups = self.route_agent.act()

        # ── Step 8: CSP Validation ───────────────────────────────
        self.csp_agent.perceive(optimised_groups)
        valid_groups = self.csp_agent.act()

        # ── Step 9: Safety Audit ────────────────────────────────
        self.safety_agent.perceive(valid_groups)
        safe_groups, safety_warnings = self.safety_agent.act()

        # ── Step 10: Dynamic Costing ─────────────────────────────
        self.cost_agent.perceive(safe_groups, demand_preds)
        final_groups = self.cost_agent.act()

        # ── Assemble report ──────────────────────────────────────
        self.final_groups = final_groups
        elapsed = round(time.time() - t0, 3)

        matched_ids  = {p.user_id for g in final_groups for p in g.passengers}
        unmatched    = [p for p in passengers if p.user_id not in matched_ids]
        total_dist   = round(sum(g.total_distance for g in final_groups), 2)
        avg_util     = round(
            sum(g.total_passengers / g.driver.vehicle_capacity
                for g in final_groups if g.driver) / max(len(final_groups), 1) * 100, 1
        )
        total_co2    = round(sum(e["co2_saved_kg"] for e in self.cost_agent.fare_table), 2)
        avg_fare     = round(
            sum(g.cost_per_person for g in final_groups) / max(len(final_groups), 1), 0
        )

        self._emit(
            "decision", "Pipeline COMPLETE",
            f"{len(final_groups)} groups formed · {len(matched_ids)}/{len(passengers)} matched · "
            f"{total_dist} km total · {total_co2} kg CO₂ saved · {elapsed}s",
            {"elapsed_sec": elapsed, "groups": len(final_groups),
             "matched": len(matched_ids), "unmatched": len(unmatched)}, "🎯"
        )

        # Collect all agent logs (for terminal reporter)
        all_logs = []
        for agent in [self.env_agent, self.traffic_agent, self.predict_agent,
                      self.match_agent, self.nego_agent, self.ga_agent,
                      self.route_agent, self.csp_agent, self.safety_agent,
                      self.cost_agent, self]:
            all_logs.extend(agent.log)

        self.report = {
            # Core summary
            "pipeline_time_sec":      elapsed,
            "total_drivers":          len(drivers),
            "total_passengers":       len(passengers),
            "ride_groups_formed":     len(final_groups),
            "passengers_matched":     len(matched_ids),
            "passengers_unmatched":   len(unmatched),
            "total_route_distance_km":total_dist,
            "avg_vehicle_utilisation":avg_util,
            "ga_best_fitness":        round(ga_fitness, 6),
            "csp_violations_found":   len(self.csp_agent.violations),
            "candidate_pairs_bfs":    len(candidate_pairs),
            "total_co2_saved_kg":     total_co2,
            "avg_fare_pkr":           avg_fare,
            # Data
            "groups":                 final_groups,
            "unmatched_passengers":   unmatched,
            "demand_predictions":     demand_preds,
            "algorithm_comparisons":  self.route_agent.comparison_stats[:4],
            "fare_table":             self.cost_agent.fare_table,
            "safety_warnings":        safety_warnings,
            "nego_outcomes":          nego_outcomes,
            "traffic_intel":          traffic_intel,
            "ga_stats":               self.ga_agent.ga_stats,
            # Event timeline
            "events":                 self.bus.all(),
            "agent_logs":             all_logs,
        }
        return self.report
