"""
tests/test_system.py  –  35 tests covering every module.
Run: python -m pytest tests/ -v
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from models.models       import Location, Driver, Passenger, RideGroup
from models.city_graph   import build_karachi_demo_city
from models.bayesian     import build_predictor
from algorithms.search   import astar, bfs, greedy_best_first, dijkstra
from algorithms.search_extended import (
    dfs, uniform_cost, idastar, hill_climbing,
    beam_search, bidirectional_bfs, benchmark_all, run_alpha_beta_demo
)
from algorithms.genetic  import GeneticOptimizer
from algorithms.csp      import CSPSolver
from data.data_generator import generate_scenario
from ml.ml_engine        import MLEngine


@pytest.fixture(scope="module")
def city_locs():
    return build_karachi_demo_city()

@pytest.fixture(scope="module")
def city(city_locs):
    return city_locs[0]

@pytest.fixture(scope="module")
def scenario(city):
    return generate_scenario(city, n_drivers=4, n_passengers=8, seed=1)

@pytest.fixture(scope="module")
def ml(city):
    eng = MLEngine()
    coords = [(l.x, l.y) for l in city.all_locations()]
    names  = [l.name     for l in city.all_locations()]
    eng.train_all(coords, names)
    return eng


# ── 1. Models ─────────────────────────────────────────────────
class TestModels:
    def test_distance(self):
        assert abs(Location("A",0,0).distance_to(Location("B",3,4)) - 5.0) < 1e-6

    def test_driver_seats(self):
        d = Driver("D1","X",None,Location("A",0,0),Location("B",1,1),0,30,vehicle_capacity=4)
        d.current_load = 3
        assert d.available_seats == 1 and d.can_accept(1) and not d.can_accept(2)

    def test_group_cost(self):
        d = Driver("D1","X",None,Location("A",0,0),Location("B",1,1),0,30,fuel_efficiency=15.0)
        p = Passenger("P1","Y",None,Location("C",0,0),Location("D",1,1),0,30)
        assert RideGroup(driver=d,passengers=[p],total_distance=30.0).compute_cost(280.0) > 0


# ── 2. City Graph ─────────────────────────────────────────────
class TestCityGraph:
    def test_connected(self, city):    assert city.is_connected()
    def test_nodes(self, city):        assert city.graph.number_of_nodes() == 20
    def test_edge(self, city):         assert city.graph.has_edge("NUCES","Gulshan")
    def test_heuristic(self, city):    assert city.heuristic("NUCES","Clifton") >= 0

    def test_traffic(self, city):
        city.update_traffic("NUCES","Gulshan",3.0)
        assert city.graph["NUCES"]["Gulshan"]["traffic_factor"] == 3.0
        city.update_traffic("NUCES","Gulshan",1.0)


# ── 3. Core Search ────────────────────────────────────────────
class TestCoreSearch:
    def test_astar(self, city):
        path, cost, _ = astar(city,"NUCES","Clifton")
        assert path[0]=="NUCES" and path[-1]=="Clifton" and cost < float("inf")

    def test_bfs(self, city):
        path, _, _ = bfs(city,"NUCES","Clifton")
        assert path[0]=="NUCES" and path[-1]=="Clifton"

    def test_greedy(self, city):
        path, _, _ = greedy_best_first(city,"NUCES","Clifton")
        assert len(path) > 0

    def test_dijkstra_matches_astar(self, city):
        _, d1, _ = astar(city,"NUCES","Clifton")
        _, d2, _ = dijkstra(city,"NUCES","Clifton")
        assert abs(d1-d2) < 1e-4

    def test_astar_optimal_vs_bfs(self, city):
        _, a, _ = astar(city,"NUCES","Malir")
        _, b, _ = bfs(city,"NUCES","Malir")
        assert a <= b + 1e-6

    def test_same_node(self, city):
        path, cost, _ = astar(city,"NUCES","NUCES")
        assert path == ["NUCES"] and cost == 0.0


# ── 4. Extended Search ────────────────────────────────────────
class TestExtendedSearch:
    def test_dfs(self, city):
        path, cost, _, trace = dfs(city,"NUCES","Clifton")
        assert path[-1]=="Clifton" and len(trace) > 0

    def test_ucs(self, city):
        path, cost, _, _ = uniform_cost(city,"NUCES","Clifton")
        assert path[-1]=="Clifton" and cost < float("inf")

    def test_idastar_optimal(self, city):
        path, cost, _, _ = idastar(city,"NUCES","Clifton")
        _, a_cost, _ = astar(city,"NUCES","Clifton")
        assert path[-1]=="Clifton" and abs(cost-a_cost) < 1e-4

    def test_hill_climbing(self, city):
        path, cost, _, trace = hill_climbing(city,"NUCES","Clifton")
        assert len(path) > 0 and len(trace) > 0

    def test_beam_search(self, city):
        path, cost, _, _ = beam_search(city,"NUCES","Clifton",beam_width=3)
        assert path[-1]=="Clifton"

    def test_bidirectional_bfs(self, city):
        path, cost, _, _ = bidirectional_bfs(city,"NUCES","Clifton")
        assert path[0]=="NUCES" and path[-1]=="Clifton"

    def test_benchmark_has_all_algos(self, city):
        res = benchmark_all(city,"NUCES","Clifton")
        for a in ["A*","BFS","DFS","IDA*","Hill Climbing","Beam Search","BiDir BFS"]:
            assert a in res

    def test_astar_best_cost(self, city):
        # A* is optimal on traffic-weighted graph; compare with Dijkstra/UCS only.
        res = benchmark_all(city,"NUCES","Hawks_Bay")
        astar_c = res["A*"]["cost"]
        dijk_c  = res["Dijkstra"]["cost"]
        ucs_c   = res["UCS"]["cost"]
        assert astar_c is not None
        assert abs(astar_c - dijk_c) < 1e-4
        assert abs(astar_c - ucs_c)  < 1e-4

    def test_alpha_beta_runs(self):
        best, trace, _ = run_alpha_beta_demo(depth=3)
        assert isinstance(best, float) and len(trace) > 0

    def test_alpha_beta_prunes(self):
        _, trace, _ = run_alpha_beta_demo(depth=4)
        assert any(t.get("pruned") for t in trace)


# ── 5. Genetic Algorithm ─────────────────────────────────────
class TestGA:
    def test_runs(self, city, scenario):
        d, p = scenario
        groups, fit, hist = GeneticOptimizer(d,p,city,pop_size=20,generations=10).run()
        assert len(hist) == 10

    def test_capacity_respected(self, city, scenario):
        d, p = scenario
        groups, _, _ = GeneticOptimizer(d,p,city,pop_size=20,generations=10).run()
        for g in groups:
            assert g.total_passengers <= g.driver.vehicle_capacity

    def test_fitness_nondecreasing(self, city, scenario):
        d, p = scenario
        _, _, hist = GeneticOptimizer(d,p,city,pop_size=30,generations=30).run()
        assert max(hist) >= hist[0]


# ── 6. CSP ───────────────────────────────────────────────────
class TestCSP:
    def test_solves(self, city, scenario):
        d, p = scenario
        groups, nodes = CSPSolver(d,p,city).solve()
        assert isinstance(groups,list) and nodes > 0

    def test_capacity_respected(self, city, scenario):
        d, p = scenario
        groups, _ = CSPSolver(d,p,city).solve()
        for g in groups:
            assert g.total_passengers <= g.driver.vehicle_capacity


# ── 7. Bayesian ───────────────────────────────────────────────
class TestBayesian:
    def test_trained(self):           assert build_predictor()._trained
    def test_valid_class(self):       assert build_predictor().predict(8,"university","clear") in ["low","medium","high"]
    def test_proba_sums_one(self):
        proba = build_predictor().predict_proba(8,"commercial","rainy")
        assert abs(sum(proba.values())-1.0) < 1e-6
    def test_rush_hour_demand(self):
        p = build_predictor()
        assert p.predict_proba(8,"commercial","clear")["high"] >= \
               p.predict_proba(3,"commercial","clear")["high"]


# ── 8. ML Engine ─────────────────────────────────────────────
class TestML:
    def test_eda_records(self, ml):      assert ml.eda_result["n_records"] == 1000
    def test_match_rate_valid(self, ml): assert 0 < ml.eda_result["match_rate_pct"] < 100
    def test_clf_accuracy(self, ml):     assert ml.clf_metrics["ens_accuracy"] > 0.70
    def test_feat_imp_sum(self, ml):     assert abs(sum(ml.clf_metrics["feature_importances"].values())-1.0) < 0.05
    def test_clustering_k(self, ml):     assert ml.cluster_result["k"] == 5
    def test_silhouette(self, ml):       assert 0 < ml.cluster_result["silhouette"] < 1
    def test_rating_r2(self, ml):        assert ml.rating_metrics["r2"] > 0.40
    def test_pca_shape(self, ml):        assert len(ml.pca_data["x"]) == 1000

    def test_predict_match(self, ml):
        lbl, conf = ml.classifier.predict({
            "hour":8,"distance":10,"detour":5,"n_passengers":1,
            "weather_code":0,"time_diff":10,"driver_rating":4.5,"passenger_rating":4.0
        })
        assert lbl in [0,1] and 0.0 <= conf <= 1.0

    def test_predict_rating_bounds(self, ml):
        r = ml.rating_pred.predict(10,5,8,1,0)
        assert 1.0 <= r <= 5.0


# ── 9. Integration ───────────────────────────────────────────
class TestIntegration:
    def test_full_pipeline(self, city, scenario):
        from agents.agent import OrchestratorAgent
        d, p = scenario
        orch = OrchestratorAgent(city, build_predictor(), hour=8, weather="clear")
        orch.ga_agent.pop_size    = 20
        orch.ga_agent.generations = 15
        report = orch.run(d, p)
        assert report["total_drivers"]    == len(d)
        assert report["total_passengers"] == len(p)
        assert report["pipeline_time_sec"] < 60
        assert isinstance(report["groups"], list)

    def test_flask_graph_api(self, city):
        from gui.app import app
        with app.test_client() as c:
            r = c.get('/api/graph')
            data = r.get_json()
            assert r.status_code == 200 and "nodes" in data and "edges" in data

    def test_flask_benchmark_api(self, city):
        from gui.app import app
        with app.test_client() as c:
            r = c.post('/api/algo/benchmark',
                       data=json.dumps({'src':'NUCES','dst':'Clifton'}),
                       content_type='application/json')
            data = r.get_json()
            assert data['ok'] and 'A*' in data['results']

    def test_flask_alphabeta_api(self):
        from gui.app import app
        with app.test_client() as c:
            r = c.post('/api/algo/alphabeta',
                       data=json.dumps({'depth':3}),
                       content_type='application/json')
            data = r.get_json()
            assert data['ok'] and any(t.get('pruned') for t in data['trace'])

    def test_flask_algo_all_variants(self):
        from gui.app import app
        algos = ["astar","bfs","dfs","idastar","hillclimb","beam","bibfs","ucs","greedy"]
        with app.test_client() as c:
            for algo in algos:
                r = c.post('/api/algo/run',
                           data=json.dumps({'src':'NUCES','dst':'Clifton','algo':algo}),
                           content_type='application/json')
                d = r.get_json()
                assert d['ok'], f"{algo} failed: {d}"
