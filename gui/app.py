"""
gui/app.py  –  Enhanced Flask Backend with SQLite DB – Carpooling AI
"""
from __future__ import annotations
import sys, os, base64, threading, traceback, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify
from database.db import (
    init_db, get_analytics, get_users, get_user, create_user,
    get_rides, get_ride, create_ride, create_booking,
    get_bookings_for_ride, get_user_rides, add_review,
    create_payment, add_notification, get_notifications,
    mark_notifications_read, save_algo_run, save_match_session
)
from models.city_graph   import build_karachi_demo_city
from models.bayesian     import build_predictor
from models.models       import Driver, Passenger, RideGroup
from data.data_generator import generate_scenario
from agents.agent        import OrchestratorAgent
from ml.ml_engine        import MLEngine
from algorithms.search_extended import (
    dfs, uniform_cost, idastar, hill_climbing,
    beam_search, bidirectional_bfs, benchmark_all, run_alpha_beta_demo
)
from algorithms.search import astar, bfs, greedy_best_first, dijkstra

app = Flask(__name__, template_folder="templates", static_folder="static")

# ── Init DB ───────────────────────────────────────────────────
init_db()

# ── Singletons ────────────────────────────────────────────────
city, locs  = build_karachi_demo_city()
predictor   = build_predictor()
ml_engine   = MLEngine()
LOCATION_NAMES = sorted(locs.keys())

def _train_ml():
    coords = [(l.x, l.y) for l in city.all_locations()]
    names  = [l.name     for l in city.all_locations()]
    ml_engine.train_all(coords, names)

threading.Thread(target=_train_ml, daemon=True).start()

_state = {
    "drivers":[], "passengers":[], "report":None,
    "running":False, "ga_hist":[], "events":[]
}

# ── Helpers ───────────────────────────────────────────────────
def _loc(n): return locs[n]

def _b64(fname):
    p = os.path.join(os.path.dirname(__file__), "..", "output", fname)
    if not os.path.exists(p): return None
    with open(p,"rb") as f: return base64.b64encode(f.read()).decode()

def _ser_group(g: RideGroup):
    return {
        "group_id":    g.group_id,
        "driver_name": g.driver.name if g.driver else "—",
        "driver_cap":  g.driver.vehicle_capacity if g.driver else 0,
        "driver_rating": g.driver.rating if g.driver else 0,
        "passengers":  [{"name":p.name,"pickup":p.pickup.name,
                          "dropoff":p.dropoff.name,"seats":p.num_people,"rating":p.rating}
                         for p in g.passengers],
        "route":       [w.name for w in g.route],
        "distance_km": round(g.total_distance,2),
        "time_min":    round(g.estimated_time,1),
        "cost_pkr":    round(g.cost_per_person),
        "seats_used":  g.total_passengers,
        "seats_total": g.driver.vehicle_capacity if g.driver else 0,
        "fitness":     round(g.fitness_score,4),
        "status":      g.status.value,
    }

def _graph_json():
    nodes=[{"id":l.name,"x":l.x,"y":l.y} for l in city.all_locations()]
    edges=[{"source":u,"target":v,
            "distance":round(d["distance"],2),"traffic":d.get("traffic_factor",1.0)}
           for u,v,d in city.graph.edges(data=True)]
    return {"nodes":nodes,"edges":edges}

# ══ Core Routes ════════════════════════════════════════════════

@app.route("/")
def index(): return render_template("index.html", locations=LOCATION_NAMES)

@app.route("/api/locations")
def api_locations(): return jsonify(LOCATION_NAMES)

@app.route("/api/graph")
def api_graph(): return jsonify(_graph_json())

@app.route("/api/state")
def api_state():
    return jsonify({
        "drivers":   [{"id":d.user_id,"name":d.name,"pickup":d.pickup.name,
                        "dropoff":d.dropoff.name,"capacity":d.vehicle_capacity,
                        "rating":d.rating,"tw":f"{d.time_window_start:.0f}–{d.time_window_end:.0f}"}
                       for d in _state["drivers"]],
        "passengers":[{"id":p.user_id,"name":p.name,"pickup":p.pickup.name,
                        "dropoff":p.dropoff.name,"seats":p.num_people,"rating":p.rating,
                        "tw":f"{p.time_window_start:.0f}–{p.time_window_end:.0f}"}
                       for p in _state["passengers"]],
        "running":   _state["running"],
        "has_result":_state["report"] is not None,
    })

@app.route("/api/add_driver", methods=["POST"])
def add_driver():
    d=request.json; idx=len(_state["drivers"])+1
    obj=Driver(user_id=f"D{idx:02d}",name=d["name"],role=None,
               pickup=_loc(d["pickup"]),dropoff=_loc(d["dropoff"]),
               time_window_start=float(d["tw_start"]),time_window_end=float(d["tw_end"]),
               rating=float(d.get("rating",4.5)),vehicle_capacity=int(d["capacity"]),
               fuel_efficiency=float(d.get("fuel_eff",15.0)))
    _state["drivers"].append(obj)
    return jsonify({"ok":True,"id":obj.user_id,"count":len(_state["drivers"])})

@app.route("/api/add_passenger", methods=["POST"])
def add_passenger():
    p=request.json; idx=len(_state["passengers"])+1
    obj=Passenger(user_id=f"P{idx:02d}",name=p["name"],role=None,
                  pickup=_loc(p["pickup"]),dropoff=_loc(p["dropoff"]),
                  time_window_start=float(p["tw_start"]),time_window_end=float(p["tw_end"]),
                  rating=float(p.get("rating",4.0)),num_people=int(p.get("seats",1)),
                  max_detour_minutes=float(p.get("max_detour",15.0)))
    _state["passengers"].append(obj)
    return jsonify({"ok":True,"id":obj.user_id,"count":len(_state["passengers"])})

@app.route("/api/remove_driver/<uid>",    methods=["DELETE"])
def remove_driver(uid):
    _state["drivers"]=[d for d in _state["drivers"] if d.user_id!=uid]
    return jsonify({"ok":True})

@app.route("/api/remove_passenger/<uid>", methods=["DELETE"])
def remove_passenger(uid):
    _state["passengers"]=[p for p in _state["passengers"] if p.user_id!=uid]
    return jsonify({"ok":True})

@app.route("/api/clear", methods=["POST"])
def clear_all():
    _state.update({"drivers":[],"passengers":[],"report":None,
                    "running":False,"ga_hist":[],"events":[]})
    return jsonify({"ok":True})

@app.route("/api/generate_demo", methods=["POST"])
def generate_demo():
    b=request.json or {}
    dr,pa=generate_scenario(city,n_drivers=int(b.get("drivers",5)),
                             n_passengers=int(b.get("passengers",12)),
                             hour=int(b.get("hour",8)),seed=int(b.get("seed",42)))
    _state.update({"drivers":dr,"passengers":pa,"report":None,"ga_hist":[],"events":[]})
    return jsonify({"ok":True,"drivers":len(dr),"passengers":len(pa)})

@app.route("/api/run", methods=["POST"])
def run_pipeline():
    if _state["running"]: return jsonify({"ok":False,"error":"Already running"})
    if not _state["drivers"] or not _state["passengers"]:
        return jsonify({"ok":False,"error":"Need ≥1 driver and ≥1 passenger"})
    b=request.json or {}
    def _run():
        _state["running"]=True
        t0 = time.time()
        try:
            orch=OrchestratorAgent(city,predictor,
                                   hour=int(b.get("hour",8)),
                                   weather=b.get("weather","clear"))
            orch.ga_agent.pop_size    =int(b.get("pop_size",60))
            orch.ga_agent.generations =int(b.get("generations",80))
            report=orch.run(list(_state["drivers"]),list(_state["passengers"]))
            try:
                from dashboard.visualiser import render_all
                render_all(city,report,orch.ga_agent.ga_history,predictor)
            except Exception: pass
            _state["report"] =report
            _state["ga_hist"]=orch.ga_agent.ga_history
            _state["events"] =report.get("events",[])
            # Persist to DB
            save_match_session({
                "n_drivers": len(_state["drivers"]),
                "n_passengers": len(_state["passengers"]),
                "groups_formed": report.get("ride_groups_formed",0),
                "matched": report.get("passengers_matched",0),
                "unmatched": report.get("passengers_unmatched",0),
                "co2_saved_kg": report.get("total_co2_saved_kg",0),
                "total_dist_km": report.get("total_route_distance_km",0),
                "avg_fare_pkr": report.get("avg_fare_pkr",0),
                "ga_fitness": report.get("ga_best_fitness",0),
                "pipeline_sec": time.time()-t0,
                "weather": b.get("weather","clear"),
                "hour": int(b.get("hour",8)),
            })
        except Exception as e:
            _state["report"]={"error":str(e),"tb":traceback.format_exc()}
        finally:
            _state["running"]=False
    threading.Thread(target=_run,daemon=True).start()
    return jsonify({"ok":True})

@app.route("/api/status")
def api_status():
    r=_state.get("report") or {}
    return jsonify({"running":_state["running"],
                    "has_result":bool(_state["report"]) and "error" not in r})

@app.route("/api/results")
def api_results():
    rep=_state.get("report")
    if rep is None:    return jsonify({"ready":False})
    if "error" in rep: return jsonify({"ready":False,"error":rep["error"]})
    groups=[_ser_group(g) for g in rep.get("groups",[])]
    unmatched=[{"name":p.name,"pickup":p.pickup.name,"dropoff":p.dropoff.name}
               for p in rep.get("unmatched_passengers",[])]
    algo_rows=[{"route":f"{c['from']} → {c['to']}","astar":c.get("A*",{}),
                "bfs":c.get("BFS",{}),"greedy":c.get("Greedy",{})}
               for c in rep.get("algorithm_comparisons",[])]
    return jsonify({
        "ready":True,
        "summary":{
            "pipeline_time":    rep.get("pipeline_time_sec"),
            "ride_groups":      rep.get("ride_groups_formed"),
            "matched":          rep.get("passengers_matched"),
            "unmatched":        rep.get("passengers_unmatched"),
            "total_distance":   rep.get("total_route_distance_km"),
            "avg_utilisation":  rep.get("avg_vehicle_utilisation"),
            "ga_fitness":       rep.get("ga_best_fitness"),
            "csp_violations":   rep.get("csp_violations_found"),
            "candidate_pairs":  rep.get("candidate_pairs_bfs"),
            "total_drivers":    rep.get("total_drivers"),
            "total_passengers": rep.get("total_passengers"),
            "co2_saved":        rep.get("total_co2_saved_kg"),
            "avg_fare_pkr":     rep.get("avg_fare_pkr"),
            "safety_warnings":  len(rep.get("safety_warnings",[])),
        },
        "groups":groups,"unmatched":unmatched,"algo_cmp":algo_rows,
        "demand":rep.get("demand_predictions",{}),
        "ga_hist":_state.get("ga_hist",[]),
        "ga_stats":rep.get("ga_stats",{}),
        "agent_logs":rep.get("agent_logs",[]),
        "events":rep.get("events",[]),
        "fare_table":rep.get("fare_table",[]),
        "safety_warnings":rep.get("safety_warnings",[]),
        "nego_outcomes":rep.get("nego_outcomes",[]),
        "traffic_intel":rep.get("traffic_intel",{}),
        "charts":{
            "city_routes":    _b64("city_routes.png"),
            "ga_convergence": _b64("ga_convergence.png"),
            "algo_comparison":_b64("algo_comparison.png"),
            "demand_heatmap": _b64("demand_heatmap.png"),
        },
    })

# ══ DATABASE API ═══════════════════════════════════════════════

@app.route("/api/db/analytics")
def db_analytics():
    return jsonify(get_analytics())

@app.route("/api/db/users")
def db_users():
    role = request.args.get("role")
    return jsonify(get_users(role=role))

@app.route("/api/db/users/<uid>")
def db_user(uid):
    u = get_user(uid)
    if not u: return jsonify({"error":"Not found"}), 404
    rides = get_user_rides(uid)
    notifs = get_notifications(uid)
    return jsonify({**u, "rides": rides, "notifications": notifs})

@app.route("/api/db/users", methods=["POST"])
def db_create_user():
    d = request.json or {}
    if not d.get("name") or not d.get("email"):
        return jsonify({"error":"name and email required"}), 400
    uid = create_user(d["name"], d["email"], d.get("phone",""), d.get("role","passenger"), d.get("avatar_color","#4f8ef7"))
    add_notification(uid, "welcome", "Welcome to RideKaro!", f"Hi {d['name']}, your account is ready. Start finding rides!")
    return jsonify({"ok":True,"id":uid})

@app.route("/api/db/rides")
def db_rides():
    status = request.args.get("status")
    origin = request.args.get("origin")
    dest   = request.args.get("destination")
    return jsonify(get_rides(status=status, origin=origin, destination=dest))

@app.route("/api/db/rides/<rid>")
def db_ride(rid):
    ride = get_ride(rid)
    if not ride: return jsonify({"error":"Not found"}), 404
    bookings = get_bookings_for_ride(rid)
    return jsonify({**ride, "bookings": bookings})

@app.route("/api/db/rides", methods=["POST"])
def db_create_ride():
    d = request.json or {}
    required = ["driver_id","vehicle_id","origin","destination","departure_time","seats_total","fare_per_seat"]
    missing = [f for f in required if not d.get(f)]
    if missing: return jsonify({"error":f"Missing: {missing}"}), 400
    rid = create_ride(
        driver_id=d["driver_id"], vehicle_id=d["vehicle_id"],
        origin=d["origin"], destination=d["destination"],
        departure_time=d["departure_time"],
        seats_total=int(d["seats_total"]),
        fare_per_seat=float(d["fare_per_seat"]),
        distance_km=float(d.get("distance_km",0)),
        duration_min=float(d.get("duration_min",0)),
        weather=d.get("weather","clear"),
        notes=d.get("notes",""),
        waypoints=d.get("waypoints",[])
    )
    driver = get_user(d["driver_id"])
    if driver:
        add_notification(d["driver_id"], "ride_created", "Ride Posted!", f"Your ride from {d['origin']} to {d['destination']} is now live.")
    return jsonify({"ok":True,"id":rid})

@app.route("/api/db/rides/<rid>/status", methods=["PATCH"])
def db_update_ride_status(rid):
    d = request.json or {}
    status = d.get("status")
    if status not in ("scheduled","active","completed","cancelled"):
        return jsonify({"error":"Invalid status"}), 400
    from database.db import get_conn
    with get_conn() as conn:
        conn.execute("UPDATE rides SET status=? WHERE id=?", (status, rid))
    return jsonify({"ok":True})

@app.route("/api/db/bookings", methods=["POST"])
def db_create_booking():
    d = request.json or {}
    required = ["ride_id","passenger_id","seats","pickup_stop","dropoff_stop","fare_paid"]
    missing = [f for f in required if d.get(f) is None]
    if missing: return jsonify({"error":f"Missing: {missing}"}), 400
    ride = get_ride(d["ride_id"])
    if not ride: return jsonify({"error":"Ride not found"}), 404
    if ride["seats_available"] < int(d["seats"]):
        return jsonify({"error":"Not enough seats available"}), 400
    bid = create_booking(
        ride_id=d["ride_id"], passenger_id=d["passenger_id"],
        seats=int(d["seats"]), pickup_stop=d["pickup_stop"],
        dropoff_stop=d["dropoff_stop"], fare_paid=float(d["fare_paid"])
    )
    passenger = get_user(d["passenger_id"])
    if passenger:
        add_notification(d["passenger_id"], "booking", "Booking Confirmed!", f"Your seat on the {ride['origin']} → {ride['destination']} ride is confirmed.")
    if ride.get("driver_id"):
        add_notification(ride["driver_id"], "new_passenger", "New Passenger!", f"{passenger['name'] if passenger else 'Someone'} booked a seat on your ride.")
    return jsonify({"ok":True,"id":bid})

@app.route("/api/db/reviews", methods=["POST"])
def db_add_review():
    d = request.json or {}
    if not all(k in d for k in ["ride_id","reviewer_id","reviewee_id","rating"]):
        return jsonify({"error":"Missing fields"}), 400
    rid = add_review(d["ride_id"], d["reviewer_id"], d["reviewee_id"], float(d["rating"]), d.get("comment",""))
    return jsonify({"ok":True,"id":rid})

@app.route("/api/db/payments", methods=["POST"])
def db_create_payment():
    d = request.json or {}
    if not d.get("booking_id") or not d.get("amount"):
        return jsonify({"error":"booking_id and amount required"}), 400
    result = create_payment(d["booking_id"], float(d["amount"]), d.get("method","cash"))
    return jsonify({"ok":True, **result})

@app.route("/api/db/notifications/<uid>")
def db_notifications(uid):
    unread_only = request.args.get("unread") == "1"
    notifs = get_notifications(uid, unread_only=unread_only)
    return jsonify(notifs)

@app.route("/api/db/notifications/<uid>/read", methods=["POST"])
def db_mark_read(uid):
    mark_notifications_read(uid)
    return jsonify({"ok":True})

# ══ Algorithm Visualiser ═══════════════════════════════════════

@app.route("/api/algo/run", methods=["POST"])
def algo_run():
    b=request.json or {}
    src=b.get("src","NUCES"); dst=b.get("dst","Clifton")
    alg=b.get("algo","astar")
    if src not in locs or dst not in locs:
        return jsonify({"ok":False,"error":"Invalid location"})
    MAP={
        "astar":    lambda:astar(city,src,dst)[:3]+([],),
        "bfs":      lambda:bfs(city,src,dst)[:3]+([],),
        "greedy":   lambda:greedy_best_first(city,src,dst)[:3]+([],),
        "dijkstra": lambda:dijkstra(city,src,dst)[:3]+([],),
        "dfs":      lambda:dfs(city,src,dst),
        "ucs":      lambda:uniform_cost(city,src,dst),
        "idastar":  lambda:idastar(city,src,dst),
        "hillclimb":lambda:hill_climbing(city,src,dst),
        "beam":     lambda:beam_search(city,src,dst),
        "bibfs":    lambda:bidirectional_bfs(city,src,dst),
    }
    fn=MAP.get(alg)
    if not fn: return jsonify({"ok":False,"error":"Unknown algorithm"})
    t0 = time.time()
    path,cost,explored,trace=fn()
    dur = (time.time()-t0)*1000
    try: save_algo_run(alg, src, dst, path, cost if cost!=float("inf") else None, explored, dur)
    except Exception: pass
    return jsonify({"ok":True,"path":path,
                    "cost":round(cost,3) if cost!=float("inf") else None,
                    "explored":explored,"trace":trace[:200],
                    "graph":_graph_json(),"duration_ms":round(dur,1)})

@app.route("/api/algo/benchmark", methods=["POST"])
def algo_benchmark():
    b=request.json or {}
    src=b.get("src","NUCES"); dst=b.get("dst","Clifton")
    if src not in locs or dst not in locs:
        return jsonify({"ok":False,"error":"Invalid location"})
    return jsonify({"ok":True,"src":src,"dst":dst,
                    "results":benchmark_all(city,src,dst)})

@app.route("/api/algo/alphabeta", methods=["POST"])
def algo_alphabeta():
    b=request.json or {}
    depth=max(2,min(5,int(b.get("depth",3))))
    best,trace,leaves=run_alpha_beta_demo(depth)
    return jsonify({"ok":True,"best":best,"trace":trace[:300],
                    "leaf_values":leaves[:32],"depth":depth})

# ══ ML Endpoints ═══════════════════════════════════════════════

@app.route("/api/ml/status")
def ml_status(): return jsonify({"ready":ml_engine.ready})

@app.route("/api/ml/all")
def ml_all():
    if not ml_engine.ready: return jsonify({"ready":False})
    return jsonify({"ready":True,
                    "eda":    ml_engine.eda_result,
                    "clf":    ml_engine.clf_metrics,
                    "pca":    ml_engine.pca_data,
                    "cluster":ml_engine.cluster_result,
                    "rating": ml_engine.rating_metrics,
                    "graph":  _graph_json()})

@app.route("/api/ml/predict_match", methods=["POST"])
def ml_predict_match():
    if not ml_engine.ready: return jsonify({"ready":False})
    lbl,conf=ml_engine.classifier.predict(request.json or {})
    return jsonify({"label":lbl,"confidence":conf,
                    "verdict":"✓ MATCH" if lbl==1 else "✗ NO MATCH"})

@app.route("/api/ml/predict_rating", methods=["POST"])
def ml_predict_rating():
    if not ml_engine.ready: return jsonify({"ready":False})
    b=request.json or {}
    r=ml_engine.rating_pred.predict(
        b.get("distance",10),b.get("detour",5),
        b.get("time_diff",10),b.get("n_pass",1),b.get("weather",0))
    return jsonify({"predicted_rating":r})

# ══ RAG ENDPOINTS ══════════════════════════════════════════════

from rag.rag_engine import get_rag_engine
from flask import Response, stream_with_context
import uuid as _uuid

# Warm up RAG engine on startup (builds FAISS index in background)
_rag = get_rag_engine()

@app.route("/api/rag/status")
def rag_status():
    return jsonify(get_rag_engine().get_stats())

@app.route("/api/rag/chat", methods=["POST"])
def rag_chat():
    """Non-streaming RAG chat endpoint."""
    b = request.json or {}
    query      = b.get("query", "").strip()
    session_id = b.get("session_id", "default")
    if not query:
        return jsonify({"error": "query is required"}), 400
    try:
        live_stats = get_analytics()
        result = get_rag_engine().generate(
            query=query,
            session_id=session_id,
            live_stats=live_stats,
            top_k=4,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "answer": f"Sorry, I encountered an error: {str(e)}"}), 500

@app.route("/api/rag/stream", methods=["POST"])
def rag_stream():
    """Streaming RAG chat endpoint — Server-Sent Events."""
    b = request.json or {}
    query      = b.get("query", "").strip()
    session_id = b.get("session_id", "default")
    if not query:
        return jsonify({"error": "query is required"}), 400

    live_stats = get_analytics()

    def generate():
        try:
            for chunk in get_rag_engine().stream_generate(
                query=query,
                session_id=session_id,
                live_stats=live_stats,
                top_k=4,
            ):
                yield chunk
        except Exception as e:
            import json as _json
            yield f"data: {_json.dumps({'type':'error','content':str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":   "no-cache",
            "X-Accel-Buffering":"no",
            "Connection":      "keep-alive",
        }
    )

@app.route("/api/rag/clear", methods=["POST"])
def rag_clear():
    """Clear conversation history for a session."""
    b = request.json or {}
    session_id = b.get("session_id", "default")
    get_rag_engine().clear_session(session_id)
    return jsonify({"ok": True, "cleared": session_id})

@app.route("/api/rag/new_session", methods=["POST"])
def rag_new_session():
    """Generate a new unique session ID."""
    sid = str(_uuid.uuid4())[:12]
    return jsonify({"session_id": sid})

@app.route("/api/rag/retrieve", methods=["POST"])
def rag_retrieve():
    """Retrieve relevant documents without generating an answer (debug/inspect)."""
    b = request.json or {}
    query = b.get("query", "").strip()
    top_k = int(b.get("top_k", 4))
    if not query:
        return jsonify({"error": "query is required"}), 400
    docs = get_rag_engine().retrieve(query, top_k=top_k)
    return jsonify({
        "query": query,
        "docs": [{"title": d["title"], "category": d["category"],
                  "score": round(d["score"], 3), "excerpt": d["text"][:200] + "..."}
                 for d in docs]
    })

if __name__ == "__main__":
    print("\n  🚗  RideKaro AI Platform → http://127.0.0.1:5000\n")
    app.run(debug=False,port=5000,threaded=True)
