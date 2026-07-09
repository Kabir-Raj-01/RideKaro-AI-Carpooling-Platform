"""
utils/reporter.py
Terminal dashboard with coloured, formatted output.
Uses only colorama + tabulate (no external UI deps).
"""
from __future__ import annotations
from typing import Dict, List
from colorama import Fore, Back, Style, init as colorama_init
from tabulate import tabulate

colorama_init(autoreset=True)

HEADER = Fore.CYAN + Style.BRIGHT
INFO   = Fore.GREEN
WARN   = Fore.YELLOW
ERROR  = Fore.RED
BOLD   = Style.BRIGHT
RESET  = Style.RESET_ALL


def _banner():
    print(HEADER + "=" * 72)
    print(HEADER + "   INTELLIGENT CARPOOLING OPTIMISATION SYSTEM  |  AI Lab Project")
    print(HEADER + "   NUCES – Kabir Raj (23K-0702) & Hassnain Aziz (23K-0905)")
    print(HEADER + "=" * 72 + RESET)


def _section(title: str):
    print()
    print(Fore.MAGENTA + Style.BRIGHT + f"┌─ {title} " + "─" * (60 - len(title)) + "┐" + RESET)


def print_pipeline_summary(report: Dict):
    _banner()
    _section("PIPELINE SUMMARY")
    rows = [
        ["Pipeline Time",         f"{report['pipeline_time_sec']}s"],
        ["Total Drivers",         report["total_drivers"]],
        ["Total Passengers",      report["total_passengers"]],
        ["Ride Groups Formed",    report["ride_groups_formed"]],
        ["Passengers Matched",    INFO + str(report["passengers_matched"]) + RESET],
        ["Passengers Unmatched",  WARN + str(report["passengers_unmatched"]) + RESET],
        ["Total Route Distance",  f"{report['total_route_distance_km']} km"],
        ["Avg Vehicle Utilisation", f"{report['avg_vehicle_utilisation']}%"],
        ["GA Best Fitness",       f"{report['ga_best_fitness']}"],
        ["CSP Violations Found",  report["csp_violations_found"]],
        ["BFS Candidate Pairs",   report["candidate_pairs_bfs"]],
    ]
    print(tabulate(rows, tablefmt="fancy_grid",
                   headers=[BOLD + "Metric" + RESET, BOLD + "Value" + RESET]))


def print_ride_groups(report: Dict):
    _section("RIDE GROUPS")
    groups = report.get("groups", [])
    if not groups:
        print(WARN + "  No ride groups formed." + RESET)
        return

    for i, grp in enumerate(groups, 1):
        driver = grp.driver
        pnames = ", ".join(p.name for p in grp.passengers)
        route_str = " → ".join(w.name for w in grp.route) if grp.route else "N/A"
        seats_used = grp.total_passengers
        seats_total = driver.vehicle_capacity if driver else "?"

        print()
        print(INFO + f"  ● Group {i}  [ID: {grp.group_id}]" + RESET)
        rows = [
            ["Driver",           driver.name if driver else "None"],
            ["Vehicle Capacity", f"{seats_used}/{seats_total} seats"],
            ["Passengers",       pnames or "None"],
            ["Route",            route_str],
            ["Distance",         f"{grp.total_distance:.2f} km"],
            ["Est. Time",        f"{grp.estimated_time:.1f} min"],
            ["Cost / Person",    f"PKR {grp.cost_per_person:.0f}"],
            ["Fitness Score",    f"{grp.fitness_score:.4f}"],
            ["Status",           INFO + grp.status.value.upper() + RESET],
        ]
        print(tabulate(rows, tablefmt="simple",
                       headers=["", ""], colalign=("right", "left")))


def print_unmatched(report: Dict):
    unmatched = report.get("unmatched_passengers", [])
    if not unmatched:
        return
    _section("UNMATCHED PASSENGERS")
    rows = [[p.user_id, p.name, p.pickup.name, p.dropoff.name,
             f"{p.time_window_start:.0f}–{p.time_window_end:.0f} min"]
            for p in unmatched]
    print(tabulate(rows,
                   headers=["ID", "Name", "Pickup", "Dropoff", "Time Window"],
                   tablefmt="fancy_grid"))


def print_demand_predictions(report: Dict):
    _section("BAYESIAN DEMAND PREDICTIONS")
    preds = report.get("demand_predictions", {})
    rows = []
    for zone, data in preds.items():
        demand = data["demand"]
        colour = {"high": ERROR, "medium": WARN, "low": INFO}.get(demand, "")
        prob_str = "  ".join(f"{k}:{v:.2f}" for k, v in data["probabilities"].items())
        rows.append([
            zone,
            colour + demand.upper() + RESET,
            f"×{data['eta_factor']}",
            f"×{data['surge_multiplier']}",
            prob_str,
        ])
    print(tabulate(rows,
                   headers=["Zone", "Demand", "ETA Factor", "Surge", "Probabilities"],
                   tablefmt="fancy_grid"))


def print_algorithm_comparison(report: Dict):
    _section("ALGORITHM COMPARISON  (A* vs BFS vs Greedy)")
    comparisons = report.get("algorithm_comparisons", [])
    if not comparisons:
        print(WARN + "  No comparison data available." + RESET)
        return

    for comp in comparisons:
        print(f"\n  Route: {BOLD}{comp['from']}{RESET} → {BOLD}{comp['to']}{RESET}")
        rows = []
        for algo in ["A*", "BFS", "Greedy"]:
            d = comp.get(algo, {})
            star = INFO + " ✓ OPTIMAL" + RESET if algo == "A*" else ""
            rows.append([
                algo,
                d.get("dist", "N/A"),
                d.get("explored", "N/A"),
                d.get("path_len", "N/A"),
                star,
            ])
        print(tabulate(rows,
                       headers=["Algorithm", "Distance", "Nodes Explored",
                                 "Path Length", "Note"],
                       tablefmt="simple"))


def print_agent_logs(report: Dict):
    _section("AGENT ACTIVITY LOG")
    for entry in report.get("agent_logs", []):
        if "Orchestrator" in entry:
            print(HEADER + "  " + entry + RESET)
        elif "GA" in entry or "Genetic" in entry:
            print(Fore.BLUE + "  " + entry + RESET)
        elif "CSP" in entry or "Constraint" in entry:
            print(Fore.YELLOW + "  " + entry + RESET)
        elif "Route" in entry or "A*" in entry:
            print(Fore.CYAN + "  " + entry + RESET)
        elif "Bayesian" in entry or "Prediction" in entry or "ETA" in entry:
            print(Fore.GREEN + "  " + entry + RESET)
        else:
            print("  " + entry)


def print_ga_convergence(history: List[float]):
    if not history:
        return
    _section("GENETIC ALGORITHM CONVERGENCE")
    # ASCII sparkline
    mn, mx = min(history), max(history)
    span = mx - mn or 1
    blocks = " ▁▂▃▄▅▆▇█"
    sparkline = ""
    for val in history:
        idx = int((val - mn) / span * (len(blocks) - 1))
        sparkline += blocks[idx]
    print(f"  Gen 0 → {len(history)}   "
          f"Best: {BOLD}{mx:.6f}{RESET}   "
          f"Start: {mn:.6f}")
    print(f"  {INFO}{sparkline}{RESET}")
    print(f"  Improvement: {INFO}+{((mx - mn) / max(abs(mn), 1e-9) * 100):.1f}%{RESET}")


def full_report(report: Dict, ga_history: List[float] = None):
    print_pipeline_summary(report)
    print_demand_predictions(report)
    print_algorithm_comparison(report)
    print_ride_groups(report)
    print_unmatched(report)
    if ga_history:
        print_ga_convergence(ga_history)
    print_agent_logs(report)
    print()
    print(HEADER + "=" * 72 + RESET)
    print(INFO + "  System run complete. All agents terminated successfully." + RESET)
    print(HEADER + "=" * 72 + RESET)
