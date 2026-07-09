"""
dashboard/visualiser.py
Matplotlib-based visualisation module.

Produces four figures:
  1. City road network with ride-group routes overlaid
  2. GA convergence curve
  3. Algorithm comparison bar chart
  4. Demand heatmap by zone & hour
"""
from __future__ import annotations
import os
import math
from typing import List, Dict, Optional

try:
    import matplotlib
    matplotlib.use("Agg")   # headless / file-only rendering
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from models.models import RideGroup
from models.city_graph import CityGraph

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


def _ensure_output():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _path(name: str) -> str:
    return os.path.join(OUTPUT_DIR, name)


# ─────────────────────────────────────────────────────────────
# 1. City Network + Routes
# ─────────────────────────────────────────────────────────────

def plot_city_routes(city: CityGraph,
                     groups: List[RideGroup],
                     filename: str = "city_routes.png"):
    if not HAS_MPL:
        print("matplotlib not available — skipping city route plot.")
        return
    _ensure_output()

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # Draw all edges
    for u, v, data in city.graph.edges(data=True):
        loc_u = city.get_location(u)
        loc_v = city.get_location(v)
        tf = data.get("traffic_factor", 1.0)
        colour = "#ff4444" if tf >= 1.8 else "#ffaa00" if tf >= 1.3 else "#334455"
        ax.plot([loc_u.x, loc_v.x], [loc_u.y, loc_v.y],
                color=colour, linewidth=1.0, alpha=0.6, zorder=1)

    # Draw all nodes
    for loc in city.all_locations():
        ax.scatter(loc.x, loc.y, s=60, color="#7ec8e3", zorder=3, edgecolors="#ffffff",
                   linewidths=0.5)
        ax.annotate(loc.name.replace("_", " "), (loc.x, loc.y),
                    fontsize=6.5, color="#ccddee", ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points", zorder=4)

    # Draw ride-group routes
    colours = plt.cm.tab10.colors
    for gi, grp in enumerate(groups):
        if not grp.route or grp.driver is None:
            continue
        col = colours[gi % len(colours)]
        xs = [w.x for w in grp.route]
        ys = [w.y for w in grp.route]
        ax.plot(xs, ys, color=col, linewidth=2.5, zorder=5, alpha=0.85,
                label=f"Group {gi+1}: {grp.driver.name}")
        # Mark driver pickup
        ax.scatter(xs[0], ys[0], s=120, color=col, marker="^",
                   edgecolors="white", linewidths=1.2, zorder=6)
        # Mark final dropoff
        ax.scatter(xs[-1], ys[-1], s=120, color=col, marker="s",
                   edgecolors="white", linewidths=1.2, zorder=6)

    ax.set_title("Karachi Metro – Ride Group Routes (A* Optimised)",
                 color="white", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("X (longitude-proxy)", color="#aabbcc")
    ax.set_ylabel("Y (latitude-proxy)", color="#aabbcc")
    ax.tick_params(colors="#aabbcc")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334455")

    legend = ax.legend(loc="lower left", fontsize=7.5, framealpha=0.3,
                       facecolor="#111827", edgecolor="#334455",
                       labelcolor="white")

    # Legend for edge colours
    patches = [
        mpatches.Patch(color="#ff4444", label="Heavy traffic (tf≥1.8)"),
        mpatches.Patch(color="#ffaa00", label="Moderate (tf≥1.3)"),
        mpatches.Patch(color="#334455", label="Clear road"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=7, framealpha=0.3,
              facecolor="#111827", edgecolor="#334455", labelcolor="white")

    plt.tight_layout()
    plt.savefig(_path(filename), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Saved: output/{filename}")


# ─────────────────────────────────────────────────────────────
# 2. GA Convergence
# ─────────────────────────────────────────────────────────────

def plot_ga_convergence(history: List[float], filename: str = "ga_convergence.png"):
    if not HAS_MPL or not history:
        return
    _ensure_output()

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#111827")

    gens = list(range(len(history)))
    ax.plot(gens, history, color="#00e5ff", linewidth=2, label="Best Fitness")
    ax.fill_between(gens, history, min(history),
                    color="#00e5ff", alpha=0.12)

    # Rolling average
    window = max(1, len(history) // 10)
    roll = [
        sum(history[max(0, i-window):i+1]) / len(history[max(0, i-window):i+1])
        for i in range(len(history))
    ]
    ax.plot(gens, roll, color="#ff9800", linewidth=1.5,
            linestyle="--", label=f"Rolling avg (w={window})")

    ax.set_title("Genetic Algorithm – Fitness Convergence",
                 color="white", fontsize=13, fontweight="bold")
    ax.set_xlabel("Generation", color="#aabbcc")
    ax.set_ylabel("Fitness Score", color="#aabbcc")
    ax.tick_params(colors="#aabbcc")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334455")
    ax.grid(True, color="#1e2a3a", linewidth=0.6)
    ax.legend(fontsize=9, framealpha=0.3, facecolor="#111827",
              edgecolor="#334455", labelcolor="white")

    plt.tight_layout()
    plt.savefig(_path(filename), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Saved: output/{filename}")


# ─────────────────────────────────────────────────────────────
# 3. Algorithm Comparison
# ─────────────────────────────────────────────────────────────

def plot_algorithm_comparison(comparisons: List[Dict],
                               filename: str = "algo_comparison.png"):
    if not HAS_MPL or not comparisons:
        return
    _ensure_output()

    algos = ["A*", "BFS", "Greedy"]
    colours = {"A*": "#00e5ff", "BFS": "#ff9800", "Greedy": "#e040fb"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#aabbcc")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334455")

    # Distance comparison
    x = list(range(len(comparisons)))
    width = 0.25
    for ai, algo in enumerate(algos):
        dists = [c.get(algo, {}).get("dist", 0) for c in comparisons]
        axes[0].bar([xi + ai * width for xi in x], dists, width,
                    label=algo, color=colours[algo], alpha=0.85)
    axes[0].set_title("Route Distance Comparison", color="white", fontsize=11)
    axes[0].set_ylabel("Distance (km)", color="#aabbcc")
    labels = [f"{c['from'][:4]}→{c['to'][:4]}" for c in comparisons]
    axes[0].set_xticks([xi + width for xi in x])
    axes[0].set_xticklabels(labels, color="#aabbcc", fontsize=8)
    axes[0].legend(fontsize=8, framealpha=0.3, facecolor="#111827",
                   edgecolor="#334455", labelcolor="white")

    # Nodes explored comparison
    for ai, algo in enumerate(algos):
        explored = [c.get(algo, {}).get("explored", 0) for c in comparisons]
        axes[1].bar([xi + ai * width for xi in x], explored, width,
                    label=algo, color=colours[algo], alpha=0.85)
    axes[1].set_title("Nodes Explored", color="white", fontsize=11)
    axes[1].set_ylabel("Nodes Explored", color="#aabbcc")
    axes[1].set_xticks([xi + width for xi in x])
    axes[1].set_xticklabels(labels, color="#aabbcc", fontsize=8)
    axes[1].legend(fontsize=8, framealpha=0.3, facecolor="#111827",
                   edgecolor="#334455", labelcolor="white")

    fig.suptitle("A* vs BFS vs Greedy – Efficiency Analysis",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(_path(filename), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Saved: output/{filename}")


# ─────────────────────────────────────────────────────────────
# 4. Demand Heatmap
# ─────────────────────────────────────────────────────────────

def plot_demand_heatmap(predictor, filename: str = "demand_heatmap.png"):
    if not HAS_MPL:
        return
    _ensure_output()

    zones = ["residential", "commercial", "university", "industrial"]
    hours = list(range(6, 23))
    demand_map = {"low": 0, "medium": 1, "high": 2}

    matrix = []
    for zone in zones:
        row = []
        for hour in hours:
            d = predictor.predict(hour, zone, "clear")
            row.append(demand_map[d])
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    import numpy as np
    data = np.array(matrix, dtype=float)
    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=2)

    ax.set_xticks(range(len(hours)))
    ax.set_xticklabels([f"{h:02d}:00" for h in hours], color="#aabbcc", fontsize=8)
    ax.set_yticks(range(len(zones)))
    ax.set_yticklabels([z.capitalize() for z in zones], color="#aabbcc", fontsize=9)

    # Annotate cells
    demand_labels = ["Low", "Med", "High"]
    for r in range(len(zones)):
        for c in range(len(hours)):
            ax.text(c, r, demand_labels[int(data[r, c])],
                    ha="center", va="center", fontsize=6.5, color="white")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(["Low", "Medium", "High"])
    cbar.ax.yaxis.set_tick_params(color="#aabbcc")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#aabbcc")

    ax.set_title("Bayesian Demand Prediction Heatmap (Clear Weather)",
                 color="white", fontsize=12, fontweight="bold")
    ax.set_xlabel("Hour of Day", color="#aabbcc")
    ax.set_ylabel("Zone Type", color="#aabbcc")

    plt.tight_layout()
    plt.savefig(_path(filename), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Saved: output/{filename}")


# ─────────────────────────────────────────────────────────────
# Master render function
# ─────────────────────────────────────────────────────────────

def render_all(city: CityGraph,
               report: Dict,
               ga_history: List[float],
               predictor):
    print("\n  Generating visualisations...")
    plot_city_routes(city, report.get("groups", []))
    plot_ga_convergence(ga_history)
    plot_algorithm_comparison(report.get("algorithm_comparisons", []))
    plot_demand_heatmap(predictor)
