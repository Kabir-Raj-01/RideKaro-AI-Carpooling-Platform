"""
algorithms/search_extended.py
Additional search algorithms for the Carpooling AI system:
  - DFS (Depth-First Search)
  - Hill Climbing (Steepest Ascent & Random Restart)
  - Beam Search
  - IDA* (Iterative Deepening A*)
  - Bidirectional BFS
  - Uniform Cost Search
Each returns (path, cost, nodes_explored, algorithm_trace) for visualisation.
"""
from __future__ import annotations
import heapq, random, math
from collections import deque
from typing import List, Tuple, Dict, Optional, Any

from models.city_graph import CityGraph

Trace = List[Dict]   # list of {step, node, action, g, h, f}


# ──────────────────────────────────────────────
# DFS
# ──────────────────────────────────────────────
def dfs(graph: CityGraph, start: str, goal: str,
        max_depth: int = 30) -> Tuple[List[str], float, int, Trace]:
    stack = [(start, [start], 0.0, 0)]
    visited: set = set()
    nodes_explored = 0
    trace: Trace = []

    while stack:
        node, path, cost, depth = stack.pop()
        if node in visited or depth > max_depth:
            continue
        visited.add(node)
        nodes_explored += 1
        trace.append({"step": nodes_explored, "node": node,
                       "action": "visit", "depth": depth, "cost": round(cost, 3)})

        if node == goal:
            return path, cost, nodes_explored, trace

        for nb in reversed(graph.neighbors(node)):
            if nb not in visited:
                edge_cost = graph.graph[node][nb]["weight"]
                stack.append((nb, path + [nb], cost + edge_cost, depth + 1))

    return [], float("inf"), nodes_explored, trace


# ──────────────────────────────────────────────
# Uniform Cost Search
# ──────────────────────────────────────────────
def uniform_cost(graph: CityGraph, start: str, goal: str) -> Tuple[List[str], float, int, Trace]:
    pq = [(0.0, start, [start])]
    best: Dict[str, float] = {start: 0.0}
    nodes_explored = 0
    trace: Trace = []

    while pq:
        cost, node, path = heapq.heappop(pq)
        nodes_explored += 1
        trace.append({"step": nodes_explored, "node": node, "cost": round(cost, 3)})

        if node == goal:
            return path, cost, nodes_explored, trace
        if cost > best.get(node, float("inf")):
            continue

        for nb in graph.neighbors(node):
            nc = cost + graph.graph[node][nb]["weight"]
            if nc < best.get(nb, float("inf")):
                best[nb] = nc
                heapq.heappush(pq, (nc, nb, path + [nb]))

    return [], float("inf"), nodes_explored, trace


# ──────────────────────────────────────────────
# IDA* (Iterative Deepening A*)
# ──────────────────────────────────────────────
def idastar(graph: CityGraph, start: str, goal: str) -> Tuple[List[str], float, int, Trace]:
    nodes_explored = 0
    trace: Trace = []

    def search(path, g, bound):
        nonlocal nodes_explored
        node = path[-1]
        f = g + graph.heuristic(node, goal)
        if f > bound:
            return f, None
        nodes_explored += 1
        trace.append({"step": nodes_explored, "node": node,
                       "g": round(g, 3), "h": round(graph.heuristic(node, goal), 3),
                       "f": round(f, 3), "bound": round(bound, 3)})
        if node == goal:
            return -1, path[:]

        minimum = float("inf")
        for nb in graph.neighbors(node):
            if nb not in path:
                edge_cost = graph.graph[node][nb]["weight"]
                t, result = search(path + [nb], g + edge_cost, bound)
                if t == -1:
                    return -1, result
                minimum = min(minimum, t)

        return minimum, None

    bound = graph.heuristic(start, goal)
    path  = [start]
    for _ in range(50):
        t, result = search(path, 0.0, bound)
        if t == -1:
            total_cost = sum(
                graph.graph[result[i]][result[i+1]]["weight"]
                for i in range(len(result)-1)
            )
            return result, total_cost, nodes_explored, trace
        if t == float("inf"):
            break
        bound = t

    return [], float("inf"), nodes_explored, trace


# ──────────────────────────────────────────────
# Hill Climbing (Steepest-Ascent with restarts)
# ──────────────────────────────────────────────
def hill_climbing(graph: CityGraph, start: str, goal: str,
                  restarts: int = 5) -> Tuple[List[str], float, int, Trace]:
    """
    Steepest-ascent hill climbing using negative distance-to-goal as value.
    With random restarts to escape local optima.
    """
    nodes_explored = 0
    trace: Trace = []
    best_path: List[str] = []
    best_cost = float("inf")

    for restart in range(restarts):
        current = start
        path    = [start]
        cost    = 0.0
        visited = {start}

        for step in range(100):
            nodes_explored += 1
            h = graph.heuristic(current, goal)
            trace.append({"step": nodes_explored, "node": current,
                           "restart": restart, "h": round(h, 3), "cost": round(cost, 3)})

            if current == goal:
                if cost < best_cost:
                    best_cost = cost
                    best_path = path[:]
                break

            neighbours = [
                (nb, graph.graph[current][nb]["weight"])
                for nb in graph.neighbors(current)
                if nb not in visited
            ]
            if not neighbours:
                break

            # Pick neighbour with smallest h (steepest ascent toward goal)
            neighbours.sort(key=lambda x: graph.heuristic(x[0], goal))
            next_node, edge_cost = neighbours[0]

            # Escape flat shoulders with slight randomness
            if random.random() < 0.15 and len(neighbours) > 1:
                next_node, edge_cost = random.choice(neighbours[:3])

            current = next_node
            path.append(current)
            cost += edge_cost
            visited.add(current)

    return best_path, best_cost, nodes_explored, trace


# ──────────────────────────────────────────────
# Beam Search
# ──────────────────────────────────────────────
def beam_search(graph: CityGraph, start: str, goal: str,
                beam_width: int = 3) -> Tuple[List[str], float, int, Trace]:
    """
    Beam Search: at each level keep only the best `beam_width` partial paths
    ranked by h(n) = heuristic to goal.
    """
    # Each beam item: (h_value, path, cost)
    beam = [(graph.heuristic(start, goal), [start], 0.0)]
    nodes_explored = 0
    trace: Trace = []

    for depth in range(100):
        if not beam:
            break
        candidates = []
        for h_val, path, cost in beam:
            node = path[-1]
            nodes_explored += 1
            trace.append({"step": nodes_explored, "node": node, "depth": depth,
                           "h": round(h_val, 3), "beam_size": len(beam)})

            if node == goal:
                return path, cost, nodes_explored, trace

            for nb in graph.neighbors(node):
                if nb not in path:
                    ec   = graph.graph[node][nb]["weight"]
                    new_h = graph.heuristic(nb, goal)
                    candidates.append((new_h, path + [nb], cost + ec))

        # Keep best beam_width candidates
        candidates.sort(key=lambda x: x[0])
        beam = candidates[:beam_width]

    if beam:
        best = min(beam, key=lambda x: x[0])
        return best[1], best[2], nodes_explored, trace
    return [], float("inf"), nodes_explored, trace


# ──────────────────────────────────────────────
# Bidirectional BFS
# ──────────────────────────────────────────────
def bidirectional_bfs(graph: CityGraph, start: str,
                      goal: str) -> Tuple[List[str], float, int, Trace]:
    """
    Bidirectional BFS: expand from both start and goal simultaneously.
    Terminates when frontiers meet.
    """
    if start == goal:
        return [start], 0.0, 0, []

    front_visited  = {start: [start]}
    back_visited   = {goal:  [goal]}
    front_queue    = deque([start])
    back_queue     = deque([goal])
    nodes_explored = 0
    trace: Trace   = []

    def _path_cost(path):
        return sum(graph.graph[path[i]][path[i+1]]["distance"]
                   for i in range(len(path)-1))

    for step in range(200):
        if front_queue:
            node = front_queue.popleft()
            nodes_explored += 1
            trace.append({"step": nodes_explored, "node": node, "direction": "forward"})
            for nb in graph.neighbors(node):
                if nb not in front_visited:
                    front_visited[nb] = front_visited[node] + [nb]
                    front_queue.append(nb)
                if nb in back_visited:
                    full = front_visited[nb] + list(reversed(back_visited[nb][:-1]))
                    return full, _path_cost(full), nodes_explored, trace

        if back_queue:
            node = back_queue.popleft()
            nodes_explored += 1
            trace.append({"step": nodes_explored, "node": node, "direction": "backward"})
            for nb in graph.neighbors(node):
                if nb not in back_visited:
                    back_visited[nb] = back_visited[node] + [nb]
                    back_queue.append(nb)
                if nb in front_visited:
                    full = front_visited[nb] + list(reversed(back_visited[nb][:-1]))
                    return full, _path_cost(full), nodes_explored, trace

    return [], float("inf"), nodes_explored, trace


# ──────────────────────────────────────────────
# Alpha-Beta Pruning (Minimax for ride negotiation)
# ──────────────────────────────────────────────
def minimax_alpha_beta(depth: int, node_idx: int, is_max: bool,
                       values: List[float], alpha: float, beta: float,
                       trace: List[Dict], step_counter: List[int]) -> float:
    """
    Generic Alpha-Beta pruning on a game tree encoded as flat list of leaf values.
    Used to demonstrate driver-passenger negotiation:
      MAX player = driver (maximises route value)
      MIN player = passenger (minimises detour cost)

    Parameters
    ----------
    depth       : current depth (0 = leaf)
    node_idx    : index into values list
    is_max      : True if maximiser's turn
    values      : leaf utility values
    alpha, beta : pruning bounds
    trace       : list to append step info for visualisation
    step_counter: [int] mutable counter
    """
    n_leaves = len(values)
    branching = 2   # binary tree for simplicity

    if depth == 0 or node_idx >= n_leaves:
        val = values[node_idx] if node_idx < n_leaves else 0.0
        step_counter[0] += 1
        trace.append({"step": step_counter[0], "node": node_idx,
                       "depth": depth, "value": round(val, 3),
                       "type": "leaf", "pruned": False})
        return val

    if is_max:
        best = float("-inf")
        for i in range(branching):
            child_idx = node_idx * branching + i + 1
            val = minimax_alpha_beta(depth-1, child_idx, False,
                                     values, alpha, beta, trace, step_counter)
            best  = max(best, val)
            alpha = max(alpha, best)
            step_counter[0] += 1
            pruned = beta <= alpha
            trace.append({"step": step_counter[0], "node": node_idx,
                           "depth": depth, "value": round(best, 3),
                           "alpha": round(alpha, 3), "beta": round(beta, 3),
                           "type": "max", "pruned": pruned})
            if pruned:
                break
        return best
    else:
        best = float("inf")
        for i in range(branching):
            child_idx = node_idx * branching + i + 1
            val = minimax_alpha_beta(depth-1, child_idx, True,
                                     values, alpha, beta, trace, step_counter)
            best = min(best, val)
            beta = min(beta, best)
            step_counter[0] += 1
            pruned = beta <= alpha
            trace.append({"step": step_counter[0], "node": node_idx,
                           "depth": depth, "value": round(best, 3),
                           "alpha": round(alpha, 3), "beta": round(beta, 3),
                           "type": "min", "pruned": pruned})
            if pruned:
                break
        return best


def run_alpha_beta_demo(depth: int = 4) -> Tuple[float, List[Dict]]:
    """Run alpha-beta on a randomly generated game tree and return (best_value, trace)."""
    random.seed(99)
    n_leaves = 2 ** depth
    values   = [round(random.uniform(-10, 10), 2) for _ in range(n_leaves * 2)]
    trace: List[Dict] = []
    counter = [0]
    best = minimax_alpha_beta(depth, 0, True, values, float("-inf"), float("inf"),
                               trace, counter)
    return best, trace, values


# ──────────────────────────────────────────────
# Algorithm benchmark helper
# ──────────────────────────────────────────────
def benchmark_all(graph: CityGraph, src: str, dst: str) -> Dict:
    """
    Run all search algorithms between src and dst, return comparison dict.
    """
    from algorithms.search import astar, bfs, greedy_best_first, dijkstra

    results = {}
    algos = [
        ("A*",             lambda: astar(graph, src, dst)[:3]),
        ("BFS",            lambda: bfs(graph, src, dst)[:3]),
        ("Greedy",         lambda: greedy_best_first(graph, src, dst)[:3]),
        ("Dijkstra",       lambda: dijkstra(graph, src, dst)[:3]),
        ("DFS",            lambda: dfs(graph, src, dst)[:3]),
        ("UCS",            lambda: uniform_cost(graph, src, dst)[:3]),
        ("IDA*",           lambda: idastar(graph, src, dst)[:3]),
        ("Hill Climbing",  lambda: hill_climbing(graph, src, dst)[:3]),
        ("Beam Search",    lambda: beam_search(graph, src, dst)[:3]),
        ("BiDir BFS",      lambda: bidirectional_bfs(graph, src, dst)[:3]),
    ]

    for name, fn in algos:
        try:
            path, cost, explored = fn()
            results[name] = {
                "path":     path,
                "cost":     round(cost, 3) if cost != float("inf") else None,
                "explored": explored,
                "hops":     len(path),
            }
        except Exception as e:
            results[name] = {"path": [], "cost": None, "explored": 0,
                             "hops": 0, "error": str(e)}

    return results
