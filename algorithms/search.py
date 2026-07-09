"""
algorithms/search.py
Informed Search  : A* (with Euclidean heuristic)
Uninformed Search: BFS, Greedy Best-First
All algorithms operate on CityGraph and return (path, total_distance, nodes_explored).
"""
from __future__ import annotations
import heapq
from collections import deque
from typing import Dict, List, Optional, Tuple

from models.city_graph import CityGraph


# ──────────────────────────────────────────────
# A* (Informed Search)
# ──────────────────────────────────────────────

def astar(graph: CityGraph, start: str, goal: str) -> Tuple[List[str], float, int]:
    """
    A* search using Euclidean distance as the admissible heuristic.

    Returns
    -------
    path            : ordered list of node names from start → goal
    total_distance  : actual road distance along the path
    nodes_explored  : number of nodes popped from the open set
    """
    # Priority queue: (f = g + h, g, node, parent_path)
    open_set: list = []
    heapq.heappush(open_set, (0.0, 0.0, start, [start]))

    # Best known g-cost to each node
    g_costs: Dict[str, float] = {start: 0.0}
    nodes_explored = 0

    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        nodes_explored += 1

        if current == goal:
            return path, g, nodes_explored

        # Skip if we already found a cheaper path
        if g > g_costs.get(current, float("inf")):
            continue

        for neighbor in graph.neighbors(current):
            edge_cost = graph.graph[current][neighbor]["weight"]
            new_g = g + edge_cost
            if new_g < g_costs.get(neighbor, float("inf")):
                g_costs[neighbor] = new_g
                h = graph.heuristic(neighbor, goal)
                f_new = new_g + h
                heapq.heappush(open_set, (f_new, new_g, neighbor, path + [neighbor]))

    return [], float("inf"), nodes_explored  # no path


# ──────────────────────────────────────────────
# BFS (Uninformed Search)
# ──────────────────────────────────────────────

def bfs(graph: CityGraph, start: str, goal: str) -> Tuple[List[str], float, int]:
    """
    Breadth-First Search — finds shortest hop-count path (not distance-optimal).

    Returns
    -------
    path            : ordered list of node names
    total_distance  : road distance along the BFS path
    nodes_explored  : nodes dequeued
    """
    queue: deque = deque([(start, [start])])
    visited = {start}
    nodes_explored = 0

    while queue:
        current, path = queue.popleft()
        nodes_explored += 1

        if current == goal:
            dist = _path_distance(graph, path)
            return path, dist, nodes_explored

        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return [], float("inf"), nodes_explored


# ──────────────────────────────────────────────
# Greedy Best-First Search (Informed but not optimal)
# ──────────────────────────────────────────────

def greedy_best_first(graph: CityGraph, start: str, goal: str) -> Tuple[List[str], float, int]:
    """
    Greedy Best-First Search: always expand the node with lowest h(n).
    Fast but not guaranteed optimal.
    """
    open_set: list = []
    heapq.heappush(open_set, (graph.heuristic(start, goal), start, [start]))
    visited = {start}
    nodes_explored = 0

    while open_set:
        h, current, path = heapq.heappop(open_set)
        nodes_explored += 1

        if current == goal:
            dist = _path_distance(graph, path)
            return path, dist, nodes_explored

        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                h_n = graph.heuristic(neighbor, goal)
                heapq.heappush(open_set, (h_n, neighbor, path + [neighbor]))

    return [], float("inf"), nodes_explored


# ──────────────────────────────────────────────
# Dijkstra (for comparison / CSP support)
# ──────────────────────────────────────────────

def dijkstra(graph: CityGraph, start: str, goal: str) -> Tuple[List[str], float, int]:
    """Standard Dijkstra's algorithm (A* with h=0)."""
    open_set: list = []
    heapq.heappush(open_set, (0.0, start, [start]))
    g_costs: Dict[str, float] = {start: 0.0}
    nodes_explored = 0

    while open_set:
        g, current, path = heapq.heappop(open_set)
        nodes_explored += 1

        if current == goal:
            return path, g, nodes_explored

        if g > g_costs.get(current, float("inf")):
            continue

        for neighbor in graph.neighbors(current):
            edge_cost = graph.graph[current][neighbor]["weight"]
            new_g = g + edge_cost
            if new_g < g_costs.get(neighbor, float("inf")):
                g_costs[neighbor] = new_g
                heapq.heappush(open_set, (new_g, neighbor, path + [neighbor]))

    return [], float("inf"), nodes_explored


# ──────────────────────────────────────────────
# Multi-stop route distance helper
# ──────────────────────────────────────────────

def route_distance(graph: CityGraph, waypoints: List[str]) -> float:
    """
    Compute total A*-optimal road distance for an ordered list of waypoints.
    Used by the Genetic Algorithm fitness function.
    """
    if len(waypoints) < 2:
        return 0.0
    total = 0.0
    for i in range(len(waypoints) - 1):
        _, dist, _ = astar(graph, waypoints[i], waypoints[i + 1])
        if dist == float("inf"):
            return float("inf")
        total += dist
    return total


# ──────────────────────────────────────────────
# Internal helper
# ──────────────────────────────────────────────

def _path_distance(graph: CityGraph, path: List[str]) -> float:
    total = 0.0
    for i in range(len(path) - 1):
        total += graph.graph[path[i]][path[i + 1]]["distance"]
    return total
