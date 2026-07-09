"""
models/city_graph.py
Weighted city road network represented as a NetworkX graph.
Nodes = named intersections/locations, Edges = roads with distance & traffic weight.
"""
from __future__ import annotations
import networkx as nx
import math
from typing import Dict, List, Tuple, Optional
from models.models import Location


class CityGraph:
    """
    Weighted undirected graph representing urban road network.
    Supports dynamic traffic weights, road closures, and multi-attribute edges.
    """

    def __init__(self, name: str = "City"):
        self.name = name
        self.graph: nx.Graph = nx.Graph()
        self._locations: Dict[str, Location] = {}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_location(self, loc: Location) -> None:
        self._locations[loc.name] = loc
        self.graph.add_node(loc.name, pos=(loc.x, loc.y), location=loc)

    def add_road(self, loc_a: Location, loc_b: Location,
                 traffic_factor: float = 1.0) -> None:
        """
        Add a bidirectional road. Weight = euclidean distance × traffic_factor.
        traffic_factor > 1 → congested; < 1 → express / highway.
        """
        dist = loc_a.distance_to(loc_b)
        weight = dist * traffic_factor
        self.graph.add_edge(
            loc_a.name, loc_b.name,
            distance=dist,
            weight=weight,
            traffic_factor=traffic_factor
        )

    def update_traffic(self, loc_a_name: str, loc_b_name: str,
                       new_traffic_factor: float) -> None:
        """Dynamically update traffic on an existing road."""
        if self.graph.has_edge(loc_a_name, loc_b_name):
            dist = self.graph[loc_a_name][loc_b_name]["distance"]
            self.graph[loc_a_name][loc_b_name]["weight"] = dist * new_traffic_factor
            self.graph[loc_a_name][loc_b_name]["traffic_factor"] = new_traffic_factor

    def close_road(self, loc_a_name: str, loc_b_name: str) -> None:
        if self.graph.has_edge(loc_a_name, loc_b_name):
            self.graph.remove_edge(loc_a_name, loc_b_name)

    # ------------------------------------------------------------------
    # Heuristic (for A*)
    # ------------------------------------------------------------------

    def heuristic(self, node_a: str, node_b: str) -> float:
        """Admissible Euclidean heuristic for A*."""
        loc_a = self._locations[node_a]
        loc_b = self._locations[node_b]
        return loc_a.distance_to(loc_b)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_location(self, name: str) -> Optional[Location]:
        return self._locations.get(name)

    def all_locations(self) -> List[Location]:
        return list(self._locations.values())

    def get_edge_distance(self, a: str, b: str) -> float:
        if self.graph.has_edge(a, b):
            return self.graph[a][b]["distance"]
        return float("inf")

    def is_connected(self) -> bool:
        return nx.is_connected(self.graph)

    def neighbors(self, node: str) -> List[str]:
        return list(self.graph.neighbors(node))

    def __repr__(self):
        return (f"CityGraph({self.name}, nodes={self.graph.number_of_nodes()}, "
                f"edges={self.graph.number_of_edges()})")


# ------------------------------------------------------------------
# Factory: Build Karachi-inspired demo city
# ------------------------------------------------------------------

def build_karachi_demo_city() -> Tuple[CityGraph, Dict[str, Location]]:
    """
    Build a realistic Karachi-inspired urban road network with 20 named nodes.
    Returns the CityGraph and a dict of location objects for easy access.
    """
    city = CityGraph("Karachi Metro")

    raw = {
        "NUCES":       (0,   0),
        "Gulshan":     (3,   2),
        "Johar":       (6,   1),
        "North_Naz":   (8,   3),
        "Gulistan":    (5,   5),
        "Saddar":      (2,   6),
        "Clifton":     (0,   8),
        "DHA_1":       (-3,  7),
        "DHA_5":       (-5,  4),
        "Korangi":     (10,  0),
        "Landhi":      (12,  2),
        "Malir":       (14,  5),
        "Airport":     (7,   8),
        "Federal_B":   (4,   9),
        "Surjani":     (1,   12),
        "North_Kar":   (-2,  11),
        "Orangi":      (-4,  9),
        "Lyari":       (-1,  5),
        "Keamari":     (-6,  2),
        "Hawks_Bay":   (-8,  0),
    }

    locs: Dict[str, Location] = {}
    for name, (x, y) in raw.items():
        loc = Location(name=name, x=float(x), y=float(y))
        locs[name] = loc
        city.add_location(loc)

    # Road connections (adjacency) with realistic traffic factors
    roads = [
        ("NUCES",    "Gulshan",   1.0),
        ("NUCES",    "Lyari",     1.2),
        ("NUCES",    "DHA_5",     1.1),
        ("Gulshan",  "Johar",     1.0),
        ("Gulshan",  "Gulistan",  1.1),
        ("Gulshan",  "Saddar",    1.3),
        ("Johar",    "North_Naz", 1.0),
        ("Johar",    "Korangi",   1.0),
        ("North_Naz","Gulistan",  1.0),
        ("North_Naz","Malir",     1.0),
        ("Gulistan", "Airport",   1.1),
        ("Gulistan", "Federal_B", 1.0),
        ("Saddar",   "Clifton",   1.4),
        ("Saddar",   "Lyari",     1.2),
        ("Saddar",   "Federal_B", 1.0),
        ("Clifton",  "DHA_1",     1.0),
        ("Clifton",  "Lyari",     1.1),
        ("DHA_1",    "DHA_5",     1.0),
        ("DHA_5",    "Keamari",   1.0),
        ("Korangi",  "Landhi",    1.0),
        ("Landhi",   "Malir",     1.0),
        ("Airport",  "Federal_B", 1.0),
        ("Airport",  "Malir",     1.1),
        ("Federal_B","Surjani",   1.0),
        ("Surjani",  "North_Kar", 1.0),
        ("North_Kar","Orangi",    1.0),
        ("Orangi",   "Lyari",     1.1),
        ("Lyari",    "Keamari",   1.0),
        ("Keamari",  "Hawks_Bay", 1.0),
        ("DHA_5",    "Hawks_Bay", 1.2),
    ]

    for a, b, tf in roads:
        city.add_road(locs[a], locs[b], traffic_factor=tf)

    return city, locs
