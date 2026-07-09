"""
algorithms/csp.py
Constraint Satisfaction Problem (CSP) for carpooling validation and repair.

Variables   : assignment of each passenger to a driver (or "unassigned")
Domains     : each passenger → list of compatible drivers
Constraints :
  1. Capacity      : sum of passenger seats ≤ driver.vehicle_capacity
  2. Time Window   : passenger time window overlaps driver time window
  3. Max Detour    : detour to pick up passenger ≤ passenger.max_detour_minutes
  4. Rating        : only pair passengers with drivers meeting min rating threshold

Solver      : Backtracking with Arc Consistency (AC-3) preprocessing + MRV heuristic.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import copy
from models.models import Driver, Passenger, RideGroup, RideStatus
from models.city_graph import CityGraph
from algorithms.search import route_distance


Assignment = Dict[str, Optional[str]]   # passenger_id → driver_id or None


# ──────────────────────────────────────────────────────────
# Constraint checks
# ──────────────────────────────────────────────────────────

def _capacity_ok(driver: Driver, current_load: int, passenger: Passenger) -> bool:
    return current_load + passenger.num_people <= driver.vehicle_capacity


def _time_ok(driver: Driver, passenger: Passenger) -> bool:
    return not (driver.time_window_end < passenger.time_window_start or
                driver.time_window_start > passenger.time_window_end)


def _detour_ok(driver: Driver, passenger: Passenger) -> bool:
    detour = passenger.pickup.distance_to(driver.pickup)
    return detour <= passenger.max_detour_minutes


def _rating_ok(driver: Driver, passenger: Passenger, min_rating: float = 3.5) -> bool:
    return driver.rating >= min_rating


def _all_constraints(driver: Driver, current_load: int, passenger: Passenger) -> bool:
    return (_capacity_ok(driver, current_load, passenger) and
            _time_ok(driver, passenger) and
            _detour_ok(driver, passenger) and
            _rating_ok(driver, passenger))


# ──────────────────────────────────────────────────────────
# Domain computation
# ──────────────────────────────────────────────────────────

def _initial_domains(passengers: List[Passenger],
                     drivers: List[Driver]) -> Dict[str, List[str]]:
    """Compute initial domain (feasible drivers) for each passenger."""
    domains: Dict[str, List[str]] = {}
    for p in passengers:
        domains[p.user_id] = [
            d.user_id for d in drivers
            if _time_ok(d, p) and _detour_ok(d, p) and _rating_ok(d, p)
        ]
        if not domains[p.user_id]:
            domains[p.user_id] = []   # will be unassigned
    return domains


# ──────────────────────────────────────────────────────────
# AC-3 Arc Consistency
# ──────────────────────────────────────────────────────────

def _ac3(domains: Dict[str, List[str]],
         passengers: List[Passenger],
         drivers: List[Driver]) -> bool:
    """
    AC-3: remove values from domains that violate binary constraints.
    Returns False if any domain becomes empty (no solution possible).
    """
    # Build arc queue: pairs of (passenger_id, driver_id)
    driver_map = {d.user_id: d for d in drivers}
    passenger_map = {p.user_id: p for p in passengers}

    queue = [(pid, did)
             for pid, dids in domains.items()
             for did in dids]

    while queue:
        pid, did = queue.pop(0)
        if did not in domains.get(pid, []):
            continue
        p = passenger_map[pid]
        d = driver_map[did]
        # Remove if hard constraints fail (ignoring current load here)
        if not (_time_ok(d, p) and _detour_ok(d, p) and _rating_ok(d, p)):
            domains[pid].remove(did)
            if not domains[pid]:
                return False  # domain wipeout
    return True


# ──────────────────────────────────────────────────────────
# Backtracking with MRV
# ──────────────────────────────────────────────────────────

class CSPSolver:
    """
    Backtracking CSP solver with:
    - AC-3 preprocessing for arc consistency
    - MRV (Minimum Remaining Values) variable ordering heuristic
    - Least Constraining Value (LCV) value ordering heuristic
    - Forward checking after each assignment
    """

    def __init__(self,
                 drivers: List[Driver],
                 passengers: List[Passenger],
                 graph: CityGraph):
        self.drivers = drivers
        self.passengers = passengers
        self.graph = graph
        self.driver_map = {d.user_id: d for d in drivers}
        self.passenger_map = {p.user_id: p for p in passengers}
        self.nodes_explored = 0

    def solve(self) -> Tuple[List[RideGroup], int]:
        """
        Run CSP and return (ride_groups, nodes_explored).
        """
        domains = _initial_domains(self.passengers, self.drivers)
        _ac3(domains, self.passengers, self.drivers)  # preprocessing

        assignment: Assignment = {}
        load_tracker: Dict[str, int] = {d.user_id: 0 for d in self.drivers}

        result = self._backtrack(assignment, domains, load_tracker,
                                 list(self.passenger_map.keys()))

        if result is None:
            return [], self.nodes_explored

        groups = self._build_groups(result)
        return groups, self.nodes_explored

    def _backtrack(self,
                   assignment: Assignment,
                   domains: Dict[str, List[str]],
                   load_tracker: Dict[str, int],
                   unassigned: List[str]) -> Optional[Assignment]:
        if not unassigned:
            return assignment  # complete assignment

        # MRV: select passenger with fewest domain options
        pid = min(unassigned, key=lambda p: len(domains.get(p, [])))
        remaining = [p for p in unassigned if p != pid]
        self.nodes_explored += 1

        p = self.passenger_map[pid]

        # LCV: prefer driver that rules out fewest options for others
        candidates = self._lcv_order(pid, domains, load_tracker)

        for did in candidates:
            d = self.driver_map[did]
            cur_load = load_tracker[did]

            if not _all_constraints(d, cur_load, p):
                continue

            # Make assignment
            assignment[pid] = did
            load_tracker[did] += p.num_people

            # Forward checking: prune domains
            pruned: Dict[str, List[str]] = {}
            consistent = True
            for other_pid in remaining:
                other_p = self.passenger_map[other_pid]
                if did in domains.get(other_pid, []):
                    new_load = load_tracker[did]
                    if not _capacity_ok(d, new_load, other_p):
                        pruned[other_pid] = pruned.get(other_pid, []) + [did]
                        domains[other_pid].remove(did)
                        if not domains[other_pid]:
                            consistent = False
                            break

            if consistent:
                result = self._backtrack(assignment, domains, load_tracker, remaining)
                if result is not None:
                    return result

            # Undo assignment (backtrack)
            del assignment[pid]
            load_tracker[did] -= p.num_people
            for other_pid, pruned_vals in pruned.items():
                domains.setdefault(other_pid, []).extend(pruned_vals)

        # Try leaving passenger unassigned as last resort
        assignment[pid] = None
        result = self._backtrack(assignment, domains, load_tracker, remaining)
        if result is not None:
            return result
        del assignment[pid]

        return None

    def _lcv_order(self, pid: str,
                   domains: Dict[str, List[str]],
                   load_tracker: Dict[str, int]) -> List[str]:
        """Order drivers by least constraining value."""
        candidates = domains.get(pid, [])
        if not candidates:
            return []

        def count_constrained(did: str) -> int:
            d = self.driver_map[did]
            new_load = load_tracker[did] + self.passenger_map[pid].num_people
            constrained = 0
            for other_pid, other_domain in domains.items():
                if other_pid == pid:
                    continue
                other_p = self.passenger_map[other_pid]
                if did in other_domain and not _capacity_ok(d, new_load, other_p):
                    constrained += 1
            return constrained

        return sorted(candidates, key=count_constrained)

    def _build_groups(self, assignment: Assignment) -> List[RideGroup]:
        from algorithms.genetic import _build_waypoints
        groups_map: Dict[str, RideGroup] = {}

        for pid, did in assignment.items():
            if did is None:
                continue
            p = self.passenger_map[pid]
            if did not in groups_map:
                d = copy.deepcopy(self.driver_map[did])
                groups_map[did] = RideGroup(driver=d)
            groups_map[did].passengers.append(p)

        result = []
        for grp in groups_map.values():
            if not grp.passengers:
                continue
            waypoints = _build_waypoints(grp, self.graph)
            grp.route = waypoints
            grp.total_distance = route_distance(self.graph, [w.name for w in waypoints])
            grp.estimated_time = grp.total_distance * 3.0
            grp.status = RideStatus.MATCHED
            result.append(grp)

        return result
