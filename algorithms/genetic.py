"""
algorithms/genetic.py
Genetic Algorithm (GA) for global ride-sharing optimisation.

Chromosome encoding
-------------------
A chromosome is a permutation of passenger indices.
The decoder greedily assigns passengers to drivers in chromosome order,
respecting capacity and time-window constraints.

Fitness function
----------------
  F = w1/total_distance + w2*utilisation + w3*time_compliance - w4*unmatched_penalty
Higher fitness = better solution.
"""
from __future__ import annotations
import random
import math
import copy
from typing import List, Tuple, Dict, Optional
from models.models import Driver, Passenger, RideGroup, RideStatus
from models.city_graph import CityGraph
from algorithms.search import route_distance


# ──────────────────────────────────────────────────────────
# Chromosome helpers
# ──────────────────────────────────────────────────────────

Chromosome = List[int]   # permutation of passenger indices


def _decode(chromosome: Chromosome,
            drivers: List[Driver],
            passengers: List[Passenger],
            graph: CityGraph) -> List[RideGroup]:
    """
    Decode a chromosome into a list of RideGroups.
    Passengers are assigned to the first available driver that:
      1. Has capacity.
      2. Overlaps time window with the passenger.
      3. Does not exceed the passenger's max_detour_minutes.
    """
    groups: Dict[str, RideGroup] = {}
    unassigned_drivers: List[Driver] = [copy.deepcopy(d) for d in drivers]

    for pidx in chromosome:
        p = passengers[pidx]
        best_driver: Optional[Driver] = None
        best_detour = float("inf")

        for d in unassigned_drivers:
            if not d.can_accept(p.num_people):
                continue
            # Time window compatibility
            if d.time_window_end < p.time_window_start or d.time_window_start > p.time_window_end:
                continue
            # Estimate detour: driver pickup → passenger pickup detour
            detour = p.pickup.distance_to(d.pickup)
            if detour <= p.max_detour_minutes and detour < best_detour:
                best_detour = detour
                best_driver = d

        if best_driver is None:
            continue  # passenger stays unmatched

        key = best_driver.user_id
        if key not in groups:
            groups[key] = RideGroup(driver=best_driver)
        groups[key].passengers.append(p)
        best_driver.current_load += p.num_people

    # Compute route & distance for each group
    result: List[RideGroup] = []
    for grp in groups.values():
        if not grp.passengers:
            continue
        waypoints = _build_waypoints(grp, graph)
        grp.route = waypoints
        grp.total_distance = route_distance(graph, [w.name for w in waypoints])
        grp.estimated_time = grp.total_distance * 3.0     # ~3 min/km in city
        grp.status = RideStatus.MATCHED
        result.append(grp)

    return result


def _build_waypoints(grp: RideGroup, graph: CityGraph):
    """
    Build an ordered waypoint list:
    driver.pickup → [passenger pickups] → [passenger dropoffs] → driver.dropoff
    Simple nearest-neighbour ordering for pickups.
    """
    from models.models import Location
    pickups = sorted(grp.passengers, key=lambda p: p.pickup.distance_to(grp.driver.pickup))
    waypoints = [grp.driver.pickup]
    for p in pickups:
        waypoints.append(p.pickup)
    for p in pickups:
        waypoints.append(p.dropoff)
    waypoints.append(grp.driver.dropoff)
    return waypoints


def _fitness(groups: List[RideGroup],
             all_passengers: List[Passenger],
             w1: float = 1.0, w2: float = 0.5,
             w3: float = 0.3, w4: float = 2.0) -> float:
    """
    Multi-objective fitness function.
    """
    matched_ids = {p.user_id for g in groups for p in g.passengers}
    unmatched = sum(1 for p in all_passengers if p.user_id not in matched_ids)

    total_dist = sum(g.total_distance for g in groups) or 1e-6
    total_cap = sum(g.driver.vehicle_capacity for g in groups) or 1
    total_load = sum(g.total_passengers for g in groups)
    utilisation = total_load / total_cap

    time_compliance = sum(
        1 for g in groups for p in g.passengers
        if g.estimated_time <= p.time_window_end - p.time_window_start
    ) / max(len(all_passengers), 1)

    return (w1 / total_dist) + (w2 * utilisation) + (w3 * time_compliance) - (w4 * unmatched)


# ──────────────────────────────────────────────────────────
# Genetic operators
# ──────────────────────────────────────────────────────────

def _tournament_select(population: List[Tuple[Chromosome, float]], k: int = 3) -> Chromosome:
    contestants = random.sample(population, min(k, len(population)))
    return max(contestants, key=lambda x: x[1])[0]


def _order_crossover(parent1: Chromosome, parent2: Chromosome) -> Chromosome:
    """Order Crossover (OX1) — preserves relative order."""
    n = len(parent1)
    if n < 2:
        return parent1[:]
    a, b = sorted(random.sample(range(n), 2))
    child = [-1] * n
    child[a:b+1] = parent1[a:b+1]
    fill = [x for x in parent2 if x not in child]
    idx = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[idx]
            idx += 1
    return child


def _mutate(chromosome: Chromosome, rate: float) -> Chromosome:
    """Swap mutation: randomly swap two genes."""
    chrom = chromosome[:]
    for i in range(len(chrom)):
        if random.random() < rate:
            j = random.randint(0, len(chrom) - 1)
            chrom[i], chrom[j] = chrom[j], chrom[i]
    return chrom


# ──────────────────────────────────────────────────────────
# Main GA
# ──────────────────────────────────────────────────────────

class GeneticOptimizer:
    """
    Genetic Algorithm for global carpooling optimisation.

    Parameters
    ----------
    pop_size        : population size
    generations     : number of generations
    mutation_rate   : per-gene mutation probability
    elite_fraction  : fraction of top individuals carried to next gen
    """

    def __init__(self,
                 drivers: List[Driver],
                 passengers: List[Passenger],
                 graph: CityGraph,
                 pop_size: int = 60,
                 generations: int = 80,
                 mutation_rate: float = 0.05,
                 elite_fraction: float = 0.1):
        self.drivers = drivers
        self.passengers = passengers
        self.graph = graph
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = max(1, int(pop_size * elite_fraction))
        self.history: List[float] = []   # best fitness per generation

    def _random_chromosome(self) -> Chromosome:
        chrom = list(range(len(self.passengers)))
        random.shuffle(chrom)
        return chrom

    def _evaluate(self, chrom: Chromosome) -> float:
        groups = _decode(chrom, self.drivers, self.passengers, self.graph)
        return _fitness(groups, self.passengers)

    def run(self) -> Tuple[List[RideGroup], float, List[float]]:
        """
        Run the Genetic Algorithm.

        Returns
        -------
        best_groups   : decoded ride groups from the best chromosome
        best_fitness  : best fitness value achieved
        history       : best fitness per generation
        """
        if not self.passengers:
            return [], 0.0, []

        # Initialise population
        population: List[Tuple[Chromosome, float]] = []
        for _ in range(self.pop_size):
            chrom = self._random_chromosome()
            fit = self._evaluate(chrom)
            population.append((chrom, fit))

        best_chrom, best_fit = max(population, key=lambda x: x[1])
        self.history = []

        for gen in range(self.generations):
            # Sort by fitness descending
            population.sort(key=lambda x: x[1], reverse=True)
            gen_best = population[0][1]
            self.history.append(gen_best)

            if gen_best > best_fit:
                best_fit = gen_best
                best_chrom = population[0][0]

            # Elitism: keep top individuals
            new_population = [population[i] for i in range(self.elite_size)]

            # Fill rest via crossover + mutation
            while len(new_population) < self.pop_size:
                p1 = _tournament_select(population)
                p2 = _tournament_select(population)
                child = _order_crossover(p1, p2)
                child = _mutate(child, self.mutation_rate)
                fit = self._evaluate(child)
                new_population.append((child, fit))

            population = new_population

        best_groups = _decode(best_chrom, self.drivers, self.passengers, self.graph)
        for g in best_groups:
            g.fitness_score = best_fit

        return best_groups, best_fit, self.history
