"""
models/models.py
Core data models: Location, User, Driver, Passenger, Ride, RideGroup
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import math
import uuid


class UserRole(Enum):
    DRIVER = "driver"
    PASSENGER = "passenger"


class RideStatus(Enum):
    PENDING = "pending"
    MATCHED = "matched"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class Location:
    name: str
    x: float  # longitude-like coordinate
    y: float  # latitude-like coordinate

    def distance_to(self, other: Location) -> float:
        """Euclidean distance between two locations."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __repr__(self):
        return f"Location({self.name}, x={self.x:.1f}, y={self.y:.1f})"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Location) and self.name == other.name


@dataclass
class User:
    user_id: str
    name: str
    role: UserRole
    pickup: Location
    dropoff: Location
    time_window_start: float  # minutes from now
    time_window_end: float
    rating: float = 5.0

    def __repr__(self):
        return (f"User(id={self.user_id}, name={self.name}, "
                f"role={self.role.value}, pickup={self.pickup.name}→{self.dropoff.name})")

    def __hash__(self):
        return hash(self.user_id)

    def __eq__(self, other):
        return isinstance(other, User) and self.user_id == other.user_id


@dataclass
class Driver(User):
    vehicle_capacity: int = 4
    fuel_efficiency: float = 15.0   # km per litre
    current_load: int = 0

    def __post_init__(self):
        self.role = UserRole.DRIVER

    @property
    def available_seats(self) -> int:
        return self.vehicle_capacity - self.current_load

    def can_accept(self, n_passengers: int = 1) -> bool:
        return self.available_seats >= n_passengers


@dataclass
class Passenger(User):
    num_people: int = 1
    max_detour_minutes: float = 15.0
    preferred_gender: Optional[str] = None

    def __post_init__(self):
        self.role = UserRole.PASSENGER


@dataclass
class RideGroup:
    group_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    driver: Optional[Driver] = None
    passengers: List[Passenger] = field(default_factory=list)
    route: List[Location] = field(default_factory=list)
    total_distance: float = 0.0
    estimated_time: float = 0.0
    cost_per_person: float = 0.0
    status: RideStatus = RideStatus.PENDING
    fitness_score: float = 0.0

    @property
    def total_passengers(self) -> int:
        return sum(p.num_people for p in self.passengers)

    def is_valid(self) -> bool:
        if self.driver is None:
            return False
        return self.driver.can_accept(self.total_passengers)

    def compute_cost(self, fuel_price_per_litre: float = 280.0) -> float:
        """Compute shared cost in PKR."""
        if self.driver is None or self.total_distance == 0:
            return 0.0
        litres = self.total_distance / self.driver.fuel_efficiency
        total_cost = litres * fuel_price_per_litre
        n = self.total_passengers + 1  # driver + passengers
        self.cost_per_person = round(total_cost / n, 2)
        return self.cost_per_person

    def __repr__(self):
        pnames = [p.name for p in self.passengers]
        driver_name = self.driver.name if self.driver else "None"
        return (f"RideGroup(id={self.group_id}, driver={driver_name}, "
                f"passengers={pnames}, dist={self.total_distance:.2f}km, "
                f"fitness={self.fitness_score:.4f})")
