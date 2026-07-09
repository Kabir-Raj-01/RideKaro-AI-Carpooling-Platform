"""
data/data_generator.py
Realistic demo data generator for Karachi carpooling scenarios.
Produces Driver and Passenger objects placed on the Karachi city graph.
"""
from __future__ import annotations
import random
from typing import List, Tuple, Dict
from models.models import Driver, Passenger
from models.city_graph import CityGraph


_DRIVER_NAMES = [
    "Ali Hassan", "Bilal Ahmed", "Zara Khan", "Hamza Sheikh",
    "Fatima Malik", "Usman Raza", "Ayesha Siddiqui", "Tariq Mehmood",
    "Sana Butt", "Imran Chaudhry",
]

_PASSENGER_NAMES = [
    "Kabir Raj", "Hassnain Aziz", "Sara Qureshi", "Noman Akhtar",
    "Hina Mirza", "Danish Farooq", "Mehwish Baig", "Asad Javed",
    "Rida Hussain", "Kamran Iqbal", "Lubna Shah", "Fawad Anwar",
    "Zainab Rizvi", "Omer Farhan", "Naila Tahir", "Saad Mahmood",
    "Amna Nawaz", "Junaid Bhatti", "Sobia Noor", "Talha Suleman",
]

_VEHICLE_TYPES = {
    "Suzuki Alto": 4,
    "Honda City": 4,
    "Toyota Corolla": 4,
    "Suzuki Wagon R": 5,
    "Toyota Prado": 6,
    "Honda BRV": 7,
}


def generate_scenario(city: CityGraph,
                       n_drivers: int = 6,
                       n_passengers: int = 15,
                       hour: int = 8,
                       seed: int = 42) -> Tuple[List[Driver], List[Passenger]]:
    """
    Generate a realistic carpooling scenario.

    Parameters
    ----------
    city        : the CityGraph instance
    n_drivers   : number of drivers
    n_passengers: number of passengers
    hour        : simulated hour (affects time windows)
    seed        : random seed for reproducibility

    Returns
    -------
    drivers, passengers
    """
    random.seed(seed)
    loc_names = [loc.name for loc in city.all_locations()]
    locs = {name: city.get_location(name) for name in loc_names}

    def rand_loc():
        return locs[random.choice(loc_names)]

    def rand_time_window():
        start = max(0.0, float(hour * 60 + random.randint(-15, 15)))
        end = start + random.uniform(20, 45)
        return start, end

    drivers: List[Driver] = []
    driver_names = random.sample(_DRIVER_NAMES, min(n_drivers, len(_DRIVER_NAMES)))

    for i, name in enumerate(driver_names):
        vehicle, cap = random.choice(list(_VEHICLE_TYPES.items()))
        tw_s, tw_e = rand_time_window()
        pickup = rand_loc()
        dropoff = rand_loc()
        while dropoff == pickup:
            dropoff = rand_loc()

        d = Driver(
            user_id=f"D{i+1:02d}",
            name=name,
            role=None,          # set by __post_init__
            pickup=pickup,
            dropoff=dropoff,
            time_window_start=tw_s,
            time_window_end=tw_e,
            rating=round(random.uniform(3.8, 5.0), 1),
            vehicle_capacity=cap,
            fuel_efficiency=random.uniform(12.0, 18.0),
        )
        drivers.append(d)

    passengers: List[Passenger] = []
    pass_names = random.sample(_PASSENGER_NAMES, min(n_passengers, len(_PASSENGER_NAMES)))

    for i, name in enumerate(pass_names):
        tw_s, tw_e = rand_time_window()
        pickup = rand_loc()
        dropoff = rand_loc()
        while dropoff == pickup:
            dropoff = rand_loc()

        p = Passenger(
            user_id=f"P{i+1:02d}",
            name=name,
            role=None,
            pickup=pickup,
            dropoff=dropoff,
            time_window_start=tw_s,
            time_window_end=tw_e,
            rating=round(random.uniform(3.5, 5.0), 1),
            num_people=random.choices([1, 1, 1, 2], weights=[0.7, 0.1, 0.1, 0.1])[0],
            max_detour_minutes=random.uniform(8.0, 20.0),
        )
        passengers.append(p)

    return drivers, passengers
