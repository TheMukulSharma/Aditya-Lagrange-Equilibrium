"""
Tests for the Sun-Earth L1 solver.
"""

import numpy as np
from src.lagrange import CelestialBody, find_l1, G, AU, YEAR
from astropy.constants import (
    M_earth,
    M_sun,
    G,
    au,
)

# mass of celestial objects (in kg)
m_sun_kg = M_sun.value  # 1.98840987... x 10^30 kg
m_earth_kg = M_earth.value  # 5.97216787... x 10^24 kg


def test_l1_distance_matches_known_value():
    """
    L1 should land within 0.1% of our last verified solver output
    (roughly around 1,492,165.0 km from Earth). This locks in current correctness so
    future changes to shared constants or dynamics can't silently
    corrupt this result.
    """

    sun = CelestialBody("Sun", mass=m_sun_kg, position=[0.0, 0.0])
    earth = CelestialBody("Earth", mass=m_earth_kg, position=[AU, 0.0])
    omega = 2 * np.pi / YEAR

    r_l1 = find_l1(sun, earth, omega)
    expected_km = 1_492_165.0
    tolerance_km = expected_km * 0.001  # 0.1% tolerance

    assert abs(r_l1 / 1000 - expected_km) < tolerance_km


def test_l1_lies_strictly_between_sun_and_earth():
    """L1 must be a physically valid point on the line"""

    sun = CelestialBody("Sun", mass=m_sun_kg, position=[0.0, 0.0])
    earth = CelestialBody("Earth", mass=m_earth_kg, position=[AU, 0.0])
    omega = 2 * np.pi / YEAR

    r_l1 = find_l1(sun, earth, omega)
    D = earth.position[0] - sun.position[0]

    assert 0 < r_l1 < D
