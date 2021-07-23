# %%
from re import M
from numba import njit
from dataclasses import dataclass
from typing import Tuple, Dict, List
from tqdm.auto import tqdm
from geopy.distance import great_circle as geodesic
import gpxpy
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from math import sqrt, sin, atan2
%load_ext autotime
# %%

# %%


def calculate_deltas(df: pd.DataFrame) -> Dict[int, Tuple[float, float, float]]:
    """

    """
    (lat1, lon1) = (np.nan, np.nan)
    deltas = {}
    for i, (ind, row) in tqdm(enumerate(df.iterrows()),
                              total=len(df),
                              desc='Calculating deltas'):

        (lat2, lon2) = (row.lat, row.lon)
        if i > 0:
            # Calculate horizontal, vertical and actual deltas
            delta_x = geodesic((lat1, lon1), (lat1, lon2)).meters
            delta_y = geodesic((lat1, lon1), (lat2, lon1)).meters
            delta = geodesic((lat1, lon1), (lat2, lon2)).meters

            # Put a negative signal if it is west / south directed
            if lon2 < lon1:
                delta_x *= -1
            if lat2 < lat1:
                delta_y *= -1

            # Attribute tuple of deltas
            deltas[ind] = (delta, delta_x, delta_y)
        else:
            deltas[ind] = (0.0, 0.0, 0.0)
        (lat1, lon1) = (lat2, lon2)
    return deltas


def load_route(path: str) -> pd.DataFrame:
    """

    """
    # Load and parse GPX
    with open(path, 'r') as fid:
        content: str = fid.read()
    gpx = gpxpy.parse(content)

    # Get points associated with the first track and segment
    route_points = gpx.tracks[0].segments[0].points

    # Generate dataframe from points
    cols = ['lon', 'lat', 'ele']
    df = (pd.DataFrame([(p.longitude, p.latitude, p.elevation)
                        for p
                        in route_points],
                       columns=cols)
          .drop_duplicates(subset=['lat', 'lon'])
          )
    return df


def append_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """

    """
    # Get the distance difference between each sucessive point
    deltas = calculate_deltas(df)

    # Put deltas into variables
    delta = {k: v[0] for k, v in deltas.items()}
    delta_x = {k: v[1] for k, v in deltas.items()}
    delta_y = {k: v[2] for k, v in deltas.items()}

    # Append deltas        lon2 = row.lon

    df = df.join(pd.Series(delta, name='delta'))
    df = df.join(pd.Series(delta_x, name='delta_x'))
    df = df.join(pd.Series(delta_y, name='delta_y'))

    # Calculate velocities on the latitude and longitude axes
    df = df.assign(u_x=lambda df: df.delta_x / df.delta)
    df = df.assign(u_y=lambda df: df.delta_y / df.delta)

    return df


# %%
df = (load_route('../data/route.gpx')
      .pipe(append_deltas)
      .assign(total_distance=lambda df: df.delta.cumsum())
      )
# %%

elevation_interpolator = interp1d(df.total_distance, df.ele)
longitude_interpolator = interp1d(df.total_distance, df.lon)
latitude_interpolator = interp1d(df.total_distance, df.lat)
# %%


@dataclass
class State():
    v: float


@dataclass
class EnvironmentalState():
    P_in: float
    theta: float
    A: float
    w: float
    phi: float


@dataclass
class Parameters():
    m: float
    g: float
    R: float
    dt: float


def delta_energy(v: float,
                 e: EnvironmentalState,
                 p: Parameters) -> float:
    P_g = v * p.m * p.g * sin(e.theta)
    P_r = v * p.m * p.g * p.R
    P_w = v * e.A * (v - e.w * sin(e.phi)) ** 2
    P_e = e.P_in * 0.02
    P_out = P_g + P_r + P_w + P_e
    P_result = e.P_in - P_out
    return P_result * p.dt


@njit
def energy_from_speed(v, m):
    return (m * v ** 2) / 2


@njit
def speed_from_energy(K, m):
    return sqrt(2 * K / m)


def new_speed(v: float,
              e: EnvironmentalState,
              p: Parameters) -> float:
    dK = delta_energy(v, e, p)
    K = energy_from_speed(v, p.m) + dK
    v = speed_from_energy(K, p.m)
    return v


@njit
def new_speed_raw(v: float,
                  theta: float,
                  A: float,
                  w: float,
                  phi: float,
                  P_in: float,
                  m: float,
                  g: float,
                  R: float,
                  dt: float) -> float:
    P_g = v * m * g * sin(theta)
    P_r = v * m * g * R
    P_w = v * A * (v - w * sin(phi)) ** 2
    P_e = P_in * 0.02
    P_out = P_g + P_r + P_w + P_e
    P_result = P_in - P_out
    dK = P_result * dt
    K = (m * v ** 2 / 2)
    new_v = sqrt(2 * (K + dK) / m)
    return new_v


def interpolate_incline(x: float,
                        x_old: float,
                        interpolator) -> float:
    dx = x - x_old
    past_elevation = interpolator(x_old)
    current_elevation = interpolator(x)
    dy = current_elevation - past_elevation
    value = atan2(dy, dx)
    return value


def simulate_route(elevation_interpolator,
                   max_x=10,
                   dt=0.1):
    x_old = 0.0
    x = 0.0
    v = 0.0
    p = Parameters(75, 9.8, 0.005, dt)
    while x < max_x:
        e = EnvironmentalState(P_in=120,
                               theta=interpolate_incline(x,
                                                         x_old,
                                                         elevation_interpolator),
                               A=0.5,
                               w=0.0,
                               phi=0.0)
        v = new_speed(v, e, p)
        x_old = x
        x += v * p.dt
        yield (v, x)


def simulate_route_raw(elevation_interpolator,
                       max_x=10,
                       dt=0.1):
    x_old = 0.0
    x = 0.0
    v = 0.0
    m = 75
    g = 9.8
    R = 0.005
    theta = 0.0
    A = 0.42
    w = 0.0
    phi = 0.0
    P_in = 120
    while x < max_x:
        theta = interpolate_incline(x,
                                    x_old,
                                    elevation_interpolator)
        v = new_speed_raw(v,
                          theta,
                          A,
                          w,
                          phi,
                          P_in,
                          m,
                          g,
                          R,
                          dt)
        x_old = x
        x += v * dt
        yield (v, x)

        # %%
dt = 0.05
max_x = 1000 * 100
a = list(simulate_route(elevation_interpolator,
                        max_x=max_x,
                        dt=dt))
print(len(a) * dt)
# %%
dt = 0.05
max_x = 1000 * 100
a = list(simulate_route_raw(elevation_interpolator,
                            max_x=max_x,
                            dt=dt))
print(len(a) * dt)

# %%
