from typing import Tuple, Dict, List
from tqdm.auto import tqdm
from geopy.distance import great_circle as geodesic
import gpxpy
import pandas as pd
import numpy as np

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