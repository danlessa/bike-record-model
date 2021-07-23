from time import time
from typing import Any
from pandas.core.frame import DataFrame
from datetime import datetime, timedelta
from os import listdir
from pathlib import Path
from cadCAD_tools import easy_run
from cadCAD_tools.preparation import prepare_params, Param, ParamSweep
from json import dump
import papermill as pm
import os


def extrapolation_cycle(base_path: str = None,
                        historical_interval: Days = 14,
                        historical_lag: Days = 0,
                        signal_samples: int = 10,
                        extrapolation_samples: int = 1,
                        extrapolation_timesteps: int = 7 * 24,
                        use_last_data=False,
                        generate_reports=True) -> object:
    """
    Perform a entire extrapolation cycle.
    """
    t1 = time()

    # Collect wind speed data & current trajectory status
    print("0. Retrieving Data\n---")
    # TODO

    # 
    print("1. Preparing Data\n---")

    print("2. Backtesting Model\n---")


    print("3. Fitting Stochastic Processes\n---")

    print("4. Extrapolating Exogenous Signals\n---")
   

    print("5. Extrapolating Future Data\n---")
  

    print("6. Exporting results\n---")
   
    print(f"7. Done! {t2 - t1 :.2f}s\n---")

    return output
