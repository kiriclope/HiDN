import numpy as np
from scipy.stats import circmean, circstd
from src.overlap.get_cos import run_get_cos, plot_bump
from src.decode.bump import decode_bump, circcvl  

def get_map_day(mouse, day, task, **kwargs):
  
  X_df, y_df, X_, y_, theta_first = run_get_cos(mouse=mouse, day=day, task=task, **kwargs)

  return X_, y_
