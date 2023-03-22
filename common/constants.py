import os
import numpy as np
import seaborn as sns

global path, scriptdir, figdir, filedir

# in bcn
# data_path = "/home/leon/dual_task_data"
# path = "/home/leon/bebopalula/python/dual"

# on achlys
# data_path = "/home/leon/bebopalula/python/dual/data"
# path = "/home/leon/bebopalula/python/dual"

# in paris 
data_path = "/homecentral/alexandre.mahrach/bebopalula/dual_data/data"
path = "/homecentral/alexandre.mahrach/bebopalula/dual_data/"

global mouse, mice
mouse = "JawsM15"
mice = [
    # "C57_2_DualTask",
    "ChRM04",
    "JawsM15",
    "JawsM18",
    "ACCM03",
    "ACCM04",
    # "PrL",
    # "ACC",
    # "AP02",
    # "AP12",
    # "PP09",
]

figdir = path + "/figs/" + mouse
filedir = path + "/data/"

if not os.path.isdir(figdir):
    os.makedirs(figdir)

if not os.path.isdir(filedir):
    os.makedirs(filedir)

global day, days, n_days, n_first
day = 1
n_days = 6  # PrL 6, ACC 5 or multi 10

if "P" in mouse:
    n_days = 10  # PrL 6, ACC 5 or multi 10

days = np.arange(1, n_days + 1)

global n_discard
n_discard = 0
n_first = 3  # 3 or 2
n_middle = 0  # 0 or 2
if "P" in mouse:
    n_discard = 4
    n_first = 3  # 3 or 4
    n_middle = 0  # 0 or 3

global task, tasks
task = "DPA"
tasks = ["DPA", "DualGo", "DualNoGo"]

global epoch_str, task_str
epoch_str = ["Early", "Middle", "Late"]
task_str = ["DPA", "Dual Go", "Dual NoGo"]

global IF_SAVE
IF_SAVE = 1

global epochs
# epochs = ['MD']
# epochs = ['Baseline','Stim','ED','Dist','MD','CUE','LD','Test']
epochs = ["ED", "MD", "LD"]

global frame_rate, duration, time
frame_rate = 6
inv_frame = 0  # 1 / frame_rate
duration = 14  # 14, 19.2
if "P" in mouse:
    duration = 19.2

time = np.linspace(0, duration, int(duration * frame_rate))

global t_ED, t_MD, t_LD
global t_BL, t_STIM, t_TEST, t_DIST, t_CUE, t_RWD

t_BL = [0, 2]
t_STIM = [2 + inv_frame, 3]
t_ED = [3 + inv_frame, 4.5]
t_DIST = [4.5 + inv_frame, 5.5]
t_MD = [5.5 + inv_frame, 6.5]
t_CUE = [6.5 + inv_frame, 7]
t_RWD = [7 + inv_frame, 7.5]
t_LD = [7.5 + inv_frame, 9]
t_TEST = [9 + inv_frame, 10]
t_RWD2 = [11 + inv_frame, 12]

if "P" in mouse:
    t_BL = [0 + inv_frame, 2]
    t_STIM = [2 + inv_frame, 3]
    t_ED = [3 + inv_frame, 4]
    t_DIST = [4 + inv_frame, 5]
    t_MD = [5 + inv_frame, 6]
    t_CUE = [6 + inv_frame, 7]
    t_RWD = [7 + inv_frame, 8]
    t_LD = [8 + inv_frame, 9]
    t_TEST = [9 + inv_frame, 10]
    t_RWD2 = [11 + inv_frame, 12]

    # t_BL = [0 + inv_frame, 2 + inv_frame]
    # t_STIM = [2 + inv_frame, 3 + inv_frame]
    # t_ED = [3 + inv_frame, 4 + inv_frame]
    # t_DIST = [4 + inv_frame, 5 + inv_frame]
    # t_MD = [5 + inv_frame, 6 + inv_frame]
    # t_CUE = [6 + inv_frame, 7 + inv_frame]
    # t_RWD = [7 + inv_frame, 8 + inv_frame]
    # t_LD = [8 + inv_frame, 9 + inv_frame]
    # t_TEST = [9 + inv_frame, 10 + inv_frame]
    # t_RWD2 = [10 + inv_frame, 12 + inv_frame]

    # t_MD = [5, 10]
    # t_CUE = [10, 11]
    # t_RWD = [11, 12]
    # t_LD = [12, 13]
    # t_TEST = [13, 14]

global bins, bins_BL, bins_STIM, bins_ED, bins_DIST, bins_MD, bins_LD, bins_CUE, bins_RWD, bins_TEST
bins = np.arange(0, len(time))

global T_WINDOW
T_WINDOW = 0.5

bins_BL = bins[int(t_BL[0] * frame_rate) : int(t_BL[1] * frame_rate)]
bins_STIM = bins[int((t_STIM[0] + T_WINDOW) * frame_rate) : int(t_STIM[1] * frame_rate)]
bins_ED = bins[int((t_ED[0] + T_WINDOW) * frame_rate) : int(t_ED[1] * frame_rate)]
bins_DIST = bins[int((t_DIST[0] + T_WINDOW) * frame_rate) : int(t_DIST[1] * frame_rate)]
bins_MD = bins[int((t_MD[0] + T_WINDOW) * frame_rate) : int(t_MD[1] * frame_rate)]
bins_CUE = bins[int((t_CUE[0] + T_WINDOW) * frame_rate) : int(t_CUE[1] * frame_rate)]
bins_RWD = bins[int((t_RWD[0] + T_WINDOW) * frame_rate) : int(t_RWD[1] * frame_rate)]
bins_LD = bins[int((t_LD[0] + T_WINDOW) * frame_rate) : int(t_LD[1] * frame_rate)]
bins_TEST = bins[
    int((t_TEST[0] + T_WINDOW) * frame_rate) : int((t_TEST[1]) * frame_rate)
]

bins_CHOICE = bins[
    int((t_TEST[1] + T_WINDOW) * frame_rate) : int(t_RWD2[0] * frame_rate)
]

bins_RWD2 = bins[int((t_RWD2[0] + T_WINDOW) * frame_rate) : int(t_RWD2[1] * frame_rate)]

# print("bins ED", bins_ED)
# print("bins MD", bins_MD)
# print("bins LD", bins_LD)

global data_type
data_type = "raw"  # 'raw' or 'dF'

global pal
pal = [
    sns.color_palette("bright")[1],
    sns.color_palette("bright")[0],
    sns.color_palette("bright")[2],
    sns.color_palette("bright")[3],
]
