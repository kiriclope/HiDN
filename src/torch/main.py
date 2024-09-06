# import torch
import numpy as np
from sklearn.model_selection import train_test_split
from time import perf_counter
from src.common.get_data import get_X_y_days, get_X_y_S1_S2
from src.preprocess.helpers import avg_epochs

# Function to map values
def map_values(row):
    if np.isnan(row['dist_odor']):
        return np.nan
    return row['sample_odor'] * 2 + row['dist_odor']

# Apply the function to each row

def convert_seconds(seconds):
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return h, m, s

def y_to_arr(y, **options):
    if options['task'] == 'Dual':
        y['labels'] = y.apply(map_values, axis=1)
        y = y.labels.dropna().to_numpy()
    elif options['features'] == 'sample':
        y = y.sample_odor.dropna().to_numpy()
    elif options['features'] == 'distractor':
        y = y.dist_odor.dropna().to_numpy()
    elif options['features'] == 'choice':
        y = y.choice.to_numpy()

    # y[y==-1] = 0

    return y

def get_classification(model, RETURN='overlaps', **options):
    start = perf_counter()

    # dum = 0
    # if options['features'] == 'distractor':
    #     if options['task'] != 'Dual':
    #         task = options['task']
    #         options['task'] = 'Dual'
    #         dum = 1

    X_days, y_days = get_X_y_days(**options)
    X, y = get_X_y_S1_S2(X_days, y_days, **options)
    y = y_to_arr(y, **options)

    # if dum:
    #     X, X_test, y, y_test = train_test_split(X, y, test_size=0.5, stratify=y)

    if options['verbose']:
        print('X', X.shape, 'y', y.shape)

    X_avg = avg_epochs(X, **options).astype('float32')
    y_avg = y.copy()

    # if dum==0:
    #     task = 'Dual'
    #     options['features'] = 'sample'
    #     options['task'] = task
    #     X_test, y_test = get_X_y_S1_S2(X_days, y_days, **options)
    #     y_test = y_to_arr(y_test, **options)
    #     # y_test = None
    #     # print('X_test', X_test.shape, 'y_test', y_test.shape)
    # else:
    X_test, y_test = None, None

    # if options['class_weight']:
    #         pos_weight = torch.tensor(np.sum(y==0) / np.sum(y==1), device=DEVICE).to(torch.float32)
    #         print('imbalance', pos_weight)
    #         model.criterion__pos_weight = pos_weight

    if RETURN is None:
        return None
    else:
        model.fit(X_avg, y_avg)

    if 'scores' in RETURN:
        scores = model.get_cv_scores(X, y, options['scoring'], X_test=X_test, y_test=y_test)
        # scores = model.get_cv_scores(X_avg[..., np.newaxis], y_avg, options['scoring'], X_test=X_test, y_test=y_test)
        end = perf_counter()
        print("Elapsed (with compilation) = %dh %dm %ds" % convert_seconds(end - start))
        return scores
    elif 'overlaps' in RETURN:
        coefs, bias = model.get_bootstrap_coefs(X_avg, y_avg, n_boots=options['n_boots'])
        overlaps = model.get_bootstrap_overlaps(X)
        end = perf_counter()
        print("Elapsed (with compilation) = %dh %dm %ds" % convert_seconds(end - start))
        return overlaps
    elif 'coefs' in RETURN:
        coefs, bias = model.get_bootstrap_coefs(X_avg, y_avg, n_boots=options['n_boots'])
        end = perf_counter()
        print("Elapsed (with compilation) = %dh %dm %ds" % convert_seconds(end - start))
        return coefs, bias
    else:
        return None
