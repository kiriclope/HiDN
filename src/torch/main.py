# import torch
import numpy as np
import pandas as pd
from time import perf_counter
from src.common.get_data import get_X_y_days, get_X_y_S1_S2
from src.preprocess.helpers import avg_epochs

# Function to map values
def map_values(row):
    if np.isnan(row['dist_odor']):
        return np.nan
    return row['sample_odor'] * 2 + row['dist_odor']

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

    return y

def get_classification(model, RETURN='overlaps', **options):
    start = perf_counter()

    X_days, y_days = get_X_y_days(**options)
    X_test, y_test, y_labels = None, None, None

    IF_COMPO=0
    IF_GEN = 1
    if options['features'] == 'distractor':
        if 'DPA' in options['task']:
            IF_COMPO = 1
            options['features'] = 'sample'
            options['task'] = 'DPA'
            X_test, y_test = get_X_y_S1_S2(X_days, y_days, **options)
            y_labels = y_test.copy()
            y_test = y_to_arr(y_test, **options)
            print('X_test', X_test.shape, 'y_test', y_test.shape)
            options['features'] = 'distractor'

        options['task'] = 'Dual'

    X, y = get_X_y_S1_S2(X_days, y_days, **options)
    if y_labels is None:
        if options['features'] == 'distractor':
            y_labels = y.copy().dropna()
        else:
            y_labels = y.copy()

    print('y_labels', y_labels.tasks.unique())

    y = y_to_arr(y, **options)

    if options['verbose']:
        print('X', X.shape, 'y', y.shape, np.unique(y))

    X_avg = avg_epochs(X, **options).astype('float32')
    y_avg = y.copy()

    try:
        y_avg[y_avg==2]=0
        y_avg[y_avg==3]=1
    except:
        pass

    # if options['class_weight']:
    #         pos_weight = torch.tensor(np.sum(y==0) / np.sum(y==1), device=DEVICE).to(torch.float32)
    #         print('imbalance', pos_weight)
    #         model.criterion__pos_weight = pos_weight

    if RETURN is None:
        return None
    elif RETURN == 'labels':
        pass
    else:
        model.fit(X_avg, y_avg)

    if 'scores' in RETURN:
        scores = model.get_cv_scores(X, y, options['scoring'], X_test=X_test, y_test=y_test, IF_GEN=IF_GEN, IF_COMPO=IF_COMPO, cv=options['cv'])
        print('scores', scores.shape)

        # if options['features'] == 'distractor':
        #     if IF_COMPO==0:
        #         idx_Go = (y==0) | (y==2)
        #         scores_Go = scores[idx_Go]
        #         scores_NoGo = scores[~idx_Go]
        #         scores = np.stack((scores_Go, scores_NoGo), 1)
        #     else:
        #         scores = scores[:, np.newaxis]

        #     print('scores', scores.shape)

        if 'df' in RETURN:
            if IF_GEN:
                scores_list = scores.reshape(-1, 84 * 84).tolist()
            else:
                scores_list = scores.reshape(-1, 84).tolist()

            df = pd.DataFrame({'overlaps': scores_list}).reset_index(drop=True)
            print(df.shape, y_labels.shape)
            print('y_labels', y_labels.tasks.unique())
            y_labels['day'] = options['day']

            labels = pd.concat([y_labels.reset_index(drop=True), df], axis=1)
            print('df', labels.shape)
            return labels

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
    elif 'labels' in RETURN:
        return y_labels
    else:
        return None
