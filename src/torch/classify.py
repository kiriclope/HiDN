import numpy as np
import pandas as pd
from time import perf_counter
from src.common.get_data import get_X_y_days, get_X_y_S1_S2
from src.preprocess.helpers import avg_epochs


def is_list_of_nans(x):
    if isinstance(x, list) or isinstance(x, np.ndarray):
        return np.isnan(x).all()
    return False

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
    if options['features'] == 'odr_choice':
        y = y.odr_choice.to_numpy()
    elif options['task'] == 'Dual':
        y['labels'] = y.apply(map_values, axis=1)
        y = y.labels.dropna().to_numpy()
    elif options['features'] == 'sample':
        y = y.sample_odor.to_numpy()
    elif options['features'] == 'distractor':
        y = y.dist_odor.to_numpy()
    elif options['features'] == 'choice':
        y = y.choice.to_numpy()
    elif options['features'] == 'pair':
        y = y.pair.to_numpy()

    return y

def get_classification(model, RETURN='overlaps', **options):
    start = perf_counter()

    X_days, y_days = get_X_y_days(**options)

    y_days = y_days.reset_index()
    print('X_days', X_days.shape, 'y_days', y_days.shape)
    y_days['idx'] = y_days['index'].copy()
    # print(y_days.head())

    X_B, y_B, y_B_labels = None, None, None
    y_labels = None
    cv_B = True

    if (options['features'] == 'sample'):
        # test on incorret trials
        options['trials'] = 'incorrect'

        X_B, y_B = get_X_y_S1_S2(X_days, y_days, **options)
        y_B_labels = y_B.copy()
        y_B = y_to_arr(y_B, **options)
        print('X_B', X_B.shape, 'y_B', y_B.shape, np.unique(y_B), y_B_labels.tasks.unique())

        # train and test on correct trials in X_A, y_A
        options['trials'] = 'correct'

    if (options['features']=='choice') and not ('ACC' in options['mouse']):
        # test on laser ON trials
        options['laser'] = 1

        X_B, y_B = get_X_y_S1_S2(X_days, y_days, **options)
        y_B_labels = y_B.copy()
        y_B = y_to_arr(y_B, **options)
        print('X_B', X_B.shape, 'nans', np.isnan(X_B).mean(), 'y_B', y_B.shape, np.unique(y_B), y_B_labels.tasks.unique())

        # train and test on laser OFF trials in X_A, y_A
        options['laser'] = 0

    if (options['features'] == 'distractor') or (options['features'] == 'odr_choice'):
        # test on odr incorrect trials, DPA, and laser ON trials see src/common/get_data.py
        feat = options['features']
        options['trials'] = 'incorrect'

        X_B, y_B = get_X_y_S1_S2(X_days, y_days, **options)
        y_B_labels = y_B.copy()
        y_B = np.ones(X_B.shape[0]) # y_to_arr(y_B, **options) % 2

        print('X_B', X_B.shape, 'y_B', y_B.shape, np.unique(y_B), y_B_labels.tasks.unique(), y_B_labels.odr_perf.unique())

        # train on odr correct trials
        options['laser'] = 0
        options['features'] = feat
        options['task'] = 'Dual'
        options['trials'] = 'correct'

    X, y = get_X_y_S1_S2(X_days, y_days, **options)
    y_labels = y.copy()

    if (options['features'] == 'distractor') or (options['features'] == 'odr_choice'):
        y_labels = y_labels.dropna()
        # y_labels = y_labels[y_labels.tasks!='DPA']

    print('y_labels', y_labels.shape, y_labels.tasks.unique())
    y = y_to_arr(y, **options) % 2

    if options['verbose']:
        print('X', X.shape, 'nans', np.isnan(X).mean(), 'y', y.shape, np.unique(y))

    X_avg = avg_epochs(X, **options).astype('float32')
    y_avg = y.copy()
    y_avg = y_avg % 2

    if RETURN is None:
        return None
    elif RETURN == 'DATA':
        return X_avg, y_labels
    elif RETURN == 'labels':
        pass
    elif 'bolasso' in RETURN:
        coefs = model.get_bolasso_coefs(X_avg, y_avg, n_boots=options['n_boots'], penalty=options['bolasso_penalty'], pval=options['pval'], confidence=options['bolasso_pval'])
    else:
        model.fit(X_avg, y_avg)

    if 'scores' in RETURN:
        scores, probas, coefs, _ = model.get_cv_scores(X, y, options["scoring"], cv=options['cv'], X_B=X_B, y_B=y_B, cv_B=cv_B)

        try:
            scores = np.array(scores)
            print('scores', scores.shape, np.nanmean(scores)) # 'probas', np.array(probas).shape, 'coefs', coefs.shape)
            probas = np.array(probas)
            coefs = np.array(coefs)
        except:
            pass

        if 'all' in RETURN:
            probas[y==1, 0, ..., 0] = probas[y==1, 0, ..., 1]
            coefs = coefs[..., 0, :]
            coefs = coefs.reshape(-1, coefs.shape[1] * coefs.shape[-1])

        if y_B_labels is not None:
            scores_A = np.array(scores[:y_labels.shape[0]][0])
            if 'all' in RETURN:
                probas_A = probas[:y_labels.shape[0], 0]
        else:
            scores_A = scores
            if 'all' in RETURN:
                probas_A = probas[:, 0]

        if options['cv'] is None:
            y_labels = y_labels[:scores_A.shape[0]]

        if 'df' in RETURN:
            if options['mne_estimator'] == 'generalizing':
                scores_A = scores_A.reshape(-1, X.shape[-1] * X.shape[-1])
                if 'all' in RETURN:
                    probas_A = probas_A.reshape(-1, X.shape[-1] * X.shape[-1])
            else:
                scores_A = scores.reshape(-1, X.shape[-1])
                if 'all' in RETURN:
                    probas_A = probas.reshape(-1, X.shape[-1])

            df = y_labels.reset_index(drop=True)
            df['overlaps'] = pd.DataFrame({'overlaps': scores_A.tolist()}).reset_index(drop=True)['overlaps']
            if 'all' in RETURN:
                df['probas'] = pd.DataFrame({'probas': probas_A.tolist()}).reset_index(drop=True)['probas']
                df['coefs'] = pd.DataFrame({'coefs': coefs.tolist()}).reset_index(drop=True)['coefs']

            print('df_A', df.shape, 'scores', scores_A.shape, 'labels', y_labels.shape)

            if y_B_labels is not None:
                scores_B = np.array(scores[:y_B_labels.shape[0]][1])
                print('scores_B', scores_B.shape)

                if 'all' in RETURN:
                    probas_B = probas[:y_B_labels.shape[0], 1]

                if options['mne_estimator'] == 'generalizing':
                    scores_B = scores_B.reshape(-1, X.shape[-1] * X.shape[-1])
                    if 'all' in RETURN:
                        probas_B = probas_B.reshape(-1, X.shape[-1] * X.shape[-1])
                else:
                    scores_B = scores_B.reshape(-1, X.shape[-1])
                    if 'all' in RETURN:
                        probas_B = probas_B.reshape(-1, X.shape[-1])

                df_B = y_B_labels.reset_index(drop=True)
                df_B['overlaps'] = pd.DataFrame({'overlaps': scores_B.tolist()}).reset_index(drop=True)['overlaps']
                if 'all' in RETURN:
                    df_B['probas'] = pd.DataFrame({'probas': probas_B.tolist()}).reset_index(drop=True)['probas']
                print('df_B', df_B.shape, 'scores', scores_B.shape, 'labels', y_B_labels.shape)

                df = pd.concat((df_B, df)).reset_index(drop=True)

            df['day'] = options['day']

            print('df', df.shape)
            return df


        end = perf_counter()
        print("Elapsed (with compilation) = %dh %dm %ds" % convert_seconds(end - start))
        return scores

    elif 'coefs' in RETURN:
        coefs, bias = model.get_bootstrap_coefs(X_avg, y_avg, n_boots=options['n_boots'])
        end = perf_counter()
        print("Elapsed (with compilation) = %dh %dm %ds" % convert_seconds(end - start))
        return coefs, bias
    elif 'bolasso' in RETURN:
        # coefs = model.get_bolasso_coefs(X_avg, y_avg, n_boots=options['n_boots'], penalty='l2', pval=options['pval'], confidence=0.05)
        return coefs, 0
    else:
        return None
