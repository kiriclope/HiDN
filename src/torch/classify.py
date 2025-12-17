import numpy as np
import pandas as pd
from time import perf_counter
from src.common.get_data import get_X_y_days, get_X_y_S1_S2, get_X_y_mice
from src.preprocess.helpers import avg_epochs
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

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
    elif options['features'] == 'test':
        y = y.test_odor.to_numpy()
    elif options['features'] == 'distractor':
        y = y.dist_odor.to_numpy()
    elif options['features'] == 'choice':
        y = y.choice.to_numpy()
    elif options['features'] == 'pair':
        y = y.pair.to_numpy()

    return y

def get_classification(model, RETURN='overlaps', **options):
    start = perf_counter()

    if options['mouse']=='all':
        X_days, y_days = get_X_y_mice(**options)
    else:
        X_days, y_days = get_X_y_days(**options)

    y_days = y_days.reset_index()

    print('X_days', X_days.shape, 'y_days', y_days.shape)
    y_days['idx'] = y_days['index'].copy()

    X_B, y_B, y_B_labels = None, None, None
    y_labels = None
    cv_B = options['cv_B']

    if (options['features'] == 'sample'):
        if cv_B:
            options['trials'] = 'incorrect'

            X_B, y_B = get_X_y_S1_S2(X_days, y_days, **options)
            y_B_labels = y_B.copy()
            y_B = y_to_arr(y_B, **options)

            print('Test set: X_B', X_B.shape, 'y_B', y_B.shape, np.unique(y_B), y_B_labels.tasks.unique())

            options['trials'] = 'correct'

    if (options['features']=='choice') or (options['features'] == 'test'):
        if ('ACC' in options['mouse']):
            cv_B = False

        if cv_B:
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
        # options['laser'] = 0
        options['features'] = feat
        options['task'] = 'Dual'
        options['trials'] = 'correct'

    # load training data
    X, y = get_X_y_S1_S2(X_days, y_days, **options)
    y_labels = y.copy()

    if (options['features'] == 'distractor') or (options['features'] == 'odr_choice'):
        y_labels = y_labels.dropna()

    y = y_to_arr(y, **options)

    print('Training set: X', X.shape, 'y_labels', y_labels.shape, y_labels.tasks.unique(), y.shape)

    X_flat = avg_epochs(X, **options).astype('float32')
    y_flat = y.copy()

    # X_task = X[..., options['bins_' + options['epochs'][0]]]
    # y_flat = np.repeat(y, X_task.shape[-1])
    # X_flat = X_task.transpose(0, 2, 1).reshape(-1, X_task.shape[1])

    if 'bolasso' in RETURN:
        coefs = model.get_bolasso_coefs(X_flat, y_flat,
                                        n_boots=options['n_boots'],
                                        penalty=options['bolasso_penalty'],
                                        pval=options['pval'],
                                        confidence=options['bolasso_pval'])

        output = coefs, 0

    elif 'pca' in RETURN:

        if options['epochs'][0] == 'all':
            X_task = X.copy()
        else:
            X_task = X[..., options['bins_' + options['epochs'][0]]]

        # full pca
        # X_flat = X_task.transpose(0, 2, 1).reshape(-1, X_task.shape[1])
        # X_pca = model.fit_pca(X_flat).reshape(X.shape[0], X_task.shape[-1], -1)

        # X_flat = X.copy().transpose(0, 2, 1).reshape(-1, X.shape[1])
        # X_pca = model.transform_pca(X_flat).reshape(X.shape[0], X.shape[-1], -1)

        # Condition averaged pca
        # X_pca = []

        X_avg = []
        for i in range(4):
            idx = (y_labels.odor_pair==i)

            if options['trials'] == 'correct':
                correct = ~y_labels.response.str.contains("incorrect")
                idx  = (y_labels.odor_pair==i) & correct

            X_avg.append(np.mean(X_task[idx], 0))
            # for k in range(len(y_labels.tasks.unique())):
            #     task = y_labels.tasks.unique()[k]
            #     if options['trials'] == 'correct':
            #         correct = ~y_labels.response.str.contains("incorrect")
            #     else:
            #         correct = True

            #     idx = (y_labels.odor_pair==i) & (y_labels.tasks==task) & correct

            #     if 'Dual' in task:
            #         if options['trials'] == 'correct':
            #             idx = (y_labels.odor_pair==i) & (y_labels.tasks==task) & (y_labels.odr_perf==1) & correct
            #         else:
            #             idx = (y_labels.odor_pair==i) & (y_labels.tasks==task)

            #     if idx.mean()>0:
            #         X_avg.append(np.mean(X_task[idx], 0))

        X_avg = np.array(X_avg)

        X_mean = np.mean(X_task, 0)[np.newaxis]
        X_avg -= X_mean

        X_std = np.std(X_task, 0)[np.newaxis]
        X_avg /= X_std

        X_flat = X_avg.transpose(0, 2, 1).reshape(-1, X_avg.shape[1])
        X_pca = model.fit_pca(X_flat)

        X_test = X.copy()
        X_mean = np.mean(X_test, 0)[np.newaxis]
        X_test -= X_mean

        X_std = np.std(X_test, 0)[np.newaxis]
        X_test /= X_std

        X_flat = X_test.transpose(0, 2, 1).reshape(-1, X.shape[1])
        X_pca = model.transform_pca(X_flat).reshape(X.shape[0], X.shape[-1], -1)

        output = X_pca, y_labels, model.components_, model.explained_variance_

    elif 'bootstrap' in RETURN:

        coefs, bias = model.get_bootstrap_coefs(X_flat, y_flat, n_boots=options['n_boots'])
        output = coefs, bias

    else:
        model.fit(X_flat, y_flat)

    if 'scores' in RETURN:
        features = options['features']
        if (options['features']=='sample') or (options['features']=='test'):
            features +='_odor'

        if options['features'] =='distractor':
            features = 'dist_odor'

        scores, perm_scores, probas, coefs, _ = model.get_cv_scores(X, y_labels,
                                                                    options["scoring"],
                                                                    features=features,
                                                                    cv=options['cv'],
                                                                    X_B=X_B, y_B=y_B, cv_B=cv_B)
        try:
            coefs = np.array(coefs)
            coefs = coefs[..., 0, :]
            coefs = coefs.reshape(-1, coefs.shape[1] * coefs.shape[-1])
            print('coefs', coefs.shape)
        except:
            pass

        try:
            scores = np.array(scores)
            print('scores', scores.shape, np.nanmean(scores))
        except:
            pass

        try:
            perm_scores = np.array(perm_scores)
            print('perm_scores', perm_scores.shape, np.nanmean(perm_scores))
        except:
            pass


        try:
            probas = np.array(probas)
            probas[y==1, 0, ..., 0] = probas[y==1, 0, ..., 1]
            print('probas', probas.shape)
        except:
            pass

        if y_B_labels is not None:
            scores_A = np.array(scores[:y_labels.shape[0]][0])
            perm_scores_A = np.array(perm_scores[:y_labels.shape[0]][0])
        else:
            scores_A = scores
            perm_scores_A = perm_scores

        if options['cv'] is None:
            y_labels = y_labels[:scores_A.shape[0]]

        if 'df' in RETURN:
            if options['mne_estimator'] == 'generalizing':
                scores_A = scores_A.reshape(-1, X.shape[-1] * X.shape[-1])
                perm_scores_A = perm_scores_A.reshape(-1, X.shape[-1] * X.shape[-1])
            else:
                scores_A = scores.reshape(-1, X.shape[-1])
                perm_scores_A = perm_scores.reshape(-1, X.shape[-1])

            df = y_labels.reset_index(drop=True)
            df['overlaps'] = pd.DataFrame({'overlaps': scores_A.tolist()}).reset_index(drop=True)['overlaps']
            df['perm_overlaps'] = pd.DataFrame({'perm_overlaps': perm_scores_A.tolist()}).reset_index(drop=True)['perm_overlaps']

            if 'all' in RETURN:
                df['coefs'] = pd.DataFrame({'coefs': coefs.tolist()}).reset_index(drop=True)['coefs']

            print('df_A', df.shape, 'scores', scores_A.shape, 'labels', y_labels.shape)

            if y_B_labels is not None:
                scores_B = np.array(scores[:y_B_labels.shape[0]][1])
                print('scores_B', scores_B.shape)

                if options['mne_estimator'] == 'generalizing':
                    scores_B = scores_B.reshape(-1, X.shape[-1] * X.shape[-1])
                else:
                    scores_B = scores_B.reshape(-1, X.shape[-1])

                df_B = y_B_labels.reset_index(drop=True)
                df_B['overlaps'] = pd.DataFrame({'overlaps': scores_B.tolist()}).reset_index(drop=True)['overlaps']

                print('df_B', df_B.shape, 'scores', scores_B.shape, 'labels', y_B_labels.shape)

                df = pd.concat((df_B, df)).reset_index(drop=True)

            df['day'] = options['day']

            print('df', df.shape)
            output = df


    end = perf_counter()
    print("Elapsed (with compilation) = %dh %dm %ds" % convert_seconds(end - start))
    return output
