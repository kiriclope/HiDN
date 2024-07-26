from src.common.get_data import get_X_y_days, get_X_y_S1_S2
from src.preprocess.helpers import avg_epochs

def get_classification(model, RETURN='overlaps', **options):
        start = perf_counter()

        dum = 0
        if options['features'] == 'distractor':
                if options['task'] != 'Dual':
                        task = options['task']
                        options['task'] = 'Dual'
                        dum = 1

        X_days, y_days = get_X_y_days(**options)
        X, y = get_X_y_S1_S2(X_days, y_days, **options)
        y[y==-1] = 0
        if options['verbose']:
            print('X', X.shape, 'y', y.shape)

        X_avg = avg_epochs(X, **options).astype('float32')
        if dum:
                options['features'] = 'sample'
                options['task'] = task
                X, _ = get_X_y_S1_S2(X_days, y_days, **options)

        index = mice.index(options['mouse'])
        model.num_features = N_NEURONS[index]

        if options['class_weight']:
                pos_weight = torch.tensor(np.sum(y==0) / np.sum(y==1), device=DEVICE).to(torch.float32)
                print('imbalance', pos_weight)
                model.criterion__pos_weight = pos_weight

        model.fit(X_avg, y)

        if 'scores' in RETURN:
            scores = model.get_cv_scores(X, y, options['scoring'])
            end = perf_counter()
            print("Elapsed (with compilation) = %dh %dm %ds" % convert_seconds(end - start))
            return scores
        if 'overlaps' in RETURN:
            coefs, bias = model.get_bootstrap_coefs(X_avg, y, n_boots=options['n_boots'])
            overlaps = model.get_bootstrap_overlaps(X)
            end = perf_counter()
            print("Elapsed (with compilation) = %dh %dm %ds" % convert_seconds(end - start))
            return overlaps
        if 'coefs' in RETURN:
            coefs, bias = model.get_bootstrap_coefs(X_avg, y, n_boots=options['n_boots'])
            end = perf_counter()
            print("Elapsed (with compilation) = %dh %dm %ds" % convert_seconds(end - start))
            return coefs, bias
