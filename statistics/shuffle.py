import numpy as np
from joblib import Parallel, delayed

from . import progressbar as pgb


def shuffle_parloop(X, statfunc):

    np.random.seed(None)

    X_shuffle = X.copy()
    np.random.shuffle(X_shuffle)
    res = statfunc(X_shuffle)

    return res


def my_shuffle(X, statfunc, n_samples=1000, n_jobs=-1):

    with pgb.tqdm_joblib(pgb.tqdm(desc='shuffle', total=n_samples)) as progress_bar:
        res = Parallel(n_jobs=n_jobs)(delayed(shuffle_parloop)(X, statfunc)
                                      for _ in range(n_samples))

    res = np.asarray(res)

    return res
