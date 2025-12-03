import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, LeaveOneOut

class ChoiceLimitedStratifiedKFold:
    def __init__(self, n_splits=5, stratify_col='sample_odor', shuffle=True, random_state=None):
        # self.skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.skf = LeaveOneOut()
        self.stratify_col = stratify_col

    def split(self, X, y_df, groups=None):
        y_sample = y_df[self.stratify_col].astype(int).values
        y_choice = y_df['choice'].astype(int).values

        for train_idx, test_idx in self.skf.split(X, y_sample, groups):
            train_choices = y_choice[train_idx]
            # Figure out the minimum class count among "choice" classes in train
            choice_counts = np.bincount(train_choices)
            min_count = choice_counts.min()
            # For each "sample" class, select up to min_count samples (randomly) from train
            sample_train_labels = y_sample[train_idx]
            balanced_train_idx = []
            rng = np.random.default_rng()
            for sample_class in np.unique(sample_train_labels):
                idx_in_class = np.where(sample_train_labels == sample_class)[0]
                if len(idx_in_class) > min_count:
                    chosen = rng.choice(idx_in_class, min_count, replace=False)
                else:
                    chosen = idx_in_class
                # Map back into the train_idx space
                balanced_train_idx.extend(train_idx[chosen])
            balanced_train_idx = np.array(balanced_train_idx)
            yield balanced_train_idx, test_idx
