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

        for train_idx, test_idx in self.skf.split(X, y_sample):
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


class BalancedStratifiedKFold:
    def __init__(self, n_splits=5, random_state=None, stratify_col='sample_odor', shuffle=True):
        # self.skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.skf = LeaveOneOut()
        self.stratify_col = stratify_col

    def split(self, X, y_df, groups=None):
        # y_df: pandas DataFrame with columns ["sample", "test", "choice"]
        y_sample = y_df[self.stratify_col].astype(int).values
        y_choice = y_df['choice'].astype(int).values

        for train_idx, test_idx in self.skf.split(X, y_sample):
            # Select only indices corresponding to the minority class in 'choice'
            choice_test_labels = y_choice[test_idx]
            # Count number in each choice class
            counts = np.bincount(choice_test_labels)
            min_count = counts.min()
            # For each sample class, randomly select min_count indices with that label and in the test set
            sample_test_labels = y_sample[test_idx]
            selected_test_idx = []
            rng = np.random.default_rng()
            for sample_class in np.unique(sample_test_labels):
                mask = (sample_test_labels == sample_class)
                idx_in_class = np.where(mask)[0]
                if len(idx_in_class) >= min_count:
                    chosen = rng.choice(idx_in_class, min_count, replace=False)
                else:
                    chosen = idx_in_class  # If not enough, just take all
                selected_test_idx.extend(test_idx[chosen])
            selected_test_idx = np.array(selected_test_idx)
            yield train_idx, selected_test_idx


class StratifiedSubSampleKFold:
    """
    A cross-validator that:
      - splits like StratifiedKFold on col for overall stratification
      - under-samples train indices to balance a given 'balance_col'
        within (optional) group cols
    """
    def __init__(self, n_splits=5, n_repeats=10, stratify_col='test_odor',
                 balance_col='choice', group_cols='test_odor', random_state=None, shuffle=True):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.stratify_col = stratify_col
        self.balance_col = balance_col
        self.group_cols = group_cols
        self.random_state = random_state
        self.shuffle = shuffle

    def split(self, X, y, groups=None):
        # y: DataFrame
        skf = StratifiedKFold(n_splits=self.n_splits, random_state=self.random_state)

        stratify_vals = y[self.stratify_col]

        for train_idx, test_idx in skf.split(X, stratify_vals, groups):
            y_train = y.iloc[train_idx].copy()
            idxs_to_keep = []

            # Group and under-sample to balance 'balance_col' in all groups
            if self.group_cols is None:
                groupby = [self.balance_col]
            else:
                groupby = list(self.group_cols) + [self.balance_col]

            if self.group_cols is None:
                # Just balance across balance_col
                min_count = y_train[self.balance_col].value_counts().min()
                for val in y_train[self.balance_col].unique():
                    sub = y_train[y_train[self.balance_col] == val]
                    sample = sub.sample(n=min_count, random_state=self.random_state)
                    idxs_to_keep.extend(sample.index.tolist())
            else:
                # Balance within each group defined by group_cols
                grouped = y_train.groupby(self.group_cols)
                for name, group in grouped:
                    counts = group[self.balance_col].value_counts()
                    min_count = counts.min()
                    for bal_val in group[self.balance_col].unique():
                        sub = group[group[self.balance_col] == bal_val]
                        if len(sub) >= min_count:
                            sample = sub.sample(n=min_count, random_state=self.random_state)
                            idxs_to_keep.extend(sample.index.tolist())
                        # else: drop (not enough to sample)

            idxs_to_keep = sorted(set(idxs_to_keep)) # De-duplicated, sorted (not strictly necessary)
            yield np.array(idxs_to_keep), test_idx


class StratifiedOverSampleKFold:
    """
    Like StratifiedKFold, but up-samples the minority class (balance_col) in each train split,
    either globally or within each group defined in group_cols.
    """
    def __init__(self, n_splits=5, n_repeats=10, stratify_col='test',
                 balance_col='choice', group_cols=None, random_state=None, shuffle=True):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.stratify_col = stratify_col
        self.balance_col = balance_col
        self.group_cols = group_cols
        self.random_state = random_state
        self.shuffle = shuffle

    def split(self, X, y, groups=None):
        rng = np.random.RandomState(self.random_state)
        skf = RepeatedStratifiedKFold(
            n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state)
        stratify_vals = y[self.stratify_col]

        for train_idx, test_idx in skf.split(X, stratify_vals, groups):
            y_train = y.iloc[train_idx].copy()
            idxs_to_keep = []

            # Over-sample within each group or globally
            if self.group_cols is None:
                # Just balance globally across balance_col
                counts = y_train[self.balance_col].value_counts()
                max_count = counts.max()
                for val in y_train[self.balance_col].unique():
                    sub = y_train[y_train[self.balance_col] == val]
                    # Sample with replacement to bring all up to max_count
                    sample = sub.sample(
                        n=max_count,
                        replace=True,
                        random_state=self.random_state
                    )
                    idxs_to_keep.extend(sample.index.tolist())
            else:
                # Balance within each group
                grouped = y_train.groupby(self.group_cols)
                for name, group in grouped:
                    counts = group[self.balance_col].value_counts()
                    max_count = counts.max()
                    for bal_val in group[self.balance_col].unique():
                        sub = group[group[self.balance_col] == bal_val]
                        sample = sub.sample(
                            n=max_count,
                            replace=True,
                            random_state=self.random_state
                        )
                        idxs_to_keep.extend(sample.index.tolist())

            # No deduplication needed since over-sampling intentionally allows repeats
            yield np.array(idxs_to_keep), test_idx
