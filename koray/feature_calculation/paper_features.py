
import inspect
import warnings

import numpy as np
import pandas as pd

from koray.feature_calculation.feature_funcs import FeatureFunctions

try:
    from tqdm import TqdmExperimentalWarning

    # Remove experimental warning
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    from tqdm.rich import tqdm
except ImportError:
    import tqdm
tqdm.pandas()


def extract_overwrite_dtype(func):
    # Get the source code of the function
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        return None  # Could not retrieve source

    # Check for assignment to __overwrite_dtype__
    for line in source.splitlines():
        line = line.strip()
        if line.startswith("__overwrite_dtype__"):
            # Extract the assigned value (e.g., np.float64)
            parts = line.split("=", 1)
            if len(parts) == 2:
                # Return the assigned value as a string
                return parts[1].strip()
    return None


def get_dtype_safe(dtype_str):
    # Safe resolution of the dtype from globals() or locals()
    try:
        # Attempt to retrieve the dtype object from globals
        dtype = eval(dtype_str, {"np": np})
        return dtype
    except NameError:
        # If eval fails, return None
        return None


class FeatureExtractor:
    def __init__(self, paper_df: pd.DataFrame, review_df: pd.DataFrame, other_replies_df: pd.DataFrame):
        self.paper_df = paper_df
        self.review_df = review_df
        self.other_replies_df = other_replies_df

        self.feature_df = self.get_base_feature_df()

    def get_base_feature_df(self):
        feature_df = self.paper_df.iloc[:, [0]].copy()
        feature_df.rename(columns={'id': 'paper_id'}, inplace=True)
        return feature_df

    def _extract_features(self):
        def _impl(group):

            paper_id = group['replyto'].values[0]

            # create sub dataframes
            _reviewers = group.signatures.apply(lambda x: any('Reviewer_' in s for s in x))
            kwargs = {
                'paper_df': self.paper_df[self.paper_df['id'] == paper_id],
                'review_df': group[_reviewers],
                'other_replies_df': group[~_reviewers],
                'master_paper_df': self.paper_df,
                'master_review_df': self.review_df,
                'master_other_replies_df': self.other_replies_df,

            }

            # extract features
            features = {}
            for func_name in dir(FeatureFunctions):
                if func_name.startswith('ff_'):
                    func = getattr(FeatureFunctions, func_name)
                    feature_name = func_name[len("ff_"):]
                    features[feature_name] = func(**kwargs)

            return pd.Series(features)

        return _impl

    def extract_features(self):
        """main function to extract features"""

        feature_df = pd.concat(
            (self.review_df, self.other_replies_df),
            ignore_index=True
        ).groupby('replyto').progress_apply(
            self._extract_features()
        ).reset_index()
        self.feature_df = feature_df.rename(columns={'replyto': 'paper_id'})
        
        self.overwrite_dtypes()

    def overwrite_dtypes(self):
        for col in self.feature_df.columns:
            try:
                feature_func = getattr(FeatureFunctions, f"ff_{col}")
            except AttributeError:
                continue

            dtype_str = extract_overwrite_dtype(feature_func)
            if dtype_str:
                dtype = get_dtype_safe(dtype_str)
                if dtype:
                    self.feature_df[col] = self.feature_df[col].astype(dtype)
