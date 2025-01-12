
import warnings

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
            paper_df = self.paper_df[self.paper_df['id'] == paper_id]
            _reviewers = group.signatures.apply(lambda x: any('Reviewer_' in s for s in x))
            review_df = group[_reviewers]
            other_replies_df = group[~_reviewers]

            # extract features
            features = {}
            for func_name in dir(FeatureFunctions):
                if func_name.startswith('ff_'):
                    func = getattr(FeatureFunctions, func_name)
                    feature_name = func_name[len("ff_"):]

                    features[feature_name] = func(paper_df, review_df, other_replies_df)

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
