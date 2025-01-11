
import re

import numpy as np
import pandas as pd


class FeatureExtractor:
    def __init__(self, paper_df, review_df, other_replies_df):
        self.paper_df = paper_df
        self.review_df = review_df
        self.other_replies_df = other_replies_df

        self.feature_df = self.get_base_feature_df()

    def get_base_feature_df(self):
        feature_df = self.paper_df.iloc[:, [0]].copy()
        feature_df.rename(columns={'id': 'paper_id'}, inplace=True)
        return feature_df

    def extract_features(self):
        """main function to extract features"""
        self.calculate_basic_features()
        self.calculate_compex_features()

    def calculate_basic_features(self):
        self.feature_df['title_length'] = self.paper_df['content'].apply(lambda x: len(x.get('title', '')))
        self.feature_df['abstract_length'] = self.paper_df['content'].apply(lambda x: len(x.get('abstract', '')))
        self.feature_df['tldr_length'] = self.paper_df['content'].apply(lambda x: len(x.get('TL;DR', '')))

        self.feature_df['author_count'] = self.paper_df['content'].apply(lambda x: len(x.get('authors', [])))
        self.feature_df['keyword_count'] = self.paper_df['content'].apply(lambda x: len(x.get('keywords', [])))

        self.feature_df = self.feature_df.merge(self._get_commitee_decision(), on='paper_id', how='left')

    def calculate_compex_features(self):
        self.feature_df = self.feature_df.merge(self.get_reviewer_features(), on='paper_id', how='left')

    def _get_commitee_decision(self):
        """is paper accepted or not"""

        def get_decision(group_df: 'pd.Series'):
            decision_note = group_df[group_df['invitation'].apply(lambda x: '/Decision' in x)]
            if len(decision_note) == 1:
                decision = decision_note['content'].iloc[0]['decision']
                return 'Accept' in decision
            else:
                # display(group_df)
                # display(decision_note)
                assert False

        def get_comment_length(group_df: 'pd.Series'):
            decision_note = group_df[group_df['invitation'].apply(lambda x: '/Meta' in x)]
            if len(decision_note) == 1:
                metareview = decision_note['content'].iloc[0]['metareview']
                return len(metareview)
            # paper might not have a metareview
            return None

        decision_agg = self.other_replies_df.groupby('replyto').progress_apply(
            lambda group: pd.Series(
                {
                    # we are working on a group.
                    'is_accepted': get_decision(group),
                    'metareview_length': get_comment_length(group),
                    # you can add more here
                }),
        ).reset_index()

        decision_agg = decision_agg.rename(columns={'replyto': 'paper_id'})
        return decision_agg

    def get_reviewer_features(self):
        # first we define the helper and main functions
        def _extract_numeric_prefix(maybe_string: object) -> int:
            """
            Extracts the leading integer from a string.
            If no integer is found, returns NaN.
            """
            string = str(maybe_string)
            match = re.match(r'^(\d+)', string.strip())
            return int(match.group(1)) if match else np.nan

        def numeric_prefix_agg(series: 'pd.Series', fieldname: str, agg_func: callable):
            field_values = series.apply(lambda x: _extract_numeric_prefix(
                x.get(fieldname)) if isinstance(x, dict) else None)

            if field_values.isnull().all():
                return np.nan  # if cannot extract numeric prefix from any value, return NaN
            return agg_func(field_values)

        # then we agg it. Note that this functions are applied to the 'content' field of the reviews.
        agg_functions_on_df = {
            'review_count': 'count',  # Count reviews
            #          example entry. The dict will have (key,value) pairs like this:
            #          'recommendation_nanmean': lambda s: numeric_prefix_agg(s, 'rating', np.nanmean),
        }

        fields = ['confidence', 'correctness', 'technical_novelty_and_significance',
                  'empirical_novelty_and_significance', 'recommendation']
        agg_functions = [np.nanmean, np.nanstd,
                         np.nanmin, np.nanmax, np.nanmedian, np.nansum, np.nanvar, np.nanprod
                         ]

        # add the aggregation functions to the dictionary
        for field in fields:
            for func in agg_functions:
                agg_functions_on_df[f'review_{field}_{func.__name__}'] = lambda s, field=field, func=func: numeric_prefix_agg(
                    s, field, func)

        # this is just 'review_df.groupby('replyto').agg(agg_functions_on_df).reset_index()' but I want tqdm. Thus the complexity.
        review_agg = self.review_df.groupby('replyto').progress_apply(
            lambda group: group['content'].agg(agg_functions_on_df)
        ).reset_index()

        review_agg = review_agg.rename(columns={'replyto': 'paper_id'}, inplace=True)
        return review_agg
