
import re

import numpy as np
import pandas as pd

# ff_ is short for feature_function and the prefix is necessary for the feature_extractor.py to find the function
# these functions are used to calculate columns in feature df
# the inputs are  papers' reviews and nonreviews. you can extract features from them.

"""
def ff_<your_feature_name>(review_df: 'pd.DataFrame', other_replies_df: 'pd.DataFrame'):
    # write your feature calculation here

    return 1234  # this will be the value in the feature_df
"""

# ----------------------------------

# some utility functions


def _extract_numeric_prefix(maybe_string: object) -> int:
    """
    Extracts the leading integer from a string.
    If no integer is found, returns NaN.
    """
    string = str(maybe_string)
    match = re.match(r'^(\d+)', string.strip())
    return int(match.group(1)) if match else np.nan

# ----------------------------------


class FeatureFunctions:
    @staticmethod
    def ff_title_length(paper_df: 'pd.DataFrame', review_df: 'pd.DataFrame', other_replies_df: 'pd.DataFrame'):
        return paper_df['content'].apply(lambda x: len(x.get('title', '')))

    @staticmethod
    def ff_abstract_length(paper_df: 'pd.DataFrame', review_df: 'pd.DataFrame', other_replies_df: 'pd.DataFrame'):
        return paper_df['content'].apply(lambda x: len(x.get('abstract', '')))

    @staticmethod
    def ff_tldr_length(paper_df: 'pd.DataFrame', review_df: 'pd.DataFrame', other_replies_df: 'pd.DataFrame'):
        return paper_df['content'].apply(lambda x: len(x.get('TL;DR', '')))

    @staticmethod
    def ff_author_count(paper_df: 'pd.DataFrame', review_df: 'pd.DataFrame', other_replies_df: 'pd.DataFrame'):
        return paper_df['content'].apply(lambda x: len(x.get('authors', [])))

    @staticmethod
    def ff_keyword_count(paper_df: 'pd.DataFrame', review_df: 'pd.DataFrame', other_replies_df: 'pd.DataFrame'):
        return paper_df['content'].apply(lambda x: len(x.get('keywords', [])))

    @staticmethod
    def ff_is_accepted(paper_df: 'pd.DataFrame', review_df: 'pd.DataFrame', other_replies_df: 'pd.DataFrame'):
        decision_note = other_replies_df[other_replies_df['invitation'].apply(lambda x: '/Decision' in x)]
        if len(decision_note) == 1:
            decision = decision_note['content'].iloc[0]['decision']
            return 'Accept' in decision
        else:
            # display(group_df)
            # display(decision_note)
            assert False

    @staticmethod
    def ff_metareview_length(paper_df: 'pd.DataFrame', review_df: 'pd.DataFrame', other_replies_df: 'pd.DataFrame'):
        decision_note = other_replies_df[other_replies_df['invitation'].apply(lambda x: '/Meta' in x)]
        if len(decision_note) == 1:
            metareview = decision_note['content'].iloc[0]['metareview']
            return len(metareview)
        # paper might not have a metareview
        return None

# ----------------------------------


# TODO: populate the namespace with such functions.
# the cartesian product of
"""
        fields = ['confidence', 'correctness', 'technical_novelty_and_significance',
                  'empirical_novelty_and_significance', 'recommendation']
        agg_functions = [np.nanmean, np.nanstd,
                         np.nanmin, np.nanmax, np.nanmedian, np.nansum, np.nanvar, np.nanprod
                         ]

"""


def ff_confidence_nanmean(paper_df: 'pd.DataFrame', review_df: 'pd.DataFrame', other_replies_df: 'pd.DataFrame'):
    fieldname = 'confidence'
    agg_func = np.nanmean

    # generic func below

    field_values = review_df['content'].apply(lambda x: _extract_numeric_prefix(
        x.get(fieldname)) if isinstance(x, dict) else None)

    if field_values.isnull().all():
        return np.nan  # if cannot extract numeric prefix from any value, return NaN
    return agg_func(field_values)
