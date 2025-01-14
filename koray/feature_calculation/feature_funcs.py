
import re

import numpy as np
import pandas as pd

# ff_ is short for feature_function and the prefix is necessary for the feature_extractor.py to find the function
# these functions are used to calculate columns in feature df
# the inputs are  papers' reviews and nonreviews. you can extract features from them.


def ff_your_feature_name(
    paper_df: 'pd.DataFrame',  # i'th paper's data
    review_df: 'pd.DataFrame',  # i'th paper's reviews
    other_replies_df: 'pd.DataFrame',  # i'th paper's nonreviews
    master_paper_df: 'pd.DataFrame',  # all papers' data in the conference
    master_review_df: 'pd.DataFrame',  # all papers' reviews in the conference
    master_other_replies_df: 'pd.DataFrame',  # all papers' nonreviews in the conference
):
    # Note: in the current implementation master_ dataframmes are for 1 conference.
    # However, this behavior depends on how FeatureExtractor is used.

    # these overwrite the dtype of the feature_df column
    __overwrite_dtype__ = np.float64
    __overwrite_dtype__ = np.int64

    # write your feature calculation here
    ...

    return 1234  # this will be the value in the feature_df

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


def get_field_length(content: dict, field: str, default: int = 0) -> int:
    """
    Utility function to calculate the length of a specified field in a dictionary.

    Args:
        content (dict): The dictionary containing the fields.
        field (str): The field name to calculate the length for.
        default (int): The default length to return if the field is missing or empty.

    Returns:
        int: The length of the specified field, or the default value.
    """
    return len(content.get(field, '')) if content.get(field) else default


# ----------------------------------

class FeatureFunctions:
    @staticmethod
    def ff_title_length(paper_df: 'pd.DataFrame', **kwargs):
        __overwrite_dtype__ = np.float64
        return paper_df['content'].apply(lambda x: get_field_length(x, 'title'))

    @staticmethod
    def ff_abstract_length(paper_df: 'pd.DataFrame', **kwargs):
        __overwrite_dtype__ = np.float64
        return paper_df['content'].apply(lambda x: get_field_length(x, 'abstract'))

    @staticmethod
    def ff_tldr_length(paper_df: 'pd.DataFrame', **kwargs):
        __overwrite_dtype__ = np.float64
        return paper_df['content'].apply(lambda x: get_field_length(x, 'TL;DR'))

    @staticmethod
    def ff_author_count(paper_df: 'pd.DataFrame', **kwargs):
        __overwrite_dtype__ = np.float64
        return paper_df['content'].apply(lambda x: get_field_length(x, 'authors'))

    @staticmethod
    def ff_keyword_count(paper_df: 'pd.DataFrame', **kwargs):
        __overwrite_dtype__ = np.float64
        return paper_df['content'].apply(lambda x: get_field_length(x, 'keywords'))

    @staticmethod
    def ff_is_accepted(other_replies_df: 'pd.DataFrame', **kwargs):
        decision_note = other_replies_df[other_replies_df['invitation'].apply(lambda x: '/Decision' in x)]
        if len(decision_note) == 1:
            decision = decision_note['content'].iloc[0]['decision']
            return 'Accept' in decision
        else:
            # display(group_df)
            # display(decision_note)
            assert False

    @staticmethod
    def ff_metareview_length(other_replies_df: 'pd.DataFrame', **kwargs):
        __overwrite_dtype__ = np.float64
        decision_note = other_replies_df[other_replies_df['invitation'].apply(lambda x: '/Meta' in x)]
        if len(decision_note) == 1:
            metareview = decision_note['content'].iloc[0]['metareview']
            return len(metareview)
        # paper might not have a metareview
        return np.nan

    @staticmethod
    def ff_conference_name(paper_df: 'pd.DataFrame', **kwargs):
        invitation = paper_df['invitation'].iloc[0]
        return invitation.split("/")[0]

    @staticmethod
    def ff_conference_year(paper_df: 'pd.DataFrame', **kwargs):
        invitation = paper_df['invitation'].iloc[0]
        return int(invitation.split("/")[1])

    @staticmethod
    def ff_review_count(paper_df: 'pd.DataFrame', review_df: 'pd.DataFrame', **kwargs):
        __overwrite_dtype__ = np.int64
        return len(review_df)

# ----------------------------------
# extending the FeatureFunctions class with numeric functions for reviewer fields


def reviewer_numeric_agg(review_df: 'pd.DataFrame', fieldname: str, agg_func: callable):
    field_values = review_df['content'].apply(lambda x: _extract_numeric_prefix(
        x.get(fieldname)) if isinstance(x, dict) else None)

    if field_values.isnull().all():
        return np.nan  # if cannot extract numeric prefix from any value, return NaN
    return agg_func(field_values)


def yusuf_max_diff(arr):
    return np.max(arr) - np.min(arr)


fields = ['confidence', 'correctness', 'technical_novelty_and_significance',
          'empirical_novelty_and_significance', 'recommendation']
agg_functions = [
    np.nanmean,
    np.nanvar,
    np.nanstd,
    #
    np.nanmin,
    np.nanmax,
    np.nanmedian,
    #
    yusuf_max_diff,
]


for field in fields:
    for agg_func in agg_functions:
        func_name = f'ff_reviewer_{field}_{agg_func.__name__}'

        def func(review_df, field=field, agg_func=agg_func, **kwargs):
            return reviewer_numeric_agg(review_df, field, agg_func)

        setattr(FeatureFunctions, func_name, staticmethod(func))
