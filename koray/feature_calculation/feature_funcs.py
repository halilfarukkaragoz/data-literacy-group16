import re
from collections import Counter

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

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

    # @staticmethod
    # def ff_metareview_length(other_replies_df: 'pd.DataFrame', **kwargs):
    #     __overwrite_dtype__ = np.float64
    #     decision_note = other_replies_df[other_replies_df['invitation'].apply(lambda x: '/Meta' in x)]
    #     if len(decision_note) == 1:
    #         metareview = decision_note['content'].iloc[0]['metareview']
    #         return len(metareview)
    #     # paper might not have a metareview
    #     return np.nan

    @staticmethod
    def ff_conference_name(paper_df: 'pd.DataFrame', **kwargs):
        invitation = paper_df['invitation'].iloc[0]
        return invitation.split("/")[0]

    @staticmethod
    def ff_conference_year(paper_df: 'pd.DataFrame', **kwargs):
        invitation = paper_df['invitation'].iloc[0]
        return int(invitation.split("/")[1])

    @staticmethod
    def ff_web_url(paper_df: 'pd.DataFrame', **kwargs):
        return f"https://openreview.net/forum?id={paper_df.iloc[0]['id']}"

    @staticmethod
    def ff_review_count(paper_df: 'pd.DataFrame', review_df: 'pd.DataFrame', **kwargs):
        __overwrite_dtype__ = np.int32
        return len(review_df)

    @staticmethod
    def ff_individual_confidence_scores(review_df: 'pd.DataFrame', **kwargs):
        confidence_scores = list(review_df['content'].apply(lambda x: _extract_numeric_prefix(x.get('confidence'))))
        return confidence_scores

    @staticmethod
    def ff_individual_recommendation_scores(review_df: 'pd.DataFrame', **kwargs):
        recommendation_scores = list(review_df['content'].apply(
            lambda x: _extract_numeric_prefix(x.get('recommendation'))))
        return recommendation_scores

    @staticmethod
    def ff_paper_area(paper_df: 'pd.DataFrame', **kwargs):
        area = paper_df.iloc[0]['content'].get('Please_choose_the_closest_area_that_your_submission_falls_into', '')
        return area

    @staticmethod
    def ff_keywords(paper_df: 'pd.DataFrame', **kwargs):
        keywords = paper_df.iloc[0]['content'].get('keywords', '')
        return keywords

    @staticmethod
    def ff_is_high_discrepancy(review_df: 'pd.DataFrame', **kwargs):
        recommendation_scores = list(review_df['content'].apply(
            lambda x: _extract_numeric_prefix(x.get('recommendation'))))

        discrepancy = max(recommendation_scores) - min(recommendation_scores)
        return discrepancy >= 4

    @staticmethod
    def ff_paper_topic_salience(paper_df: 'pd.DataFrame', master_paper_df: 'pd.DataFrame', **kwargs):
        def _normalize_keywords(df:  'pd.DataFrame'):
            # paper keywords
            keywords = df['content'].apply(lambda x: x.get('keywords', ''))
            keywords = list(keywords)
            keywords = [kw.lower().split(" ") for item in keywords for kw in item]
            keywords = [item.strip() for sublist in keywords for item in sublist]
            keywords = [kw for kw in keywords if kw not in ['', 'learning', 'deep', 'neural',
                                                            'networks', 'network', 'model', 'models', 'data', 'machine', 'generation', 'vision']]
            return keywords

        paper_kw = _normalize_keywords(paper_df)
        all_paper_kw = _normalize_keywords(master_paper_df)  # not optimal but works
        freq_counter = Counter(all_paper_kw)

        return sum(freq_counter[k] for k in paper_kw)

    @staticmethod
    def ff_time_to_deadline_list_agg(paper_df: 'pd.DataFrame', review_df: 'pd.DataFrame', **kwargs):
        deadline_delta = pd.to_datetime('2022-11-05 01:00:00') - review_df['cdate']
        return list(deadline_delta)

    @staticmethod
    def ff_time_to_deadline_mean_seconds(paper_df: 'pd.DataFrame', review_df: 'pd.DataFrame', **kwargs):
        deadline_delta = pd.to_datetime('2022-11-05 01:00:00') - review_df['cdate']
        return deadline_delta.mean().total_seconds()

    @staticmethod
    def ff_time_to_deadline_mean_days(paper_df: 'pd.DataFrame', review_df: 'pd.DataFrame', **kwargs):
        deadline_delta = pd.to_datetime('2022-11-05 01:00:00') - review_df['cdate']
        return deadline_delta.mean().days

    # @staticmethod
    # def ff_sentiment_analysis_scores(review_df: 'pd.DataFrame', **kwargs):
    #     reviews = list(review_df['content'])
    #     sentiment_scores = [analyzer.polarity_scores(str(review)) for review in reviews]

    #     return sentiment_scores

    # @staticmethod
    # def ff_sentiment_analysis_score_negative_mean(review_df: 'pd.DataFrame', **kwargs):
    #     # this is slow but works for now.
    #     sentiment_scores = FeatureFunctions.ff_sentiment_analysis_scores(review_df, **kwargs)
    #     negative_scores = [score['neg'] for score in sentiment_scores]
    #     return np.mean(negative_scores)

    # @staticmethod
    # def ff_sentiment_analysis_score_neutral_mean(review_df: 'pd.DataFrame', **kwargs):
    #     # this is slow but works for now.
    #     sentiment_scores = FeatureFunctions.ff_sentiment_analysis_scores(review_df, **kwargs)
    #     neutral_scores = [score['neu'] for score in sentiment_scores]
    #     return np.mean(neutral_scores)

    # @staticmethod
    # def ff_sentiment_analysis_score_positive_mean(review_df: 'pd.DataFrame', **kwargs):
    #     # this is slow but works for now.
    #     sentiment_scores = FeatureFunctions.ff_sentiment_analysis_scores(review_df, **kwargs)
    #     positive_scores = [score['pos'] for score in sentiment_scores]
    #     return np.mean(positive_scores)

    # @staticmethod
    # def ff_sentiment_analysis_score_negative_var(review_df: 'pd.DataFrame', **kwargs):
    #     # this is slow but works for now.
    #     sentiment_scores = FeatureFunctions.ff_sentiment_analysis_scores(review_df, **kwargs)
    #     negative_scores = [score['neg'] for score in sentiment_scores]
    #     return np.var(negative_scores)

    # @staticmethod
    # def ff_sentiment_analysis_score_neutral_var(review_df: 'pd.DataFrame', **kwargs):
    #     # this is slow but works for now.
    #     sentiment_scores = FeatureFunctions.ff_sentiment_analysis_scores(review_df, **kwargs)
    #     neutral_scores = [score['neu'] for score in sentiment_scores]
    #     return np.var(neutral_scores)

    # @staticmethod
    # def ff_sentiment_analysis_score_positive_var(review_df: 'pd.DataFrame', **kwargs):
    #     # this is slow but works for now.
    #     sentiment_scores = FeatureFunctions.ff_sentiment_analysis_scores(review_df, **kwargs)
    #     positive_scores = [score['pos'] for score in sentiment_scores]
    #     return np.var(positive_scores)


# ----------------------------------
# extending the FeatureFunctions class with numeric functions for reviewer fields


def reviewer_numeric_agg(review_df: 'pd.DataFrame', fieldname: str, agg_func: callable):
    field_values = review_df['content'].apply(lambda x: _extract_numeric_prefix(
        x.get(fieldname)) if isinstance(x, dict) else None)

    if field_values.isnull().all():
        return np.nan  # if cannot extract numeric prefix from any value, return NaN
    return agg_func(field_values)


def yusuf_max_diff(series: 'pd.Series'):
    return np.max(series) - np.min(series)


def list_agg(series: 'pd.Series'):
    return series.tolist()


def unique_value_frequency(series: 'pd.Series'):
    return series.value_counts()


def num_unique_values(series: 'pd.Series'):
    return len(series.unique())


def var_to_mean_ratio(series: 'pd.Series'):
    return series.var() / series.mean()


def mean_absolute_deviation(series: 'pd.Series'):
    return (series - series.mean()).abs().mean()


def coefficient_of_variation(series: 'pd.Series'):
    return series.std() / series.mean()


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
    list_agg,
    unique_value_frequency,
    num_unique_values,
    var_to_mean_ratio,
    mean_absolute_deviation,
    coefficient_of_variation,
]


for field in fields:
    for agg_func in agg_functions:
        func_name = f'ff_reviewer_{field}_{agg_func.__name__}'

        def func(review_df, field=field, agg_func=agg_func, **kwargs):
            return reviewer_numeric_agg(review_df, field, agg_func)

        setattr(FeatureFunctions, func_name, staticmethod(func))
