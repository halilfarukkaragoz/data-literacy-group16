

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from const import API_STORE
from scrapers.scraper import get_paper_and_review_df, query_openreview

if TYPE_CHECKING:
    import openreview.api


def get_invitation_path(invitation: str):

    return API_STORE / Path(invitation) / 'data.pkl'


def _load_df_from_cache(invitation: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    picklefile = get_invitation_path(invitation)
    with open(picklefile, 'rb') as f:
        data = pickle.load(f)

        paper_df = pd.DataFrame(data['paper_df'])
        review_df = pd.DataFrame(data['review_df'])
        other_replies_df = pd.DataFrame(data['other_replies_df'])

    return paper_df, review_df, other_replies_df


def _save_papers_to_cache(invitation: str, papers: list['openreview.api.Note']):
    # get dataframes
    paper_df, review_df, other_replies_df = get_paper_and_review_df(papers)

    # set types
    for df in [paper_df, review_df, other_replies_df]:
        for col in df.columns:
            if "date" in col.lower():
                df[col] = pd.to_datetime(df[col], unit='ms')

    # save to cache
    picklefile = get_invitation_path(invitation)
    picklefile.parent.mkdir(parents=True, exist_ok=True)
    with open(picklefile, 'wb') as f:
        data = {
            'paper_df': paper_df.to_dict(),
            'review_df': review_df.to_dict(),
            'other_replies_df': other_replies_df.to_dict()
        }
        pickle.dump(data, f)


def get_dataframes(conference_invitations: list[str]):
    for invitation in conference_invitations:
        if not get_invitation_path(invitation).exists():
            papers = list(query_openreview([invitation]))
            _save_papers_to_cache(invitation, papers)
        yield _load_df_from_cache(invitation)


def get_dataframes_concatd(conference_invitations: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dfs = list(get_dataframes(conference_invitations))
    grouped_dfs = zip(*dfs)
    master_dfs = [pd.concat(group, ignore_index=True) for group in grouped_dfs]
    return tuple(master_dfs)
