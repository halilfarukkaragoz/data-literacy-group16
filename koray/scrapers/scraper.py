import itertools
from typing import Iterable, List

import openreview
import pandas as pd

# pip install openreview-py

def query_openreview(conference_invitations: List[str]) -> Iterable['openreview.api.Note']:
    client = openreview.Client(baseurl='https://api.openreview.net')
    for invitation in conference_invitations:
        print(f'Fetching papers for invitation {invitation}')
        paper_iterable = client.get_all_notes(
            invitation=invitation,
            details='directReplies'
        )

        yield from paper_iterable


def get_paper_and_review_df(papers: List['openreview.api.Note']) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # note that to_json() method discards directReplies
    paper_df = pd.DataFrame([paper.to_json() for paper in papers])

    # we can join this with the paper_df on paper_df.id == review_df.forum
    flattened_reviews = itertools.chain.from_iterable(paper.details['directReplies'] for paper in papers)
    reply_df = pd.DataFrame(list(flattened_reviews))

    # when a reply is a review, it has 'Reviewer_' in ['signatures'], which is a list of strings
    review_df = reply_df[reply_df.signatures.apply(lambda x: any('Reviewer_' in s for s in x))]

    # other replies are replies that are not reviews, e.g. decision, meta-review
    other_replies_df = reply_df[~reply_df.index.isin(review_df.index)]

    return paper_df, review_df, other_replies_df
