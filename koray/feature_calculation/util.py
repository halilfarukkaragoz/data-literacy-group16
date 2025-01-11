import pickle
from pathlib import Path

import pandas as pd
from datastore.util import get_dataframes_concatd
from feature_calculation.paper_features import FeatureExtractor

from koray.const import FEAT_STORE


def get_feature_path(invitation: str):
    return FEAT_STORE / Path(invitation) / 'features.pkl'


def get_features(conference_invitations):
    for invitation in conference_invitations:
        invitation_path = get_feature_path(invitation)

        if not invitation_path.exists():
            # calculate features
            paper_df, review_df, other_replies_df = get_dataframes_concatd([invitation])
            feature_extractor = FeatureExtractor(paper_df, review_df, other_replies_df)
            feature_extractor.extract_features()

            # save to cache
            invitation_path.parent.mkdir(parents=True, exist_ok=True)
            with open(invitation_path, 'wb') as f:
                pickle.dump(feature_extractor.feature_df, f)

        # load from path
        with open(invitation_path, 'rb') as f:
            feature_df = pickle.load(f)

        feature_df = pd.DataFrame(feature_df)
        yield feature_df


def get_features_concatd(conference_invitations):
    feature_dfs = list(get_features(conference_invitations))
    return pd.concat(feature_dfs)
