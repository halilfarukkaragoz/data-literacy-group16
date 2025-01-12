
from koray.datastore.util import get_dataframes_concatd
from koray.feature_calculation.paper_features import FeatureExtractor
from koray.feature_calculation.util import get_features


def main():
    conference_invitations = [
        'NeurIPS.cc/2022/Conference/-/Blind_Submission',
        'ICLR.cc/2023/Conference/-/Blind_Submission',
    ]
    

    feature_df = list(get_features(conference_invitations, disable_cache=True))
    print(feature_df)
    

if __name__ == "__main__":
    main()
