
from datastore.util import get_dataframes_concatd
from feature_calculation.paper_features import FeatureExtractor


def example():
    conference_invitations = [
        'NeurIPS.cc/2022/Conference/-/Blind_Submission',
        'ICLR.cc/2023/Conference/-/Blind_Submission',
    ]

    paper_df, review_df, other_replies_df = get_dataframes_concatd(conference_invitations)

    feature_extractor = FeatureExtractor(paper_df, review_df, other_replies_df)
    feature_extractor.extract_features()

    print(feature_extractor.feature_df)


def main():
    example()


if __name__ == "__main__":
    main()
