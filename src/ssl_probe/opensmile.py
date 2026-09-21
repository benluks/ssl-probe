import opensmile


def init_opensmile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
):
    return opensmile.Smile(
        feature_set=feature_set,
        feature_level=feature_level,
    )
