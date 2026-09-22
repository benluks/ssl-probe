import opensmile

OPENSMILE_LLD_FRAME_HZ = 100.0


def init_opensmile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
):
    return opensmile.Smile(
        feature_set=feature_set,
        feature_level=feature_level,
    )
