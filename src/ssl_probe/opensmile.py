import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import opensmile

OPENSMILE_LLD_FRAME_HZ = 100.0


def validate_jitter_min_periods(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError("jitter_min_periods must be an integer >= 2.")
    return value


def init_opensmile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    *,
    jitter_min_periods: int = 2,
):
    validate_jitter_min_periods(jitter_min_periods)
    if jitter_min_periods == 2:
        return opensmile.Smile(feature_set=feature_set, feature_level=feature_level)
    if feature_set != opensmile.FeatureSet.eGeMAPSv02:
        raise ValueError("Custom jitter_min_periods requires eGeMAPSv02.")

    # Copy the installed configuration tree to preserve relative includes without
    # modifying package files. Keep it alive for subsequent process_signal calls.
    temporary = TemporaryDirectory(prefix="ssl-probe-opensmile-")
    try:
        config_root = Path(temporary.name) / "config"
        shutil.copytree(Path(opensmile.__file__).parent / "core/config", config_root)
        core = config_root / "gemaps/v01b/GeMAPSv01b_core.lld.conf.inc"
        text, count = re.subn(
            r"(?m)^minNumPeriods\s*=\s*2\s*$",
            f"minNumPeriods = {jitter_min_periods}",
            core.read_text(),
        )
        if count != 1:
            raise RuntimeError("Installed eGeMAPS jitter configuration has changed.")
        core.write_text(text)
        smile = opensmile.Smile(
            feature_set=str(config_root / "egemaps/v02/eGeMAPSv02.conf"),
            feature_level=feature_level,
        )
        smile._jitter_config_directory = temporary
        return smile
    except Exception:
        temporary.cleanup()
        raise
