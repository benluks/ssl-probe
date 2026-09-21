from scipy.stats import pearsonr

from ..opensmile import init_opensmile


def compute_pearson_from_smile(orig_smile, resynth_smile):
    # Resynthesis may differ slightly in duration
    n_frames = min(len(orig_smile), len(resynth_smile))

    orig = orig_smile.iloc[:n_frames]
    resynth = resynth_smile.iloc[:n_frames]

    r, p = pearsonr(
        orig.to_numpy(),
        resynth.to_numpy(),
    )
    return r, p


def compute_pearson_from_file(orig_wav, resynth_wav, sr=16000):
    smile = init_opensmile()
    orig_smile = smile.process_file(orig_wav)
    resynth_smile = smile.process_file(resynth_wav)

    return compute_pearson_from_smile(orig_smile, resynth_smile)


# def compute_pearson_from_signal():
