from scipy.stats import pearsonr

from ..opensmile_factory import init_opensmile


def compute_pearson(orig_wav, resynth_wav, sr=16000):

    smile = init_opensmile
    orig_smile = smile.process_file(orig_wav)
    resynth_smile = smile.process_file(resynth_wav)

    # Resynthesis may differ slightly in duration
    n_frames = min(len(orig_smile), len(resynth_smile))

    orig = orig_smile.iloc[:n_frames]
    resynth = resynth_smile.iloc[:n_frames]

    r, p = pearsonr(
        orig.to_numpy(),
        resynth.to_numpy(),
    )
    return r, p
