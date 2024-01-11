from mne.decoding import TimeFrequency

from constants import FREQ_BANDS


def time_freq_filter_init(sfreq: float = 1.0) -> TimeFrequency:
    """Initialize the time frequency filter.
    Uses the frequency bands defined in `constants.py`.

    Parameters
    ----------
    sfreq
        The sample frequency of the data. By default, 1.0.

    Returns
    -------
    The time frequency filter.
    """
    return TimeFrequency(freqs=FREQ_BANDS, sfreq=sfreq)
