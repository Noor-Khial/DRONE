# CA-CFAR utilities with gain-aware peak injection for ablations
# [R8-54] processing gain G_proc affects peak strength

import numpy as np

def ca_cfar(signal, guard_cells, reference_cells, pfa):
    N = len(signal)
    threshold = np.zeros(N)
    for i in range(N):
        start = max(0, i - reference_cells - guard_cells)
        end = min(N, i + reference_cells + guard_cells + 1)

        left = signal[start:max(0, i - guard_cells)]
        right = signal[min(N, i + guard_cells + 1):end]
        reference = np.concatenate((left, right))

        if len(reference) > 0:
            average = np.mean(reference)
            threshold[i] = average * (-np.log(pfa) * (2 / max(reference_cells, 1)))
    return threshold

def get_cacfar_detection(
    distance,
    N=100,
    target_location=50,
    guard_cells=2,
    reference_cells=10,
    pfa=1e-4,
    G_proc=1.0     # [R8-54]
):
    """Returns a CFAR-based amplification factor (~SNR-like)."""
    # Avoid degenerate distances
    max_distance = 10.0
    d = 0.1 + ((distance - 0.1) * 1 / max_distance)

    # Rayleigh clutter
    signal = np.random.rayleigh(scale=1, size=N)

    # Peak amplitude scales with processing gain and distance
    peak_strength_factor = max(G_proc, 0.0) / (d ** 4)  # [R8-54]
    base_peak_amplitude = 0.1
    signal[int(np.clip(target_location, 0, N-1))] = base_peak_amplitude * peak_strength_factor

    # CFAR thresholding
    threshold = ca_cfar(signal, guard_cells, reference_cells, pfa)
    peak_index = int(np.argmax(signal))

    # Ratio > 1 => detection likely
    x = float(signal[peak_index] / max(threshold[peak_index], 1e-9))
    return x
