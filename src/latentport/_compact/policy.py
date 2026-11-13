"""The frozen E129 span policy, independent of the inference backend."""
import math

MIN_CONTEXT = 65536


def budget(physical, ratio):
    if type(ratio) is not int or ratio not in (4, 8):
        raise ValueError('ratio must be 4 or 8')
    if type(physical) is not int or physical < 0:
        raise ValueError('Invalid physical occupancy')
    return min(physical, max(1024, 16 * (physical // (16 * ratio))))


def choose(scores, target):
    """Match E129 including mean-score ordering and high-index tie breaks."""
    raw = [float(value) for value in scores]
    n = len(raw)
    if not 0 <= target <= n or (target < n and target < 1024):
        raise ValueError('Invalid retained budget')
    if not all(math.isfinite(value) for value in raw):
        raise ValueError('Nonfinite selector output')
    if n <= target:
        return list(range(n))
    runs = []
    for i in range(512, n - 512):
        if raw[i] <= 0:
            continue
        if runs and i - runs[-1][1] <= 4:
            runs[-1][1] = i + 1
        else:
            runs.append([i, i + 1])
    merged = []
    for a, b in runs:
        a, b = max(512, a - 2), min(n - 512, b + 2)
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    merged.sort(key=lambda ab: (sum(raw[ab[0]:ab[1]]) / (ab[1] - ab[0]), ab[0]), reverse=True)
    kept = set(range(512)) | set(range(n - 512, n))
    for a, b in merged:
        if len(kept) + b - a <= target:
            kept.update(range(a, b))
    for i in sorted(range(512, n - 512), key=lambda j: (raw[j], j), reverse=True):
        if len(kept) == target:
            break
        kept.add(i)
    return sorted(kept)
