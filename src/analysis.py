# statistics for the 12 monthly maps
# all changes in class share are in percentage points (pp),
# e.g. built-up 41.2 % -> 43.0 % is +1.8 pp

from pathlib import Path

import numpy as np
import pandas as pd

from .classes import CLASS_IDS, CLASS_NAMES

NAMES = [CLASS_NAMES[c] for c in CLASS_IDS]


def make_stack(maps, months):
    # put the monthly maps into one array with shape (months, H, W)
    if len(months) < 2:
        raise ValueError("Need at least 2 months to calculate changes.")
    for m in months:
        if m not in maps:
            raise ValueError("Missing monthly map: " + m)

    stack = np.stack([np.asarray(maps[m]) for m in months])

    if stack.ndim != 3 or stack.shape[1] == 0 or stack.shape[2] == 0:
        raise ValueError("Monthly maps must be non-empty 2-D arrays of equal shape.")

    for value in np.unique(stack):
        if value not in CLASS_IDS:
            raise ValueError("Unexpected class id: " + str(value))

    return stack


def class_percentages(stack, months):
    # % of the area covered by each class, one row per month
    total_pixels = stack[0].size
    rows = []
    for i in range(len(months)):
        row = []
        for c in CLASS_IDS:
            count = np.sum(stack[i] == c)
            row.append(count / total_pixels * 100)
        rows.append(row)

    df = pd.DataFrame(rows, index=months, columns=NAMES)
    df.index.name = "month"
    return df


def month_to_month_change(percentages, stack=None):
    # change compared to the previous month, in pp
    months = list(percentages.index)

    rows = []
    labels = []
    for i in range(1, len(months)):
        change = percentages.iloc[i] - percentages.iloc[i - 1]
        rows.append(change.values)
        labels.append(months[i - 1] + " → " + months[i])

    df = pd.DataFrame(rows, index=labels, columns=percentages.columns)
    df.index.name = "period"

    # gross change: how many pixels got a different class than last month
    if stack is not None:
        changed = []
        for i in range(1, len(stack)):
            changed.append(np.mean(stack[i] != stack[i - 1]) * 100)
        df["Pixels changed (%)"] = changed

    return df


def total_change(percentages):
    # first month vs last month
    first = percentages.index[0]
    last = percentages.index[-1]

    df = pd.DataFrame({
        first + " (%)": percentages.iloc[0],
        last + " (%)": percentages.iloc[-1],
        "Change (pp)": percentages.iloc[-1] - percentages.iloc[0],
    })
    df.index.name = "class"
    return df


def transition_matrix(first_map, last_map):
    # rows = class in the first map, columns = class in the last map, values = pixel counts
    n = len(CLASS_IDS)
    matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            matrix[i, j] = np.sum((first_map == CLASS_IDS[i]) & (last_map == CLASS_IDS[j]))

    df = pd.DataFrame(matrix, index=NAMES, columns=NAMES)
    df.index.name = "from \\ to"
    return df


def temporal_stability(stack):
    # how often each pixel switched class, and how many different classes it had
    switches = np.zeros(stack.shape[1:], dtype=int)
    for i in range(1, len(stack)):
        switches += (stack[i] != stack[i - 1])

    distinct = np.zeros(stack.shape[1:], dtype=int)
    for c in CLASS_IDS:
        distinct += np.any(stack == c, axis=0)

    return switches, distinct


def analyze(maps, months, pixel_size_m=10, output_dir=None):
    stack = make_stack(maps, months)
    total_pixels = int(stack[0].size)
    pixel_area_ha = pixel_size_m * pixel_size_m / 10000

    percentages = class_percentages(stack, months)
    m2m = month_to_month_change(percentages, stack)
    total = total_change(percentages)
    transitions = transition_matrix(stack[0], stack[-1])
    switches, distinct = temporal_stability(stack)

    changed_pixels = int(np.sum(stack[0] != stack[-1]))
    changes = total["Change (pp)"]

    summary = {
        "months": months,
        "total_pixels": total_pixels,
        "area_km2": total_pixels * pixel_area_ha / 100,
        "pixel_size_m": pixel_size_m,
        "changed_pixels_first_last": changed_pixels,
        "changed_pct_first_last": changed_pixels / total_pixels * 100,
        "changed_area_ha_first_last": changed_pixels * pixel_area_ha,
        "stable_all_months_pct": float(np.mean(switches == 0) * 100),
        "mean_switches_per_pixel": float(np.mean(switches)),
        "largest_net_gain": (changes.idxmax(), float(changes.max())),
        "largest_net_loss": (changes.idxmin(), float(changes.min())),
    }

    results = {
        "summary": summary,
        "percentages": percentages,
        "month_to_month_pp": m2m,
        "total_change_pp": total,
        "transition_first_last": transitions,
        "switches": switches,
        "distinct_classes": distinct,
        "stack": stack,
    }

    if output_dir is not None:
        save_tables(results, output_dir)

    return results


def save_tables(results, output_dir):
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)

    results["percentages"].round(3).to_csv(folder / "class_percentages_by_month.csv")
    results["month_to_month_pp"].round(3).to_csv(folder / "month_to_month_change_pp.csv")
    results["total_change_pp"].round(3).to_csv(folder / "total_change_first_to_last_pp.csv")
    results["transition_first_last"].to_csv(folder / "transition_matrix_first_to_last_pixels.csv")

    return folder
