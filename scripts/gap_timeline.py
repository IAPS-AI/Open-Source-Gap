"""Gap-by-date timelines: the day-by-day "months behind" series, the score
gap series, and the context statistics the Gap Over Time chart surfaces.

The site's headline "Average Gap" is the mean of a day-by-day series computed
in ``scripts/update_data.py::calculate_gap_metrics`` (mirroring Epoch AI's
open-vs-SOTA methodology): on each day, the gap is the number of months that
have elapsed since the *most recent* leader-side frontier (running-max) model
that the day's best laggard-side model has plausibly caught up to. Only the
mean and the 5th/95th percentiles of that series were published. This module
publishes the series itself, in a compact form, together with the "context"
statistics the chart surfaces, and two companions:

* a **lower bound** per segment -- months since the *earliest* leader model
  the laggard's best has NOT caught up to (0 when it has matched the newest
  one). The laggard demonstrably trailed that model's level from its release
  onward, so the true backward-looking gap lies between the lower bound and
  the matched lead;
* an optional **expected lead** series (``expected_fn`` injected by the
  pipeline: the survival-analysis current-gap estimate recomputed as of a past
  date, sampled every few days). It is a model-based forecast that also counts
  unmatched leader models by their age, and is labelled as such;
* a **score gap** block (``build_score_gap_timeline``): best leader score
  minus best laggard score on each day, in the benchmark's own units. Unlike
  the time series it rises the day a leader releases an unmatched model.

Representation. Between frontier releases the reference model is fixed, so
the time gap grows linearly at exactly one month per month. The series is
therefore a list of ``segments``; within a segment the value on day ``t`` is
``(t - reference_date) / DAYS_PER_MONTH``. Segment ``end`` is exclusive
(it equals the next segment's ``start``) except for the last segment, whose
``end`` is ``as_of`` inclusive. Discontinuities (drops) are listed under
``events``. Expanding the segments to daily resolution reproduces the
``calculate_gap_metrics`` loop exactly; ``tests/test_gap_timeline.py`` pins
that invariant. The score gap block uses constant ``steps`` with the same
end-exclusive convention.

Inherent to the methodology, and worth labelling on the chart: the time gap
drops when the laggard releases a model that catches a newer leader model,
and it drops to zero on the day a leader releases a model that the laggard's
best had already plausibly caught up to. It can also step *up* -- rarely --
when a new laggard record with a higher point estimate fails the "plausibly
caught up" test against a leader model the previous record had passed (the
paired bootstrap and the CI test are significance tests, not monotone in the
point estimate). Such events carry ``reversal: true`` so the chart can mark
them.

The "caught up" predicate is injected (``caught_up(open_score, open_std,
sota_score, sota_std, open_name, sota_name) -> bool``) so this module has no
dependency on ``update_data`` (which imports it) and is trivially testable;
``update_data`` wires ``_open_caught_up`` with its threshold and, for ECI,
the paired bootstrap.

Input convention matches the rest of the pipeline: a per-group frontier
DataFrame with ``date``, a score column, an optional ``<score>_std`` column,
a model-name column, and a boolean laggard column (``Open`` by default; True
marks the LAGGARD group -- open-weight models, or Chinese models in the
China-vs-US framing).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

DAYS_PER_MONTH = 365.25 / 12  # matches scripts/update_data.py
_EPS = 1e-9

CaughtUp = Callable[[float, float, float, float, Optional[str], Optional[str]], bool]
ExpectedFn = Callable[[pd.Timestamp], Optional[float]]


def _iso(ts: Any) -> Optional[str]:
    if ts is None or pd.isna(ts):
        return None
    return pd.Timestamp(ts).isoformat()


def _months(later: Any, earlier: Any) -> float:
    return (pd.Timestamp(later) - pd.Timestamp(earlier)).days / DAYS_PER_MONTH


def _r1(v: Any) -> Optional[float]:
    return None if v is None else round(float(v), 1)


def _r2(v: Any) -> Optional[float]:
    return None if v is None or pd.isna(v) else round(float(v), 2)


def _running_max(rows: pd.DataFrame, score_col: str, std_col: str, model_col: str) -> list[dict]:
    """Records that set a new running-max score within a group, by date."""
    out: list[dict] = []
    run = -np.inf
    has_std = std_col in rows.columns
    for _, r in rows.iterrows():
        s = float(r[score_col])
        if s > run:
            run = s
            std = float(r[std_col]) if has_std and pd.notna(r.get(std_col)) else np.nan
            out.append({
                "date": pd.Timestamp(r["date"]),
                "score": s,
                "std": std,
                "name": r.get(model_col, r.get("model")),
            })
    return out


def _prepare(df: pd.DataFrame, score_col: str, model_col: str, laggard_col: str, as_of: Any):
    """Shared input handling. Returns (d, sota, lag_best, ws, as_of_ts) or
    None when the inputs cannot support a series (empty, one-sided, or
    ``as_of`` before the first day on which both groups exist)."""
    if df is None or df.empty or laggard_col not in df.columns:
        return None
    std_col = f"{score_col}_std"
    d = df.dropna(subset=["date", score_col]).copy()
    if d.empty:
        return None
    d["date"] = pd.to_datetime(d["date"])
    as_of_ts = pd.Timestamp(as_of).normalize()
    d = d[d["date"] <= as_of_ts].sort_values("date", kind="mergesort")
    is_lag = d[laggard_col].astype(bool)
    lag_rows = d[is_lag]
    led_rows = d[~is_lag]
    if lag_rows.empty or led_rows.empty:
        return None
    sota = _running_max(led_rows, score_col, std_col, model_col)
    lag_best = _running_max(lag_rows, score_col, std_col, model_col)
    # Window start mirrors calculate_gap_metrics: the first day both groups exist.
    ws = max(lag_rows["date"].min(), sota[0]["date"])
    if ws > as_of_ts:
        return None
    return d, sota, lag_best, ws, as_of_ts


def _event_dates(ws: pd.Timestamp, lag_best: list[dict], sota: list[dict]) -> list[pd.Timestamp]:
    """The state only changes on frontier release dates."""
    return sorted(
        {ws}
        | {r["date"] for r in lag_best if r["date"] >= ws}
        | {s["date"] for s in sota if s["date"] >= ws}
    )


def _latest_before(records: list[dict], day: pd.Timestamp) -> Optional[dict]:
    best = None
    for r in records:
        if r["date"] <= day:
            best = r
        else:
            break
    return best


def _state_at(day: pd.Timestamp, lag_best: list[dict], sota: list[dict],
              caught_up: CaughtUp) -> Optional[tuple[dict, dict, bool, Optional[dict]]]:
    """(best laggard, reference leader, matched?, next unmatched leader) on
    ``day``. The reference is the most recent leader record the best laggard
    has plausibly caught up to; the next unmatched leader is the earliest
    leader record released after the reference (None if the laggard has
    matched the newest one)."""
    best = _latest_before(lag_best, day)
    avail = [s for s in sota if s["date"] <= day]
    if best is None or not avail:
        return None
    for i in range(len(avail) - 1, -1, -1):
        s = avail[i]
        if caught_up(best["score"], best["std"], s["score"], s["std"],
                     best["name"], s["name"]):
            nxt = avail[i + 1] if i + 1 < len(avail) else None
            return best, s, True, nxt
    # Laggard has not caught even the earliest leader record: the lag is the
    # full history (same floor as calculate_gap_metrics). The earliest record
    # is itself the oldest unmatched one.
    return best, avail[0], False, avail[0]


def _lookback(g: np.ndarray, i0: int, pred: Callable[[np.ndarray], np.ndarray]) -> tuple[Optional[int], Optional[int]]:
    """Return (index of the most recent day before the current run where
    ``pred`` holds, index where the current run of ``pred`` days began).

    The "current run" is the contiguous block of days immediately before
    ``i0`` on which ``pred`` holds (empty -> None). We skip over it, then over
    the days on which ``pred`` fails, and report the next day on which it
    holds again (None if there is none).
    """
    ok = pred(g)
    j = i0 - 1
    run_since = None
    while j >= 0 and ok[j]:
        run_since = j
        j -= 1
    while j >= 0 and not ok[j]:
        j -= 1
    return (j if j >= 0 else None), run_since


def _daily_context(g: np.ndarray, days: pd.DatetimeIndex, as_of_ts: pd.Timestamp,
                   vk: str, ck: str, models_at: Callable[[int], dict]) -> dict:
    """Context statistics for any daily series ``g``. ``vk`` names the value
    key (``gap_months`` / ``gap_points``), ``ck`` the change key
    (``change_months`` / ``change_points``)."""
    n = len(g)
    i0 = n - 1
    cur = float(g[i0])

    def lookback_block(pred) -> dict:
        j, run = _lookback(g, i0, pred)
        return {
            "date": _iso(days[j]) if j is not None else None,
            vk: _r1(g[j]) if j is not None else None,
            "months_ago": _r1(_months(as_of_ts, days[j])) if j is not None else None,
            "run_since": _iso(days[run]) if run is not None else None,
        }

    prior = g[:i0]
    i_peak = int(np.argmax(g))
    i_trough = int(np.argmin(g))

    year_ago = None
    ya_date = as_of_ts - pd.DateOffset(years=1)
    if ya_date >= days[0]:
        k = int((ya_date - days[0]).days)
        year_ago = {"date": _iso(ya_date), vk: _r1(g[k]), ck: _r1(cur - g[k])}

    # A same-day-a-year-ago comparison flips sign with the anchor on a
    # sawtooth; the trailing-365-day mean against the 365 days before it is
    # the phase-robust version (None until a full year of data exists).
    trailing_12m = None
    if n >= 365:
        t_mean = float(np.mean(g[n - 365:]))
        p_mean = float(np.mean(g[n - 730:n - 365])) if n >= 730 else None
        trailing_12m = {
            f"mean_{vk}": _r1(t_mean),
            f"prior_mean_{vk}": _r1(p_mean) if p_mean is not None else None,
            ck: _r1(t_mean - p_mean) if p_mean is not None else None,
        }

    return {
        f"current_{vk}": _r1(cur),
        "last_at_least": lookback_block(lambda a: a >= cur - _EPS),
        "last_at_most": lookback_block(lambda a: a <= cur + _EPS),
        "is_record_high": bool(prior.size == 0 or cur >= prior.max() - _EPS),
        "is_record_low": bool(prior.size == 0 or cur <= prior.min() + _EPS),
        "peak": {vk: _r1(g[i_peak]), "date": _iso(days[i_peak]), **models_at(i_peak)},
        "trough": {vk: _r1(g[i_trough]), "date": _iso(days[i_trough]), **models_at(i_trough)},
        f"mean_{vk}": _r1(float(np.mean(g))),
        f"median_{vk}": _r1(float(np.median(g))),
        "percentile_of_current": int(round(100.0 * float(np.mean(g <= cur + _EPS)))),
        "year_ago": year_ago,
        "trailing_12m": trailing_12m,
    }


def _frontier_list(records: list[dict]) -> list[dict]:
    return [{"model": r["name"], "date": _iso(r["date"]), "score": _r2(r["score"])}
            for r in records]


def _sample_expected(expected_fn: ExpectedFn, ws: pd.Timestamp, as_of_ts: pd.Timestamp,
                     every_days: int) -> list[dict]:
    dates = list(pd.date_range(ws, as_of_ts, freq=f"{max(1, int(every_days))}D"))
    if not dates or dates[-1] != as_of_ts:
        dates.append(as_of_ts)
    out = []
    for t in dates:
        try:
            v = expected_fn(t)
        except Exception:  # a single bad sample must not kill the series
            v = None
        if v is None or not np.isfinite(float(v)):
            continue
        out.append({"date": _iso(t), "months": _r1(v)})
    return out


def build_gap_timeline(
    df: pd.DataFrame,
    *,
    score_col: str,
    model_col: str,
    caught_up: CaughtUp,
    as_of: Any,
    laggard_col: str = "Open",
    expected_fn: Optional[ExpectedFn] = None,
    expected_every_days: int = 7,
) -> Optional[dict]:
    """Build the gap-by-date timeline block (months behind). Returns None
    when the inputs cannot support a series."""
    prep = _prepare(df, score_col, model_col, laggard_col, as_of)
    if prep is None:
        return None
    d, sota, lag_best, ws, as_of_ts = prep

    # ---- segments: one per (laggard, reference, next unmatched leader) ----
    segments: list[dict] = []
    events: list[dict] = []
    for e in _event_dates(ws, lag_best, sota):
        st = _state_at(e, lag_best, sota, caught_up)
        if st is None:  # cannot happen for e >= ws, defensive
            continue
        best, ref, matched, nxt = st
        if segments and segments[-1]["_lag"] is best and segments[-1]["_ref"] is ref \
                and segments[-1]["_nxt"] is nxt:
            continue
        if segments:
            prev = segments[-1]
            prev["_end"] = e
            before = _months(e, prev["_ref"]["date"])
            after = _months(e, ref["date"])
            if abs(before - after) > _EPS:
                kind = "laggard_release" if prev["_lag"] is not best else "leader_release"
                events.append({
                    "date": _iso(e),
                    "kind": kind,
                    "model": best["name"] if kind == "laggard_release" else ref["name"],
                    "gap_before": _r1(before),
                    "gap_after": _r1(after),
                    # True when the reference moved to an EARLIER leader model
                    # (the gap stepped up): see the module docstring.
                    "reversal": bool(after > before + _EPS),
                    "laggard_model": best["name"],
                    "reference_model": ref["name"],
                })
        segments.append({"_start": e, "_end": None, "_lag": best, "_ref": ref,
                         "_matched": matched, "_nxt": nxt})
    if not segments:
        return None
    segments[-1]["_end"] = as_of_ts

    # ---- daily expansion (exactly the calculate_gap_metrics loop) ----
    days = pd.date_range(ws, as_of_ts)
    n = len(days)
    g = np.empty(n, dtype=float)
    lower = np.zeros(n, dtype=float)
    seg_idx = np.empty(n, dtype=int)
    for i, seg in enumerate(segments):
        last = i == len(segments) - 1
        lo = int((seg["_start"] - ws).days)
        hi = int((seg["_end"] - ws).days) + (1 if last else 0)
        ref_offset_days = (seg["_start"] - seg["_ref"]["date"]).days
        g[lo:hi] = (ref_offset_days + np.arange(hi - lo)) / DAYS_PER_MONTH
        if seg["_nxt"] is not None:
            nxt_offset_days = (seg["_start"] - seg["_nxt"]["date"]).days
            lower[lo:hi] = (nxt_offset_days + np.arange(hi - lo)) / DAYS_PER_MONTH
        seg_idx[lo:hi] = i

    def models_at(i: int) -> dict:
        s = segments[int(seg_idx[i])]
        return {"laggard_model": s["_lag"]["name"], "reference_model": s["_ref"]["name"]}

    context = _daily_context(g, days, as_of_ts, "gap_months", "change_months", models_at)
    last = segments[-1]
    nxt = last["_nxt"]
    context.update({
        "current_since": _iso(last["_start"]),
        "current_laggard_model": last["_lag"]["name"],
        "current_laggard_date": _iso(last["_lag"]["date"]),
        "current_laggard_score": _r2(last["_lag"]["score"]),
        "current_reference_model": last["_ref"]["name"],
        "current_reference_date": _iso(last["_ref"]["date"]),
        "current_reference_score": _r2(last["_ref"]["score"]),
        "current_reference_matched": bool(last["_matched"]),
        # Lower bound of the measured range: months since the oldest leader
        # model the laggard has not matched (0 if it has matched the newest).
        "current_lower_months": _r1(float(lower[n - 1])),
        "current_next_leader_model": nxt["name"] if nxt is not None else None,
        "current_next_leader_date": _iso(nxt["date"]) if nxt is not None else None,
        "current_next_leader_score": _r2(nxt["score"]) if nxt is not None else None,
        "mean_lower_months": _r1(float(np.mean(lower))),
    })

    out_segments = []
    for seg in segments:
        nx = seg["_nxt"]
        out_segments.append({
            "start": _iso(seg["_start"]),
            "end": _iso(seg["_end"]),
            "gap_start": _r1(_months(seg["_start"], seg["_ref"]["date"])),
            "gap_end": _r1(_months(seg["_end"], seg["_ref"]["date"])),
            "laggard_model": seg["_lag"]["name"],
            "laggard_date": _iso(seg["_lag"]["date"]),
            "laggard_score": _r2(seg["_lag"]["score"]),
            "reference_model": seg["_ref"]["name"],
            "reference_date": _iso(seg["_ref"]["date"]),
            "reference_score": _r2(seg["_ref"]["score"]),
            "reference_matched": bool(seg["_matched"]),
            # Lower bound: months since next_leader_date (0 when null).
            "next_leader_model": nx["name"] if nx is not None else None,
            "next_leader_date": _iso(nx["date"]) if nx is not None else None,
            "next_leader_score": _r2(nx["score"]) if nx is not None else None,
            "lower_start": _r1(_months(seg["_start"], nx["date"])) if nx is not None else 0.0,
            "lower_end": _r1(_months(seg["_end"], nx["date"])) if nx is not None else 0.0,
        })

    expected = _sample_expected(expected_fn, ws, as_of_ts, expected_every_days) if expected_fn else []

    return {
        "method": "epoch_daily",
        "days_per_month": DAYS_PER_MONTH,
        "as_of": _iso(as_of_ts),
        "start": _iso(ws),
        # Latest scored release (either group) on or before as_of. The series
        # keeps accruing after it, so a long gap between this date and as_of
        # means the tail reflects missing scores, not measured progress.
        "last_model_date": _iso(d["date"].max()),
        "n_days": int(n),
        "segments": out_segments,
        "events": events,
        # Running-max release sequences of both groups (all dates <= as_of):
        # the chart draws leader releases as ticks, including recent ones the
        # laggard has not matched yet.
        "leader_frontier": _frontier_list(sota),
        "laggard_frontier": _frontier_list(lag_best),
        # Survival-analysis current-gap estimate recomputed as of each sample
        # date (a forecast that counts unmatched leader models by their age).
        "expected_lead": {"method": "survival", "every_days": int(expected_every_days), "points": expected},
        "context": context,
    }


def build_score_gap_timeline(
    df: pd.DataFrame,
    *,
    score_col: str,
    model_col: str,
    as_of: Any,
    laggard_col: str = "Open",
) -> Optional[dict]:
    """Best leader score minus best laggard score on each day, in the
    benchmark's own units (signed: negative means the laggard leads). Steps
    change only on frontier releases: up on a leader record, down on a
    laggard record. ``calculate_gap_metrics``' ``avg_vertical_gap`` is the
    mean of this series clamped at zero."""
    prep = _prepare(df, score_col, model_col, laggard_col, as_of)
    if prep is None:
        return None
    d, sota, lag_best, ws, as_of_ts = prep

    steps: list[dict] = []
    events: list[dict] = []
    for e in _event_dates(ws, lag_best, sota):
        led = _latest_before(sota, e)
        lag = _latest_before(lag_best, e)
        if led is None or lag is None:
            continue
        if steps and steps[-1]["_led"] is led and steps[-1]["_lag"] is lag:
            continue
        gap = led["score"] - lag["score"]
        if steps:
            prev = steps[-1]
            prev["_end"] = e
            led_changed = prev["_led"] is not led
            events.append({
                "date": _iso(e),
                "kind": "leader_release" if led_changed else "laggard_release",
                "model": led["name"] if led_changed else lag["name"],
                "gap_before": _r2(prev["_gap"]),
                "gap_after": _r2(gap),
                "leader_model": led["name"],
                "laggard_model": lag["name"],
            })
        steps.append({"_start": e, "_end": None, "_led": led, "_lag": lag, "_gap": gap})
    if not steps:
        return None
    steps[-1]["_end"] = as_of_ts

    days = pd.date_range(ws, as_of_ts)
    n = len(days)
    g = np.empty(n, dtype=float)
    step_idx = np.empty(n, dtype=int)
    for i, st in enumerate(steps):
        last = i == len(steps) - 1
        lo = int((st["_start"] - ws).days)
        hi = int((st["_end"] - ws).days) + (1 if last else 0)
        g[lo:hi] = st["_gap"]
        step_idx[lo:hi] = i

    def models_at(i: int) -> dict:
        s = steps[int(step_idx[i])]
        return {"leader_model": s["_led"]["name"], "laggard_model": s["_lag"]["name"]}

    context = _daily_context(g, days, as_of_ts, "gap_points", "change_points", models_at)
    last = steps[-1]
    context.update({
        "current_since": _iso(last["_start"]),
        "current_leader_model": last["_led"]["name"],
        "current_leader_date": _iso(last["_led"]["date"]),
        "current_leader_score": _r2(last["_led"]["score"]),
        "current_laggard_model": last["_lag"]["name"],
        "current_laggard_date": _iso(last["_lag"]["date"]),
        "current_laggard_score": _r2(last["_lag"]["score"]),
        "laggard_leads": bool(last["_gap"] < -_EPS),
    })

    out_steps = [{
        "start": _iso(st["_start"]),
        "end": _iso(st["_end"]),
        "gap": _r2(st["_gap"]),
        "leader_model": st["_led"]["name"],
        "leader_date": _iso(st["_led"]["date"]),
        "leader_score": _r2(st["_led"]["score"]),
        "laggard_model": st["_lag"]["name"],
        "laggard_date": _iso(st["_lag"]["date"]),
        "laggard_score": _r2(st["_lag"]["score"]),
    } for st in steps]

    return {
        "method": "score_gap",
        "as_of": _iso(as_of_ts),
        "start": _iso(ws),
        "last_model_date": _iso(d["date"].max()),
        "n_days": int(n),
        "steps": out_steps,
        "events": events,
        "leader_frontier": _frontier_list(sota),
        "laggard_frontier": _frontier_list(lag_best),
        "context": context,
    }
