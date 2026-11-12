"""How far one group of models trails another on the Epoch Capabilities Index
(ECI), measured day by day.

The method is Epoch AI's ("Open models lag state-of-the-art closed models by 4
months", 29 May 2026, https://epoch.ai/data-insights/open-closed-eci-gap),
written here for any two groups: a leader group and a laggard group.

* On each day, take the laggard model with the highest ECI released so far.
* Find the most recent leader model that set a new ECI record and that the
  laggard model has plausibly caught up to. The time gap that day is the time
  since that leader model was released.
* The ECI gap that day is the best leader score minus the best laggard score.

"Plausibly caught up" means the laggard model scores higher in at least 5% of
paired bootstrap resamples of the benchmark data (`Criterion`). The headline
is the daily time gap averaged over the trailing 12 months, and
`rolling_average` gives that average as it stood on every earlier day, with
how much the daily gap varied around it.

Two comparisons are published (`COMPARISONS`): open-weight models against
closed ones, which is Epoch's, and Chinese models against U.S. ones, which
applies the same method to Epoch's developer-country label with open and
closed models counted alike. The China vs U.S. comparison also carries a
second, smoother measure, `trend_lag`: the distance between straight-line
fits to each country's record-setters since 2024.

Between releases nothing changes except the calendar, so the series is stored
as segments: within one, the gap on day `t` is the time since the segment's
reference model was released. Two consequences of the method are worth
knowing when reading the series. It falls when a laggard model catches up to
a newer leader record, and it falls to zero on the day a leader sets a record
the best laggard model had already caught up to. And because the test is
statistical, a new laggard record can fail a comparison its predecessor
passed, which moves the reference back and the gap up.

Pure Python, no third-party imports: the paired bootstrap arrives as an
object with `prob_exceeds(a, b)` (see `eci_bootstrap.py`).
"""
from __future__ import annotations

import calendar
import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, Iterable, Iterator, Optional, Sequence

DAYS_PER_MONTH = 365.25 / 12

# Epoch's rule: caught up when ahead in at least this share of paired resamples.
MIN_PAIRED_SHARE = 0.05
# Published intervals are 90% intervals, so sd = width / (2 * 1.645).
CI_Z = 1.645
# One-sided 5% critical value, Phi^-1(0.95): the same rule on a normal curve.
Z_ONE_SIDED_5PCT = 1.6448536269514722
# Last resort when a model has no interval at all: within this many ECI points.
POINT_MARGIN = 1.0

HEADLINE_WINDOW_MONTHS = 12

# The trendline comparison fits each group's record-setters from this date on,
# and needs at least this many of them in each group.
TREND_SINCE = date(2024, 1, 1)
TREND_MIN_RECORDS = 3

CHINA = "China"
UNITED_STATES = "United States of America"


@dataclass(frozen=True)
class Model:
    """One model as the ECI scores it (a row of Epoch's `model_group`)."""

    name: str
    released: date
    eci: float
    is_open: bool
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    organization: str = ""
    country: str = ""               # Epoch's "Country (of organization)"

    @property
    def sd(self) -> Optional[float]:
        if self.ci_low is None or self.ci_high is None:
            return None
        return (self.ci_high - self.ci_low) / (2 * CI_Z)


# name -> (is this model in the laggard group?, is it in the leader group?)
COMPARISONS: dict[str, tuple[Callable[[Model], bool], Callable[[Model], bool]]] = {
    "open_closed": (lambda m: m.is_open, lambda m: not m.is_open),
    "china_us": (lambda m: m.country == CHINA, lambda m: m.country == UNITED_STATES),
}
# Comparisons that also publish the trendline measure.
TRENDLINES = {"china_us"}


class Criterion:
    """Has `challenger` plausibly caught up to `record`?

    Three tests, strongest available first:

    bootstrap   ahead in >= 5% of paired resamples (Epoch's test);
    analytical  the same rule on independent normal curves built from the two
                published intervals, used when a model is missing from the
                bootstrap fit or the fit is unavailable. Ignoring the pairing
                overstates the spread, so it is slightly more permissive;
    threshold   within one ECI point, when a model has no interval.
    """

    def __init__(self, draws=None):
        self.draws = draws
        self._memo: dict[tuple[str, str], tuple[bool, str]] = {}

    def test(self, challenger: Model, record: Model) -> tuple[bool, str]:
        """(caught up?, which test decided)."""
        key = (challenger.name, record.name)
        if key not in self._memo:
            self._memo[key] = self._decide(challenger, record)
        return self._memo[key]

    def caught_up(self, challenger: Model, record: Model) -> bool:
        return self.test(challenger, record)[0]

    def _decide(self, challenger: Model, record: Model) -> tuple[bool, str]:
        if self.draws is not None:
            share = self.draws.prob_exceeds(challenger.name, record.name)
            if share is not None:
                return share >= MIN_PAIRED_SHARE, "bootstrap"
        sd_a, sd_b = challenger.sd, record.sd
        if sd_a is not None and sd_b is not None and (sd_a > 0 or sd_b > 0):
            spread = Z_ONE_SIDED_5PCT * math.hypot(sd_a, sd_b)
            return record.eci - challenger.eci <= spread, "analytical"
        return challenger.eci >= record.eci - POINT_MARGIN, "threshold"


def record_setters(models: Iterable[Model]) -> list[Model]:
    """The models that beat every earlier model in the list, in release order.
    Of several released on one day, only the best can set the record."""
    out: list[Model] = []
    best = -math.inf
    for m in sorted(models, key=lambda m: (m.released, -m.eci, m.name)):
        if m.eci > best:
            best = m.eci
            out.append(m)
    return out


@dataclass(frozen=True)
class Segment:
    """A run of days on which nothing changed: the same best laggard model,
    the same leader, and the same reference. Runs from `start` to the day
    before the next segment starts (the last one runs to the build date)."""

    start: date
    laggard: Model                  # best laggard model released so far
    leader: Model                   # best leader model released so far
    reference: Model                # leader record the gap is measured from
    matched: bool                   # False: not caught up to any leader record yet
    unmatched: Optional[Model]      # oldest leader record not yet caught up to

    def months(self, day: date) -> float:
        return (day - self.reference.released).days / DAYS_PER_MONTH

    @property
    def eci_points(self) -> float:
        """Best leader minus best laggard; negative if the laggard is ahead."""
        return self.leader.eci - self.laggard.eci


def _latest(records: Sequence[Model], day: date) -> Optional[Model]:
    found = None
    for m in records:
        if m.released > day:
            break
        found = m
    return found


def build_segments(laggards: Sequence[Model], leaders: Sequence[Model],
                   criterion: Criterion, as_of: date) -> list[Segment]:
    """The day-by-day series from the first day both groups exist to `as_of`.
    `laggards` and `leaders` are each group's record-setters."""
    if not laggards or not leaders:
        raise ValueError("need at least one model in each group")
    start = max(laggards[0].released, leaders[0].released)
    if start > as_of:
        raise ValueError(f"no day on or before {as_of} has a model in both groups")

    release_days = {m.released for m in (*laggards, *leaders) if start <= m.released <= as_of}
    segments: list[Segment] = []
    for day in sorted(release_days | {start}):
        best = _latest(laggards, day)
        available = [m for m in leaders if m.released <= day]
        # Most recent leader record the best laggard model has caught up to.
        for i in range(len(available) - 1, -1, -1):
            if criterion.caught_up(best, available[i]):
                unmatched = available[i + 1] if i + 1 < len(available) else None
                state = (best, available[-1], available[i], True, unmatched)
                break
        else:
            # Not caught up to any of them: count from the first leader record.
            state = (best, available[-1], available[0], False, available[0])
        segment = Segment(day, *state)
        if segments and _state(segments[-1]) == _state(segment):
            continue
        segments.append(segment)
    return segments


def _state(s: Segment) -> tuple:
    return (s.laggard, s.leader, s.reference, s.matched, s.unmatched)


def segment_on(segments: Sequence[Segment], day: date) -> Segment:
    if day < segments[0].start:
        raise ValueError(f"{day} is before the series starts ({segments[0].start})")
    found = segments[0]
    for s in segments:
        if s.start > day:
            break
        found = s
    return found


def daily(segments: Sequence[Segment], first: date, last: date) -> Iterator[tuple[date, Segment]]:
    """Every day from `first` to `last` inclusive with the segment it falls in."""
    i = 0
    day = max(first, segments[0].start)
    while day <= last:
        while i + 1 < len(segments) and segments[i + 1].start <= day:
            i += 1
        yield day, segments[i]
        day += timedelta(days=1)


def quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile (numpy's default)."""
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def months_before(day: date, months: int) -> date:
    """`day` moved back by whole calendar months, clamped to the month's end."""
    year, month0 = divmod(day.year * 12 + (day.month - 1) - months, 12)
    last_day = calendar.monthrange(year, month0 + 1)[1]
    return date(year, month0 + 1, min(day.day, last_day))


def summarize(segments: Sequence[Segment], first: date, last: date) -> dict:
    """Mean and 5th-95th percentile of the daily gaps from `first` to `last`.

    The percentiles describe how much the gap moved over the window. They are
    not a confidence interval for the mean. The ECI gap is floored at zero each
    day, as Epoch defines it (state of the art minus the best laggard model)."""
    months, points = [], []
    for day, s in daily(segments, first, last):
        months.append(s.months(day))
        points.append(max(s.eci_points, 0.0))
    if not months:
        raise ValueError(f"no days between {first} and {last}")
    return {
        "months": sum(months) / len(months),
        "months_p05": quantile(months, 0.05),
        "months_p95": quantile(months, 0.95),
        "eci_points": sum(points) / len(points),
        "eci_points_p05": quantile(points, 0.05),
        "eci_points_p95": quantile(points, 0.95),
        "days": len(months),
    }


def headline_window(segments: Sequence[Segment], laggards: Sequence[Model],
                    leaders: Sequence[Model], as_of: date) -> tuple[date, date, bool]:
    """(first day, last day, uses the full history?) for the headline average.

    The window is the 12 months up to the latest record-setting release, so a
    stretch with no new scores does not count calendar time as lag. With less
    than 12 months of history it is everything there is."""
    last = max(m.released for m in (*laggards, *leaders) if m.released <= as_of)
    first = months_before(last, HEADLINE_WINDOW_MONTHS)
    if first > segments[0].start:
        return first, last, False
    return segments[0].start, last, True


@dataclass(frozen=True)
class Rolling:
    """The headline's figures as they stood on each day from `start`."""

    start: Optional[date]           # None when there is no day to report
    full_from: Optional[date]       # first day with a full 12 months behind it
    mean: list[float]               # one value per day
    spread: list[float]             # standard deviation of the days averaged


def rolling_average(segments: Sequence[Segment], last: date) -> Rolling:
    """The headline average as it stood on each day up to `last`, and how much
    the daily gap varied around it.

    The values for day `t` summarise the daily gap over the 12 months to `t`,
    the headline's own window: its mean, and its standard deviation. The daily
    gap is a sawtooth (a day older every day, then a drop when a newer record
    is matched), so the standard deviation says how far a single day typically
    sits from the average. Until the series is 12 months old the window is
    every day so far, which is also the headline's rule (`headline_window`).
    So the final mean is the headline when `last` is the headline's last day."""
    start = segments[0].start
    if last < start:
        return Rolling(None, None, [], [])
    gaps = [s.months(day) for day, s in daily(segments, start, last)]
    cumulative = [0.0]
    for gap in gaps:
        cumulative.append(cumulative[-1] + gap)

    full_from = None
    means, spreads = [], []
    for hi in range(len(gaps)):
        day = start + timedelta(days=hi)
        window_first = months_before(day, HEADLINE_WINDOW_MONTHS)
        lo = 0
        if window_first > start:
            lo = (window_first - start).days
            full_from = full_from or day
        window = gaps[lo:hi + 1]
        mean = (cumulative[hi + 1] - cumulative[lo]) / len(window)
        means.append(mean)
        spreads.append(math.sqrt(sum((gap - mean) ** 2 for gap in window) / len(window)))
    return Rolling(start, full_from, means, spreads)


def firm_from(segments: Sequence[Segment], first: Optional[date], last: date) -> Optional[date]:
    """First day, from `first` on, whose 12-month window is all measured days.
    `first` is the first day with a full window (`Rolling.full_from`).

    Before any leader record has been matched the daily gap is only a lower
    bound, and so is an average that includes those days."""
    if first is None:
        return None
    matched_since = next((s.start for s in segments if s.matched), None)
    if matched_since is None:
        return None
    day = first
    while day <= last:
        if months_before(day, HEADLINE_WINDOW_MONTHS) >= matched_since:
            return day
        day += timedelta(days=1)
    return None


def fit_line(points: Sequence[tuple[float, float]]) -> Optional[tuple[float, float]]:
    """Least-squares (slope, intercept) of y on x, or None if x does not vary."""
    if len(points) < 2:
        return None
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    spread = sum((x - mean_x) ** 2 for x, _ in points)
    if spread == 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / spread
    return slope, mean_y - slope * mean_x


def _r_squared(points: Sequence[tuple[float, float]], slope: float, intercept: float) -> Optional[float]:
    mean_y = sum(y for _, y in points) / len(points)
    total = sum((y - mean_y) ** 2 for _, y in points)
    if total == 0:
        return None
    return 1 - sum((y - (slope * x + intercept)) ** 2 for x, y in points) / total


def trend_lag(laggard_models: Iterable[Model], leader_models: Iterable[Model],
              since: date = TREND_SINCE) -> Optional[dict]:
    """How much earlier the leaders' trendline reached the level the laggards'
    trendline stands at on the date of the latest laggard record.

    Each trendline is a straight-line fit to its group's record-setters
    released on or after `since`. A record means beating every earlier model
    in the group, including those from before `since`: starting the count
    afresh at `since` would let the first release after it stand as a record
    however far behind the group's best it was. Fitted trends move less than
    the day-by-day gap when one release is unusually strong or weak, and the
    figure is read at the latest laggard record, so it does not grow while no
    new laggard record appears.

    Returns None when either group has fewer than three such records or the
    leaders' trend is not rising, since a flat line is never "reached"."""
    fits = []
    for models in (laggard_models, leader_models):
        records = [m for m in record_setters(models) if m.released >= since]
        if len(records) < TREND_MIN_RECORDS:
            return None
        points = [(float(m.released.toordinal()), m.eci) for m in records]
        line = fit_line(points)
        if line is None:
            return None
        fits.append((records, points, line))
    (laggards, laggard_points, (laggard_slope, laggard_at_zero)), \
        (leaders, leader_points, (leader_slope, leader_at_zero)) = fits
    if leader_slope <= 0:
        return None

    on = laggards[-1].released                       # where the figure is read
    level = laggard_slope * on.toordinal() + laggard_at_zero
    reached = (level - leader_at_zero) / leader_slope   # day the leaders' line was there
    lag = (on.toordinal() - reached) / DAYS_PER_MONTH
    end = max(laggards[-1].released, leaders[-1].released)

    def group(records, points, slope, at_zero) -> dict:
        r2 = _r_squared(points, slope, at_zero)
        return {
            "records": len(records),
            "eci_per_month": round(slope * DAYS_PER_MONTH, 2),
            "r2": None if r2 is None else round(r2, 2),
            # The fitted line from the group's first record to the later group's last.
            "line": [[day.isoformat(), round(slope * day.toordinal() + at_zero, 2)]
                     for day in (records[0].released, end)],
        }

    in_range = date.min.toordinal() <= reached <= date.max.toordinal()
    return {
        "since": since.isoformat(),
        "lag_months": round(lag, 2),
        # Positive: the leaders' line is the steeper one, so the gap is widening.
        "change_months_per_year": round((1 - laggard_slope / leader_slope) * 12, 2),
        "on": on.isoformat(),
        "laggard_model": laggards[-1].name,
        "level": round(level, 2),
        "leader_reached": date.fromordinal(round(reached)).isoformat() if in_range else None,
        "laggard": group(laggards, laggard_points, laggard_slope, laggard_at_zero),
        "leader": group(leaders, leader_points, leader_slope, leader_at_zero),
    }


@dataclass(frozen=True)
class Match:
    record: Model
    by: Optional[Model]         # first laggard model to have caught up, if any
    on: Optional[date]          # the day it had (the record's own release day if already there)
    test: Optional[str]

    @property
    def lag_months(self) -> Optional[float]:
        return None if self.on is None else (self.on - self.record.released).days / DAYS_PER_MONTH


def first_matches(laggards: Sequence[Model], leaders: Sequence[Model],
                  criterion: Criterion) -> list[Match]:
    """For each leader record, the first day the best laggard model had caught up."""
    out = []
    for record in leaders:
        incumbent = _latest(laggards, record.released)
        contenders = [(incumbent, record.released)] if incumbent else []
        contenders += [(m, m.released) for m in laggards if m.released > record.released]
        for challenger, day in contenders:
            ok, how = criterion.test(challenger, record)
            if ok:
                out.append(Match(record, challenger, day, how))
                break
        else:
            out.append(Match(record, None, None, None))
    return out


def _r(value: Optional[float], digits: int = 1) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def _iso(day: Optional[date]) -> Optional[str]:
    return day.isoformat() if day else None


def _model_json(m: Model) -> dict:
    return {
        "model": m.name,
        "released": m.released.isoformat(),
        "eci": _r(m.eci, 2),
        "ci_low": _r(m.ci_low, 2),
        "ci_high": _r(m.ci_high, 2),
        "organization": m.organization,
    }


def _model_row(m: Model) -> dict:
    """A model as the page's list of every model shows it: its score and the
    two labels the comparisons group it by."""
    return {**_model_json(m), "country": m.country, "open": m.is_open}


def build_comparison(laggard_models: Iterable[Model], leader_models: Iterable[Model],
                     as_of: date, criterion: Criterion) -> dict:
    """One comparison's figures: headline, today, the daily series, the rolling
    average, and both groups' record-setters."""
    laggard_models, leader_models = list(laggard_models), list(leader_models)
    laggards = record_setters(laggard_models)
    leaders = record_setters(leader_models)
    segments = build_segments(laggards, leaders, criterion, as_of)

    first, last, full_history = headline_window(segments, laggards, leaders, as_of)
    headline = summarize(segments, first, last)
    now = segments[-1]
    rolling = rolling_average(segments, last)

    leaders_json = []
    for match in first_matches(laggards, leaders, criterion):
        leaders_json.append({
            **_model_json(match.record),
            "matched_by": match.by.name if match.by else None,
            # The matching model's own score. A match is statistical, so this
            # can be lower than the record's.
            "matched_eci": _r(match.by.eci, 2) if match.by else None,
            "matched_on": _iso(match.on),
            "lag_months": _r(match.lag_months),
            "test": match.test,
        })

    firm = firm_from(segments, rolling.full_from, last)
    return {
        "headline": {
            "months": _r(headline["months"]),
            "months_p05": _r(headline["months_p05"]),
            "months_p95": _r(headline["months_p95"]),
            "eci_points": _r(headline["eci_points"]),
            "eci_points_p05": _r(headline["eci_points_p05"]),
            "eci_points_p95": _r(headline["eci_points_p95"]),
            "window_start": first.isoformat(),
            "window_end": last.isoformat(),
            "days": headline["days"],
            "window_months": HEADLINE_WINDOW_MONTHS,
            "full_history": full_history,
        },
        "today": {
            "months": _r(now.months(as_of)),
            "eci_points": _r(now.eci_points),
            "laggard_model": now.laggard.name,
            "leader_model": now.leader.name,
            "reference_model": now.reference.name,
            "matched": now.matched,
            "unmatched_model": now.unmatched.name if now.unmatched else None,
        },
        "segments": [{
            "start": s.start.isoformat(),
            "laggard_model": s.laggard.name,
            "leader_model": s.leader.name,
            "reference_model": s.reference.name,
            "matched": s.matched,
            "unmatched_model": s.unmatched.name if s.unmatched else None,
        } for s in segments],
        # One value per day from `start` in each list; the last average is the headline.
        "average": {
            "start": _iso(rolling.start),
            "full_from": _iso(rolling.full_from),
            "firm_from": _iso(firm),
            # Three decimals, so a value shown to one decimal rounds as the headline does.
            "months": [round(v, 3) for v in rolling.mean],
            "spread": [round(v, 2) for v in rolling.spread],
        },
        "leader_frontier": leaders_json,
        "laggard_frontier": [_model_json(m) for m in laggards],
        "models_scored": {"laggard": len(laggard_models), "leader": len(leader_models)},
    }


def build_payload(models: Iterable[Model], as_of: date, draws=None) -> dict:
    """Everything the site shows, ready for `json.dump`."""
    models = [m for m in models if m.released <= as_of]
    criterion = Criterion(draws)
    comparisons = {}
    for name, (is_laggard, is_leader) in COMPARISONS.items():
        laggard_models = [m for m in models if is_laggard(m)]
        leader_models = [m for m in models if is_leader(m)]
        comparison = build_comparison(laggard_models, leader_models, as_of, criterion)
        comparison["trend"] = trend_lag(laggard_models, leader_models) if name in TRENDLINES else None
        comparisons[name] = comparison
    return {
        "schema": 2,
        "as_of": as_of.isoformat(),
        "days_per_month": DAYS_PER_MONTH,
        "test": {
            "paired_bootstrap": draws is not None,
            "resamples": getattr(draws, "n_samples", None),
            "min_share": MIN_PAIRED_SHARE,
        },
        "comparisons": comparisons,
        # Every model the comparisons draw on, newest first.
        "models": [_model_row(m) for m in
                   sorted(models, key=lambda m: (-m.released.toordinal(), -m.eci, m.name))],
    }
