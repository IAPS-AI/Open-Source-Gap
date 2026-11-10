/* Open Source Gap. Loads data.json and draws the page. No dependencies.

   data.json is built by scripts/update_data.py; the gap arithmetic lives in
   scripts/gap.py. This file only reads the result: on any day the gap is the
   time since the release of that day's reference model.

   The page shows one comparison at a time, chosen by the URL hash: a laggard
   group (open-weight models, or Chinese models) against a leader group
   (closed models, or U.S. models). */
(() => {
    'use strict';

    const SCHEMA = 2;
    const DAY = 86400000;
    const SVG_NS = 'http://www.w3.org/2000/svg';

    const WORDS = {
        open_closed: {
            laggard: 'open-weight',
            leader: 'closed',
            laggards: 'Open-weight models',
            leaders: 'Closed models',
            frontier: 'the closed frontier',
            title: 'Open Source Gap',
        },
        china_us: {
            laggard: 'Chinese',
            leader: 'U.S.',
            laggards: 'Chinese models',
            leaders: 'U.S. models',
            frontier: 'the U.S. frontier',
            title: 'China vs U.S. | Open Source Gap',
        },
    };
    const HASHES = { '#open-closed': 'open_closed', '#china-us': 'china_us' };

    const $ = (id) => document.getElementById(id);
    const all = (selector) => Array.from(document.querySelectorAll(selector));
    const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

    const toTime = (iso) => Date.UTC(+iso.slice(0, 4), +iso.slice(5, 7) - 1, +iso.slice(8, 10));
    const LONG = new Intl.DateTimeFormat('en-US', { year: 'numeric', month: 'long', day: 'numeric', timeZone: 'UTC' });
    const SHORT = new Intl.DateTimeFormat('en-US', { year: 'numeric', month: 'short', day: 'numeric', timeZone: 'UTC' });
    const longDate = (t) => LONG.format(new Date(t));
    const shortDate = (t) => SHORT.format(new Date(t));
    const one = (value) => value.toFixed(1);

    /* ------------------------------------------------------------- data */

    function prepare(data) {
        if (!data || data.schema !== SCHEMA || !data.comparisons) {
            throw new Error('data.json has an unexpected format');
        }
        const asOf = toTime(data.as_of);
        const site = {
            asOf,
            asOfIso: data.as_of,
            // The build runs daily. If it has been failing, "today" would be wrong.
            fresh: Date.now() - asOf < 2 * DAY,
            daysPerMonth: data.days_per_month,
            test: data.test,
            views: {},
        };
        for (const key of Object.keys(WORDS)) {
            if (!data.comparisons[key]) throw new Error(`data.json has no ${key} comparison`);
            site.views[key] = prepareView(site, key, data.comparisons[key]);
        }
        return site;
    }

    function prepareView(site, key, comparison) {
        const index = (list) => {
            const byName = new Map();
            const records = list.map((m) => ({ ...m, t: toTime(m.released) }));
            records.forEach((m) => byName.set(m.model, m));
            return { records, byName };
        };
        const leaders = index(comparison.leader_frontier);
        const laggards = index(comparison.laggard_frontier);
        const find = (group, name) => {
            if (name === null) return null;
            const record = group.byName.get(name);
            if (!record) throw new Error(`data.json names an unknown model: ${name}`);
            return record;
        };
        const segments = comparison.segments.map((s) => ({
            t: toTime(s.start),
            laggard: find(laggards, s.laggard_model),
            leader: find(leaders, s.leader_model),
            reference: find(leaders, s.reference_model),
            matched: s.matched,
            unmatched: find(leaders, s.unmatched_model),
        }));
        if (!segments.length) throw new Error(`data.json has no ${key} series`);
        const average = comparison.average;
        return {
            key,
            words: WORDS[key],
            asOf: site.asOf,
            fresh: site.fresh,
            daysPerMonth: site.daysPerMonth,
            headline: comparison.headline,
            today: comparison.today,
            leaders: leaders.records,
            laggards: laggards.records,
            segments,
            start: segments[0].t,
            windowEnd: toTime(comparison.headline.window_end),
            average: {
                start: average.start ? toTime(average.start) : null,
                firmFrom: average.firm_from ? toTime(average.firm_from) : null,
                months: average.months,
            },
            averageName: comparison.headline.full_history
                ? 'average so far'
                : `${comparison.headline.window_months}-month average`,
            trend: prepareTrend(comparison.trend),
        };
    }

    /** The trendline comparison, where a view has one: it then leads the page. */
    function prepareTrend(trend) {
        if (!trend) return null;
        const line = (group) => group.line.map(([day, eci]) => [toTime(day), eci]);
        return {
            months: trend.lag_months,
            perYear: trend.change_months_per_year,
            since: toTime(trend.since),
            on: toTime(trend.on),
            level: trend.level,
            reached: trend.leader_reached ? toTime(trend.leader_reached) : null,
            leaderLine: line(trend.leader),
            laggardLine: line(trend.laggard),
        };
    }

    /** The state of the world on day `t`: who leads, who is matched, how far. */
    function stateAt(view, t) {
        let segment = view.segments[0];
        for (const s of view.segments) {
            if (s.t > t) break;
            segment = s;
        }
        return {
            t,
            segment,
            months: (t - segment.reference.t) / DAY / view.daysPerMonth,
            average: averageAt(view, t),
        };
    }

    /** The headline average as it stood on day `t`, or null before there was one. */
    function averageAt(view, t) {
        // From its last day the headline holds until the next record-setting release.
        if (t >= view.windowEnd) return view.headline.months;
        const { start, months } = view.average;
        if (start === null || t < start) return null;
        return months[Math.round((t - start) / DAY)];
    }

    /** ECI of the latest record released on or before `t`, or null. */
    function levelAt(records, t) {
        let level = null;
        for (const m of records) {
            if (m.t > t) break;
            level = m.eci;
        }
        return level;
    }

    /* ------------------------------------------------------------- text */

    function setParts(node, parts) {
        node.replaceChildren(...parts.map((part) => {
            if (typeof part === 'string') return document.createTextNode(part);
            const strong = document.createElement('strong');
            strong.textContent = part.strong;
            return strong;
        }));
    }

    function renderText(site, view) {
        const { headline, today, words, trend } = view;
        const { laggard, reference } = stateAt(view, view.asOf).segment;

        // A view with a trendline leads with it; otherwise with the 12-month average.
        const figure = trend ? trend.months : headline.months;
        document.title = words.title;
        $('headline-who').textContent = words.laggards;
        $('headline-months').textContent = `about ${one(Math.abs(figure))} months`;
        $('headline-side').textContent = figure < 0 ? 'ahead of' : 'behind';
        $('headline-whom').textContent = words.frontier;

        const updated = $('updated');
        updated.textContent = shortDate(site.asOf);
        updated.setAttribute('datetime', site.asOfIso);

        const parts = [];
        let lead = site.fresh ? 'Today the gap is' : `On ${longDate(site.asOf)}, the gap was`;
        let after = '';
        if (trend) {
            parts.push(trend.months < 0
                ? `On ${longDate(trend.on)}, the date of the latest ${words.laggard} record, the ${words.laggard} `
                    + `trendline on the Epoch Capabilities Index (ECI) stood above the ${words.leader} one, at a `
                    + `level the ${words.leader} trendline reaches that much later. `
                : `That is how much earlier the ${words.leader} trendline on the Epoch Capabilities Index (ECI) `
                    + `reached the level the ${words.laggard} trendline stood at on ${longDate(trend.on)}, the date `
                    + `of the latest ${words.laggard} record. `);
            const pace = Math.abs(trend.perYear);
            parts.push(pace < 0.05
                ? 'On those trends the gap is holding steady. '
                : `On those trends the gap is ${trend.perYear > 0 ? 'widening' : 'narrowing'} by about `
                    + `${one(pace)} months a year. `);
            lead = site.fresh
                ? 'Matched model by model, the gap is'
                : `Matched model by model, the gap on ${longDate(site.asOf)} was`;
            after = site.fresh ? ' today' : '';
        } else {
            // The window ends at the latest record-setting release, usually days ago.
            let span = `over the past ${headline.window_months} months`;
            if (headline.full_history) span = `since ${longDate(toTime(headline.window_start))},`;
            else if (site.asOf - view.windowEnd > 14 * DAY) span = `over the ${headline.window_months} months to ${longDate(view.windowEnd)},`;
            parts.push(`That is the average ${span} on the Epoch Capabilities Index (ECI), which combines many `
                + 'benchmarks into one score. ');
        }
        if (today.matched) {
            parts.push(`${lead} `, { strong: `${one(today.months)} months` },
                `${after}: the best ${words.laggard} model, ${laggard.model}, statistically ${site.fresh ? 'matches' : 'matched'} `
                + `${reference.model}, a ${words.leader} model released on ${longDate(reference.t)}.`);
        } else {
            parts.push(`${lead} at least `, { strong: `${one(today.months)} months` },
                `${after}: no ${words.laggard} model ${site.fresh ? 'has' : 'had'} matched ${reference.model}, `
                + `released on ${longDate(reference.t)}.`);
        }
        if (today.eci_points < 0) {
            parts.push(` On raw score, ${laggard.model} leads ${words.frontier} by ${one(-today.eci_points)} ECI points.`);
        }
        setParts($('lede'), parts);

        $('legend-leader').textContent = words.leaders;
        $('legend-laggard').textContent = words.laggards;

        const lowerBound = view.segments.some((s) => !s.matched);
        const top = trend
            ? 'Top: each dot is a model that set a new record on the Epoch Capabilities Index. The dashed lines '
                + `are straight-line fits to the ${words.leader} and ${words.laggard} records since `
                + `${new Date(trend.since).getUTCFullYear()}, and the bracket shows how much earlier the `
                + `${words.leader} line reached the level the ${words.laggard} line stood at on ${shortDate(trend.on)}. `
                + 'Bottom: the gap on each day, matched model by model, and its average '
            : 'Top: each dot is a model that set a new record on the Epoch Capabilities Index, and the bracket '
                + `runs from the release of the newest ${words.leader} model that the best ${words.laggard} model `
                + 'has matched to the selected day. Bottom: the gap on each day and its average ';
        $('chart-caption').textContent = `${top}over the ${headline.window_months} months before.`
            + (lowerBound ? ` Paler stretches are lower bounds, from before any ${words.leader} model had been matched.` : '')
            + ' Point at or tap the chart to pick a day.';

        $('releases-title').textContent = `Every ${words.leader} record and the ${words.laggard} model that matched it`;
        $('releases-leader').textContent = `${words.leader[0].toUpperCase()}${words.leader.slice(1)} model`;

        for (const notes of all('.notes')) notes.hidden = notes.dataset.view !== view.key;
        for (const link of all('[data-view-link]')) {
            if (link.dataset.viewLink === view.key) link.setAttribute('aria-current', 'true');
            else link.removeAttribute('aria-current');
        }
    }

    /** Numbers in the method notes, which do not depend on the view on screen. */
    function fillNotes(site) {
        for (const notes of all('.notes')) {
            const view = site.views[notes.dataset.view];
            for (const node of notes.querySelectorAll('[data-fill="eci-points"]')) {
                node.textContent = one(view.headline.eci_points);
            }
            for (const node of notes.querySelectorAll('[data-fill="average-months"]')) {
                node.textContent = one(view.headline.months);
            }
            for (const node of notes.querySelectorAll('[data-show="points"]')) node.hidden = false;
            // Notes written for a trendline headline fall back if there is no trendline to show.
            for (const node of notes.querySelectorAll('[data-show="trend"]')) node.hidden = !view.trend;
            for (const node of notes.querySelectorAll('[data-show="no-trend"]')) node.hidden = Boolean(view.trend);
        }
        if (site.test.resamples) {
            for (const node of all('[data-fill="resamples"]')) node.textContent = String(site.test.resamples);
        }
        for (const node of all('[data-show="fallback"]')) node.hidden = site.test.paired_bootstrap;
    }

    function cell(row, className) {
        const td = document.createElement('td');
        if (className) td.className = className;
        row.appendChild(td);
        return td;
    }

    function line(parent, text, className) {
        const span = document.createElement('span');
        if (className) span.className = className;
        span.textContent = text;
        parent.appendChild(span);
        return span;
    }

    function renderTable(view) {
        const rows = [...view.leaders].reverse().map((m) => {
            const tr = document.createElement('tr');

            const name = cell(tr, 'keyed');
            line(name, '', 'swatch swatch-leader');
            name.appendChild(document.createTextNode(m.model));
            const released = line(name, '', 'sub');
            if (m.organization) line(released, `${m.organization}, `, 'org');
            released.appendChild(document.createTextNode(shortDate(m.t)));

            cell(tr, 'num').textContent = one(m.eci);

            const by = cell(tr);
            const lag = cell(tr, 'num');
            if (m.matched_by) {
                by.className = 'keyed';
                line(by, '', 'swatch swatch-laggard');
                by.appendChild(document.createTextNode(m.matched_by));
                const sameDay = m.matched_on === m.released;
                line(by, sameDay ? 'Already level at release' : shortDate(toTime(m.matched_on)), 'sub');
                lag.textContent = one(m.lag_months);
            } else {
                by.className = 'pending';
                by.textContent = 'Not yet matched';
                lag.className = 'num pending';
                lag.textContent = one((view.asOf - m.t) / DAY / view.daysPerMonth);
                line(lag, 'so far', 'sub');
            }
            return tr;
        });
        $('releases-body').replaceChildren(...rows);
        $('releases').hidden = false;
    }

    function describe(view, state) {
        const { laggard, reference, matched } = state.segment;
        const when = state.t === view.asOf && view.fresh ? 'today' : `on ${shortDate(state.t)}`;
        return {
            value: `${matched ? '' : 'At least '}${one(state.months)} months behind ${when}`,
            aside: state.average === null ? '' : ` (${view.averageName} ${one(state.average)})`,
            detail: matched
                ? `${laggard.model} matches ${reference.model}, released ${shortDate(reference.t)}`
                : `${laggard.model} has not matched ${reference.model}, released ${shortDate(reference.t)}`,
        };
    }

    /* ------------------------------------------------------------ chart */

    function svg(name, attrs, parent) {
        const node = document.createElementNS(SVG_NS, name);
        for (const key in attrs) node.setAttribute(key, attrs[key]);
        if (parent) parent.appendChild(node);
        return node;
    }

    function label(parent, text, attrs) {
        const node = svg('text', attrs, parent);
        node.textContent = text;
        return node;
    }

    /** Round tick values covering [low, high] in at most `count` steps. */
    function niceTicks(low, high, count) {
        const span = high - low || 1;
        const power = Math.pow(10, Math.floor(Math.log10(span / count)));
        const step = [1, 2, 5, 10].map((m) => m * power).find((s) => span / s <= count);
        const first = Math.floor(low / step) * step;
        const last = Math.ceil(high / step) * step;
        const ticks = [];
        for (let v = first; v <= last + step / 2; v += step) ticks.push(Math.round(v * 1e6) / 1e6);
        return ticks;
    }

    const trace = (points) => points.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join('');

    /** Two labels a fixed distance apart at least, keeping their order. */
    function spread(a, b, apart) {
        if (Math.abs(b - a) >= apart) return [a, b];
        const middle = (a + b) / 2;
        const side = b >= a ? 1 : -1;
        return [middle - (side * apart) / 2, middle + (side * apart) / 2];
    }

    function createChart(container, onSelect) {
        let view = null;       // the comparison on screen
        let domain = null;     // what its axes cover
        let selected = 0;
        let frame = null;      // geometry of the current render
        let dynamic = null;    // <g> holding everything that follows the selection

        /** Width of `text` set in `className`, measured in the page's own typeface. */
        function textWidth(text, className) {
            const probe = svg('svg', { width: 0, height: 0, 'aria-hidden': 'true', style: 'position:absolute;visibility:hidden' }, container);
            const measured = label(probe, text, { class: className }).getComputedTextLength();
            probe.remove();
            return measured;
        }

        function show(next) {
            view = next;
            selected = view.asOf;
            const models = view.leaders.concat(view.laggards);
            // The gap peaks on the last day of a segment, just before the next release.
            const peak = view.segments.reduce((most, s, i) => {
                const end = i + 1 < view.segments.length ? view.segments[i + 1].t : view.asOf;
                return Math.max(most, (end - s.reference.t) / DAY / view.daysPerMonth);
            }, 0);
            domain = {
                tMin: Math.min(view.leaders[0].t, view.laggards[0].t),
                eciTicks: niceTicks(Math.min(...models.map((m) => m.eci)), Math.max(...models.map((m) => m.eci)), 5),
                gapTicks: niceTicks(0, peak, 3),
                ends: [view.leaders[view.leaders.length - 1], view.laggards[view.laggards.length - 1]],
            };
            if (container.clientWidth) render();
        }

        function render() {
            if (!view) return;
            const { leaders, laggards, segments, asOf, headline, average } = view;
            const { tMin, eciTicks, gapTicks, ends } = domain;
            const now = stateAt(view, asOf);

            const width = Math.max(280, Math.floor(container.clientWidth));
            const narrow = width < 520;
            // Wide charts label the ends of their lines in a margin on the right.
            const endLabelWidth = narrow ? 0 : Math.min(
                Math.ceil(Math.max(
                    ...ends.map((m) => textWidth(m.model, 'mark-label')),
                    textWidth(view.averageName, 'tick'),
                )),
                Math.round(width * 0.26),
            );
            const margin = { left: 38, right: narrow ? 6 : endLabelWidth + 16, top: 24, bottom: 26 };
            // On a narrow chart the lower panel's key takes a row of its own.
            const between = narrow ? 68 : 46;
            const eciHeight = clamp(Math.round(width * 0.34), 220, 360);
            const gapHeight = clamp(Math.round(width * 0.22), 150, 230);
            const height = margin.top + eciHeight + between + gapHeight + margin.bottom;

            const x0 = margin.left;
            const x1 = width - margin.right;
            const xSpan = x1 - x0 - (narrow ? 6 : 0);
            const x = (t) => x0 + ((t - tMin) / (asOf - tMin)) * xSpan;
            const tAt = (px) => tMin + ((px - x0) / xSpan) * (asOf - tMin);

            const eciTop = margin.top;
            const eciBottom = eciTop + eciHeight;
            const eciLow = eciTicks[0];
            const eciHigh = eciTicks[eciTicks.length - 1];
            const yEci = (v) => eciBottom - ((v - eciLow) / (eciHigh - eciLow)) * eciHeight;

            const gapTop = eciBottom + between;
            const gapBottom = gapTop + gapHeight;
            const gapHigh = gapTicks[gapTicks.length - 1];
            const yGap = (v) => gapBottom - (v / gapHigh) * gapHeight;
            const months = (s, t) => (t - s.reference.t) / DAY / view.daysPerMonth;

            const root = svg('svg', {
                viewBox: `0 0 ${width} ${height}`,
                width,
                height,
                role: 'img',
                'aria-label': `Two aligned charts from ${longDate(tMin)} to ${longDate(asOf)}. Top: the highest ECI `
                    + `score among ${view.words.leader} models rose from ${one(leaders[0].eci)} to ${one(ends[0].eci)}, `
                    + `and among ${view.words.laggard} models from ${one(laggards[0].eci)} to ${one(ends[1].eci)}. `
                    + `Bottom: months behind, ${one(now.months)} on the last day, with a ${view.averageName} of `
                    + `${one(headline.months)}.`,
            });

            /* grid, axes, titles */
            const chrome = svg('g', {}, root);
            label(chrome, 'ECI score', { class: 'panel-title', x: 0, y: eciTop - 10 });
            label(chrome, 'Months behind', { class: 'panel-title', x: 0, y: gapTop - (narrow ? 32 : 10) });
            for (const v of eciTicks) {
                svg('line', { class: 'grid', x1: x0, x2: x1, y1: yEci(v), y2: yEci(v) }, chrome);
                label(chrome, String(v), { class: 'tick', x: x0 - 8, y: yEci(v) + 4, 'text-anchor': 'end' });
            }
            for (const v of gapTicks) {
                svg('line', { class: v === 0 ? 'baseline' : 'grid', x1: x0, x2: x1, y1: yGap(v), y2: yGap(v) }, chrome);
                label(chrome, String(v), { class: 'tick', x: x0 - 8, y: yGap(v) + 4, 'text-anchor': 'end' });
            }
            const firstYear = new Date(tMin).getUTCFullYear();
            for (let year = firstYear + 1; Date.UTC(year, 0, 1) <= asOf; year += 1) {
                const px = x(Date.UTC(year, 0, 1));
                svg('line', { class: 'grid', x1: px, x2: px, y1: eciTop, y2: eciBottom }, chrome);
                svg('line', { class: 'grid', x1: px, x2: px, y1: gapTop, y2: gapBottom }, chrome);
                label(chrome, String(year), { class: 'tick', x: px + 5, y: gapBottom + 17 });
            }
            if (x(Date.UTC(firstYear + 1, 0, 1)) - x0 > 44) {
                label(chrome, String(firstYear), { class: 'tick', x: x0, y: gapBottom + 17 });
            }

            // Key to the lower panel: beside its title, or under it on a narrow chart.
            const keys = [['area', 'Each day'], ['average', view.averageName]];
            const keyWidths = keys.map(([, text]) => 24 + textWidth(text, 'panel-key'));
            let keyX = narrow ? 0 : x1 - keyWidths.reduce((sum, w) => sum + w, 0) - 18 * (keys.length - 1);
            const keyY = gapTop - 10;
            keys.forEach(([kind, text], i) => {
                if (kind === 'area') {
                    svg('rect', { class: 'gap-area', x: keyX, y: keyY - 9, width: 16, height: 9 }, chrome);
                    svg('path', { class: 'gap-line', d: `M${keyX},${keyY - 9}h16` }, chrome);
                } else {
                    svg('path', { class: 'avg-line', d: `M${keyX},${keyY - 4}h16` }, chrome);
                }
                label(chrome, text, { class: 'panel-key', x: keyX + 22, y: keyY });
                keyX += keyWidths[i] + 18;
            });

            const marks = svg('g', {}, root);
            const trend = view.trend;
            if (trend) {
                // The straight-line fits sit behind the frontiers they summarise.
                for (const [points, side] of [[trend.leaderLine, 'leader'], [trend.laggardLine, 'laggard']]) {
                    svg('path', {
                        class: `trendline line-${side}`,
                        d: trace(points.map(([t, eci]) => [x(t), yEci(eci)])),
                    }, marks);
                }
            }

            /* frontier lines: each record holds until the next one */
            const steps = (records) => records.map((m, i) => (
                `${i ? `H${x(m.t).toFixed(1)}V` : `M${x(m.t).toFixed(1)},`}${yEci(m.eci).toFixed(1)}`
            )).join('') + `H${x(asOf).toFixed(1)}`;
            svg('path', { class: 'line line-leader', d: steps(leaders) }, marks);
            svg('path', { class: 'line line-laggard', d: steps(laggards) }, marks);
            // A filled dot loses its 2px ring and a hollow one gains its stroke, so
            // these radii give both the same visible size.
            const size = narrow ? 3.25 : 4.25;
            for (const m of leaders) svg('circle', { class: 'dot-leader', cx: x(m.t), cy: yEci(m.eci), r: size + 1 }, marks);
            for (const m of laggards) svg('circle', { class: 'dot-laggard', cx: x(m.t), cy: yEci(m.eci), r: size - 1 }, marks);

            const fitted = [];   // labels to cut short if they overrun the margin
            if (!narrow) {
                spread(yEci(ends[0].eci), yEci(ends[1].eci), 36).forEach((y, i) => {
                    fitted.push(label(marks, ends[i].model, { class: 'mark-label', x: x1 + 10, y: y - 1 }));
                    label(marks, one(ends[i].eci), { class: 'tick', x: x1 + 10, y: y + 13 });
                });
            }

            /* The trendline gap, dimensioned as on a drawing: the two points at one level
               (where the leaders' line was, where the laggards' line is), extension lines
               up from each, and the measurement on a line above the frontier. */
            let trendLabel = null;
            if (trend && trend.months >= 0 && trend.reached !== null && trend.reached >= tMin) {
                const xa = x(trend.reached);
                const xb = x(trend.on);
                const level = yEci(trend.level);
                const clear = yEci(Math.max(levelAt(leaders, trend.on) ?? trend.level, trend.level));
                const yDim = Math.max(eciTop + 16, clear - 26);
                svg('path', { class: 'trend-level', d: `M${xa.toFixed(1)},${level.toFixed(1)}H${xb.toFixed(1)}` }, marks);
                svg('path', {
                    class: 'bracket',
                    d: `M${xa.toFixed(1)},${level.toFixed(1)}V${(yDim - 5).toFixed(1)}`
                        + `M${xb.toFixed(1)},${level.toFixed(1)}V${(yDim - 5).toFixed(1)}`
                        + `M${xa.toFixed(1)},${yDim.toFixed(1)}H${xb.toFixed(1)}`
                        + `M${(xa - 4).toFixed(1)},${(yDim + 4).toFixed(1)}l8,-8`
                        + `M${(xb - 4).toFixed(1)},${(yDim + 4).toFixed(1)}l8,-8`,
                }, marks);
                svg('circle', { class: 'trend-anchor', cx: xa, cy: level, r: 3 }, marks);
                svg('circle', { class: 'trend-anchor', cx: xb, cy: level, r: 3 }, marks);
                trendLabel = label(marks, `${one(trend.months)} months`, {
                    class: 'trend-label', x: (xa + xb) / 2, y: yDim - 8, 'text-anchor': 'middle',
                });
            }

            /* gap on each day: climbs a day per day and falls when a newer record is matched */
            const runs = [];   // consecutive segments with the same matched flag share a shape
            segments.forEach((s, i) => {
                const end = i + 1 < segments.length ? segments[i + 1].t : asOf;
                const points = [[x(s.t), yGap(months(s, s.t))], [x(end), yGap(months(s, end))]];
                const last = runs[runs.length - 1];
                if (last && last.matched === s.matched) last.points.push(...points);
                else runs.push({ matched: s.matched, points: last ? [last.points[last.points.length - 1], ...points] : points });
            });
            for (const run of runs) {
                const soft = run.matched ? '' : ' gap-soft';
                const first = run.points[0];
                const last = run.points[run.points.length - 1];
                svg('path', {
                    class: `gap-area${soft}`,
                    d: `${trace(run.points)}L${last[0].toFixed(1)},${gapBottom}L${first[0].toFixed(1)},${gapBottom}Z`,
                }, marks);
                svg('path', { class: `gap-line${soft}`, d: trace(run.points) }, marks);
            }

            /* its 12-month average, ending on the headline */
            const count = average.months.length;
            if (average.start !== null && count > 1) {
                const point = (i) => [x(average.start + i * DAY), yGap(average.months[i])];
                // One value a day is finer than the screen: draw about one point per pixel and a half.
                const stride = Math.max(1, Math.floor(count / Math.max(40, xSpan / 1.5)));
                const stretch = (from, to) => {
                    const points = [];
                    for (let i = from; i < to; i += stride) points.push(point(i));
                    points.push(point(to));
                    return points;
                };
                const last = count - 1;
                const firm = average.firmFrom === null
                    ? last
                    : clamp(Math.round((average.firmFrom - average.start) / DAY), 0, last);
                if (firm > 0) svg('path', { class: 'avg-line avg-soft', d: trace(stretch(0, firm)) }, marks);
                if (firm < last) svg('path', { class: 'avg-line', d: trace(stretch(firm, last)) }, marks);
                svg('circle', { class: 'avg-dot', cx: x(view.windowEnd), cy: yGap(headline.months), r: 4.5 }, marks);
            }
            if (!narrow) {
                const [yAverage, yDay] = spread(yGap(headline.months), yGap(now.months), 34);
                label(marks, one(headline.months), { class: 'mark-label', x: x1 + 10, y: yAverage - 1 });
                fitted.push(label(marks, view.averageName, { class: 'tick', x: x1 + 10, y: yAverage + 13 }));
                label(marks, one(now.months), { class: 'mark-label', x: x1 + 10, y: yDay - 1 });
                label(marks, view.fresh ? 'today' : shortDate(asOf), { class: 'tick', x: x1 + 10, y: yDay + 13 });
            }

            dynamic = svg('g', {}, root);

            container.replaceChildren(root);
            frame = { root, width, narrow, size, x, tAt, yEci, yGap, x0, eciTop, gapBottom };

            // A label too long for the margin is cut short; the readout has it in full.
            for (const node of fitted) {
                let text = node.textContent;
                while (text.length > 4 && node.getComputedTextLength() > endLabelWidth) {
                    text = text.slice(0, -2);
                    node.textContent = `${text.trimEnd()}…`;
                }
            }
            // Keep the trendline measurement inside the plot when its bracket is near an edge.
            if (trendLabel) {
                const half = trendLabel.getComputedTextLength() / 2;
                const centre = Number(trendLabel.getAttribute('x'));
                trendLabel.setAttribute('x', clamp(centre, x0 + half, Math.max(x0 + half, x(asOf) - half)));
            }

            draw();
        }

        /** Everything that follows the selected day. */
        function draw() {
            const f = frame;
            const { leaders } = view;
            const state = stateAt(view, selected);
            const { laggard: best, reference } = state.segment;
            const xt = f.x(selected);
            const xr = f.x(reference.t);
            const yr = f.yEci(reference.eci);
            const xo = f.x(best.t);
            const yo = f.yEci(best.eci);
            const g = dynamic;
            g.replaceChildren();

            svg('line', { class: 'crosshair', x1: xt, x2: xt, y1: f.eciTop, y2: f.gapBottom }, g);

            // The same day in the lower panel: its average (until the line ends), then the day itself.
            if (state.average !== null && selected < view.windowEnd) {
                svg('circle', { class: 'avg-dot', cx: xt, cy: f.yGap(state.average), r: 4 }, g);
            }
            svg('circle', { class: 'gap-dot', cx: xt, cy: f.yGap(state.months), r: 4 }, g);

            onSelect(view, state);
            // Where the top panel dimensions the trendline gap, it carries no second
            // bracket for the day's matched pair: the readout names them.
            if (view.trend) return;

            // A dimension line, as on a drawing: along the matched model's score from its
            // release to the selected day, then down (or up) to the laggard's line, with a
            // tick where the two meet.
            svg('path', {
                class: 'bracket',
                d: `M${xr.toFixed(1)},${yr.toFixed(1)}H${xt.toFixed(1)}V${yo.toFixed(1)}`
                    + `M${(xt - 4).toFixed(1)},${(yr + 4).toFixed(1)}l8,-8`,
            }, g);
            svg('circle', { class: 'dot-leader', cx: xr, cy: yr, r: f.size + 3 }, g);
            svg('circle', { class: 'dot-laggard dot-selected', cx: xo, cy: yo, r: f.size + 0.75 }, g);

            if (!f.narrow) {
                // Label the bracket where it starts: the matched model (unless its line-end
                // label already names it) and how long before the selected day it came out.
                // Above a leader record and to its left is always clear of the leader line.
                const lines = [`${one(state.months)} months earlier`];
                if (reference !== domain.ends[0]) lines.unshift(reference.model);
                const nodes = lines.map((text, i) => label(g, text, {
                    class: i < lines.length - 1 ? 'mark-label' : 'mark-note',
                    x: xr - 10,
                    y: yr - 10 - 15 * (lines.length - 1 - i),
                    'text-anchor': 'end',
                }));
                const widest = Math.max(...nodes.map((node) => node.getComputedTextLength()));
                if (xr - 10 - widest < f.x0) {
                    // No room on the left: sit above the line on the right instead.
                    const level = levelAt(leaders, f.tAt(xr + 10 + widest + 8)) ?? reference.eci;
                    const base = Math.max(f.eciTop + 11 + 15 * (lines.length - 1), f.yEci(level) - 10);
                    nodes.forEach((node, i) => {
                        node.setAttribute('text-anchor', 'start');
                        node.setAttribute('x', xr + 10);
                        node.setAttribute('y', base - 15 * (lines.length - 1 - i));
                    });
                }
            }
        }

        function select(t) {
            if (!frame) return;
            const { tMin } = domain;
            const day = tMin + Math.round((t - tMin) / DAY) * DAY;
            const next = clamp(day, view.start, view.asOf);
            if (next === selected) return;
            selected = next;
            draw();
        }

        function fromPointer(event) {
            const box = frame.root.getBoundingClientRect();
            return frame.tAt(((event.clientX - box.left) / box.width) * frame.width);
        }

        container.addEventListener('pointermove', (event) => frame && select(fromPointer(event)));
        container.addEventListener('pointerdown', (event) => frame && select(fromPointer(event)));
        container.addEventListener('pointerleave', (event) => {
            if (frame && event.pointerType === 'mouse') select(view.asOf);
        });

        container.addEventListener('keydown', (event) => {
            if (!frame) return;
            const forward = event.key === 'ArrowRight';
            if (forward || event.key === 'ArrowLeft') {
                if (event.shiftKey) {
                    const starts = view.segments.map((s) => s.t);
                    const target = forward
                        ? starts.find((t) => t > selected)
                        : [...starts].reverse().find((t) => t < selected);
                    select(target ?? (forward ? view.asOf : view.start));
                } else {
                    select(selected + (forward ? 7 : -7) * DAY);
                }
            } else if (event.key === 'Home') {
                select(view.start);
            } else if (event.key === 'End' || event.key === 'Escape') {
                select(view.asOf);
            } else {
                return;
            }
            event.preventDefault();
        });

        let lastWidth = 0;
        const refresh = () => {
            if (container.clientWidth && container.clientWidth !== lastWidth) {
                lastWidth = container.clientWidth;
                render();
            }
        };
        if ('ResizeObserver' in window) new ResizeObserver(refresh).observe(container);
        else window.addEventListener('resize', refresh);
        // Labels are measured, so measure again once the typeface has loaded.
        if (document.fonts && document.fonts.ready) document.fonts.ready.then(render);

        return { show };
    }

    /* ------------------------------------------------------------- page */

    async function main() {
        const lede = $('lede');
        let site;
        try {
            const response = await fetch('data.json', { cache: 'no-cache' });
            if (!response.ok) throw new Error(`data.json returned ${response.status}`);
            site = prepare(await response.json());
        } catch (error) {
            console.error(error);
            lede.textContent = 'The latest figures did not load. Reload the page to try again.';
            $('figure').hidden = true;
            return;
        }

        const value = $('readout-value');
        const detail = $('readout-detail');
        const chartNode = $('chart');
        const chart = createChart(chartNode, (view, state) => {
            const text = describe(view, state);
            value.replaceChildren(document.createTextNode(text.value));
            if (text.aside) {
                const aside = document.createElement('span');
                aside.className = 'readout-aside';
                aside.textContent = text.aside;
                value.appendChild(aside);
            }
            detail.textContent = text.detail;
        });
        // Announce the readout only while someone is steering the chart by keyboard.
        const readout = $('readout');
        chartNode.addEventListener('focus', () => readout.setAttribute('aria-live', 'polite'));
        chartNode.addEventListener('blur', () => readout.removeAttribute('aria-live'));

        fillNotes(site);
        const showView = () => {
            const view = site.views[HASHES[window.location.hash] || 'open_closed'];
            document.documentElement.dataset.view = view.key;
            renderText(site, view);
            renderTable(view);
            chart.show(view);
        };
        window.addEventListener('hashchange', showView);
        showView();
    }

    main();
})();
