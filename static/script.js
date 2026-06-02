/**
 * ECI Accessibility Gap - Interactive Chart Visualization
 *
 * Creates a Plotly.js visualization showing the performance gap between
 * open and closed-source AI models on the Epoch AI ECI index.
 */

// Color scheme matching reference image
const COLORS = {
    closed: '#e53935',
    closedUnmatched: '#e53935',
    open: '#5c6bc0',
    connector: '#5c6bc0',
    annotation: '#6b7280',
    gridline: '#e5e7eb',
    china: '#e53935',  // Red for China
    us: '#5c6bc0',     // Blue for US/other
};

/**
 * Simple linear regression calculation
 * Returns slope (per year), intercept, and helper to predict values
 */
function linearRegression(dates, scores, useLog = false) {
    if (dates.length < 2) return null;

    // Filter out non-positive scores if using log
    let filteredDates = dates;
    let filteredScores = scores;
    if (useLog) {
        const valid = scores.map((s, i) => [s, i]).filter(([s]) => s > 0);
        if (valid.length < 2) return null;
        filteredDates = valid.map(([, i]) => dates[i]);
        filteredScores = valid.map(([s]) => s);
    }

    // Convert dates to numeric (days since first date)
    const firstDate = new Date(filteredDates[0]).getTime();
    const x = filteredDates.map(d => (new Date(d).getTime() - firstDate) / (1000 * 60 * 60 * 24));
    // For log mode, regress on ln(score) so the line is straight on a log axis
    const y = useLog ? filteredScores.map(s => Math.log(s)) : filteredScores;

    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((total, xi, i) => total + xi * y[i], 0);
    const sumXX = x.reduce((total, xi) => total + xi * xi, 0);

    const denominator = (n * sumXX - sumX * sumX);
    if (denominator === 0) return null;

    const slope = (n * sumXY - sumX * sumY) / denominator;
    const intercept = (sumY - slope * sumX) / n;

    // Convert slope to per-year (365.25 accounts for leap years, matching Python backend)
    const slopePerYear = slope * 365.25;

    return {
        slope: useLog ? Math.exp(slopePerYear) : slopePerYear, // For log: multiplicative factor per year
        slopeRaw: slopePerYear,
        intercept,
        useLog,
        predict: (date) => {
            const daysSinceFirst = (new Date(date).getTime() - firstDate) / (1000 * 60 * 60 * 24);
            const val = intercept + slope * daysSinceFirst;
            return useLog ? Math.exp(val) : val;
        },
        startDate: filteredDates[0],
        endDate: filteredDates[filteredDates.length - 1]
    };
}

// Global state
let appState = {
    data: null,
    gapMetric: 'average', // 'average' or 'current'
    framing: 'open', // 'open' or 'china'
    currentBenchmark: 'eci', // Current selected benchmark
};

/**
 * Get the score field name for the current benchmark
 * ECI uses 'eci', other benchmarks use 'score'
 */
function getScoreField() {
    return appState.currentBenchmark === 'eci' ? 'eci' : 'score';
}

/**
 * Get the score standard deviation field name for the current benchmark
 */
function getScoreStdField() {
    return appState.currentBenchmark === 'eci' ? 'eci_std' : 'score_std';
}

/**
 * Get the benchmark metadata
 */
function getBenchmarkMetadata() {
    const benchmarks = appState.data?.benchmarks;
    if (!benchmarks) return null;
    return benchmarks[appState.currentBenchmark]?.metadata || null;
}

/**
 * Fetch data from the API and render the chart
 */
async function init() {
    try {
        const url = 'data.json';
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error: ${response.status} ${response.statusText} at ${response.url}`);
        }
        const data = await response.json();

        // Store data globally
        appState.data = data;

        // Set default benchmark
        appState.currentBenchmark = data.default_benchmark || 'eci';

        // Hide loading indicator
        document.getElementById('loading').classList.add('hidden');

        // Populate benchmark selector
        populateBenchmarkSelector();

        // Set up toggle button handlers
        setupToggleHandlers();

        // Wire PNG download for gap-over-time chart
        setupHistoricalChartDownload();

        // Render chart and update UI
        if (data) {
            renderAll();
        }

    } catch (error) {
        console.error('Failed to load data:', error);

        let errorMsg = 'Failed to load data.';
        if (error.message.includes('HTTP error')) {
            errorMsg += ` Server responded with ${error.message.replace('HTTP error:', 'Status')}.`;
            errorMsg += '<br><br><b>Troubleshooting:</b><br>1. Check if data.json exists.<br>2. Check console for URL.<br>3. Hard refresh (Ctrl+F5).';
        } else {
            errorMsg += ` Error: ${error.message}`;
        }

        document.getElementById('loading').innerHTML = `
            <div style="color: #e53935; text-align: center;">
                <p><strong>${errorMsg}</strong></p>
                <p style="font-size: 0.9em; margin-top: 10px; color: #666;">If this persists on GitHub Pages, verify the repository permissions and that the deployment action succeeded.</p>
            </div>
        `;
    }
}

/**
 * Populate the benchmark selector dropdown with available benchmarks
 */
function populateBenchmarkSelector() {
    const select = document.getElementById('benchmark-select');
    if (!select || !appState.data?.benchmarks) return;

    // Clear existing options
    select.innerHTML = '';

    // Add options for each benchmark
    const benchmarks = appState.data.benchmarks;
    const benchmarkOrder = ['eci', 'metr_time_horizon', 'gpqa_diamond', 'math_level_5', 'otis_mock_aime', 'swe_bench_verified', 'simpleqa_verified', 'frontiermath_public', 'chess_puzzles'];

    // Sort benchmarks with ECI first, then alphabetically
    const sortedBenchmarks = Object.keys(benchmarks).sort((a, b) => {
        const aIndex = benchmarkOrder.indexOf(a);
        const bIndex = benchmarkOrder.indexOf(b);
        if (aIndex === -1 && bIndex === -1) return a.localeCompare(b);
        if (aIndex === -1) return 1;
        if (bIndex === -1) return -1;
        return aIndex - bIndex;
    });

    for (const benchmarkId of sortedBenchmarks) {
        const benchmark = benchmarks[benchmarkId];
        const metadata = benchmark.metadata;
        const option = document.createElement('option');
        option.value = benchmarkId;
        option.textContent = metadata?.name || benchmarkId;
        if (benchmarkId === appState.currentBenchmark) {
            option.selected = true;
        }
        select.appendChild(option);
    }

    // Add change handler
    select.addEventListener('change', function() {
        appState.currentBenchmark = this.value;
        updateBenchmarkDisplay();
        renderAll();
    });
}

/**
 * Update the UI elements that depend on the current benchmark
 */
function updateBenchmarkDisplay() {
    const metadata = getBenchmarkMetadata();
    if (!metadata) return;

    // Update title benchmark name
    const benchmarkNameEl = document.getElementById('benchmark-name');
    if (benchmarkNameEl) {
        benchmarkNameEl.textContent = metadata.name || appState.currentBenchmark;
    }

    // Update source subtitle
    const sourceEl = document.getElementById('benchmark-source');
    if (sourceEl) {
        if (appState.currentBenchmark === 'metr_time_horizon') {
            const dt = metadata.post_2023_doubling_time_days;
            const dtStr = dt ? ` Capability doubling time: ~${Math.round(dt)} days (post-2023).` : '';
            sourceEl.textContent = `Source: METR Time Horizon v1.0 + v1.1 (Frontier Models only).${dtStr}`;
        } else {
            sourceEl.textContent = `Source: Epoch AI ${metadata.name} (Frontier Models only).`;
        }
    }

    // Update score column header
    const scoreHeader = document.getElementById('score-header');
    if (scoreHeader) {
        if (appState.currentBenchmark === 'eci') {
            scoreHeader.textContent = 'ECI';
        } else if (appState.currentBenchmark === 'metr_time_horizon') {
            scoreHeader.textContent = 'p50 Horizon (min)';
        } else {
            scoreHeader.textContent = metadata.unit || 'Score';
        }
    }

    // Update trend chart title
    const trendTitle = document.getElementById('trend-title');
    if (trendTitle) {
        const unit = appState.currentBenchmark === 'eci' ? 'ECI' : 'Score';
        trendTitle.textContent = `${unit} Growth Trends (Pre vs Post April 2024)`;
    }

    const trendSubtitle = document.getElementById('trend-subtitle');
    if (trendSubtitle) {
        const unit = appState.currentBenchmark === 'eci' ? 'ECI' : 'score';
        trendSubtitle.textContent = `Comparing the rate of ${unit} increases before and after April 2024 (All Models).`;
    }

    // Update data summary
    const dataSummary = document.getElementById('data-summary');
    if (dataSummary) {
        dataSummary.textContent = `View Raw ${metadata.name || 'Benchmark'} Data (All Models)`;
    }

    // Update chart note with actual threshold value and appropriate wording
    const chartNote = document.getElementById('chart-note');
    if (chartNote && metadata?.threshold !== undefined) {
        const thresholdValue = metadata.threshold;

        if (appState.currentBenchmark === 'metr_time_horizon') {
            chartNote.innerHTML = `Note: Score is the <strong>p50 time horizon</strong> &mdash; the task duration (in human-expert minutes) where the model achieves 50% success probability.<br>
                A model is deemed to have <strong>caught up</strong> if its p50 horizon meets or exceeds the reference model's.<br>
                <em>Data merged from METR Time Horizon v1.0 and v1.1, with v1.1 taking precedence for shared models.<br>
                Matched/Unmatched counts reflect all reference models shown on the chart.</em>`;
        } else if (appState.currentBenchmark === 'eci') {
            // ECI uses Epoch AI's paired-bootstrap criterion, not a fixed threshold.
            chartNote.innerHTML = `Note: A model is deemed to have <strong>caught up</strong> when its capability exceeds the reference model's in at least 5% of Epoch AI's paired bootstrap resamples (their published criterion).<br>
                <em>The average gap is measured day by day across the overlapping history &mdash; the open frontier versus the most recent state-of-the-art model it has caught up to. Plotted scores are Epoch's published ECI point estimates.<br>
                Matched/Unmatched counts reflect all reference models shown on the chart.</em>`;
        } else {
            const thresholdDesc = `${thresholdValue} percentage point${thresholdValue !== 1 ? 's' : ''}`;
            chartNote.innerHTML = `Note: A model is deemed to have caught up if its score is <strong>within ${thresholdDesc}</strong> of the reference model.<br>
                <em>Average gap is computed by sampling 100 score levels and measuring time-to-match at each level, starting from the level where reference models first appear.<br>
                Matched/Unmatched counts reflect all reference models shown on the chart.</em>`;
        }
    }
}

/**
 * Set up toggle button event handlers
 */
function setupToggleHandlers() {
    // Gap metric toggle
    document.querySelectorAll('#gap-toggle .toggle-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            document.querySelectorAll('#gap-toggle .toggle-btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            appState.gapMetric = this.dataset.value;
            updateDisplay();
        });
    });

    // Framing toggle
    document.querySelectorAll('#framing-toggle .toggle-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            document.querySelectorAll('#framing-toggle .toggle-btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            appState.framing = this.dataset.value;
            updateFramingLabels();
            renderAll();
        });
    });
}

/**
 * Get labels based on current framing
 */
function getFramingLabels() {
    const metadata = getBenchmarkMetadata();
    const scoreName = metadata?.name || 'score';

    if (appState.framing === 'china') {
        return {
            open: 'Chinese',
            closed: 'US',
            openModel: 'Chinese model',
            closedModel: 'US model',
            unmatched: 'US model not yet matched by Chinese models',
            connector: `First Chinese model to match ${scoreName}`,
        };
    } else {
        return {
            open: 'open',
            closed: 'closed-source',
            openModel: 'Open model',
            closedModel: 'Closed model',
            unmatched: 'Closed model not yet matched by open models',
            connector: `First open model to match ${scoreName}`,
        };
    }
}

/**
 * Update the framing labels in the title and legend
 */
function updateFramingLabels() {
    const labels = getFramingLabels();

    // Update title
    const openLabel = document.getElementById('category-open');
    const closedLabel = document.getElementById('category-closed');
    if (openLabel) openLabel.textContent = labels.open;
    if (closedLabel) closedLabel.textContent = labels.closed;

    // Update legend
    const legendClosed = document.getElementById('legend-closed');
    const legendUnmatched = document.getElementById('legend-unmatched');
    const legendOpen = document.getElementById('legend-open');
    const legendConnector = document.getElementById('legend-connector');

    if (legendClosed) legendClosed.textContent = labels.closedModel;
    if (legendUnmatched) legendUnmatched.textContent = labels.unmatched;
    if (legendOpen) legendOpen.textContent = labels.openModel;
    if (legendConnector) legendConnector.textContent = labels.connector;
}

/**
 * Get the current data based on benchmark and framing selection
 */
function getCurrentData() {
    const data = appState.data;
    if (!data) return null;

    // Get benchmark-specific data
    const benchmarkData = data.benchmarks?.[appState.currentBenchmark];
    if (!benchmarkData) {
        console.warn(`Benchmark ${appState.currentBenchmark} not found`);
        return null;
    }

    // Base data from the selected benchmark
    let result = {
        models: benchmarkData.models,
        trend_models: benchmarkData.trend_models,
        gaps: benchmarkData.gaps,
        statistics: benchmarkData.statistics,
        trends: benchmarkData.trends,
        historical_gaps: benchmarkData.historical_gaps,
        metadata: benchmarkData.metadata,
        last_updated: data.last_updated,
    };

    // Apply China framing if selected and available
    if (appState.framing === 'china' && benchmarkData.china_framing) {
        result = {
            ...result,
            gaps: benchmarkData.china_framing.gaps || result.gaps,
            statistics: benchmarkData.china_framing.statistics || result.statistics,
            historical_gaps: benchmarkData.china_framing.historical_gaps || result.historical_gaps,
        };
    }

    return result;
}

/**
 * Update display based on current gap metric selection
 */
function updateDisplay() {
    const currentData = getCurrentData();
    const stats = currentData.statistics;
    const gaps = currentData.gaps;

    if (appState.gapMetric === 'current') {
        const estimate = stats.current_gap_estimate || {};
        const gapValue = estimate.estimated_current_gap || stats.avg_horizontal_gap_months;
        updateTitle(gapValue);
        updateStatsCurrentGap(stats);
        showExplainer(stats, gaps);
    } else {
        updateTitle(stats.avg_horizontal_gap_months);
        updateStats(stats);
        hideExplainer();
    }

    // Re-render historical chart to reflect gap metric mode
    renderHistoricalChart(currentData);
}

/**
 * Show the current gap explainer panel with data
 */
function showExplainer(stats, gaps) {
    const explainer = document.getElementById('current-gap-explainer');
    if (!explainer) return;

    const unmatched = gaps.filter(g => !g.matched);
    const estimate = stats.current_gap_estimate || {};
    const method = estimate.method || 'survival_analysis_mle';

    const scoreField = appState.currentBenchmark === 'eci' ? 'closed_eci' : 'closed_score';
    const metadata = getBenchmarkMetadata();
    const scoreName = metadata?.unit || 'Score';

    const unmatchedListHtml = unmatched
        .sort((a, b) => b.gap_months - a.gap_months)
        .map(g => {
            const score = g[scoreField] ?? g.closed_eci ?? g.closed_score;
            return `<li><strong>${g.closed_model}</strong>: ${g.gap_months} months old (${scoreName}: ${score?.toFixed(1) || 'N/A'})</li>`;
        })
        .join('');

    // Build explainer content based on method
    const content = explainer.querySelector('.explainer-content');
    if (!content) return;

    const avgGap = stats.avg_horizontal_gap_months?.toFixed(1) || '--';
    const minBoundVal = estimate.min_current_gap || '--';
    const priorFiltered = estimate.prior_from_first_match;

    if (method === 'oldest_unmatched') {
        content.innerHTML = `
            <p>The <strong>estimated current gap</strong> is the age of the <strong>oldest unmatched</strong> frontier model &mdash; the minimum time the open-source frontier has been behind.</p>
            <p>These frontier models have not yet been matched by an open-source model:</p>
            <ul>${unmatchedListHtml}</ul>
            <p>The oldest unmatched model has been waiting <strong>${minBoundVal} months</strong>, so the current gap is at least this long.</p>
        `;
    } else if (priorFiltered) {
        // Survival analysis with prior filtered to competitive era
        const priorMean = estimate.prior_params?.prior_mean_months?.toFixed(1) || '--';
        content.innerHTML = `
            <p>These frontier models have not yet been matched by an open-source model:</p>
            <ul>${unmatchedListHtml}</ul>
            <p><strong>Methodology:</strong> We use survival analysis, fitting a log-normal prior only to matched gaps from the <strong>competitive era</strong> &mdash; closed models released after the first open model matched any closed model. This excludes early gaps from before open-source alternatives existed.</p>
            <ol>
                <li><strong>Fit prior:</strong> A log-normal distribution is fit to recent matched gaps (prior mean: ${priorMean} months)</li>
                <li><strong>Censored likelihood:</strong> Each unmatched model tells us the gap is <em>at least</em> as long as its age</li>
                <li><strong>Truncated expectation:</strong> For each, compute E[gap | gap &gt; age] using the fitted distribution</li>
                <li><strong>Weighted estimate:</strong> Combine expectations, weighting older unmatched models more heavily</li>
            </ol>
            <p class="explainer-note">
                <strong>Minimum bound:</strong> ${minBoundVal} months &mdash; the gap cannot be shorter than the age of the oldest unmatched model.
            </p>
            <div class="distribution-section">
                <p><strong>Gap Distribution (Log-Normal Prior):</strong></p>
                <div id="distribution-chart" style="width: 100%; height: 200px;"></div>
            </div>
        `;
    } else {
        // Standard survival analysis explanation
        content.innerHTML = `
            <p>The <strong>average gap</strong> (${avgGap} months) is based on historical matched pairs, but this may underestimate the <em>current</em> gap because it doesn't fully account for models that haven't been matched yet.</p>
            <p>The <strong>estimated current gap</strong> treats unmatched frontier models as <strong>censored observations</strong> &mdash; they tell us the gap is <em>at least</em> as long as their age:</p>
            <ul>${unmatchedListHtml}</ul>
            <p><strong>Methodology:</strong> We use survival analysis with Maximum Likelihood Estimation:</p>
            <ol>
                <li><strong>Fit prior:</strong> A log-normal distribution is fit to historical matched gaps</li>
                <li><strong>Censored likelihood:</strong> Unmatched models contribute P(gap &gt; observed_age) to the likelihood</li>
                <li><strong>Truncated expectation:</strong> For each unmatched model, compute E[gap | gap &gt; age] using the truncated log-normal formula</li>
                <li><strong>Weighted estimate:</strong> Combine expectations, weighting older models more heavily (they're more informative)</li>
            </ol>
            <p class="explainer-note">
                <strong>Minimum bound:</strong> ${minBoundVal} months &mdash; the gap cannot be shorter than the age of the oldest unmatched model.
            </p>
            <div class="distribution-section">
                <p><strong>Gap Distribution (Log-Normal Prior):</strong></p>
                <div id="distribution-chart" style="width: 100%; height: 200px;"></div>
            </div>
        `;
    }

    explainer.classList.remove('hidden');

    // Store estimate for later rendering when details is opened
    explainer._currentEstimate = estimate;

    // Add toggle listener to render chart when opened (Plotly needs visible container)
    if (!explainer._hasToggleListener) {
        explainer.addEventListener('toggle', function() {
            if (this.open && this._currentEstimate && this._currentEstimate.prior_params) {
                requestAnimationFrame(() => {
                    renderDistributionChart(this._currentEstimate);
                });
            }
        });
        explainer._hasToggleListener = true;
    }
}

/**
 * Render the log-normal distribution chart in the explainer panel
 */
function renderDistributionChart(estimate) {
    const chartDiv = document.getElementById('distribution-chart');
    if (!chartDiv) {
        console.warn('Distribution chart div not found');
        return;
    }
    if (!estimate || !estimate.prior_params) {
        console.warn('No prior_params in estimate:', estimate);
        chartDiv.innerHTML = '<p style="color: #6b7280; font-size: 0.9em; text-align: center;">Distribution data not available.</p>';
        return;
    }

    const { mu, sigma, prior_mean_months } = estimate.prior_params;
    const estimatedGap = estimate.estimated_current_gap;
    const minBound = estimate.min_current_gap;

    // Generate log-normal PDF points
    const xMin = 0.1;
    const xMax = 25;
    const numPoints = 200;
    const xVals = [];
    const yVals = [];

    for (let i = 0; i < numPoints; i++) {
        const x = xMin + (xMax - xMin) * (i / (numPoints - 1));
        // Log-normal PDF: (1/(x*sigma*sqrt(2*pi))) * exp(-(ln(x)-mu)^2 / (2*sigma^2))
        const pdf = (1 / (x * sigma * Math.sqrt(2 * Math.PI))) *
                    Math.exp(-Math.pow(Math.log(x) - mu, 2) / (2 * sigma * sigma));
        xVals.push(x);
        yVals.push(pdf);
    }

    const traces = [
        // Distribution curve
        {
            x: xVals,
            y: yVals,
            type: 'scatter',
            mode: 'lines',
            fill: 'tozeroy',
            fillcolor: 'rgba(92, 107, 192, 0.2)',
            line: { color: COLORS.open, width: 2 },
            name: 'Prior Distribution',
            hoverinfo: 'skip',
        },
        // Prior mean marker
        {
            x: [prior_mean_months],
            y: [0],
            type: 'scatter',
            mode: 'markers+text',
            marker: { color: COLORS.annotation, size: 10, symbol: 'triangle-up' },
            text: [`Prior Mean: ${prior_mean_months.toFixed(1)} mo`],
            textposition: 'top center',
            textfont: { size: 10, color: COLORS.annotation },
            name: 'Prior Mean',
            showlegend: false,
        },
        // Min bound marker
        {
            x: [minBound],
            y: [0],
            type: 'scatter',
            mode: 'markers+text',
            marker: { color: COLORS.closed, size: 10, symbol: 'triangle-up' },
            text: [`Min: ${minBound.toFixed(1)} mo`],
            textposition: 'top center',
            textfont: { size: 10, color: COLORS.closed },
            name: 'Minimum Bound',
            showlegend: false,
        },
        // Estimated gap marker
        {
            x: [estimatedGap],
            y: [0],
            type: 'scatter',
            mode: 'markers+text',
            marker: { color: '#2e7d32', size: 12, symbol: 'star' },
            text: [`Estimate: ${estimatedGap.toFixed(1)} mo`],
            textposition: 'top center',
            textfont: { size: 10, color: '#2e7d32' },
            name: 'Estimated Gap',
            showlegend: false,
        },
    ];

    const layout = {
        margin: { l: 40, r: 20, t: 10, b: 40 },
        height: 200,
        xaxis: {
            title: 'Gap (Months)',
            titlefont: { size: 11 },
            tickfont: { size: 10 },
            range: [0, 20],
        },
        yaxis: {
            title: 'Density',
            titlefont: { size: 11 },
            tickfont: { size: 10 },
            showticklabels: false,
        },
        showlegend: false,
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
    };

    const config = {
        responsive: true,
        displayModeBar: false,
    };

    Plotly.newPlot('distribution-chart', traces, layout, config);
}

/**
 * Hide the current gap explainer panel
 */
function hideExplainer() {
    const explainer = document.getElementById('current-gap-explainer');
    if (explainer) {
        explainer.classList.add('hidden');
    }
}

/**
 * Render all charts and update all displays
 */
function renderAll() {
    const currentData = getCurrentData();
    if (!currentData) {
        console.warn('No data available for rendering');
        return;
    }

    updateBenchmarkDisplay();
    updateFramingLabels();
    renderChart(currentData);
    renderTrendChart(currentData);
    renderHistoricalChart(currentData);
    updateDisplay();
    updateLastUpdated(currentData.last_updated);
    renderTable(currentData.trend_models || currentData.models);
}

/**
 * Render the trend chart with dynamic trend lines based on benchmark data
 */
function renderTrendChart(data) {
    const models = data.trend_models || data.models;
    const isTrendMetr = appState.currentBenchmark === 'metr_time_horizon';
    const traces = [];
    const annotations = [];
    const scoreField = getScoreField();

    if (!models || models.length === 0) {
        document.getElementById('trend-chart').innerHTML =
            '<p style="text-align: center; color: #6b7280; padding: 2rem;">No trend data available.</p>';
        return;
    }

    const getScore = (m) => m[scoreField] ?? m.eci ?? m.score;
    const benchmarkMetadata = getBenchmarkMetadata();
    const unitName = appState.currentBenchmark === 'eci' ? 'ECI points' : (benchmarkMetadata?.unit || 'points');

    // Filter models based on current framing
    let category1Models, category2Models;
    let cat1Name, cat2Name, cat1Color, cat2Color;

    if (appState.framing === 'china') {
        category1Models = models.filter(m => m.is_china && m.date && getScore(m) !== undefined);
        category2Models = models.filter(m => !m.is_china && m.date && getScore(m) !== undefined);
        cat1Name = 'China';
        cat2Name = 'US/Other';
        cat1Color = COLORS.china;
        cat2Color = COLORS.us;
    } else {
        category1Models = models.filter(m => !m.is_open && m.date && getScore(m) !== undefined);
        category2Models = models.filter(m => m.is_open && m.date && getScore(m) !== undefined);
        cat1Name = 'Closed';
        cat2Name = 'Open';
        cat1Color = COLORS.closed;
        cat2Color = COLORS.open;
    }

    // Sort models by date for trend calculation
    category1Models.sort((a, b) => new Date(a.date) - new Date(b.date));
    category2Models.sort((a, b) => new Date(a.date) - new Date(b.date));

    // Add scatter traces for both categories
    traces.push({
        x: category1Models.map(m => m.date),
        y: category1Models.map(m => getScore(m)),
        mode: 'markers',
        type: 'scatter',
        name: cat1Name,
        marker: { color: cat1Color, size: 8, opacity: 0.6 },
        text: category1Models.map(m => m.display_name),
    });

    traces.push({
        x: category2Models.map(m => m.date),
        y: category2Models.map(m => getScore(m)),
        mode: 'markers',
        type: 'scatter',
        name: cat2Name,
        marker: { color: cat2Color, size: 8, opacity: 0.6 },
        text: category2Models.map(m => m.display_name),
    });

    // Calculate trend lines for each category
    // For METR (log-scale axis), use log-linear regression so the trend line is straight
    const cat1Regression = category1Models.length >= 3
        ? linearRegression(category1Models.map(m => m.date), category1Models.map(m => getScore(m)), isTrendMetr)
        : null;
    const cat2Regression = category2Models.length >= 3
        ? linearRegression(category2Models.map(m => m.date), category2Models.map(m => getScore(m)), isTrendMetr)
        : null;

    // Track which annotations belong to which trend lines
    const trendAnnotationMap = {}; // Maps trace name to annotation index

    // Add trend line for category 1
    if (cat1Regression) {
        // For log-scale regression, generate multiple points so the line renders
        // as a straight line on the log axis (Plotly draws linear segments between points)
        const nPts = cat1Regression.useLog ? 20 : 2;
        const t0 = new Date(cat1Regression.startDate).getTime();
        const t1 = new Date(cat1Regression.endDate).getTime();
        const trendXs = Array.from({length: nPts}, (_, i) => new Date(t0 + (t1 - t0) * i / (nPts - 1)).toISOString().split('T')[0]);
        const trendYs = trendXs.map(d => cat1Regression.predict(d));
        const startY = trendYs[0];
        const endY = trendYs[trendYs.length - 1];
        const trendName = `${cat1Name} Trend`;
        traces.push({
            x: trendXs,
            y: trendYs,
            mode: 'lines',
            type: 'scatter',
            name: trendName,
            line: { width: 3, dash: 'dot', color: cat1Color }
        });

        // Annotation for category 1 trend
        const growthText = cat1Regression.useLog
            ? `${cat1Regression.slope.toFixed(1)}x/year`
            : `+${cat1Regression.slope.toFixed(1)} ${unitName}/year`;
        trendAnnotationMap[trendName] = annotations.length; // Track annotation index
        annotations.push({
            x: cat1Regression.endDate,
            y: endY,
            text: `<b>${cat1Name} Growth</b><br>${growthText}`,
            showarrow: true,
            arrowhead: 2,
            ax: -100,
            ay: -40,
            bgcolor: 'rgba(255, 255, 255, 0.9)',
            borderpad: 4,
            bordercolor: cat1Color,
            borderwidth: 1,
            align: 'left',
            font: { size: 11, color: '#333' },
            visible: true
        });
    }

    // Add trend line for category 2
    if (cat2Regression) {
        const nPts2 = cat2Regression.useLog ? 20 : 2;
        const t0_2 = new Date(cat2Regression.startDate).getTime();
        const t1_2 = new Date(cat2Regression.endDate).getTime();
        const trendXs2 = Array.from({length: nPts2}, (_, i) => new Date(t0_2 + (t1_2 - t0_2) * i / (nPts2 - 1)).toISOString().split('T')[0]);
        const trendYs2 = trendXs2.map(d => cat2Regression.predict(d));
        const startY = trendYs2[0];
        const endY = trendYs2[trendYs2.length - 1];
        const trendName = `${cat2Name} Trend`;
        traces.push({
            x: trendXs2,
            y: trendYs2,
            mode: 'lines',
            type: 'scatter',
            name: trendName,
            line: { width: 3, dash: 'solid', color: cat2Color }
        });

        // Annotation for category 2 trend with comparison
        const growthText2 = cat2Regression.useLog
            ? `${cat2Regression.slope.toFixed(1)}x/year`
            : `+${cat2Regression.slope.toFixed(1)} ${unitName}/year`;
        let annotationText = `<b>${cat2Name} Growth</b><br>${growthText2}`;

        if (cat1Regression && cat1Regression.slope > 0) {
            const factor = (cat2Regression.slope / cat1Regression.slope).toFixed(1);
            if (factor > 1) {
                annotationText += `<br>${factor}x faster than ${cat1Name}`;
            } else if (factor < 1 && factor > 0) {
                annotationText += `<br>${(1/factor).toFixed(1)}x slower than ${cat1Name}`;
            }
        }

        trendAnnotationMap[trendName] = annotations.length; // Track annotation index
        annotations.push({
            x: cat2Regression.endDate,
            y: endY,
            text: annotationText,
            showarrow: true,
            arrowhead: 2,
            ax: 100,
            ay: 40,
            bgcolor: 'rgba(255, 255, 255, 0.9)',
            borderpad: 4,
            bordercolor: cat2Color,
            borderwidth: 1,
            align: 'left',
            font: { size: 11, color: '#333' },
            visible: true
        });
    }

    // Update trend chart title based on framing
    const trendTitle = document.getElementById('trend-title');
    const trendSubtitle = document.getElementById('trend-subtitle');
    if (trendTitle && trendSubtitle) {
        const scoreLabel = appState.currentBenchmark === 'eci' ? 'ECI' : 'Score';
        const framingLabel = appState.framing === 'china' ? 'China vs US/Other' : 'Open vs Closed';
        trendTitle.textContent = `${scoreLabel} Growth Trends: ${framingLabel}`;
        const scoreLabelLower = appState.currentBenchmark === 'eci' ? 'ECI' : 'score';
        trendSubtitle.textContent = `Comparing ${scoreLabelLower} growth rates between ${cat1Name.toLowerCase()} and ${cat2Name.toLowerCase()} models.`;
    }

    // Add "current date" vertical line logic
    const today = new Date().toISOString().split('T')[0];
    const now = new Date();
    const quarter = Math.ceil((now.getMonth() + 1) / 3);
    const quarterLabel = `Q${quarter} ${now.getFullYear()}`;

    annotations.push({
        x: today,
        y: 1,
        yref: 'paper',
        text: quarterLabel,
        showarrow: false,
        font: {
            size: 11,
            color: COLORS.annotation,
        },
        yshift: 10,
    });

    // Y-axis title and scale based on benchmark
    const yAxisTitle = appState.currentBenchmark === 'eci' ? 'ECI Score' :
                       appState.currentBenchmark === 'metr_time_horizon' ? 'p50 Time Horizon (minutes)' :
                       (benchmarkMetadata?.unit || 'Score');

    // ECI-specific reference annotations (only show for ECI benchmark)
    const eciAnnotations = appState.currentBenchmark === 'eci' ? [
        {
            x: 1,
            y: 130,
            xref: 'paper',
            yref: 'y',
            text: 'Claude 3.5 Sonnet (130)',
            showarrow: false,
            xanchor: 'right',
            yanchor: 'bottom',
            font: { size: 10, color: '#888' }
        },
        {
            x: 1,
            y: 150,
            xref: 'paper',
            yref: 'y',
            text: 'GPT-5 (150)',
            showarrow: false,
            xanchor: 'right',
            yanchor: 'bottom',
            font: { size: 10, color: '#888' }
        }
    ] : [];

    const layout = {
        title: { text: '', font: { size: 16 } },
        margin: { l: 60, r: 60, t: 40, b: 60 },
        height: 500,
        xaxis: { title: 'Model Release Date' },
        yaxis: {
            title: yAxisTitle,
            ...(isTrendMetr ? {
                type: 'log',
                range: [-2, 3], // 0.01 to 1000 minutes
                dtick: 1, // Show only powers of 10
            } : {}),
        },
        annotations: [
            ...annotations,
            ...eciAnnotations
        ],
        shapes: [
            // ECI-specific reference lines (only for ECI benchmark)
            ...(appState.currentBenchmark === 'eci' ? [
                {
                    type: 'line',
                    y0: 130,
                    y1: 130,
                    x0: 0,
                    x1: 1,
                    xref: 'paper',
                    line: {
                        color: 'rgba(150, 150, 150, 0.4)',
                        width: 1,
                        dash: 'dot'
                    }
                },
                {
                    type: 'line',
                    y0: 150,
                    y1: 150,
                    x0: 0,
                    x1: 1,
                    xref: 'paper',
                    line: {
                        color: 'rgba(150, 150, 150, 0.4)',
                        width: 1,
                        dash: 'dot'
                    }
                }
            ] : []),
            // Today marker (always show)
            {
                type: 'line',
                x0: today,
                x1: today,
                y0: 0,
                y1: 1,
                yref: 'paper',
                line: {
                    color: COLORS.gridline,
                    width: 1,
                    dash: 'dash',
                },
            }
        ],
        hovermode: 'closest',
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
    };

    const config = {
        responsive: true,
        displayModeBar: 'hover',
        modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'],
        displaylogo: false,
    };

    Plotly.newPlot('trend-chart', traces, layout, config).then(function(gd) {
        // Handle legend clicks to show/hide corresponding annotations
        gd.on('plotly_restyle', function(eventData) {
            if (!eventData || !eventData[0] || eventData[0].visible === undefined) return;

            const visibilityChanges = eventData[0].visible;
            const traceIndices = eventData[1];

            // Build new annotations array with updated visibility
            const newAnnotations = [...layout.annotations];

            traceIndices.forEach((traceIdx, i) => {
                const trace = gd.data[traceIdx];
                if (!trace) return;

                const annotationIdx = trendAnnotationMap[trace.name];
                if (annotationIdx === undefined) return;

                // Get visibility state (can be true, false, or 'legendonly')
                const isVisible = Array.isArray(visibilityChanges)
                    ? visibilityChanges[i] !== 'legendonly' && visibilityChanges[i] !== false
                    : visibilityChanges !== 'legendonly' && visibilityChanges !== false;

                if (newAnnotations[annotationIdx]) {
                    newAnnotations[annotationIdx] = {
                        ...newAnnotations[annotationIdx],
                        visible: isVisible
                    };
                }
            });

            Plotly.relayout(gd, { annotations: newAnnotations });
        });
    });
}

/**
 * Render the main chart using Plotly.js
 * Style: Only show closed models and the specific open models that matched them
 */
function renderChart(data) {
    const { models, gaps, statistics, metadata } = data;
    const isMetr = appState.currentBenchmark === 'metr_time_horizon';
    const labels = getFramingLabels();
    const scoreField = getScoreField();
    const scoreName = metadata?.unit || (appState.currentBenchmark === 'eci' ? 'ECI' : 'Score');

    // Helper to get score from model or gap
    const getModelScore = (m) => m[scoreField] ?? m.eci ?? m.score;
    const getGapClosedScore = (g) => g.closed_eci ?? g.closed_score;
    const getGapOpenScore = (g) => g.open_eci ?? g.open_score;

    // Create traces
    const traces = [];
    const annotations = [];
    const shapes = [];

    // Get closed models from gaps data
    const closedModels = models.filter(m => !m.is_open && m.date && getModelScore(m));

    // Matched closed models (solid red circles)
    const matchedClosed = closedModels.filter(m =>
        gaps.some(g => g.closed_model === m.model && g.matched)
    );

    if (matchedClosed.length > 0) {
        traces.push({
            x: matchedClosed.map(m => m.date),
            y: matchedClosed.map(m => getModelScore(m)),
            mode: 'markers',
            type: 'scatter',
            name: labels.closedModel,
            marker: {
                color: COLORS.closed,
                size: 12,
                symbol: 'circle',
            },
            hovertemplate: `<b>%{text}</b><br>${scoreName}: %{y:.1f}<br>Date: %{x}<extra></extra>`,
            text: matchedClosed.map(m => m.display_name || m.model),
        });
    }

    // Unmatched closed models (dashed outline)
    const unmatchedClosed = closedModels.filter(m =>
        gaps.some(g => g.closed_model === m.model && !g.matched)
    );

    if (unmatchedClosed.length > 0) {
        // Helper function to add months properly (handles month boundaries correctly)
        const addMonths = (date, months) => {
            const result = new Date(date);
            const wholeMonths = Math.floor(months);
            const fractionalDays = (months - wholeMonths) * 30.4375;  // Convert fractional month to days
            result.setMonth(result.getMonth() + wholeMonths);
            result.setDate(result.getDate() + Math.round(fractionalDays));
            return result;
        };

        const hasCIData = statistics?.ci_90_low !== undefined && statistics?.ci_90_high !== undefined;

        const hoverTexts = unmatchedClosed.map(m => {
            const name = m.display_name || m.model;
            const score = getModelScore(m);
            if (!hasCIData) {
                return `<b>${name}</b><br>${scoreName}: ${score?.toFixed(1) || 'N/A'}<br>Date: ${new Date(m.date).toLocaleDateString()}<br><i>Not yet matched</i>`;
            }
            const releaseDate = new Date(m.date);
            const expectedLow = addMonths(releaseDate, statistics.ci_90_low);
            const expectedHigh = addMonths(releaseDate, statistics.ci_90_high);
            const formatDate = (d) => d.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
            return `<b>${name}</b><br>${scoreName}: ${score?.toFixed(1) || 'N/A'}<br>Released: ${new Date(m.date).toLocaleDateString()}<br><i>Not yet matched</i><br><br><b>Expected catch-up (90% CI):</b><br>${formatDate(expectedLow)} – ${formatDate(expectedHigh)}`;
        });

        traces.push({
            x: unmatchedClosed.map(m => m.date),
            y: unmatchedClosed.map(m => getModelScore(m)),
            mode: 'markers',
            type: 'scatter',
            name: `${labels.closedModel} (unmatched)`,
            marker: {
                color: COLORS.closedUnmatched,
                size: 12,
                symbol: 'circle-open',
                line: { width: 2 },
            },
            hovertemplate: hoverTexts.map(t => t + '<extra></extra>'),
            text: unmatchedClosed.map(m => m.display_name || m.model),
        });
    }

    // Only show open models that matched a closed model (blue squares)
    // Position them at the END of the connector line (at closed model's score level)
    const matchedGaps = gaps.filter(g => g.matched);

    if (matchedGaps.length > 0) {
        traces.push({
            x: matchedGaps.map(g => g.open_date),
            y: matchedGaps.map(g => getGapClosedScore(g)),  // Position at closed model's score to align with connector line
            mode: 'markers',
            type: 'scatter',
            name: labels.openModel,
            marker: {
                color: COLORS.open,
                size: 10,
                symbol: 'square',
            },
            hovertemplate: matchedGaps.map(g =>
                `<b>${g.open_model}</b><br>${scoreName}: ${getGapOpenScore(g)?.toFixed(1)} (matched ${getGapClosedScore(g)?.toFixed(1)})<br>Gap: ${g.gap_months} months<br>Date: %{x}<extra></extra>`
            ),
        });
    }

    // Add horizontal connector lines for matched gaps
    // For METR log-scale, collect label data for a text trace instead of annotations
    const gapLabelX = [];
    const gapLabelY = [];
    const gapLabelText = [];
    const sortedMatchedGaps = [...matchedGaps].sort((a, b) => getGapClosedScore(a) - getGapClosedScore(b));
    let lastLabelledLogScore = -Infinity;

    sortedMatchedGaps.forEach(gap => {
        const closedDate = new Date(gap.closed_date);
        const openDate = new Date(gap.open_date);
        const midDate = new Date((closedDate.getTime() + openDate.getTime()) / 2);
        const closedScore = getGapClosedScore(gap);

        // Horizontal connector line
        shapes.push({
            type: 'line',
            x0: gap.closed_date,
            x1: gap.open_date,
            y0: closedScore,
            y1: closedScore,
            line: {
                color: COLORS.connector,
                width: 2,
            },
        });

        if (isMetr) {
            // For log-scale, skip labels too close together (< 0.4 decades apart)
            const logScore = Math.log10(closedScore);
            if ((logScore - lastLabelledLogScore) >= 0.4) {
                lastLabelledLogScore = logScore;
                // Offset the label Y upward by a factor (multiply score for log-scale offset)
                gapLabelX.push(midDate.toISOString().split('T')[0]);
                gapLabelY.push(closedScore * 1.8);
                gapLabelText.push(`${gap.gap_months} mo`);
            }
        } else {
            annotations.push({
                x: midDate.toISOString().split('T')[0],
                y: closedScore,
                text: `${gap.gap_months} mo`,
                showarrow: false,
                font: { size: 11, color: COLORS.open },
                yshift: 15,
            });
        }
    });

    // Add gap labels as a text trace for METR (more reliable on log axes)
    if (isMetr && gapLabelX.length > 0) {
        traces.push({
            x: gapLabelX,
            y: gapLabelY,
            mode: 'text',
            text: gapLabelText,
            textfont: { size: 11, color: COLORS.open },
            textposition: 'top center',
            hoverinfo: 'skip',
            showlegend: false,
        });
    }

    // Add dashed extensions for unmatched closed models
    const today = new Date().toISOString().split('T')[0];

    // ECI-specific reference lines (only for ECI benchmark)
    if (appState.currentBenchmark === 'eci') {
        shapes.push(
            {
                type: 'line',
                y0: 130,
                y1: 130,
                x0: 0,
                x1: 1,
                xref: 'paper',
                line: {
                    color: 'rgba(150, 150, 150, 0.4)',
                    width: 1,
                    dash: 'dot'
                }
            },
            {
                type: 'line',
                y0: 150,
                y1: 150,
                x0: 0,
                x1: 1,
                xref: 'paper',
                line: {
                    color: 'rgba(150, 150, 150, 0.4)',
                    width: 1,
                    dash: 'dot'
                }
            }
        );
    }

    gaps.filter(g => !g.matched).forEach(gap => {
        const closedScore = getGapClosedScore(gap);
        shapes.push({
            type: 'line',
            x0: gap.closed_date,
            x1: today,
            y0: closedScore,
            y1: closedScore,
            line: {
                color: COLORS.closedUnmatched,
                width: 2,
                dash: 'dot',
            },
        });

        // Annotation for unmatched
        annotations.push({
            x: today,
            y: closedScore,
            text: `${gap.gap_months} mo`,
            showarrow: false,
            font: {
                size: 11,
                color: COLORS.closed,
            },
            xshift: 25,
        });
    });

    // Add "current date" vertical line
    shapes.push({
        type: 'line',
        x0: today,
        x1: today,
        y0: 0,
        y1: 1,
        yref: 'paper',
        line: {
            color: COLORS.gridline,
            width: 1,
            dash: 'dash',
        },
    });

    // Current quarter annotation
    const now = new Date();
    const quarter = Math.ceil((now.getMonth() + 1) / 3);
    const quarterLabel = `Q${quarter} ${now.getFullYear()}`;

    annotations.push({
        x: today,
        y: 1,
        yref: 'paper',
        text: quarterLabel,
        showarrow: false,
        font: {
            size: 11,
            color: COLORS.annotation,
        },
        yshift: 10,
    });

    // Labels for reference lines (ECI only)
    if (appState.currentBenchmark === 'eci') {
        annotations.push(
            {
                x: 1,
                y: 130,
                xref: 'paper',
                yref: 'y',
                text: 'Claude 3.5 Sonnet (130)',
                showarrow: false,
                xanchor: 'right',
                yanchor: 'bottom',
                font: { size: 10, color: '#aaa' }
            },
            {
                x: 1,
                y: 150,
                xref: 'paper',
                yref: 'y',
                text: 'GPT-5 (150)',
                showarrow: false,
                xanchor: 'right',
                yanchor: 'bottom',
                font: { size: 10, color: '#aaa' }
            }
        );
    }

    // Y-axis title based on benchmark
    const chartYAxisTitle = appState.currentBenchmark === 'eci' ? 'ECI Score' :
                            appState.currentBenchmark === 'metr_time_horizon' ? 'p50 Time Horizon (minutes)' :
                            (metadata?.unit || 'Score');

    // Layout
    const layout = {
        showlegend: false,
        margin: { l: 60, r: 100, t: 20, b: 60 },
        xaxis: {
            title: 'Model Release Date',
            titlefont: { size: 12, color: COLORS.annotation },
            tickfont: { size: 11, color: COLORS.annotation },
            gridcolor: COLORS.gridline,
            zeroline: false,
        },
        yaxis: {
            title: chartYAxisTitle,
            titlefont: { size: 12, color: COLORS.annotation },
            tickfont: { size: 11, color: COLORS.annotation },
            gridcolor: COLORS.gridline,
            zeroline: false,
            ...(isMetr ? {
                type: 'log',
                range: [-2, 3], // 0.01 to 1000 minutes
                dtick: 1, // Show only powers of 10
            } : {
                tickformat: '.0f',
            }),
        },
        shapes: shapes,
        annotations: annotations,
        hovermode: 'closest',
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
    };

    const config = {
        responsive: true,
        displayModeBar: 'hover',
        modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'],
        displaylogo: false,
    };

    Plotly.newPlot('chart', traces, layout, config);
}

/**
 * Render the historical gap chart
 * Dynamic based on gap metric selection (average vs current)
 */
/* ============================================================
 * Gap-over-time chart — inline SVG (McNair-style)
 * ============================================================ */

const SVG_NS = 'http://www.w3.org/2000/svg';

function svgEl(name, attrs, kids) {
    const e = document.createElementNS(SVG_NS, name);
    if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (kids) for (const c of kids) e.appendChild(c);
    return e;
}

function parseUTCDate(d) {
    if (d instanceof Date) return d;
    const s = String(d);
    if (s.length === 10) return new Date(s + 'T00:00:00Z');
    // API serializes ISO timestamps without a 'Z'; force UTC so dates don't
    // shift by viewer's local-time offset.
    if (!/[Zz]|[+-]\d{2}:?\d{2}$/.test(s)) return new Date(s + 'Z');
    return new Date(s);
}

function fmtMonthShort(d) {
    return d.toLocaleDateString('en-US', { month: 'short', timeZone: 'UTC' });
}

function fmtYear(d) { return d.getUTCFullYear(); }

function fmtDayLong(d) {
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' });
}

function shortenModelName(name) {
    if (!name) return '';
    return String(name).replace(/\s*\(.*?\)\s*$/, '').trim();
}

function escapeHTML(s) {
    return String(s).replace(/[&<>"']/g, ch => (
        { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]
    ));
}

function getLaggardAccent(framing) {
    return framing === 'china'
        ? { dot: '#e53935', area: 'rgba(229, 57, 53, 0.18)' }   // red (China laggard)
        : { dot: '#5c6bc0', area: 'rgba(92, 107, 192, 0.18)' }; // indigo (open laggard)
}

function getHistoricalLabels(framing) {
    if (framing === 'china') {
        return {
            laggard: 'Chinese',
            leader: 'US',
            laggardModel: 'Chinese release',
            leaderModelMatched: 'First US model to reach this score',
            yAxisTitle: 'Months China lagged the US frontier at each Chinese release',
            xAxisTitle: 'Chinese frontier release date',
            behindHeader: 'months behind US frontier',
        };
    }
    return {
        laggard: 'Open',
        leader: 'Closed',
        laggardModel: 'Open release',
        leaderModelMatched: 'First closed model to reach this score',
        yAxisTitle: 'Months open-source lagged the closed frontier at each open release',
        xAxisTitle: 'Open-source frontier release date',
        behindHeader: 'months behind closed frontier',
    };
}

/**
 * Build the laggard/leader frontier event arrays from per-model records.
 * A "frontier" release is one that raises the running max score within its
 * side. Returns dates as Date objects keyed `_d`, preserves model/score.
 */
function buildFrontierEvents(models, framing, parityStart) {
    const isLaggard = framing === 'china'
        ? (m) => m.is_china === true
        : (m) => m.is_open === true;
    const isLeader = framing === 'china'
        ? (m) => m.is_china !== true
        : (m) => m.is_open !== true;
    const scoreField = getScoreField();

    const byDate = (a, b) => parseUTCDate(a.date) - parseUTCDate(b.date);
    const laggard = [];
    const leader = [];
    let maxLag = -Infinity;
    let maxLed = -Infinity;
    for (const m of [...models].sort(byDate)) {
        const s = m[scoreField];
        if (s == null) continue;
        const d = parseUTCDate(m.date);
        if (parityStart && d < parityStart) {
            // still update running max so post-parity frontier baseline is correct
            if (isLaggard(m) && s > maxLag) maxLag = s;
            if (isLeader(m) && s > maxLed) maxLed = s;
            continue;
        }
        if (isLaggard(m) && s > maxLag) {
            maxLag = s;
            laggard.push({ model: m.model, display: m.display_name || m.model, date: m.date, _d: d, score: s });
        } else if (isLeader(m) && s > maxLed) {
            maxLed = s;
            leader.push({ model: m.model, display: m.display_name || m.model, date: m.date, _d: d, score: s });
        }
    }
    return { laggard, leader };
}

function renderHistoricalChart(data) {
    const root = document.getElementById('historical-chart');
    const tooltipEl = document.getElementById('historical-tooltip');
    const container = document.getElementById('historical-chart-container');
    if (!root) return;
    root.innerHTML = '';
    if (tooltipEl) tooltipEl.hidden = true;

    const historicalGaps = (data.historical_gaps || []).slice()
        .filter(g => g && g.date != null && g.gap_months != null);

    if (historicalGaps.length === 0) {
        root.innerHTML =
            '<p style="text-align: center; color: var(--color-text-muted); padding: 2rem;">No historical data available.</p>';
        return;
    }

    const framing = appState.framing;
    const labels = getHistoricalLabels(framing);
    const accent = getLaggardAccent(framing);

    // expose accent via CSS variables on the wrapper so all elements pick it up
    if (container) {
        container.style.setProperty('--c-cn', accent.dot);
        container.style.setProperty('--c-cn-soft', accent.area);
    }

    const stats = data.statistics || {};
    const isCurrentGapMode = appState.gapMetric === 'current';
    const currentEstimate = isCurrentGapMode ? (stats.current_gap_estimate || null) : null;

    // ---- derive frontier events ----
    const benchmark = appState.data?.benchmarks?.[appState.currentBenchmark];
    const allModels = benchmark?.models || [];
    let { laggard: laggardFrontierRaw, leader: leaderFrontier } =
        buildFrontierEvents(allModels, framing, null);

    // Match the same way the main chart does (Python's
    // calculate_historical_gaps): the matched closed model is the FIRST
    // closed (by date) whose score >= laggard.score - threshold. The gap
    // is the months between the matched closed release and the laggard
    // release. Releases with no matched closed, or where the matched
    // closed is later than the laggard, are dropped.
    const DAYS_PER_MONTH = 365.25 / 12;
    const matchThreshold = Number(data.metadata?.threshold) || 0;
    function firstLeaderToReach(score) {
        for (const ld of leaderFrontier) {
            if (ld.score >= score - matchThreshold) return ld;
        }
        return null;
    }

    // ECI ships a server-computed bootstrap match map; other benchmarks match
    // client-side via the threshold above. Map framing -> server key.
    const fkey = framing === 'china' ? 'china' : 'default';
    const serverMatches = benchmark?.frontier_matches?.[fkey] || null;

    // Idempotent methodology note for the bootstrap-matched (ECI) view.
    let methodNote = document.getElementById('historical-method-note');
    if (serverMatches) {
        if (!methodNote && container) {
            methodNote = document.createElement('p');
            methodNote.id = 'historical-method-note';
            methodNote.className = 'subtitle';
            methodNote.style.textAlign = 'center';
            methodNote.style.marginTop = 'var(--spacing-sm)';
            container.appendChild(methodNote);
        }
        if (methodNote) {
            methodNote.textContent =
                "Matches use Epoch AI's paired bootstrap (open model ahead in at least 5% of resamples). " +
                "Plotted scores are Epoch's published ECI point estimates.";
            methodNote.hidden = false;
        }
    } else if (methodNote) {
        methodNote.hidden = true;
    }

    // For ECI, use the server's bootstrap match; otherwise the JS threshold.
    // Named distinctly from the tooltip helper matchedLeaderFor(lag) defined
    // later: function declarations hoist, so a duplicate name would override
    // this one and the loop would silently call the wrong function.
    function pickMatchedLeader(ev) {
        if (serverMatches && Object.prototype.hasOwnProperty.call(serverMatches, ev.model)) {
            const lm = serverMatches[ev.model];
            if (lm === null) return null;               // server: no leader caught up
            const found = leaderFrontier.find(l => l.model === lm);
            if (found) return found;
            // Server named a leader not present in this frontier view: degrade
            // gracefully to the threshold match rather than dropping the laggard.
            return firstLeaderToReach(ev.score);
        }
        return firstLeaderToReach(ev.score);
    }

    const laggardFrontier = [];
    for (const ev of laggardFrontierRaw) {
        const ld = pickMatchedLeader(ev);
        if (!ld) continue;
        const gapMonths = (ev._d - ld._d) / (1000 * 60 * 60 * 24) / DAYS_PER_MONTH;
        if (gapMonths < 0) continue;
        ev.gap = gapMonths;
        ev.matchedLeader = ld;
        laggardFrontier.push(ev);
    }
    laggardFrontier.forEach((ev, i) => {
        ev.prevGap = i > 0 ? laggardFrontier[i - 1].gap : null;
    });

    if (laggardFrontier.length === 0) {
        root.innerHTML =
            '<p style="text-align: center; color: var(--color-text-muted); padding: 2rem;">Not enough laggard-frontier releases to render chart.</p>';
        return;
    }

    // ---- geometry ----
    const W = 1200;
    const H = 680;
    // Bottom margin reserves room for: tick (8) + gap (6) + rotated leader
    // labels (up to ~65px diagonal extent) + breathing room (10) + x-axis
    // title (~14) + safety (10). Total ≈ 175.
    const M = { top: 76, right: 110, bottom: 175, left: 84 };
    const innerW = W - M.left - M.right;
    const innerH = H - M.top - M.bottom;

    // X range: first → last laggard release (or today in current mode).
    // We give the chart a small horizontal pad on each side so the first/last
    // dots don't sit flush against the plot edges.
    const firstLaggardDate = laggardFrontier[0]._d;
    const lastLaggardDate = laggardFrontier[laggardFrontier.length - 1]._d;
    const today = new Date();
    const rawMin = firstLaggardDate;
    const rawMax = isCurrentGapMode && currentEstimate ? today : lastLaggardDate;
    if (rawMax <= rawMin) {
        root.innerHTML =
            '<p style="text-align: center; color: var(--color-text-muted); padding: 2rem;">Not enough data to render chart.</p>';
        return;
    }
    const pad = (rawMax - rawMin) * 0.03;
    const xMin = new Date(rawMin.getTime() - pad);
    const xMax = new Date(rawMax.getTime() + pad);

    // Y range
    const yValues = laggardFrontier.map(p => p.gap);
    if (currentEstimate) yValues.push(currentEstimate.estimated_current_gap, currentEstimate.min_current_gap);
    const yMaxRaw = Math.max(...yValues, 1);
    const yMax = Math.ceil(yMaxRaw / 3) * 3 + 1;
    const yMin = 0;

    const x = (d) => M.left + (innerW * (d - xMin)) / (xMax - xMin);
    const y = (v) => M.top + innerH - (innerH * (v - yMin)) / (yMax - yMin);

    // ---- SVG root ----
    // width/height attributes give the SVG a well-defined intrinsic aspect
    // ratio. Without them, some browsers compute `height: auto` to a default
    // (~150px) instead of deriving height from the viewBox, which lets the
    // chart's bottom content overflow into the next section.
    const svg = svgEl('svg', {
        class: 'gap-chart',
        viewBox: `0 0 ${W} ${H}`,
        width: W,
        height: H,
        preserveAspectRatio: 'xMidYMid meet',
        role: 'img',
        'aria-label': `Gap in months between ${labels.leader} and ${labels.laggard} frontier models over time`,
    });

    // y tick labels (no gridlines)
    for (let v = 0; v <= yMax; v += 3) {
        const t = svgEl('text', { class: 'axis-label', x: M.left - 8, y: y(v) + 3.5, 'text-anchor': 'end' });
        t.appendChild(document.createTextNode(v));
        svg.appendChild(t);
    }

    // x tick labels (quarterly, no gridlines)
    const xTicks = [];
    let cursor = new Date(Date.UTC(xMin.getUTCFullYear(), Math.floor(xMin.getUTCMonth() / 3) * 3, 1));
    while (cursor <= xMax) {
        if (cursor >= xMin) xTicks.push(new Date(cursor));
        cursor = new Date(Date.UTC(cursor.getUTCFullYear(), cursor.getUTCMonth() + 3, 1));
    }
    for (const d of xTicks) {
        const xp = x(d);
        const isJan = d.getUTCMonth() === 0;
        const t = svgEl('text', {
            class: 'axis-label',
            x: xp,
            y: M.top + innerH + 16,
            'text-anchor': 'middle',
            'font-weight': isJan ? '600' : '400',
        });
        t.appendChild(document.createTextNode(isJan ? fmtYear(d) : fmtMonthShort(d)));
        svg.appendChild(t);
    }

    // axis titles
    const yTitleX = 24;
    const yTitleY = M.top + innerH / 2;
    const yTitle = svgEl('text', {
        class: 'axis-title',
        x: yTitleX,
        y: yTitleY,
        'text-anchor': 'middle',
        transform: `rotate(-90 ${yTitleX} ${yTitleY})`,
    });
    yTitle.appendChild(document.createTextNode(labels.yAxisTitle));
    svg.appendChild(yTitle);

    const xTitle = svgEl('text', {
        class: 'axis-title',
        x: M.left + innerW / 2,
        y: H - 22,
        'text-anchor': 'middle',
    });
    xTitle.appendChild(document.createTextNode(labels.xAxisTitle));
    svg.appendChild(xTitle);

    // ---- gap area + line (step function) ----
    // The gap is constant between consecutive laggard releases (since the
    // matched leader doesn't change until the next laggard release advances
    // the frontier), so the line is a step function: horizontal at the
    // previous gap until the next release, then a vertical step to the new
    // gap. Each dot still anchors a vertex. The final segment extends flat
    // to xMax (today, or last release + pad in average mode).
    const stepXY = [];
    for (let i = 0; i < laggardFrontier.length; i++) {
        const ev = laggardFrontier[i];
        if (i === 0) {
            stepXY.push([x(ev._d), y(ev.gap)]);
        } else {
            const prev = laggardFrontier[i - 1];
            stepXY.push([x(ev._d), y(prev.gap)]);
            stepXY.push([x(ev._d), y(ev.gap)]);
        }
    }
    const lastEv = laggardFrontier[laggardFrontier.length - 1];
    if (xMax > lastEv._d) {
        stepXY.push([x(xMax), y(lastEv.gap)]);
    }
    const linePts = stepXY.map(p => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ');
    if (laggardFrontier.length >= 1) {
        const leftX = x(laggardFrontier[0]._d);
        const rightX = stepXY[stepXY.length - 1][0];
        const areaPts =
            `${leftX.toFixed(1)},${y(0).toFixed(1)} ` +
            linePts +
            ` ${rightX.toFixed(1)},${y(0).toFixed(1)}`;
        svg.appendChild(svgEl('polygon', { class: 'gap-area', points: areaPts }));
        svg.appendChild(svgEl('polyline', { class: 'gap-line', points: linePts }));
    }

    // ---- current-gap projection (only in current mode) ----
    if (currentEstimate && laggardFrontier.length) {
        const last = laggardFrontier[laggardFrontier.length - 1];
        const minBound = currentEstimate.min_current_gap;
        const est = currentEstimate.estimated_current_gap;
        // Soft wedge: from last release at minBound up to today at est, back along baseline to last
        const wedgePts =
            `${x(last._d).toFixed(1)},${y(last.gap).toFixed(1)} ` +
            `${x(today).toFixed(1)},${y(est).toFixed(1)} ` +
            `${x(today).toFixed(1)},${y(minBound).toFixed(1)} ` +
            `${x(last._d).toFixed(1)},${y(last.gap).toFixed(1)}`;
        svg.appendChild(svgEl('polygon', {
            class: 'gap-area is-projection',
            points: wedgePts,
        }));
        // Dashed projection line to the estimate
        svg.appendChild(svgEl('polyline', {
            class: 'gap-line is-projection',
            points: `${x(last._d).toFixed(1)},${y(last.gap).toFixed(1)} ${x(today).toFixed(1)},${y(est).toFixed(1)}`,
        }));
        // Star marker at today (rendered after dots so it stays on top)
    }

    // ---- leader frontier rotated bottom ticks ----
    const LEADER_SKIP = new Set([]); // could be wired up via UI later
    const tickTopY = M.top + innerH;
    const ROT_DEG = 40;
    const ROT_COS = Math.cos((ROT_DEG * Math.PI) / 180);
    const TICK_TOP = tickTopY + 28;
    const TICK_BOTTOM = tickTopY + 36;
    const LEADER_LABEL_Y = tickTopY + 42;
    let leaderRightX = -Infinity;
    for (const ev of leaderFrontier) {
        const xp = x(ev._d);
        if (xp < M.left || xp > W - M.right) continue;
        const labelText = shortenModelName(ev.display);
        if (LEADER_SKIP.has(labelText)) continue;
        const estW = labelText.length * 6.6 * ROT_COS + 8;
        // Skip if the rotated label would clip past the chart's right edge.
        if (xp + estW > W - 6) continue;
        if (xp < leaderRightX + 6) continue;
        leaderRightX = xp + estW;
        svg.appendChild(svgEl('line', {
            class: 'us-tick',
            x1: xp, x2: xp, y1: TICK_TOP, y2: TICK_BOTTOM,
        }));
        const labelW = labelText.length * 6.6 + 4;
        svg.appendChild(svgEl('rect', {
            class: 'label-us-box',
            x: xp - 2,
            y: LEADER_LABEL_Y - 10,
            width: labelW,
            height: 14,
            transform: `rotate(${ROT_DEG} ${xp} ${LEADER_LABEL_Y})`,
        }));
        const t = svgEl('text', {
            class: 'label-us',
            x: xp,
            y: LEADER_LABEL_Y,
            'text-anchor': 'start',
            transform: `rotate(${ROT_DEG} ${xp} ${LEADER_LABEL_Y})`,
        });
        t.appendChild(document.createTextNode(labelText));
        svg.appendChild(t);
    }

    // ---- laggard frontier dots + 2-row labels above ----
    const ROWS = [
        { y: M.top - 60, rightX: -Infinity }, // upper
        { y: M.top - 28, rightX: -Infinity }, // lower (preferred)
    ];
    const placed = [];
    for (const p of laggardFrontier) {
        const xp = x(p._d);
        if (xp < M.left || xp > W - M.right) continue;
        const labelText = shortenModelName(p.display);
        const gapText = p.prevGap != null && Number.isFinite(p.prevGap)
            ? `${p.prevGap.toFixed(1)} → ${p.gap.toFixed(1)} mo`
            : `${p.gap.toFixed(1)} mo`;
        const halfW = Math.max(labelText.length * 3.4, gapText.length * 3.1) + 6;
        let rowIdx = 1;
        if (xp - halfW < ROWS[1].rightX + 4) rowIdx = 0;
        if (rowIdx === 0 && xp - halfW < ROWS[0].rightX + 4) {
            // both rows occupied — drop this label (still draw the dot)
            placed.push({ p, xp, labelY: null, labelText, gapText });
            continue;
        }
        ROWS[rowIdx].rightX = xp + halfW;
        placed.push({ p, xp, labelY: ROWS[rowIdx].y, labelText, gapText });
    }
    const upperRowY = ROWS[0].y;
    for (const { p, xp, labelY, labelText, gapText } of placed) {
        const yp = y(p.gap);
        if (labelY != null) {
            const stubY1 = labelY === upperRowY ? M.top - 4 : labelY + 18;
            svg.appendChild(svgEl('line', { class: 'label-stub', x1: xp, x2: xp, y1: stubY1, y2: yp - 6 }));
        }
        svg.appendChild(svgEl('circle', { class: 'cn-dot', cx: xp, cy: yp, r: 4 }));
        if (labelY != null) {
            const nameEl = svgEl('text', { class: 'label-cn', x: xp, y: labelY, 'text-anchor': 'middle' });
            nameEl.appendChild(document.createTextNode(labelText));
            svg.appendChild(nameEl);
            const gapEl = svgEl('text', { class: 'label-cn-gap', x: xp, y: labelY + 14, 'text-anchor': 'middle' });
            gapEl.appendChild(document.createTextNode(gapText));
            svg.appendChild(gapEl);
        }
    }

    // ---- current-gap star marker (after dots so it sits on top) ----
    if (currentEstimate) {
        const est = currentEstimate.estimated_current_gap;
        const cx = x(today);
        const cy = y(est);
        svg.appendChild(svgEl('polygon', {
            class: 'current-star',
            points: starPoints(cx, cy, 9, 4.2, 5),
        }));
    }

    // ---- hover machinery ----
    const hoverLine = svgEl('line', {
        class: 'hover-line',
        x1: 0, x2: 0, y1: M.top, y2: M.top + innerH,
    });
    svg.appendChild(hoverLine);
    const hoverDot = svgEl('circle', { class: 'hover-dot', cx: 0, cy: 0, r: 4 });
    svg.appendChild(hoverDot);
    const hoverArea = svgEl('rect', {
        class: 'hover-target',
        x: M.left, y: M.top, width: innerW, height: innerH,
    });
    svg.appendChild(hoverArea);

    // Each laggard frontier event already carries the matched leader from
    // the catch-up computation above.
    function matchedLeaderFor(lag) {
        return lag.matchedLeader || null;
    }

    function nearestLaggard(d) {
        if (laggardFrontier.length === 0) return null;
        let best = laggardFrontier[0];
        let bd = Math.abs(d - best._d);
        for (const ev of laggardFrontier) {
            const diff = Math.abs(d - ev._d);
            if (diff < bd) { best = ev; bd = diff; }
        }
        return best;
    }

    function showTooltipAt(svgX) {
        if (!tooltipEl) return;
        const tMs = xMin.getTime() + ((svgX - M.left) / innerW) * (xMax - xMin);
        const t = new Date(tMs);
        const lag = nearestLaggard(t);
        if (!lag) {
            tooltipEl.hidden = true;
            hoverLine.classList.remove('is-active');
            hoverDot.classList.remove('is-active');
            return;
        }
        const xp = x(lag._d);
        const yp = y(lag.gap);
        hoverLine.setAttribute('x1', xp);
        hoverLine.setAttribute('x2', xp);
        hoverLine.classList.add('is-active');
        hoverDot.setAttribute('cx', xp);
        hoverDot.setAttribute('cy', yp);
        hoverDot.classList.add('is-active');
        const ld = matchedLeaderFor(lag);
        tooltipEl.innerHTML = buildTooltipHTML(lag, ld, labels);
        // Position tooltip in container coords
        const rect = svg.getBoundingClientRect();
        const scaleX = rect.width / W;
        const scaleY = rect.height / H;
        tooltipEl.style.left = `${xp * scaleX}px`;
        tooltipEl.style.top = `${yp * scaleY}px`;
        tooltipEl.hidden = false;
    }

    function buildTooltipHTML(lag, ld, lbls) {
        const lagSwatch = framing === 'china' ? 'cn' : 'cn'; // both use --c-cn token
        const leaderRow = ld ? `
            <div class="tt-row">
                <span class="tt-swatch us"></span>
                <div>
                    <div class="tt-label">${escapeHTML(lbls.leaderModelMatched)}</div>
                    <div class="tt-name">${escapeHTML(shortenModelName(ld.display))}</div>
                    <div class="tt-meta">${escapeHTML(fmtDayLong(ld._d))} · ${escapeHTML(lbls.leader === 'US' ? 'ECI' : 'Score')} ${ld.score.toFixed(1)}</div>
                </div>
            </div>` : '';
        return `
            <div class="tt-head">${lag.gap.toFixed(1)} ${escapeHTML(lbls.behindHeader)}</div>
            <div class="tt-date">at ${escapeHTML(shortenModelName(lag.display))}'s release</div>
            <div class="tt-row">
                <span class="tt-swatch ${lagSwatch}"></span>
                <div>
                    <div class="tt-label">${escapeHTML(lbls.laggardModel)}</div>
                    <div class="tt-name">${escapeHTML(shortenModelName(lag.display))}</div>
                    <div class="tt-meta">${escapeHTML(fmtDayLong(lag._d))} · ${escapeHTML(lbls.leader === 'US' ? 'ECI' : 'Score')} ${lag.score.toFixed(1)}</div>
                </div>
            </div>
            ${leaderRow}
        `;
    }

    hoverArea.addEventListener('mousemove', (ev) => {
        const rect = svg.getBoundingClientRect();
        const sx = ((ev.clientX - rect.left) * W) / rect.width;
        if (sx < M.left || sx > W - M.right) return;
        showTooltipAt(sx);
    });
    hoverArea.addEventListener('mouseleave', () => {
        if (tooltipEl) tooltipEl.hidden = true;
        hoverLine.classList.remove('is-active');
        hoverDot.classList.remove('is-active');
    });

    root.appendChild(svg);
    // Stash a serializable reference for PNG export
    root.dataset.viewbox = `0 0 ${W} ${H}`;
}

/**
 * Build N-pointed star polygon points centered at (cx, cy).
 */
function starPoints(cx, cy, outer, inner, points) {
    const step = Math.PI / points;
    const parts = [];
    for (let i = 0; i < points * 2; i++) {
        const r = i % 2 === 0 ? outer : inner;
        const a = i * step - Math.PI / 2;
        parts.push(`${(cx + r * Math.cos(a)).toFixed(2)},${(cy + r * Math.sin(a)).toFixed(2)}`);
    }
    return parts.join(' ');
}

/**
 * Serialize the gap-over-time SVG with computed styles inlined and rasterize
 * it to a PNG at 2x device pixels for sharing in reports.
 */
function setupHistoricalChartDownload() {
    const btn = document.getElementById('historical-chart-download');
    if (!btn) return;
    btn.addEventListener('click', async () => {
        const svg = document.querySelector('#historical-chart svg.gap-chart');
        if (!svg) return;
        btn.disabled = true;
        try {
            await downloadSvgAsPng(svg, gapChartFilename());
        } catch (e) {
            console.error('PNG export failed:', e);
        } finally {
            btn.disabled = false;
        }
    });
}

function gapChartFilename() {
    const today = new Date().toISOString().slice(0, 10);
    const framing = appState.framing === 'china' ? 'china-us' : 'open-closed';
    return `gap-over-time-${framing}-${today}.png`;
}

/**
 * Inline computed styles for elements we draw, then rasterize the SVG.
 * Inlining is required because the off-DOM Image loader doesn't read
 * external stylesheets.
 */
async function downloadSvgAsPng(svgEl, filename) {
    const vb = (svgEl.getAttribute('viewBox') || '0 0 1200 680').split(/\s+/).map(Number);
    const W = vb[2] || 1200;
    const H = vb[3] || 680;
    const SCALE = 2;
    const clone = svgEl.cloneNode(true);
    clone.setAttribute('xmlns', SVG_NS);
    clone.setAttribute('width', W);
    clone.setAttribute('height', H);
    inlineComputedStyles(svgEl, clone);
    // Ensure background is paper-colored (transparent rasterizes weird)
    const bg = document.createElementNS(SVG_NS, 'rect');
    bg.setAttribute('x', 0); bg.setAttribute('y', 0);
    bg.setAttribute('width', W); bg.setAttribute('height', H);
    bg.setAttribute('fill', getComputedStyle(document.body).backgroundColor || '#fff');
    clone.insertBefore(bg, clone.firstChild);
    const svgString = new XMLSerializer().serializeToString(clone);
    const blob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    try {
        const img = await loadImage(url);
        const canvas = document.createElement('canvas');
        canvas.width = W * SCALE;
        canvas.height = H * SCALE;
        const ctx = canvas.getContext('2d');
        ctx.scale(SCALE, SCALE);
        ctx.drawImage(img, 0, 0, W, H);
        const pngBlob = await new Promise(res => canvas.toBlob(res, 'image/png'));
        const pngUrl = URL.createObjectURL(pngBlob);
        const a = document.createElement('a');
        a.href = pngUrl;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(() => URL.revokeObjectURL(pngUrl), 1000);
    } finally {
        URL.revokeObjectURL(url);
    }
}

function loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = src;
    });
}

/**
 * Walk the source SVG and copy resolved SVG presentation properties onto
 * matching elements in the clone via the style attribute. We only need
 * properties that affect rendering of paint/stroke/text.
 */
const SVG_STYLE_PROPS = [
    'fill', 'fill-opacity', 'stroke', 'stroke-width', 'stroke-dasharray',
    'stroke-opacity', 'opacity', 'font-family', 'font-size', 'font-weight',
    'letter-spacing', 'text-anchor',
];
function inlineComputedStyles(source, target) {
    const srcEls = source.querySelectorAll('*');
    const tgtEls = target.querySelectorAll('*');
    if (srcEls.length !== tgtEls.length) return;
    for (let i = 0; i < srcEls.length; i++) {
        const cs = getComputedStyle(srcEls[i]);
        const parts = [];
        for (const prop of SVG_STYLE_PROPS) {
            const v = cs.getPropertyValue(prop);
            if (v && v !== 'normal' && v !== 'auto') parts.push(`${prop}:${v}`);
        }
        if (parts.length) tgtEls[i].setAttribute('style', parts.join(';'));
    }
}

/**
 * Update statistics display
 */
function updateStats(stats) {
    document.getElementById('stat-avg-gap').textContent =
        `${stats.avg_horizontal_gap_months} mo`;
    document.getElementById('stat-ci').textContent =
        `${stats.ci_90_low} - ${stats.ci_90_high} mo`;
    document.getElementById('stat-matched').textContent =
        stats.total_matched;
    document.getElementById('stat-unmatched').textContent =
        stats.total_unmatched;

    // Update stat labels for average gap view
    document.querySelector('#stat-avg-gap').closest('.stat-card').querySelector('.stat-label').textContent = 'Average Gap';
    document.querySelector('#stat-ci').closest('.stat-card').querySelector('.stat-label').textContent = '90% CI';
}

/**
 * Update statistics display for current gap view
 */
function updateStatsCurrentGap(stats) {
    const estimate = stats.current_gap_estimate || {};

    document.getElementById('stat-avg-gap').textContent =
        `${estimate.estimated_current_gap || '--'} mo`;
    document.getElementById('stat-ci').textContent =
        `≥ ${estimate.min_current_gap || '--'} mo`;
    document.getElementById('stat-matched').textContent =
        stats.total_matched;
    document.getElementById('stat-unmatched').textContent =
        stats.total_unmatched;

    // Update stat labels for current gap view
    document.querySelector('#stat-avg-gap').closest('.stat-card').querySelector('.stat-label').textContent = 'Est. Current Gap';
    document.querySelector('#stat-ci').closest('.stat-card').querySelector('.stat-label').textContent = 'Minimum Bound';
}

/**
 * Update the main title with the average gap
 */
function updateTitle(avgGap) {
    document.getElementById('gap-value').textContent = avgGap;
}

/**
 * Update the last updated timestamp
 */
function updateLastUpdated(timestamp) {
    const date = new Date(timestamp);
    const formatted = date.toLocaleString(undefined, {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        timeZoneName: 'short'
    });
    document.getElementById('last-updated').textContent = formatted;
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', init);

/**
 * Render the raw data table
 */
function renderTable(models) {
    const tableBody = document.querySelector('#eci-table tbody');
    tableBody.innerHTML = '';

    // Get score field for current benchmark
    const scoreField = getScoreField();

    // Sort models by date descending
    const sortedModels = [...models].sort((a, b) => new Date(b.date) - new Date(a.date));

    sortedModels.forEach(model => {
        const row = document.createElement('tr');

        const typeClass = model.is_open ? 'type-open' : 'type-closed';
        const typeLabel = model.is_open ? 'Open' : 'Closed';
        const score = model[scoreField] ?? model.eci ?? model.score;
        const scoreValue = score !== null && score !== undefined ? score.toFixed(1) : '-';

        // For METR, show additional info in the score cell
        let scoreDisplay = scoreValue;
        if (appState.currentBenchmark === 'metr_time_horizon' && model.average_score !== undefined) {
            scoreDisplay = `${scoreValue} min`;
        }

        // Show source version for METR
        const orgDisplay = model.source_version
            ? `${model.organization} <span style="color: #999; font-size: 0.85em;">(v${model.source_version})</span>`
            : model.organization;

        row.innerHTML = `
            <td>${model.display_name || model.model}</td>
            <td>${new Date(model.date).toLocaleDateString('en-US', { year: 'numeric', month: 'long' })}</td>
            <td>${scoreDisplay}</td>
            <td><span class=\"model-type ${typeClass}\">${typeLabel}</span></td>
            <td>${orgDisplay}</td>
        `;
        tableBody.appendChild(row);
    });
}

