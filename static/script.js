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
};

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
    const benchmarkOrder = ['eci', 'gpqa_diamond', 'math_level_5', 'otis_mock_aime', 'swe_bench_verified', 'simpleqa_verified', 'frontiermath_public', 'chess_puzzles'];

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
        sourceEl.textContent = `Source: Epoch AI ${metadata.name} (Frontier Models only).`;
    }

    // Update score column header
    const scoreHeader = document.getElementById('score-header');
    if (scoreHeader) {
        const unit = metadata.unit || 'Score';
        scoreHeader.textContent = appState.currentBenchmark === 'eci' ? 'ECI' : unit;
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

    // Update chart note with actual threshold value
    const chartNote = document.getElementById('chart-note');
    if (chartNote && metadata?.threshold !== undefined) {
        const thresholdValue = metadata.threshold;
        const unit = metadata.unit || 'points';
        chartNote.innerHTML = `Note: A model is deemed to have caught up if its score is <strong>within ${thresholdValue} ${unit}</strong> of the reference model.<br>
            <em>Average gap is computed by sampling 100 score levels and measuring time-to-match at each level, starting from the level where reference models first appear.<br>
            Matched/Unmatched counts reflect all reference models shown on the chart.</em>`;
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
    const list = document.getElementById('unmatched-ages-list');
    const minBound = document.getElementById('min-bound-value');

    if (!explainer || !list) return;

    // Get unmatched models with their ages
    const unmatched = gaps.filter(g => !g.matched);
    const estimate = stats.current_gap_estimate || {};

    // Populate the list
    const scoreField = appState.currentBenchmark === 'eci' ? 'closed_eci' : 'closed_score';
    const metadata = getBenchmarkMetadata();
    const scoreName = metadata?.unit || 'Score';

    list.innerHTML = unmatched
        .sort((a, b) => b.gap_months - a.gap_months)
        .map(g => {
            const score = g[scoreField] ?? g.closed_eci ?? g.closed_score;
            return `<li><strong>${g.closed_model}</strong>: ${g.gap_months} months old (${scoreName}: ${score?.toFixed(1) || 'N/A'})</li>`;
        })
        .join('');

    // Set minimum bound
    if (minBound) {
        minBound.textContent = estimate.min_current_gap || '--';
    }

    // Set the average gap value in the explainer text
    const explainerAvgGap = document.getElementById('explainer-avg-gap');
    if (explainerAvgGap && stats.avg_horizontal_gap_months !== undefined) {
        explainerAvgGap.textContent = stats.avg_horizontal_gap_months.toFixed(1);
    }

    explainer.classList.remove('hidden');

    // Render distribution chart after panel is visible (Plotly needs visible container)
    requestAnimationFrame(() => {
        renderDistributionChart(estimate);
    });
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
 * Render the trend chart showing pre/post 2025 regression lines
 */
function renderTrendChart(data) {
    const trends = data.trends;
    const models = data.trend_models || data.models; // Fallback only if backend not updated immediately
    const traces = [];
    const scoreField = getScoreField();

    // All models scatter - use score or eci depending on benchmark
    const closedModels = models.filter(m => !m.is_open && m.date && (m[scoreField] || m.eci || m.score));
    const openModels = models.filter(m => m.is_open && m.date && (m[scoreField] || m.eci || m.score));

    const getScore = (m) => m[scoreField] ?? m.eci ?? m.score;

    traces.push({
        x: closedModels.map(m => m.date),
        y: closedModels.map(m => getScore(m)),
        mode: 'markers',
        type: 'scatter',
        name: 'Closed',
        marker: { color: COLORS.closed, size: 8, opacity: 0.6 },
        text: closedModels.map(m => m.display_name),
    });

    traces.push({
        x: openModels.map(m => m.date),
        y: openModels.map(m => getScore(m)),
        mode: 'markers',
        type: 'scatter',
        name: 'Open',
        marker: { color: COLORS.open, size: 8, opacity: 0.6 },
        text: openModels.map(m => m.display_name),
    });

    // Trend Lines and Stats
    ['pre_apr_2024', 'post_apr_2024'].forEach((key, index) => {
        const trend = trends[key];
        if (trend) {
            // Line trace
            traces.push({
                x: [trend.start_point.date, trend.end_point.date],
                y: [trend.start_point.eci, trend.end_point.eci],
                mode: 'lines',
                type: 'scatter',
                name: `${trend.name}`,
                line: {
                    width: 4,
                    dash: key === 'post_apr_2024' ? 'solid' : 'dot',
                    color: key === 'post_apr_2024' ? COLORS.open : COLORS.annotation
                }
            });

            // Stats Annotation
            // Position: Pre-2025 near end of line, Post-2025 near start/middle
            const isPre = key === 'pre_apr_2024';
            const xPos = isPre ? trend.end_point.date : trend.start_point.date;
            const yPos = isPre ? trend.end_point.eci : trend.start_point.eci;

            const metadata = getBenchmarkMetadata();
            const unitName = appState.currentBenchmark === 'eci' ? 'ECI points' : (metadata?.unit || 'points');
            let statsText = `<b>${trend.name} Growth</b><br>+${trend.absolute_growth_per_year} ${unitName}/year`;

            if (!isPre && trends['pre_apr_2024'] && trends['post_apr_2024']) {
                const preRate = trends['pre_apr_2024'].absolute_growth_per_year;
                const postRate = trends['post_apr_2024'].absolute_growth_per_year;
                if (preRate > 0) {
                    const factor = (postRate / preRate).toFixed(1);
                    statsText = `<b>${trend.name} Growth</b><br>+${trend.absolute_growth_per_year} ${unitName}/year<br>${factor}x faster than Pre-Apr 2024`;
                }
            }

            const annotation = {
                x: xPos,
                y: yPos,
                text: statsText,
                showarrow: true,
                arrowhead: 2,
                ax: isPre ? -120 : 120, // Reduced offset for smaller box
                ay: isPre ? -30 : 30,
                bgcolor: 'rgba(255, 255, 255, 0.9)',
                borderpad: 4,
                bordercolor: COLORS.gridline,
                borderwidth: 1,
                align: 'left',
                font: { size: 12, color: COLORS.text }
            };

            // Add to layout annotations if we had access to layout, but here we construct layout below.
            // We can attach it to the traces array as a custom property or just create a separate array.
            // Wait, we can't attach to traces. We need to add to layout.annotations.
            // Let's modify the function structure slightly to accumulate annotations.

            // Temporary storage on the trend object itself to retrieve later? 
            // Better to change the loop structure. See below.
            trend.annotation = annotation;
        }
    });

    // Collect annotations
    const annotations = [];
    ['pre_apr_2024', 'post_apr_2024'].forEach(key => {
        if (trends[key] && trends[key].annotation) {
            annotations.push(trends[key].annotation);
        }
    });

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

    // Y-axis title based on benchmark
    const trendMetadata = getBenchmarkMetadata();
    const yAxisTitle = appState.currentBenchmark === 'eci' ? 'ECI Score' : (trendMetadata?.unit || 'Score');

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
        yaxis: { title: yAxisTitle },
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

    // Attempt to reuse config if possible or redefine
    const config = {
        responsive: true,
        displayModeBar: 'hover',
        modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'],
        displaylogo: false,
    };

    Plotly.newPlot('trend-chart', traces, layout, config).then(function (gd) {
        gd.on('plotly_restyle', function (data) {
            // data typically contains the update object (e.g. {visible: ['legendonly']}) and indices
            // But we can simpler just check the full state of the chart

            const currentTraces = gd.data;
            const newAnnotations = [...layout.annotations]; // Copy existing structure

            // Map known trace names to annotation indices
            // Based on our loop above: 
            // - traces[0]: Closed
            // - traces[1]: Open
            // - traces[2]: Pre-Mar 2024
            // - traces[3]: Post-Mar 2024

            // And annotations array:
            // - annotations[0]: Pre-Mar 2024
            // - annotations[1]: Post-Mar 2024

            // We need to dynamically find the trace index for each trend name
            const preTrendTraceIndex = currentTraces.findIndex(t => t.name === trends['pre_mar_2024']?.name);
            const postTrendTraceIndex = currentTraces.findIndex(t => t.name === trends['post_mar_2024']?.name);

            // Find annotation indices (assuming order matches creation order if we are consistent)
            // But better to check some property? Text content is unique enough or we rely on loop order.
            // Loop order: pre_mar_2024 pushed first, then post_mar_2024.
            const preAnnotationIndex = 0;
            const postAnnotationIndex = 1;

            if (preTrendTraceIndex !== -1 && trends['pre_mar_2024']) {
                const isVisible = currentTraces[preTrendTraceIndex].visible !== 'legendonly';
                if (newAnnotations[preAnnotationIndex]) newAnnotations[preAnnotationIndex].visible = isVisible;
            }

            if (postTrendTraceIndex !== -1 && trends['post_mar_2024']) {
                const isVisible = currentTraces[postTrendTraceIndex].visible !== 'legendonly';
                if (newAnnotations[postAnnotationIndex]) newAnnotations[postAnnotationIndex].visible = isVisible;
            }

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
        // Calculate expected catch-up dates for hover text
        const ciLowMs = (statistics?.ci_90_low || 0) * 30.5 * 24 * 60 * 60 * 1000;
        const ciHighMs = (statistics?.ci_90_high || 0) * 30.5 * 24 * 60 * 60 * 1000;
        const hasCIData = statistics?.ci_90_low !== undefined && statistics?.ci_90_high !== undefined;

        const hoverTexts = unmatchedClosed.map(m => {
            const name = m.display_name || m.model;
            const score = getModelScore(m);
            if (!hasCIData) {
                return `<b>${name}</b><br>${scoreName}: ${score?.toFixed(1) || 'N/A'}<br>Date: ${new Date(m.date).toLocaleDateString()}<br><i>Not yet matched</i>`;
            }
            const releaseDate = new Date(m.date).getTime();
            const expectedLow = new Date(releaseDate + ciLowMs);
            const expectedHigh = new Date(releaseDate + ciHighMs);
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
                `<b>${g.open_model}</b><br>${scoreName}: ${getGapOpenScore(g)?.toFixed(1)} (matched ${getGapClosedScore(g)?.toFixed(1)})<br>Date: %{x}<extra></extra>`
            ),
        });
    }

    // Add horizontal connector lines and annotations for matched gaps
    matchedGaps.forEach(gap => {
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

        // Gap annotation above the line
        annotations.push({
            x: midDate.toISOString().split('T')[0],
            y: closedScore,
            text: `${gap.gap_months} mo`,
            showarrow: false,
            font: {
                size: 11,
                color: COLORS.open,
            },
            yshift: 15,
        });
    });

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
    const chartYAxisTitle = appState.currentBenchmark === 'eci' ? 'ECI Score' : (metadata?.unit || 'Score');

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
            tickformat: '.0f',
            gridcolor: COLORS.gridline,
            zeroline: false,
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
function renderHistoricalChart(data) {
    const historicalGaps = data.historical_gaps || [];
    const labels = getFramingLabels();

    if (historicalGaps.length === 0) {
        document.getElementById('historical-chart').innerHTML =
            '<p style="text-align: center; color: #6b7280; padding: 2rem;">No historical data available.</p>';
        return;
    }

    const stats = data.statistics;
    const traces = [];
    const annotations = [];
    const isCurrentGapMode = appState.gapMetric === 'current';

    // Add 90% CI band FIRST (so it renders behind other elements)
    const startDate = historicalGaps[0]?.date;
    const endDate = historicalGaps[historicalGaps.length - 1]?.date;

    if (stats.ci_90_low !== undefined && stats.ci_90_high !== undefined && startDate && endDate) {
        traces.push({
            x: [startDate, endDate, endDate, startDate],
            y: [stats.ci_90_low, stats.ci_90_low, stats.ci_90_high, stats.ci_90_high],
            fill: 'toself',
            fillcolor: 'rgba(92, 107, 192, 0.2)',
            line: { color: 'rgba(92, 107, 192, 0.5)', width: 1, dash: 'dot' },
            type: 'scatter',
            name: `90% CI (${stats.ci_90_low} - ${stats.ci_90_high} mo)`,
            hoverinfo: 'skip',
            showlegend: true,
        });
    }

    // Add ±1 std dev band around the gap line
    if (stats.std_horizontal_gap !== undefined) {
        const stdDev = stats.std_horizontal_gap;
        const upperBound = historicalGaps.map(g => Math.max(0, g.gap_months + stdDev));
        const lowerBound = historicalGaps.map(g => Math.max(0, g.gap_months - stdDev));
        const dates = historicalGaps.map(g => g.date);

        // Create filled area for ±1 std dev
        traces.push({
            x: [...dates, ...dates.slice().reverse()],
            y: [...upperBound, ...lowerBound.slice().reverse()],
            fill: 'toself',
            fillcolor: 'rgba(92, 107, 192, 0.15)',
            line: { color: 'transparent' },
            type: 'scatter',
            name: `±1 Std Dev (${stdDev.toFixed(1)} mo)`,
            hoverinfo: 'skip',
            showlegend: true,
        });
    }

    // Main gap line
    traces.push({
        x: historicalGaps.map(g => g.date),
        y: historicalGaps.map(g => g.gap_months),
        mode: 'lines+markers',
        type: 'scatter',
        name: 'Gap at Frontier',
        line: { color: COLORS.open, width: 2 },
        marker: { color: COLORS.open, size: 6 },
        hovertemplate: historicalGaps.map(g =>
            `<b>%{x|%b %Y}</b><br>Gap: ${g.gap_months} mo ± ${(stats.std_horizontal_gap || 0).toFixed(1)}<br>` +
            `${labels.openModel} frontier: ${g.open_frontier_model || 'N/A'}<br>` +
            `${labels.closedModel} frontier: ${g.reference_model || 'N/A'}<extra></extra>`
        ),
    });

    // In "current gap" mode, add estimated current gap point and explanation
    if (isCurrentGapMode && stats.current_gap_estimate) {
        const estimate = stats.current_gap_estimate;
        const lastGap = historicalGaps[historicalGaps.length - 1];
        const today = new Date().toISOString();

        // Add estimated current gap as a separate point
        traces.push({
            x: [today],
            y: [estimate.estimated_current_gap],
            mode: 'markers',
            type: 'scatter',
            name: 'Estimated Current Gap',
            marker: {
                color: COLORS.closed,
                size: 12,
                symbol: 'star',
                line: { width: 2, color: 'white' }
            },
            hovertemplate: `<b>Estimated Current Gap</b><br>` +
                `${estimate.estimated_current_gap} months<br>` +
                `Min bound: ${estimate.min_current_gap} mo<extra></extra>`,
        });

        // Add shaded region showing uncertainty (min bound to estimate)
        if (lastGap) {
            traces.push({
                x: [lastGap.date, today, today, lastGap.date],
                y: [lastGap.gap_months, estimate.min_current_gap, estimate.estimated_current_gap, lastGap.gap_months],
                fill: 'toself',
                fillcolor: 'rgba(229, 57, 53, 0.15)',
                line: { color: 'transparent' },
                type: 'scatter',
                name: 'Uncertainty Range',
                hoverinfo: 'skip',
                showlegend: true,
            });
        }

        // Annotation for the estimate
        annotations.push({
            x: today,
            y: estimate.estimated_current_gap,
            text: `Est: ${estimate.estimated_current_gap} mo`,
            showarrow: true,
            arrowhead: 2,
            ax: -50,
            ay: -25,
            bgcolor: 'rgba(255, 255, 255, 0.9)',
            bordercolor: COLORS.closed,
            borderwidth: 1,
            font: { size: 11, color: COLORS.closed },
        });
    }

    // Add average gap reference line in average mode
    if (!isCurrentGapMode && stats.avg_horizontal_gap_months && startDate && endDate) {
        traces.push({
            x: [startDate, endDate],
            y: [stats.avg_horizontal_gap_months, stats.avg_horizontal_gap_months],
            mode: 'lines',
            type: 'scatter',
            name: `Overall Average (${stats.avg_horizontal_gap_months} mo)`,
            line: { color: COLORS.annotation, width: 2, dash: 'dash' },
            hoverinfo: 'skip',
        });
    }

    const layout = {
        title: '',
        margin: { l: 60, r: 40, t: 20, b: 60 },
        height: 400,
        xaxis: {
            title: 'Date',
            titlefont: { size: 12, color: COLORS.annotation },
            tickfont: { size: 11, color: COLORS.annotation },
            gridcolor: COLORS.gridline,
        },
        yaxis: {
            title: isCurrentGapMode ? 'Gap (Months) - with Current Estimate' : 'Gap (Months)',
            titlefont: { size: 12, color: COLORS.annotation },
            tickfont: { size: 11, color: COLORS.annotation },
            gridcolor: COLORS.gridline,
            rangemode: 'tozero',
        },
        annotations: annotations,
        hovermode: 'x unified',
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        legend: {
            orientation: 'h',
            yanchor: 'bottom',
            y: 1.02,
            xanchor: 'right',
            x: 1,
        },
    };

    const config = {
        responsive: true,
        displayModeBar: 'hover',
        modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'],
        displaylogo: false,
    };

    Plotly.newPlot('historical-chart', traces, layout, config);
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

        row.innerHTML = `
            <td>${model.model}</td>
            <td>${new Date(model.date).toLocaleDateString('en-US', { year: 'numeric', month: 'long' })}</td>
            <td>${scoreValue}</td>
            <td><span class=\"model-type ${typeClass}\">${typeLabel}</span></td>
            <td>${model.organization}</td>
        `;
        tableBody.appendChild(row);
    });
}

