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

/**
 * Fetch data from the API and render the chart
 */
async function init() {
    try {
        const response = await fetch('/api/data');
        if (!response.ok) {
            throw new Error(`HTTP error: ${response.status}`);
        }
        const data = await response.json();

        // Hide loading indicator
        document.getElementById('loading').classList.add('hidden');

        // Render chart and update UI
        renderChart(data);
        updateStats(data.statistics);
        updateTitle(data.statistics.avg_horizontal_gap_months);
        updateLastUpdated(data.last_updated);
        renderTable(data.models);

    } catch (error) {
        console.error('Failed to load data:', error);
        document.getElementById('loading').innerHTML = `
            <p style="color: #e53935;">Failed to load data. Please try refreshing.</p>
        `;
    }
}

/**
 * Render the main chart using Plotly.js
 * Style: Only show closed models and the specific open models that matched them
 */
function renderChart(data) {
    const { models, gaps } = data;

    // Create traces
    const traces = [];
    const annotations = [];
    const shapes = [];

    // Get closed models from gaps data
    const closedModels = models.filter(m => !m.is_open && m.date && m.eci);

    // Matched closed models (solid red circles)
    const matchedClosed = closedModels.filter(m =>
        gaps.some(g => g.closed_model === m.model && g.matched)
    );

    if (matchedClosed.length > 0) {
        traces.push({
            x: matchedClosed.map(m => m.date),
            y: matchedClosed.map(m => m.eci),
            mode: 'markers',
            type: 'scatter',
            name: 'Closed model',
            marker: {
                color: COLORS.closed,
                size: 12,
                symbol: 'circle',
            },
            hovertemplate: '<b>%{text}</b><br>ECI: %{y:.1f}<br>Date: %{x}<extra></extra>',
            text: matchedClosed.map(m => m.display_name || m.model),
        });
    }

    // Unmatched closed models (dashed outline)
    const unmatchedClosed = closedModels.filter(m =>
        gaps.some(g => g.closed_model === m.model && !g.matched)
    );

    if (unmatchedClosed.length > 0) {
        traces.push({
            x: unmatchedClosed.map(m => m.date),
            y: unmatchedClosed.map(m => m.eci),
            mode: 'markers',
            type: 'scatter',
            name: 'Closed (unmatched)',
            marker: {
                color: COLORS.closedUnmatched,
                size: 12,
                symbol: 'circle-open',
                line: { width: 2 },
            },
            hovertemplate: '<b>%{text}</b><br>ECI: %{y:.1f}<br>Date: %{x}<br><i>Not yet matched</i><extra></extra>',
            text: unmatchedClosed.map(m => m.display_name || m.model),
        });
    }

    // Only show open models that matched a closed model (blue squares)
    // Position them at the END of the connector line (at closed model's ECI level)
    const matchedGaps = gaps.filter(g => g.matched);

    // Separate exact and statistical matches for different hover templates
    const exactMatches = matchedGaps.filter(g => g.match_type === 'exact');
    const statisticalMatches = matchedGaps.filter(g => g.match_type === 'statistical');

    if (exactMatches.length > 0) {
        traces.push({
            x: exactMatches.map(g => g.open_date),
            y: exactMatches.map(g => g.closed_eci),
            mode: 'markers',
            type: 'scatter',
            name: 'Open model (exact)',
            marker: {
                color: COLORS.open,
                size: 10,
                symbol: 'square',
            },
            hovertemplate: exactMatches.map(g =>
                `<b>${g.open_model}</b><br>ECI: ${g.open_eci.toFixed(1)} (≥ ${g.closed_eci.toFixed(1)})<br>Date: %{x}<extra></extra>`
            ),
        });
    }

    if (statisticalMatches.length > 0) {
        traces.push({
            x: statisticalMatches.map(g => g.open_date),
            y: statisticalMatches.map(g => g.closed_eci),
            mode: 'markers',
            type: 'scatter',
            name: 'Open model (statistical)',
            marker: {
                color: COLORS.open,
                size: 10,
                symbol: 'square',
            },
            hovertemplate: statisticalMatches.map(g =>
                `<b>${g.open_model}</b><br>ECI: ${g.open_eci.toFixed(1)} (≈ ${g.closed_eci.toFixed(1)}, not statistically different)<br>Date: %{x}<extra></extra>`
            ),
        });
    }

    // Add horizontal connector lines and annotations for matched gaps
    matchedGaps.forEach(gap => {
        const closedDate = new Date(gap.closed_date);
        const openDate = new Date(gap.open_date);
        const midDate = new Date((closedDate.getTime() + openDate.getTime()) / 2);

        // Use dashed line for statistical matches, solid for exact matches
        const isStatistical = gap.match_type === 'statistical';

        // Horizontal connector line
        shapes.push({
            type: 'line',
            x0: gap.closed_date,
            x1: gap.open_date,
            y0: gap.closed_eci,
            y1: gap.closed_eci,
            line: {
                color: COLORS.connector,
                width: 2,
                dash: isStatistical ? 'dash' : 'solid',
            },
        });

        // Gap annotation above the line (add ~ for statistical matches)
        const gapLabel = isStatistical ? `~${gap.gap_months} mo` : `${gap.gap_months} mo`;
        annotations.push({
            x: midDate.toISOString().split('T')[0],
            y: gap.closed_eci,
            text: gapLabel,
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
    gaps.filter(g => !g.matched).forEach(gap => {
        shapes.push({
            type: 'line',
            x0: gap.closed_date,
            x1: today,
            y0: gap.closed_eci,
            y1: gap.closed_eci,
            line: {
                color: COLORS.closedUnmatched,
                width: 2,
                dash: 'dot',
            },
        });

        // Annotation for unmatched
        annotations.push({
            x: today,
            y: gap.closed_eci,
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

    // Layout
    const layout = {
        showlegend: false,
        margin: { l: 60, r: 100, t: 20, b: 60 },
        xaxis: {
            title: 'Model release date',
            titlefont: { size: 12, color: COLORS.annotation },
            tickfont: { size: 11, color: COLORS.annotation },
            gridcolor: COLORS.gridline,
            zeroline: false,
        },
        yaxis: {
            title: 'ECI Score',
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
        displayModeBar: true,
        modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'],
        displaylogo: false,
    };

    Plotly.newPlot('chart', traces, layout, config);
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
    const formatted = date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
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

    // Sort models by date descending
    const sortedModels = [...models].sort((a, b) => new Date(b.date) - new Date(a.date));

    sortedModels.forEach(model => {
        const row = document.createElement('tr');

        const typeClass = model.is_open ? 'type-open' : 'type-closed';
        const typeLabel = model.is_open ? 'Open' : 'Closed';
        const eciValue = model.eci !== null ? model.eci.toFixed(1) : '-';

        row.innerHTML = `
            <td>${model.display_name}</td>
            <td>${model.date}</td>
            <td>${eciValue}</td>
            <td><span class=\"model-type ${typeClass}\">${typeLabel}</span></td>
            <td>${model.organization}</td>
        `;
        tableBody.appendChild(row);
    });
}

