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
        const url = 'data.json';
        console.log(`Fetching data from: ${url}`);
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error: ${response.status} ${response.statusText} at ${response.url}`);
        }
        const data = await response.json();

        // Hide loading indicator
        document.getElementById('loading').classList.add('hidden');

        // Render chart and update UI
        if (data) {
            renderChart(data);
            renderTrendChart(data);
            updateStats(data.statistics);
            updateTitle(data.statistics.avg_horizontal_gap_months);
            updateLastUpdated(data.last_updated);
            renderTable(data.trend_models || data.models); // Use all models for table
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
 * Render the trend chart showing pre/post 2025 regression lines
 */
function renderTrendChart(data) {
    const trends = data.trends;
    const models = data.trend_models || data.models; // Fallback only if backend not updated immediately
    const traces = [];

    // All models scatter
    const closedModels = models.filter(m => !m.is_open && m.date && m.eci);
    const openModels = models.filter(m => m.is_open && m.date && m.eci);

    traces.push({
        x: closedModels.map(m => m.date),
        y: closedModels.map(m => m.eci),
        mode: 'markers',
        type: 'scatter',
        name: 'Closed',
        marker: { color: COLORS.closed, size: 8, opacity: 0.6 },
        text: closedModels.map(m => m.display_name),
    });

    traces.push({
        x: openModels.map(m => m.date),
        y: openModels.map(m => m.eci),
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

            let statsText = `<b>${trend.name} Growth</b><br>+${trend.absolute_growth_per_year} ECI points/year<br>${trend.percentage_growth_annualized}% per year`;

            if (!isPre && trends['pre_apr_2024'] && trends['post_apr_2024']) {
                const preRate = trends['pre_apr_2024'].absolute_growth_per_year;
                const postRate = trends['post_apr_2024'].absolute_growth_per_year;
                if (preRate > 0) {
                    const factor = (postRate / preRate).toFixed(1);
                    statsText = `<b>${trend.name} Growth</b><br>+${trend.absolute_growth_per_year} ECI points/year<br>${factor}x faster than Pre-Apr 2024`;
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

    const layout = {
        title: { text: '', font: { size: 16 } },
        margin: { l: 60, r: 60, t: 40, b: 60 },
        height: 500,
        xaxis: { title: 'Date' },
        yaxis: { title: 'ECI Score' },
        annotations: [
            ...annotations,
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
        ],
        shapes: [
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
        ],
        hovermode: 'closest',
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
    };

    // Attempt to reuse config if possible or redefine
    const config = {
        responsive: true,
        displayModeBar: true,
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

    if (matchedGaps.length > 0) {
        traces.push({
            x: matchedGaps.map(g => g.open_date),
            y: matchedGaps.map(g => g.closed_eci),
            mode: 'markers',
            type: 'scatter',
            name: 'Open model',
            marker: {
                color: COLORS.open,
                size: 10,
                symbol: 'square',
            },
            hovertemplate: matchedGaps.map(g =>
                `<b>${g.open_model}</b><br>ECI: ${g.open_eci.toFixed(1)} (≥ ${g.closed_eci.toFixed(1)})<br>Date: %{x}<extra></extra>`
            ),
        });
    }

    // Add horizontal connector lines and annotations for matched gaps
    matchedGaps.forEach(gap => {
        const closedDate = new Date(gap.closed_date);
        const openDate = new Date(gap.open_date);
        const midDate = new Date((closedDate.getTime() + openDate.getTime()) / 2);

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
            },
        });

        // Gap annotation above the line
        annotations.push({
            x: midDate.toISOString().split('T')[0],
            y: gap.closed_eci,
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
    // Base shapes
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

    // Labels for reference lines
    annotations.push(
        {
            x: 1,
            y: 130,
            xref: 'paper',
            yref: 'y',
            text: 'Sonnet (130)',
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
            <td>${model.model}</td>
            <td>${new Date(model.date).toLocaleDateString('en-US', { year: 'numeric', month: 'long' })}</td>
            <td>${eciValue}</td>
            <td><span class=\"model-type ${typeClass}\">${typeLabel}</span></td>
            <td>${model.organization}</td>
        `;
        tableBody.appendChild(row);
    });
}

