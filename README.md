# Open Source Gap

An interactive visualization of the performance gap between frontier open and closed-source AI models across multiple benchmarks.

## Overview

This application visualizes how long it takes for open-source models to "match" the performance of state-of-the-art closed-source models. It calculates the time gap between a closed model's release and the first open model that subsequently matches or exceeds its score on various benchmarks.

**Supported Benchmarks:**
-   **Epoch Capabilities Index (ECI)** - Comprehensive AI capabilities index from Epoch AI
-   **GPQA Diamond** - Graduate-level science questions (Diamond subset)
-   **MATH Level 5** - Competition mathematics problems (hardest level)
-   **OTIS Mock AIME** - Mock AIME competition problems
-   **SWE-Bench Verified** - Software engineering bug fixing benchmark
-   **SimpleQA Verified** - Simple factual question answering
-   **FrontierMath (Public)** - Frontier-level mathematics problems
-   **Chess Puzzles** - Chess tactical puzzles

**Key Features:**
-   **Multi-Benchmark Support:** Compare gaps across different AI capability dimensions.
-   **Interactive Timeline:** Explore model releases and performance gaps over time.
-   **Frontier Tracking:** Focuses on the "frontier" of AI capabilities.
-   **Statistical Analysis:** Automatically calculates the average gap and confidence intervals.
-   **Threshold-Crossing Analysis:** For each benchmark score threshold, how much later the first open model reached it than the first closed model (methodology ported from [open_closed_gap](https://github.com/htihle/open_closed_gap); published in `data.json` under `threshold_analysis` / `threshold_aggregate`).
-   **Release-Lag Bracket:** The instantaneous gap bracketed by the two closed-frontier releases flanking the open frontier's current level — under-estimate, over-estimate, and an interpolated central estimate (`statistics.current_lag_bracket`).
-   **Trend Gap & Catch-Up Projection:** Per-group frontier trend lines yield a backward-looking trend gap, its velocity (months of gap gained/lost per year), and a clearly labeled forward-looking catch-up projection, each with bootstrap 90% CIs (`trend_gap`).
-   **Live Data:** Fetches the latest scores daily from Epoch AI.

## Methodology Notes

Model releases are discrete, so "the gap" silently picks an estimator; this project publishes three complementary ones in `data.json` (estimator family follows the July 2026 analysis *"Have Chinese AI Models Caught Up to the US Frontier?"*):

1.  **Matching-based (day-by-day):** months since the most recent closed SOTA model the best open model has plausibly caught up to, using Epoch's paired-bootstrap / CI criterion. This drives the headline average.
2.  **Release-lag bracket:** the current gap bracketed by the closed frontier releases just below (over-estimate) and just above (under-estimate) the open frontier's level, with a central estimate interpolated between them.
3.  **Trend-regression gap:** the horizontal offset between per-group frontier trend lines, plus the gap's velocity (shrinking or growing) and a forward-looking catch-up projection. Backward-looking numbers are measurements; the projection is a trend-conditional forecast and is labeled as such.

Bounded 0–100 benchmark scores are fitted and interpolated in **logit space** (raw-space fits compress progress near the floor and ceiling); METR time horizons use log space; ECI is unbounded and fitted linearly. The "90% Range (Daily Gap)" statistic is the 5th–95th percentile spread of the day-by-day gap series — dispersion, not a confidence interval of the mean.

## Setup

### Prerequisites
-   Python 3.10+
-   pip

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/TheoBearman/Open-Source-Gap.git
    cd Open-Source-Gap
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running Locally
To run the Flask application locally:
```bash
python app.py
```
Open [http://localhost:8080](http://localhost:8080) in your browser.

## Deployment

### GitHub Pages (Automated)
This project is configured to deploy automatically to GitHub Pages.
-   **How it works**: A GitHub Actions workflow (`.github/workflows/deploy.yml`) runs daily. It fetches the latest data, builds a static version of the site, and deploys it.
-   **Static Build**: The `build_static.py` script generates a static `data.json` file so the frontend works without a backend server.

**To enable:**
1.  Go to your repository **Settings** > **Pages**.
2.  Under **Build and deployment**, set **Source** to **GitHub Actions**.

## Project Structure
-   `app.py`: Flask backend and core logic for gap calculation.
-   `static/`: CSS and JavaScript files for the frontend.
-   `templates/`: HTML templates.
-   `build_static.py`: Script to generate static files for GitHub Pages.
-   `.github/workflows/`: CI/CD configuration for automated deployment.

## Data Source & Attribution

Data is sourced from [Epoch AI](https://epoch.ai).

Epoch AI’s data is free to use, distribute, and reproduce provided the source and authors are credited under the Creative Commons Attribution license.
 
**Attribution:**
Data is provided by [Epoch AI](https://epoch.ai) and is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

If you use this data or visualization, please credit Epoch AI as the source.
