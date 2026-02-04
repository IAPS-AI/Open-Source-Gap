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
-   **Live Data:** Fetches the latest scores daily from Epoch AI.

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
