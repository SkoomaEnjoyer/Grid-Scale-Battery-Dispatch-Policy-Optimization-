# Monte Carlo Simulation for Electricity Market Analysis

## Overview

This project implements a comprehensive Monte Carlo simulation framework to analyze electricity market trends, price dynamics, and seasonal patterns. By leveraging ARIMA methodologies, such as differencing, auto-regression, and ACF/PACF analysis, along with a nested Monte Carlo simulation, the framework provides a robust analytical tool for energy market participants.

## Features

- **Data Preparation**: Cleans and preprocesses electricity generation and price data, with functionality to handle special days (e.g., holidays) and seasonal variations.
- **ARIMA Methodologies**: Utilizes ARIMA concepts like differencing and residual analysis for time-series data modeling and preparation.
- **Nested Monte Carlo Simulation**: Simulates price variations under different scenarios, such as "volatile prices" or "increased demand," by iteratively building and refining simulation layers.
- **Data Visualization**: Offers various plotting tools, including violin plots, heatmaps, and seasonal decomposition plots, to visualize trends and seasonal components.
- **Extreme Value Analysis**: Identifies and removes extreme outliers using statistical methods like Modified Z-Scores and IQR.
- **Distribution Fitting**: Tests and identifies the best-fitting probability distributions for energy price data.
- **Advanced Analytics**:
  - Aggregated data insights (hourly, daily, and monthly levels).
  - Autocorrelation and Partial Autocorrelation (ACF/PACF) analysis.
  - Seasonal decomposition using STL for trend analysis.

## Prerequisites

- Python 3.x
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`, `sklearn`, `hijri_converter`, `fitter`, `requests`, `BeautifulSoup`

## How to Use

1. **Data Preparation**:
   - Place the input Excel file (`Wind_data_dolar.xlsx`) containing the raw data in the specified directory.
   - The script processes and structures the data for analysis, excluding holidays and special events.

2. **Simulation and Methodologies**:
   - Apply ARIMA methodologies for data preparation, such as differencing and residual analysis.
   - Use the nested Monte Carlo simulation to explore various scenarios and refine insights iteratively.

3. **Visualization**:
   - Generate violin plots, heatmaps, and line graphs to analyze trends and seasonal patterns.
   - Use aggregated data to understand price behavior across different time granularities (hourly, daily, monthly).

4. **Advanced Analysis**:
   - Perform ACF/PACF analysis to examine time-series dependencies.
   - Apply seasonal decomposition to investigate trends and residuals.

## Key Outputs

- **Simulation Results**: Exported as an Excel file (`simulation_results.xlsx`) with scenario-based insights.
- **Visual Insights**: A suite of plots for trend and pattern analysis.
- **Statistical Reports**: Best-fitting probability distributions and ACF/PACF insights.

## Future Work

- Integration with real-time API data sources.
- Expansion to support additional energy markets (e.g., natural gas).
- Implementation of real-time dashboards using `Dash` or `Streamlit`.

## License

This project is open-source and available under the MIT License.

---

Let me know if you'd like any additional edits or refinements!
