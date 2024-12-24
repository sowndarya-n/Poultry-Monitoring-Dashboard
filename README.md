# Poultry Monitoring Dashboard

## Overview
The Poultry Monitoring Dashboard was developed as part of a practical exercise to analyze and visualize poultry production metrics using a simulated dataset. The goal was to explore the relationship between environmental factors, growth rates, and welfare scores, and to identify areas for operational improvement.


## Features
### 1. Interactive Dashboard
- **Scatter Plot**: Compare predicted vs. actual growth rates and identify outliers.
- **Donut Chart**: Visualize the proportion of outliers vs. normal data points.
- **Bar Chart**: Residual analysis to identify underperforming farms.
- **Summary Cards**: Highlight key metrics like average growth rate, Mean Squared Error (MSE), and outlier count.

### 2. Predictive Modeling
- Built using **Random Forest** and **Gradient Boosting** regression models.
- Provides:
  - Accurate growth rate predictions.
  - Residual analysis to highlight anomalies in data.

### 3. Insights and Recommendations
- Key Insights:
  - Environmental factors (e.g., temperature) significantly affect growth rates.
  - Outliers reveal potential operational or environmental issues.
- Recommendations:
  - Improve monitoring systems.
  - Address outliers to enhance productivity and welfare.

## Tech Stack
- **Python**:
  - Libraries: Pandas, NumPy, Scikit-learn, Matplotlib.
  - Use: Data preprocessing, predictive modeling, and analysis.
- **PowerBI**:
  - Developed an interactive dashboard for data visualization.
- **Mock Dataset**:
  - Simulates typical poultry production metrics, including environmental and growth data.

## How to Clone and Set Up
### 1. Clone the Repository
- Open your terminal or command prompt and run the following command:
  ```bash
  git clone https://github.com/sowndarya-n/Poultry-Monitoring-Dashboard.git
  cd poultry-monitoring-dashboard

### 2. Set Up a Virtual Environment
- Create a virtual environment:
  ```bash
  python -m venv venv

- Activate the virtual environment:
    - On macOS/Linux:
      ```bash
      source venv/bin/activate
    - On Windows:
      ```bash
      venv\Scripts\activate
      
### 3. Install Dependencies
- Install the required Python libraries:
   ```bash
    pip install -r requirements.txt

### 4. Run Predictive Modeling
- Execute the predictive modeling script to generate predictions and analyze residuals:
   ```bash
    python scripts/predictive_modeling.py
- The output files will be saved in the data/ folder.

### 5. Open the Dashboard
- Load the visuals/poultry_dashboard.pbix file in PowerBI.
- Use interactive slicers to explore data trends and analyze farm-specific performance.


### 6. Future Enhancements
- Integration with Real-World Data: Replace mock data with actual datasets for improved insights.
- Real-Time Monitoring: Incorporate IoT sensors for live tracking of environmental conditions and poultry behavior.
- Advanced Analytics: Use neural networks or time-series models for long-term predictions.
- Additional Metrics: Include feed quality, water consumption, and disease trends for a holistic analysis.
