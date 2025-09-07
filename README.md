# DWLR Dashboard (Streamlit)

A Streamlit app to explore and visualize DWLR sensor data.

## Features
- Load sample dataset or upload your own CSV
- Date-range filter and resampling (weekly/monthly)
- KPI cards (avg water level, temp, total rainfall, avg dissolved oxygen)
- Time-series plots for selected variables
- Rainfall bar chart
- Correlation heatmap
- Pairwise scatter plots
- Simple anomaly detection (rolling mean + z-score)
- Download the filtered dataset

## Expected Columns
- `Date` (YYYY-MM-DD)
- `Water_Level_m`
- `Temperature_C`
- `Rainfall_mm`
- `pH`
- `Dissolved_Oxygen_mg_L`

## How to Run
1. Ensure Python 3.9+ is installed.
2. Create a virtual environment (optional but recommended).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Put `DWLR_Dataset_2023.csv` in the same folder as `app.py` (already included here), or use the **Upload CSV** option.
5. Start the app:
   ```bash
   streamlit run app.py
   ```

The app will open in your browser.