import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from io import StringIO
from datetime import datetime
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="DWLR Dashboard", layout="wide")

@st.cache_data(show_spinner=False)
def load_data(file):
    with st.spinner("Loading data..."):
        df = pd.read_csv(file)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.sort_values("Date")
    return df

def add_sidebar_filters(df):
    # Remove the duplicate header here
    # st.sidebar.markdown("## 🛠️ Controls")  # <-- Remove this line

    # Date range
    min_date = pd.to_datetime(df["Date"].min())
    max_date = pd.to_datetime(df["Date"].max())
    start_date, end_date = st.sidebar.date_input(
        "📅 Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
    )
    if isinstance(start_date, tuple) or isinstance(end_date, tuple):
        start_date, end_date = start_date[0], end_date[0]
    mask = (df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))
    df = df.loc[mask].copy()

    # Resampling frequency
    freq = st.sidebar.selectbox("⏱️ Resample frequency", ["Daily (no resample)", "Weekly", "Monthly"])
    agg_map = {c:"mean" for c in df.columns if c not in ["Date"]}
    if freq == "Weekly":
        df = df.set_index("Date").resample("W").agg(agg_map).reset_index()
    elif freq == "Monthly":
        df = df.set_index("Date").resample("MS").agg(agg_map).reset_index()

    # Column selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Select variables to show")
    available_cols = [c for c in df.columns if c != "Date"]
    numeric_cols = [c for c in available_cols if pd.api.types.is_numeric_dtype(df[c])]
    y_cols = st.sidebar.multiselect("Numeric columns", numeric_cols, default=numeric_cols[:3])

    st.sidebar.markdown("---")
    st.sidebar.info("ℹ️ Use these controls to filter and explore your data. Adjust the date range and resampling to focus your analysis.")

    return df, y_cols

def kpi_cards(df):
    col1, col2, col3, col4 = st.columns(4)
    wl = df["Water_Level_m"].mean() if "Water_Level_m" in df else np.nan
    temp = df["Temperature_C"].mean() if "Temperature_C" in df else np.nan
    rain = df["Rainfall_mm"].sum() if "Rainfall_mm" in df else np.nan
    do = df["Dissolved_Oxygen_mg_L"].mean() if "Dissolved_Oxygen_mg_L" in df else np.nan

    col1.metric("🌊 Avg Water Level (m)", f"{wl:.3f}" if pd.notna(wl) else "—")
    col2.metric("🌡️ Avg Temperature (°C)", f"{temp:.2f}" if pd.notna(temp) else "—")
    col3.metric("🌧️ Total Rainfall (mm)", f"{rain:.1f}" if pd.notna(rain) else "—")
    col4.metric("🫧 Avg Dissolved O₂ (mg/L)", f"{do:.2f}" if pd.notna(do) else "—")

def time_series_section(df, y_cols):
    st.subheader("Time Series")
    if not y_cols:
        st.info("Select at least one numeric column from the sidebar.")
        return
    for col in y_cols:
        fig = px.line(df, x="Date", y=col, title=col)
        st.plotly_chart(fig, use_container_width=True)

def rainfall_bar(df):
    if "Rainfall_mm" not in df:
        return
    st.subheader("Rainfall (Bar)")
    fig = px.bar(df, x="Date", y="Rainfall_mm", labels={"Rainfall_mm": "Rainfall (mm)"}, title="Rainfall Over Time")
    st.plotly_chart(fig, use_container_width=True)

def correlation_heatmap(df):
    st.subheader("Correlation (Pearson)")
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        st.info("No numeric columns to compute correlations.")
        return
    corr = num_df.corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="Blues", title="Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)

def scatter_matrix(df):
    st.subheader("Scatter Matrix")
    cols = st.multiselect(
        "Select variables for scatter matrix:",
        [c for c in df.columns if c not in ["Date"] and pd.api.types.is_numeric_dtype(df[c])],
        default=["Rainfall_mm", "Water_Level_m"] if "Rainfall_mm" in df and "Water_Level_m" in df else []
    )
    if len(cols) >= 2:
        fig = px.scatter_matrix(df, dimensions=cols)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Select at least two variables to plot scatter matrix.")

def anomaly_detection(df):
    st.subheader("Simple Anomaly Detection (Z-score of rolling residuals)")
    target = st.selectbox("Select series for anomaly check", 
                          [c for c in df.columns if c not in ["Date"] and pd.api.types.is_numeric_dtype(df[c])])
    if target:
        work = df[["Date", target]].dropna().copy()
        work[target + "_roll"] = work[target].rolling(window=14, min_periods=7).mean()
        work["resid"] = work[target] - work[target + "_roll"]
        std = work["resid"].rolling(window=14, min_periods=7).std()
        work["z"] = work["resid"] / std
        thresh = st.slider("Z-score threshold", 1.5, 4.0, 3.0, 0.1)
        anomalies = work[np.abs(work["z"]) >= thresh]

        fig = px.line(work, x="Date", y=target, title=f"Anomalies in {target}")
        fig.add_scatter(x=work["Date"], y=work[target + "_roll"], mode="lines", name="Rolling mean (14d)")
        fig.add_scatter(x=anomalies["Date"], y=anomalies[target], mode="markers", name="Anomaly", marker=dict(color="red", size=10))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Anomaly table"):
            st.dataframe(anomalies[["Date", target, "z"]].reset_index(drop=True))

def data_download(df):
    st.subheader("Download filtered data")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, file_name="DWLR_filtered.csv", mime="text/csv")

def missingness(df):
    st.subheader("Missing Data Overview")
    miss = df.isna().sum().reset_index()
    miss.columns = ["Column", "Missing_Count"]
    st.dataframe(miss, use_container_width=True)

def data_preview(df):
    st.subheader("Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

def column_mapper(df):
    st.sidebar.subheader("Column Mapper")
    expected = ["Date", "Water_Level_m", "Temperature_C", "Rainfall_mm", "Dissolved_Oxygen_mg_L"]
    mapping = {}
    for col in expected:
        options = [None] + list(df.columns)
        selected = st.sidebar.selectbox(f"Map '{col}' to:", options, index=options.index(col) if col in df.columns else 0)
        if selected:
            mapping[col] = selected
    df = df.rename(columns={v: k for k, v in mapping.items() if v})
    return df

def local_css():
    st.markdown("""
        <style>
        .stMetric { font-size: 1.3rem; font-weight: bold; }
        .block-container { padding-top: 2rem; }
        .stButton>button { background-color: #0099ff; color: white; border-radius: 8px; }
        .stTabs [data-baseweb="tab"] { font-size: 1.1rem; }
        </style>
    """, unsafe_allow_html=True)

def hero_banner():
    st.markdown(
        """
        <div style="background: linear-gradient(90deg, #0099ff 0%, #00cc99 100%); padding: 2rem 1rem; border-radius: 12px; margin-bottom: 2rem;">
            <h1 style="color: white; margin-bottom: 0.5rem;">💧 DWLR Water Logger Dashboard</h1>
            <p style="color: white; font-size: 1.2rem;">
                Explore water-level and quality metrics interactively.<br>
                <span style="font-size: 1rem;">Powered by Streamlit</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

def footer():
    st.markdown(
        """
        <hr style="margin-top:2rem;">
        <div style="text-align:center; color:gray; font-size:0.9rem;">
            Made with ❤️ using Streamlit | <a href="#" style="color:gray;">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True
    )


def dataset_info(df):
    st.write(f"**Dataset shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")

def low_water_alert(df):
    if "Water_Level_m" in df and df["Water_Level_m"].min() < 1.5:
        st.warning("⚠️ Water level is below safe threshold! Consider reducing extraction.", icon="💧")

def historical_comparison(df):
    st.subheader("Historical Water Level Comparison")
    if "Date" in df and "Water_Level_m" in df:
        df_year = df.dropna(subset=["Date", "Water_Level_m"]).copy()
        if df_year["Date"].dt.year.nunique() > 1:
            df_year["Year"] = df_year["Date"].dt.year
            yearly = df_year.groupby("Year")["Water_Level_m"].mean().reset_index()
            fig = px.bar(yearly, x="Year", y="Water_Level_m", title="Average Water Level by Year",
                         labels={"Water_Level_m": "Avg Water Level (m)"})
        else:
            df_year["Month"] = df_year["Date"].dt.strftime("%b")
            monthly = df_year.groupby("Month")["Water_Level_m"].mean().reset_index()
            fig = px.bar(monthly, x="Month", y="Water_Level_m", title="Average Water Level by Month",
                         labels={"Water_Level_m": "Avg Water Level (m)"})
        st.plotly_chart(fig, use_container_width=True)

def illegal_extraction_detection(df):
    st.subheader("Illegal Extraction Detection")
    if "Water_Level_m" in df:
        df["diff"] = df["Water_Level_m"].diff()
        suspicious = df[df["diff"] < -2]  # Threshold for sudden drop
        if not suspicious.empty:
            st.error(f"🚨 {len(suspicious)} possible illegal extraction events detected!")
            st.dataframe(suspicious[["Date", "Water_Level_m", "diff"]])
        else:
            st.success("No suspicious extraction events detected.")

def data_quality_metrics(df):
    st.subheader("Data Quality Metrics")
    miss = df.isna().sum()
    total = len(df)
    st.write(f"**Missing values:**")
    for col, val in miss.items():
        st.write(f"- {col}: {val} ({val/total:.1%})")

def future_prediction(df):
    st.subheader("Future Water Level Prediction (Simple Linear Regression)")
    if "Date" in df and "Water_Level_m" in df:
        df_pred = df.dropna(subset=["Water_Level_m"])
        df_pred["ordinal_date"] = df_pred["Date"].map(datetime.toordinal)
        X = df_pred[["ordinal_date"]]
        y = df_pred["Water_Level_m"]
        if len(df_pred) > 10:
            model = LinearRegression()
            model.fit(X, y)
            future_days = 30
            last_date = df_pred["Date"].max()
            future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, future_days+1)]
            future_ord = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
            preds = model.predict(future_ord)
            pred_df = pd.DataFrame({"Date": future_dates, "Predicted_Water_Level_m": preds})
            fig = px.line(df_pred, x="Date", y="Water_Level_m", title="Water Level & Prediction")
            fig.add_scatter(x=pred_df["Date"], y=pred_df["Predicted_Water_Level_m"], mode="lines", name="Prediction", line=dict(dash="dot", color="red"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for prediction.")

def government_notifications(df):
    st.subheader("Government Notifications")
    if "Water_Level_m" in df and df["Water_Level_m"].min() < 1.0:
        st.error("🚨 Water level critically low! Immediate action required.")
    # Add more rules as needed

def transparency_panel(df):
    st.subheader("Transparency Panel")
    st.write("Summary of water usage, anomalies, and alerts for citizens:")
    st.write(f"- **Lowest water level:** {df['Water_Level_m'].min():.2f} m")
    st.write(f"- **Total rainfall:** {df['Rainfall_mm'].sum():.1f} mm")
    st.write(f"- **Anomalies detected:** {df['Water_Level_m'].isna().sum()} missing values")

def farmer_view(df, df_filt, y_cols):
    st.header("👩‍🌾 Farmer Dashboard")
    st.info("Get alerts on low water, cost-saving tips, and pump usage insights.", icon="🚜")
    kpi_cards(df_filt)
    low_water_alert(df_filt)
    st.subheader("Water Level Over Time")
    if "Water_Level_m" in y_cols:
        fig = px.line(df_filt, x="Date", y="Water_Level_m", title="Water Level (m)")
        st.plotly_chart(fig, use_container_width=True)
    historical_comparison(df_filt)
    st.subheader("Pump Usage Tips")
    st.markdown("- **Save costs:** Run pumps only when water level is above safe threshold.")
    st.markdown("- **Alert:** If water level drops below 1.5m, consider reducing extraction.")
    future_prediction(df_filt)
    transparency_panel(df_filt)

def researcher_view(df, df_filt, y_cols):
    st.header("🔬 Researcher Dashboard")
    st.info("Analyze anomalies, correlations, and download full datasets.", icon="📊")
    kpi_cards(df_filt)
    tab1, tab2, tab3, tab4 = st.tabs(["Time Series", "Correlation", "Anomalies", "Quality & Extraction"])
    with tab1:
        time_series_section(df_filt, y_cols)
        future_prediction(df_filt)
    with tab2:
        correlation_heatmap(df_filt)
        scatter_matrix(df_filt)
    with tab3:
        anomaly_detection(df_filt)
        missingness(df_filt)
    with tab4:
        data_quality_metrics(df_filt)
        illegal_extraction_detection(df_filt)
        government_notifications(df_filt)
    data_download(df_filt)

def main():
    local_css()
    hero_banner()
    st.caption("Interactive exploration of water-level and quality metrics.")

    # Sidebar: User type selection at the top
    st.sidebar.markdown("## 👤 Dashboard View")
    user_type = st.sidebar.selectbox("Choose your dashboard view", ["Farmer", "Researcher"])

    # Sidebar: Data source selection
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📂 Data Source")
    src = st.sidebar.radio("Select data source", ["Use sample (DWLR_Dataset_2023.csv)", "Upload CSV"])
    if src == "Upload CSV":
        uploaded = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded is None:
            st.warning("Please upload a CSV to continue, or switch to the sample dataset.")
            st.stop()
        df = load_data(uploaded)
    else:
        df = load_data("DWLR_Dataset_2023.csv")

    # Basic checks
    if "Date" not in df.columns:
        st.error("The dataset must contain a 'Date' column.")
        st.stop()

    # Sidebar: Column mapping
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🗂️ Column Mapper")
    df = column_mapper(df)

    # Data preview and info
    data_preview(df)
    dataset_info(df)

    # Sidebar: Filters (date, resample, variables)
    # The controls are now grouped under the previous sections
    df_filt, y_cols = add_sidebar_filters(df)

    # Main dashboard view
    if user_type == "Farmer":
        farmer_view(df, df_filt, y_cols)
    else:
        researcher_view(df, df_filt, y_cols)

    st.divider()
    with st.expander("About / Tips"):
        st.markdown(
            "- Use **Resample frequency** to aggregate to weekly or monthly means.\n"
            "- **Anomaly Detection** uses a 14-day rolling mean and residual z-scores.\n"
            "- If your schema differs, rename columns to match these names or adjust the code."
        )
    footer()

if __name__ == "__main__":
    main()
