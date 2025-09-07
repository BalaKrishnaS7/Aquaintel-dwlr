import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from io import StringIO
from datetime import datetime

st.set_page_config(page_title="DWLR Dashboard", layout="wide")

@st.cache_data(show_spinner=False)
def load_data(file):
    df = pd.read_csv(file)
    # Ensure Date is datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date")
    return df

def add_sidebar_filters(df):
    st.sidebar.header("🔎 Filters")
    st.sidebar.caption("Use these controls to filter and explore your data.")
    # Date range
    min_date = pd.to_datetime(df["Date"].min())
    max_date = pd.to_datetime(df["Date"].max())
    start_date, end_date = st.sidebar.date_input(
        "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
    )
    if isinstance(start_date, tuple) or isinstance(end_date, tuple):
        # Handle edge-case streamlit returns tuple
        start_date, end_date = start_date[0], end_date[0]
    mask = (df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))
    df = df.loc[mask].copy()

    # Resampling frequency
    freq = st.sidebar.selectbox("Resample frequency", ["Daily (no resample)", "Weekly", "Monthly"])
    agg_map = {c:"mean" for c in df.columns if c not in ["Date"]}
    if freq == "Weekly":
        df = df.set_index("Date").resample("W").agg(agg_map).reset_index()
    elif freq == "Monthly":
        df = df.set_index("Date").resample("MS").agg(agg_map).reset_index()

    # Column selection
    available_cols = [c for c in df.columns if c != "Date"]
    numeric_cols = [c for c in available_cols if pd.api.types.is_numeric_dtype(df[c])]
    st.sidebar.subheader("Select variables to show")
    y_cols = st.sidebar.multiselect("Numeric columns", numeric_cols, default=numeric_cols[:3])

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

        fig, ax = plt.subplots()
        ax.plot(work["Date"], work[target], label=target)
        ax.plot(work["Date"], work[target + "_roll"], label="Rolling mean (14d)")
        ax.scatter(anomalies["Date"], anomalies[target], marker="o", label="Anomaly")
        ax.set_title(f"Anomalies in {target}")
        ax.set_xlabel("Date")
        ax.set_ylabel(target)
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, clear_figure=True)

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
            Made with ❤️ using Streamlit | <a href="https://github.com/aditya" style="color:gray;">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True
    )

def main():
    local_css()
    hero_banner()
    st.title("💧 DWLR (Water Logger) Dashboard")
    st.caption("Interactive exploration of water-level and quality metrics.")

    # Data source: upload or bundled sample file
    src = st.sidebar.radio("Data Source", ["Use sample (DWLR_Dataset_2023.csv)", "Upload CSV"])
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

    # Data preview
    data_preview(df)
    df = column_mapper(df)
    df_filt, y_cols = add_sidebar_filters(df)
    kpi_cards(df_filt)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Time Series", "🌧️ Rainfall & Missing", "🔗 Correlation", "🔍 Anomalies & Scatter"])

    with tab1:
        time_series_section(df_filt, y_cols)
    with tab2:
        rainfall_bar(df_filt)
        missingness(df_filt)
    with tab3:
        correlation_heatmap(df_filt)
    with tab4:
        scatter_matrix(df_filt)
        anomaly_detection(df_filt)

    st.divider()
    data_download(df_filt)

    with st.expander("About / Tips"):
        st.markdown(
            "- Use **Resample frequency** to aggregate to weekly or monthly means.\n"
            "- **Anomaly Detection** uses a 14-day rolling mean and residual z-scores.\n"
            "- If your schema differs, rename columns to match these names or adjust the code."
        )
    footer()

if __name__ == "__main__":
    main()
