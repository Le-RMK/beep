import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh  # ✅ auto-refresh

# --- Page Config ---
st.set_page_config(page_title="Carbon Capture Dashboard", layout="wide")
st.title("TRACE's MRV Platform")

# --- Sidebar Controls ---
st.sidebar.header("Controls")
refresh_rate = st.sidebar.slider("Refresh rate (seconds)", 1, 10, 2)
show_history = st.sidebar.checkbox("Show historical graphs", True)

# --- Auto-refresh ---
st_autorefresh(interval=refresh_rate*1000, key="dashboard_autorefresh")

# --- Session State ---
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["Time", "CO2 Rate", "Pressure", "Temperature", "Cumulative CO2"])

# --- Live Data Simulation ---
def get_new_data():
    timestamp = pd.Timestamp.now()
    co2 = np.random.uniform(800, 1200)
    pressure = np.random.uniform(50, 110)
    temp = np.random.uniform(30, 110)
    cumulative = st.session_state.data["Cumulative CO2"].iloc[-1] + co2*0.001 if not st.session_state.data.empty else co2*0.001
    return {"Time": timestamp, "CO2 Rate": co2, "Pressure": pressure, "Temperature": temp, "Cumulative CO2": cumulative}

# --- System Schematic ---
def draw_system_schematic(co2_rate, pressure, temperature):
    fig = go.Figure()
    source_color = "green" if co2_rate < 1100 else "red"
    tank_color = "green" if co2_rate < 1100 else "yellow"
    pipeline_color = "green" if 70 <= pressure <= 90 else "red"
    well_color = "green"
    if pressure > 100 or temperature > 90:
        well_color = "red"
    elif pressure < 70 or temperature < 50:
        well_color = "yellow"

    fig.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1, line=dict(color="black"), fillcolor=source_color)
    fig.add_annotation(x=0.5, y=0.5, text="CO₂ Source", showarrow=False, font=dict(color="white"))
    fig.add_shape(type="rect", x0=2, y0=0, x1=3, y1=1, line=dict(color="black"), fillcolor=tank_color)
    fig.add_annotation(x=2.5, y=0.5, text="Tank", showarrow=False, font=dict(color="white"))
    fig.add_shape(type="rect", x0=4, y0=0.4, x1=6, y1=0.6, line=dict(color="black"), fillcolor=pipeline_color)
    fig.add_annotation(x=5, y=0.5, text="Pipeline", showarrow=False, font=dict(color="white"))
    fig.add_shape(type="rect", x0=7, y0=0, x1=8, y1=1, line=dict(color="black"), fillcolor=well_color)
    fig.add_annotation(x=7.5, y=0.5, text="Injection Well", showarrow=False, font=dict(color="white"))
    fig.update_xaxes(showticklabels=False, range=[-0.5,8.5])
    fig.update_yaxes(showticklabels=False, range=[-0.5,1.5])
    fig.update_layout(width=800, height=200, margin=dict(l=10, r=10, t=10, b=10))
    return fig

# --- Tabs ---
tab_overview, tab_well, tab_ai = st.tabs(["Overview", "Injection Well", "AI Predictions"])

# --- Add New Data ---
new_row = get_new_data()
st.session_state.data = pd.concat([st.session_state.data, pd.DataFrame([new_row])], ignore_index=True).tail(1000)
data = st.session_state.data

# --- Calculate Alerts ---
recent_co2 = data["CO2 Rate"].iloc[-5:]
mean_co2, std_co2 = recent_co2.mean(), recent_co2.std()
alerts = {"CO2": "✅ Normal", "Pressure": "✅ Normal", "Temperature": "✅ Normal"}
if abs(new_row["CO2 Rate"] - mean_co2) > 2*std_co2:
    alerts["CO2"] = "⚠️ CO₂ anomaly detected"
elif len(recent_co2) > 1 and new_row["CO2 Rate"] < 0.8*recent_co2.iloc[-2]:
    alerts["CO2"] = "🔻 CO₂ drop detected"
if new_row["Pressure"] > 100:
    alerts["Pressure"] = "⚠️ High Pressure"
elif new_row["Pressure"] < 70:
    alerts["Pressure"] = "🔻 Low Pressure"
if new_row["Temperature"] > 90:
    alerts["Temperature"] = "🌡 High Temperature"
elif new_row["Temperature"] < 50:
    alerts["Temperature"] = "❄ Low Temperature"

# --- Compact Overview Tab ---
with tab_overview:
    # Row 1: Alerts + Metrics
    row1_cols = st.columns([1,1,1,1,1])
    co2_status = row1_cols[0].empty()
    pressure_status = row1_cols[1].empty()
    temp_status = row1_cols[2].empty()
    overview_co2_metric = row1_cols[3].empty()
    overview_cum_metric = row1_cols[4].empty()
    co2_status.metric("CO₂ Status", alerts["CO2"])
    pressure_status.metric("Pressure Status", alerts["Pressure"])
    temp_status.metric("Temperature Status", alerts["Temperature"])
    overview_co2_metric.metric("Current CO₂ Rate (kg/hr)", f"{new_row['CO2 Rate']:.1f}")
    overview_cum_metric.metric("Cumulative CO₂ Stored (tons)", f"{new_row['Cumulative CO2']:.2f}")

    # Row 2: Historical Graphs
    row2_cols = st.columns(2)
    overview_co2_chart = row2_cols[0].empty()
    overview_cum_chart = row2_cols[1].empty()
    if show_history:
        fig_co2 = px.line(data, x="Time", y="CO2 Rate", markers=True, title="CO₂ Capture Rate Over Time")
        anomalies = data["CO2 Rate"] > mean_co2 + 2*std_co2
        if anomalies.any():
            fig_co2.add_scatter(x=data["Time"][anomalies], y=data["CO2 Rate"][anomalies],
                                mode="markers", marker=dict(color="red", size=10), name="Anomaly")
        fig_cum = px.line(data, x="Time", y="Cumulative CO2", markers=True, title="Cumulative CO₂ Stored")
        overview_co2_chart.plotly_chart(fig_co2, use_container_width=True)
        overview_cum_chart.plotly_chart(fig_cum, use_container_width=True)
    else:
        overview_co2_chart.empty()
        overview_cum_chart.empty()

# --- Compact Injection Well Tab ---
with tab_well:
    row1_cols = st.columns([1,1,1,1,1])
    well_co2_status = row1_cols[0].empty()
    well_pressure_status = row1_cols[1].empty()
    well_temp_status = row1_cols[2].empty()
    well_pressure_metric = row1_cols[3].empty()
    well_temp_metric = row1_cols[4].empty()
    well_co2_status.metric("CO₂ Status", alerts["CO2"])
    well_pressure_status.metric("Pressure Status", alerts["Pressure"])
    well_temp_status.metric("Temperature Status", alerts["Temperature"])
    well_pressure_metric.metric("Pressure (bar)", f"{new_row['Pressure']:.1f}")
    well_temp_metric.metric("Temperature (°C)", f"{new_row['Temperature']:.1f}")

    row2_cols = st.columns(2)
    well_pressure_chart = row2_cols[0].empty()
    well_temp_chart = row2_cols[1].empty()
    if show_history:
        fig_pressure = px.line(data, x="Time", y="Pressure", markers=True, title="Injection Well Pressure")
        fig_temp = px.line(data, x="Time", y="Temperature", markers=True, title="Injection Well Temperature")
        
        # High and Low Pressure
        fig_pressure.add_scatter(x=data["Time"][data["Pressure"]>100], y=data["Pressure"][data["Pressure"]>100],
                                 mode='markers', marker=dict(color='red', size=10), name="High Pressure")
        fig_pressure.add_scatter(x=data["Time"][data["Pressure"]<70], y=data["Pressure"][data["Pressure"]<70],
                                 mode='markers', marker=dict(color='green', size=10), name="Low Pressure")
        
        # High and Low Temperature
        fig_temp.add_scatter(x=data["Time"][data["Temperature"]>90], y=data["Temperature"][data["Temperature"]>90],
                             mode='markers', marker=dict(color='red', size=10), name="High Temp")
        fig_temp.add_scatter(x=data["Time"][data["Temperature"]<50], y=data["Temperature"][data["Temperature"]<50],
                             mode='markers', marker=dict(color='green', size=10), name="Low Temp")
        
        well_pressure_chart.plotly_chart(fig_pressure, use_container_width=True)
        well_temp_chart.plotly_chart(fig_temp, use_container_width=True)
    else:
        well_pressure_chart.empty()
        well_temp_chart.empty()

# --- AI Predictions Tab (reverted to original) ---
with tab_ai:
    ai_prediction_metric = tab_ai.empty()
    ai_simulation_container = tab_ai.container()
    tank_box = ai_simulation_container.empty()
    pipeline_box = ai_simulation_container.empty()
    well_box = ai_simulation_container.empty()

    if len(data) > 5:
        # Predicted CO2 Rate
        predicted_rate = LinearRegression().fit(np.arange(len(data)).reshape(-1,1), data['CO2 Rate']).predict([[len(data)+1]])[0]
        ai_prediction_metric.metric("Predicted CO₂ Rate (kg/hr)", f"{predicted_rate:.1f}")

        # System schematic
        schematic_fig = draw_system_schematic(new_row["CO2 Rate"], new_row["Pressure"], new_row["Temperature"])
        ai_simulation_container.plotly_chart(schematic_fig, use_container_width=True)

       
    else:
        ai_prediction_metric.metric("Predicted CO₂ Rate (kg/hr)", "Collecting data...")
        ai_simulation_container.empty()
        tank_box.empty()
        pipeline_box.empty()
        well_box.empty()
