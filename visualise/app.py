import streamlit as st
import pandas as pd
import plotly.graph_objects as go

### THIS FILE IS AI GENERATED ###

# Load CSV
@st.cache_data
def load_data():
    df = pd.read_csv("../controllers/supervisorGA_starter/data/robot_position_20251109-001347.csv")
    return df

df = load_data()

st.title("🤖 Robot Evolution Visualization")

# --- Sidebar Controls ---
st.sidebar.header("Controls")

# Generation slider
gen_min, gen_max = int(df["generation"].min()), int(df["generation"].max())
selected_gen = st.sidebar.slider("Select Generation", gen_min, gen_max, gen_min)

# Population (individual) selector
populations = sorted(df["population"].unique())
selected_pops = st.sidebar.multiselect(
    "Select Individuals (Population IDs)", populations, default=populations
)

# Downsampling
step = st.sidebar.slider("Render every Nth point", 1, 50, 10)

# Show orientation?
show_orientation = st.sidebar.checkbox("Show orientation vectors", value=True)

# --- Filter Data ---
filtered = df[(df["generation"] == selected_gen) & (df["population"].isin(selected_pops))]
filtered = filtered.iloc[::step]  # downsample

# --- Plot Setup ---
fig = go.Figure()

# Plot points
for pop in selected_pops:
    sub = filtered[filtered["population"] == pop]
    fig.add_trace(
        go.Scatter(
            x=sub["x"],
            y=sub["y"],
            mode="markers",
            marker=dict(size=8, color=sub["fitness"], colorscale="Viridis", showscale=True),
            name=f"Population {pop}"
        )
    )

# Optionally show orientation arrows
if show_orientation:
    for _, row in filtered.iterrows():
        fig.add_annotation(
            x=row["x"] + row["ox"] * 0.1,
            y=row["y"] + row["oy"] * 0.1,
            ax=row["x"],
            ay=row["y"],
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            opacity=0.6,
            arrowcolor="gray"
        )

# --- Layout ---
fig.update_layout(
    title=f"Generation {selected_gen} — Robot Positions",
    xaxis_title="X Position",
    yaxis_title="Y Position",
    legend_title="Population",
    height=700
)

st.plotly_chart(fig, width=True)

# --- Info ---
st.markdown(f"**Showing {len(filtered)} points** (1 out of {step})")
st.markdown("Use the sidebar to filter generations, individuals, and display options.")
