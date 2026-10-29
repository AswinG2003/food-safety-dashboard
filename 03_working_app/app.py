from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

try:
    import pydeck as pdk
    HAS_PYDECK = True
except ImportError:
    HAS_PYDECK = False

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
DATA_DIR = PARENT / "02_my_model_artifacts" / "data"
MODEL_DIR = PARENT / "02_my_model_artifacts"

GREEN = "#2D9B70"
YELLOW = "#E7AD35"
RED = "#D94B45"
DARK_BG = "#173A32"

MODEL_OPTIONS = ["Random Forest", "XGBoost"]

PROB_COL_MAP = {
    "Random Forest": "failure_probability__random_forest",
    "XGBoost": "failure_probability__xgboost",
}

st.set_page_config(
    page_title="Chicago Food Safety Risk Dashboard",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url(
      'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap'
    );

    .stApp {
        background: #F4F7F5;
        font-family: 'DM Sans', sans-serif;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #173A32;
        font-family: 'Manrope', sans-serif;
    }

    .hero {
        background: linear-gradient(115deg, #123B36, #176B57, #72A68A);
        padding: 32px;
        border-radius: 18px;
        color: white;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: white;
        font-size: 2.2rem;
        margin-bottom: 10px;
    }

    .hero p {
        color: #E4F2ED;
        font-size: 1rem;
        margin-bottom: 0;
    }

    .eyebrow {
        color: #AEE0CA;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    div[data-testid="stMetric"] {
        background: white;
        padding: 16px;
        border: 1px solid #E1EAE5;
        border-radius: 12px;
    }

    .footer {
        color: #7B8B84;
        border-top: 1px solid #DFE8E2;
        padding-top: 18px;
        margin-top: 30px;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner="Loading restaurant data…")
def load_priority_list():
    """Load the precomputed restaurant inspection priority list."""
    path = DATA_DIR / "restaurant_inspection_priority_list.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    # Parse dates
    if "latest_inspection_date" in df.columns:
        df["latest_inspection_date"] = pd.to_datetime(
            df["latest_inspection_date"], errors="coerce"
        )
    # Clean coordinates
    for col in ["Latitude", "Longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(show_spinner="Loading model metrics…")
def load_metrics():
    """Load model_metrics_all.csv and filter to standalone RF and XGBoost."""
    path = DATA_DIR / "model_metrics_all.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Filter to only standalone models
    df = df[df["Model"].isin(["Random Forest", "XGBoost"])].copy()
    return df


@st.cache_data(show_spinner="Loading feature importance…")
def load_feature_importance(model_name):
    """Load feature importance CSV for the given model."""
    if model_name == "Random Forest":
        path = DATA_DIR / "random_forest_feature_importance.csv"
    else:
        path = DATA_DIR / "xgboost_feature_importance.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner="Loading calibration data…")
def load_calibration():
    path = DATA_DIR / "calibration_curve_test.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner="Loading threshold metrics…")
def load_threshold_metrics():
    path = DATA_DIR / "threshold_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner="Loading test predictions…")
def load_test_predictions():
    path = DATA_DIR / "test_predictions_ranked.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if "Inspection Date" in df.columns:
        df["Inspection Date"] = pd.to_datetime(
            df["Inspection Date"], errors="coerce"
        )
    for col in ["Latitude", "Longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(show_spinner="Loading predict metadata…")
def load_predict_metadata():
    path = DATA_DIR / "predict_row_metadata.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if "Inspection Date" in df.columns:
        df["Inspection Date"] = pd.to_datetime(
            df["Inspection Date"], errors="coerce"
        )
    return df


@st.cache_data(show_spinner="Loading model manifest…")
def load_manifest():
    path = MODEL_DIR / "model_manifest.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# HELPERS
# ============================================================

def format_percent(value):
    if value is None:
        return "—"
    try:
        value = float(value)
        if np.isnan(value):
            return "—"
        return f"{value:.1%}"
    except (TypeError, ValueError):
        return "—"


def format_pct_value(value):
    """Format a value already in 0-100 range as a percentage string."""
    if value is None:
        return "—"
    try:
        value = float(value)
        if np.isnan(value):
            return "—"
        return f"{value:.1f}%"
    except (TypeError, ValueError):
        return "—"


def assign_priority_tier(prob):
    """Assign operational priority tier based on failure probability (0-1 scale)."""
    if pd.isna(prob):
        return "Unknown"
    if prob >= 0.50:
        return "P1 — Critical"
    elif prob >= 0.35:
        return "P2 — High"
    elif prob >= 0.20:
        return "P3 — Elevated"
    elif prob >= 0.10:
        return "P4 — Moderate"
    else:
        return "P5 — Low"


def risk_color(prob):
    if pd.isna(prob):
        return "#78918A"
    if prob >= 0.50:
        return RED
    elif prob >= 0.25:
        return YELLOW
    else:
        return GREEN


def risk_color_pydeck(prob):
    """Return [R, G, B, A] for pydeck."""
    if pd.isna(prob):
        return [120, 145, 138, 160]
    if prob >= 0.50:
        return [217, 75, 69, 200]
    elif prob >= 0.25:
        return [231, 173, 53, 200]
    else:
        return [45, 155, 112, 200]


def download_dataframe(df, filename, label):
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


def parse_violation_numbers(violations_text):
    """Extract violation numbers from the Violations or violation_excerpt text."""
    if pd.isna(violations_text) or not isinstance(violations_text, str):
        return []
    # Pattern: number at start of violation code like "21." or "1."
    return re.findall(r'(?:^|\|\s*)(\d+)\.\s', violations_text)


# ============================================================
# LOAD ALL DATA
# ============================================================

df_priority = load_priority_list()
df_metrics = load_metrics()
df_calibration = load_calibration()
df_threshold = load_threshold_metrics()
df_test = load_test_predictions()
df_meta = load_predict_metadata()
manifest = load_manifest()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("# 🍽️ Chicago FoodSafe")
st.sidebar.caption("Food Safety Risk Intelligence")
st.sidebar.divider()

# Model selector
selected_model = st.sidebar.radio(
    "Select risk model",
    MODEL_OPTIONS,
    index=1,  # Default to XGBoost
    help="Choose which standalone model's estimated failure probabilities to use for rankings and risk scores.",
)

prob_col = PROB_COL_MAP[selected_model]
prob_pct_col = f"failure_probability_pct__{prob_col.split('__')[1]}"

st.sidebar.divider()

if df_priority.empty:
    st.sidebar.error("Priority list not found.")
else:
    st.sidebar.success(f"{len(df_priority):,} restaurants loaded")
    st.sidebar.caption(f"Active model: **{selected_model}**")

st.sidebar.divider()
st.sidebar.caption(
    "**Data source:** Precomputed predictions from standalone "
    "Random Forest and XGBoost models. "
    "Probabilities are sigmoid-calibrated estimated risks, "
    "not official regulatory classifications."
)


# ============================================================
# PREPARE WORKING DATA
# ============================================================

if not df_priority.empty and prob_col in df_priority.columns:
    df = df_priority.copy()
    df["_selected_prob"] = df[prob_col]
    df["_selected_prob_pct"] = df["_selected_prob"] * 100
    df["_priority_tier"] = df["_selected_prob"].apply(assign_priority_tier)
    df["_priority_rank"] = (
        df["_selected_prob"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
else:
    df = pd.DataFrame()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">
            CHICAGO · PUBLIC HEALTH · PREDICTIVE ANALYTICS
        </div>
        <h1>Food Safety Risk Dashboard</h1>
        <p>
            Estimated inspection failure risks for Chicago restaurants,
            powered by standalone {} predictions.
            Scores are operational decision-support signals, not official
            inspection results or regulatory classifications.
        </p>
    </div>
    """.format(selected_model),
    unsafe_allow_html=True,
)


# ============================================================
# NAVIGATION
# ============================================================

tabs = st.tabs([
    "📊 Overview",
    "🗺️ Risk Map",
    "📋 Inspector Queue",
    "🏪 Restaurant Report Card",
    "🔬 What-If Simulator",
    "⚠️ Violation Analytics",
    "📅 Seasonality & Geography",
    "🧠 Model Insights",
])


# ============================================================
# TAB 1: OVERVIEW
# ============================================================

with tabs[0]:
    st.subheader("Citywide Risk Snapshot")

    if df.empty:
        st.error("Restaurant data is unavailable. Check that `02_my_model_artifacts/data/restaurant_inspection_priority_list.csv` exists.")
    else:
        total_restaurants = len(df)
        n_unique = df["Restaurant_ID"].nunique() if "Restaurant_ID" in df.columns else total_restaurants

        mean_prob = df["_selected_prob"].mean()
        n_critical = int((df["_selected_prob"] >= 0.50).sum())
        n_high = int((df["_selected_prob"] >= 0.35).sum())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Restaurants scored", f"{n_unique:,}")
        c2.metric(f"Mean est. failure prob.", format_percent(mean_prob))
        c3.metric("Critical risk (≥50%)", f"{n_critical:,}")
        c4.metric("High+ risk (≥35%)", f"{n_high:,}")

        st.caption(
            f"Model: **{selected_model}** · "
            f"Probabilities are sigmoid-calibrated estimates of next-inspection failure risk."
        )

        left, right = st.columns(2)

        with left:
            st.markdown("### Estimated Risk Distribution")

            fig = px.histogram(
                df,
                x="_selected_prob",
                nbins=40,
                color_discrete_sequence=[GREEN],
                template="plotly_white",
                labels={"_selected_prob": "Estimated Failure Probability"},
            )
            fig.update_layout(
                height=370,
                yaxis_title="Number of Restaurants",
                xaxis_tickformat=".0%",
                showlegend=False,
            )
            fig.add_vline(
                x=0.50, line_dash="dash", line_color=RED,
                annotation_text="Critical (50%)", annotation_position="top right",
            )
            fig.add_vline(
                x=0.35, line_dash="dash", line_color=YELLOW,
                annotation_text="High (35%)", annotation_position="top left",
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.markdown("### Priority Tier Breakdown")

            tier_order = ["P1 — Critical", "P2 — High", "P3 — Elevated", "P4 — Moderate", "P5 — Low", "Unknown"]
            tier_colors = {
                "P1 — Critical": "#C0392B",
                "P2 — High": RED,
                "P3 — Elevated": YELLOW,
                "P4 — Moderate": "#5DADE2",
                "P5 — Low": GREEN,
                "Unknown": "#78918A",
            }

            tier_counts = (
                df["_priority_tier"]
                .value_counts()
                .reindex(tier_order, fill_value=0)
                .reset_index()
            )
            tier_counts.columns = ["Priority Tier", "Restaurants"]
            tier_counts = tier_counts[tier_counts["Restaurants"] > 0]

            fig = px.bar(
                tier_counts,
                x="Priority Tier",
                y="Restaurants",
                color="Priority Tier",
                color_discrete_map=tier_colors,
                template="plotly_white",
            )
            fig.update_layout(height=370, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Last-inspection result summary
        if "current_result_label" in df.columns:
            st.markdown("### Last Recorded Inspection Outcome")
            outcome = (
                df["current_result_label"]
                .fillna("Unknown")
                .value_counts()
                .rename_axis("Outcome")
                .reset_index(name="Count")
            )
            fig = px.bar(
                outcome, x="Outcome", y="Count",
                color="Outcome", template="plotly_white",
            )
            fig.update_layout(height=320, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Priority Tier Guide")
        tier_guide = pd.DataFrame({
            "Tier": ["P1 — Critical", "P2 — High", "P3 — Elevated", "P4 — Moderate", "P5 — Low"],
            "Probability Range": ["≥ 50%", "35%–49%", "20%–34%", "10%–19%", "< 10%"],
            "Suggested Action": [
                "Immediate inspection recommended",
                "Schedule inspection within 2 weeks",
                "Monitor closely; inspect within 1 month",
                "Routine scheduling",
                "Standard cycle",
            ],
        })
        st.dataframe(tier_guide, use_container_width=True, hide_index=True)

        st.info(
            "⚖️ **Disclaimer:** These are estimated risks derived from historical "
            "inspection patterns. They are not official food safety ratings, "
            "regulatory decisions, or guarantees of future outcomes."
        )


# ============================================================
# TAB 2: RISK MAP
# ============================================================

with tabs[1]:
    st.subheader("Chicago Restaurant Risk Map")

    if df.empty:
        st.info("Restaurant data is unavailable.")
    elif "Latitude" not in df.columns or "Longitude" not in df.columns:
        st.info("Latitude and Longitude columns are required for the map.")
    else:
        map_df = df.dropna(subset=["Latitude", "Longitude"]).copy()
        map_df = map_df[
            map_df["Latitude"].between(41.0, 43.0) &
            map_df["Longitude"].between(-89.0, -86.0)
        ]

        # Filters
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            risk_filter = st.multiselect(
                "Filter by risk category",
                ["Risk 1 (High)", "Risk 2 (Medium)", "Risk 3 (Low)"],
                default=[],
                help="Filter by the restaurant's official risk category (not model prediction).",
            )
        with fc2:
            min_prob_map = st.slider(
                "Minimum failure probability",
                min_value=0.0, max_value=1.0, value=0.0, step=0.05,
                format="%.0%%",
                key="map_min_prob",
            )
        with fc3:
            max_points = st.select_slider(
                "Max markers on map",
                options=[500, 1000, 2000, 3000, 5000],
                value=2000,
            )

        if risk_filter and "Risk" in map_df.columns:
            map_df = map_df[map_df["Risk"].isin(risk_filter)]

        map_df = map_df[map_df["_selected_prob"] >= min_prob_map]

        if len(map_df) > max_points:
            st.caption(f"Showing a {max_points:,}-point sample from {len(map_df):,} matching restaurants.")
            map_df = map_df.nlargest(max_points, "_selected_prob")

        if map_df.empty:
            st.warning("No restaurants match the current filters.")
        elif HAS_PYDECK:
            map_df["_color"] = map_df["_selected_prob"].apply(risk_color_pydeck)
            map_df["_radius"] = (map_df["_selected_prob"] * 100 + 30).clip(30, 120)
            map_df["_prob_display"] = (map_df["_selected_prob"] * 100).round(1).astype(str) + "%"

            tooltip_cols = {
                "Restaurant_ID": "ID",
                "_prob_display": "Est. Failure Prob.",
                "_priority_tier": "Priority Tier",
                "Risk": "Risk Category",
                "current_result_label": "Last Result",
                "latest_inspection_date": "Last Inspected",
            }
            rename_map = {k: v for k, v in tooltip_cols.items() if k in map_df.columns}

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position=["Longitude", "Latitude"],
                get_color="_color",
                get_radius="_radius",
                pickable=True,
                opacity=0.75,
                radius_min_pixels=3,
                radius_max_pixels=12,
            )

            view = pdk.ViewState(
                latitude=map_df["Latitude"].median(),
                longitude=map_df["Longitude"].median(),
                zoom=10.5,
                pitch=0,
            )

            tooltip_html = "<b>Restaurant {Restaurant_ID}</b><br/>"
            tooltip_html += "Est. Failure Prob: {_prob_display}<br/>"
            tooltip_html += "Priority: {_priority_tier}<br/>"
            if "Risk" in map_df.columns:
                tooltip_html += "Risk Category: {Risk}<br/>"
            if "current_result_label" in map_df.columns:
                tooltip_html += "Last Result: {current_result_label}"

            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=view,
                map_style="mapbox://styles/mapbox/light-v11",
                tooltip={"html": tooltip_html, "style": {"backgroundColor": "#173A32", "color": "white", "fontSize": "12px"}},
            )
            st.pydeck_chart(deck)

            st.markdown(
                f"🔴 Critical (≥50%) &nbsp;&nbsp; "
                f"🟡 Elevated (25–49%) &nbsp;&nbsp; "
                f"🟢 Lower (<25%) &nbsp;&nbsp; "
                f"| Showing **{len(map_df):,}** restaurants"
            )
        else:
            # Fallback to plotly scatter_mapbox
            map_df["_prob_display"] = (map_df["_selected_prob"] * 100).round(1)
            fig = px.scatter_mapbox(
                map_df,
                lat="Latitude",
                lon="Longitude",
                color="_selected_prob",
                color_continuous_scale=[GREEN, YELLOW, RED],
                range_color=[0, 0.7],
                hover_data={"Restaurant_ID": True, "_selected_prob": ":.1%", "_priority_tier": True},
                zoom=10,
                height=600,
                mapbox_style="open-street-map",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Install `pydeck` for a more interactive map experience.")


# ============================================================
# TAB 3: INSPECTOR QUEUE
# ============================================================

with tabs[2]:
    st.subheader("Inspector Priority Queue")

    st.write(
        f"Restaurants ranked by **{selected_model}** estimated failure probability. "
        "Rankings and tiers are recalculated from the selected model's probabilities."
    )

    if df.empty:
        st.info("Restaurant data is unavailable.")
    else:
        queue = df.copy()

        # Filters
        qc1, qc2, qc3 = st.columns(3)
        with qc1:
            tier_filter = st.multiselect(
                "Filter by priority tier",
                ["P1 — Critical", "P2 — High", "P3 — Elevated", "P4 — Moderate", "P5 — Low"],
                default=[],
            )
        with qc2:
            risk_cat_filter = st.multiselect(
                "Filter by risk category",
                sorted(queue["Risk"].dropna().unique().tolist()) if "Risk" in queue.columns else [],
                default=[],
            )
        with qc3:
            age_filter = st.multiselect(
                "Filter by inspection age",
                sorted(queue["inspection_age_band"].dropna().unique().tolist()) if "inspection_age_band" in queue.columns else [],
                default=[],
            )

        if tier_filter:
            queue = queue[queue["_priority_tier"].isin(tier_filter)]
        if risk_cat_filter and "Risk" in queue.columns:
            queue = queue[queue["Risk"].isin(risk_cat_filter)]
        if age_filter and "inspection_age_band" in queue.columns:
            queue = queue[queue["inspection_age_band"].isin(age_filter)]

        queue = queue.sort_values("_selected_prob", ascending=False).reset_index(drop=True)
        queue.insert(0, "Rank", range(1, len(queue) + 1))

        display_limit = st.select_slider(
            "Restaurants to display",
            options=[25, 50, 100, 250, 500, 1000],
            value=100,
            key="queue_limit",
        )

        display_cols = [c for c in [
            "Rank",
            "Restaurant_ID",
            "_selected_prob",
            "_priority_tier",
            "Risk",
            "latest_inspection_date",
            "latest_inspection_type",
            "current_result_label",
            "Violation_Count",
            "inspection_age_band",
            "violation_excerpt",
        ] if c in queue.columns]

        col_config = {
            "_selected_prob": st.column_config.ProgressColumn(
                f"Est. Failure Prob. ({selected_model})",
                min_value=0, max_value=1, format="%.1%%",
            ),
            "_priority_tier": st.column_config.TextColumn("Priority Tier"),
            "latest_inspection_date": st.column_config.DateColumn("Last Inspection"),
            "violation_excerpt": st.column_config.TextColumn("Violation Excerpt", width="large"),
        }

        st.dataframe(
            queue[display_cols].head(display_limit),
            use_container_width=True,
            height=500,
            column_config=col_config,
        )

        download_dataframe(
            queue[display_cols],
            f"inspector_queue_{selected_model.lower().replace(' ', '_')}.csv",
            "⬇ Download full priority queue as CSV",
        )

        st.caption(
            f"{len(queue):,} restaurants in current queue · "
            f"Model: {selected_model}"
        )


# ============================================================
# TAB 4: RESTAURANT REPORT CARD
# ============================================================

with tabs[3]:
    st.subheader("Restaurant Report Card")

    if df.empty:
        st.info("Restaurant data is unavailable.")
    else:
        rc1, rc2 = st.columns([2, 1])
        with rc1:
            search_text = st.text_input(
                "Search by Restaurant ID",
                placeholder="Enter a Restaurant ID (e.g. 14887)",
                key="report_search",
            )
        with rc2:
            search_by_id = True  # We search by Restaurant_ID since names aren't in priority list

        if search_text.strip():
            try:
                search_id = int(search_text.strip())
                match = df[df["Restaurant_ID"] == search_id]
            except ValueError:
                match = pd.DataFrame()
                st.warning("Please enter a numeric Restaurant ID.")
        else:
            match = pd.DataFrame()
            # Show top 10 highest risk as suggestions
            st.markdown("**Top 10 highest-risk restaurants:**")
            top10 = df.nlargest(10, "_selected_prob")[
                [c for c in ["Restaurant_ID", "_selected_prob", "_priority_tier", "Risk",
                             "latest_inspection_date", "current_result_label"] if c in df.columns]
            ].copy()
            top10["Est. Failure Prob."] = top10["_selected_prob"].apply(format_percent)
            st.dataframe(top10, use_container_width=True, hide_index=True)

        if not match.empty:
            row = match.iloc[0]

            st.markdown(f"### Restaurant ID: {int(row['Restaurant_ID'])}")

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric(
                f"Est. Failure Prob. ({selected_model})",
                format_percent(row["_selected_prob"]),
            )
            mc2.metric("Priority Tier", row["_priority_tier"])
            mc3.metric(
                "Risk Category",
                row.get("Risk", "—"),
            )
            mc4.metric(
                "Last Result",
                row.get("current_result_label", "—"),
            )

            # Details
            st.markdown("#### Inspection Details")
            detail_items = {
                "Last Inspection Date": row.get("latest_inspection_date", "—"),
                "Inspection Type": row.get("latest_inspection_type", "—"),
                "Violation Count": row.get("Violation_Count", "—"),
                "Inspection Sequence": row.get("Inspection_Sequence", "—"),
                "Inspection Age": row.get("inspection_age_band", "—"),
                "Days Since Inspection": row.get("days_since_inspection_at_dataset_end", "—"),
                "Latitude": row.get("Latitude", "—"),
                "Longitude": row.get("Longitude", "—"),
            }
            detail_df = pd.DataFrame(
                list(detail_items.items()), columns=["Field", "Value"]
            )
            st.dataframe(detail_df, use_container_width=True, hide_index=True)

            # Model comparison for this restaurant
            st.markdown("#### Model Comparison for This Restaurant")
            model_probs = {}
            for mname, pcol in PROB_COL_MAP.items():
                if pcol in df.columns:
                    val = row.get(pcol, np.nan)
                    model_probs[mname] = val
            if model_probs:
                mp_df = pd.DataFrame(
                    [{"Model": k, "Est. Failure Probability": v} for k, v in model_probs.items()]
                )
                mp_df["Display"] = mp_df["Est. Failure Probability"].apply(format_percent)
                fig = px.bar(
                    mp_df, x="Model", y="Est. Failure Probability",
                    color="Model",
                    color_discrete_map={"Random Forest": GREEN, "XGBoost": "#2980B9"},
                    template="plotly_white",
                    text="Display",
                )
                fig.update_layout(height=300, showlegend=False, yaxis_tickformat=".0%")
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

            # Violation excerpt
            violation_text = row.get("violation_excerpt", row.get("Violations", None))
            if pd.notna(violation_text) and str(violation_text).strip():
                st.markdown("#### Violation Excerpt")
                st.text_area(
                    "Violations from last inspection",
                    value=str(violation_text)[:2000],
                    height=200,
                    disabled=True,
                    key="rc_violations",
                )

            # Inspection history from metadata if available
            if not df_meta.empty and "Restaurant_ID" in df_meta.columns:
                rest_history = df_meta[
                    df_meta["Restaurant_ID"] == int(row["Restaurant_ID"])
                ].copy()
                if not rest_history.empty and "Inspection Date" in rest_history.columns:
                    rest_history = rest_history.sort_values("Inspection Date", ascending=False)
                    st.markdown("#### Inspection History")
                    hist_cols = [c for c in [
                        "Inspection Date", "Inspection Type", "Inspection_Sequence",
                        "Results", "Target_Next_Results",
                    ] if c in rest_history.columns]
                    st.dataframe(
                        rest_history[hist_cols].head(20),
                        use_container_width=True, hide_index=True,
                    )

        elif search_text.strip():
            st.warning(f"No restaurant found with ID '{search_text.strip()}'.")


# ============================================================
# TAB 5: WHAT-IF SIMULATOR
# ============================================================

with tabs[4]:
    st.subheader("What-If Scenario Simulator")

    st.info(
        "⚙️ **About this simulator:** The standalone Random Forest and XGBoost models "
        "use 56 encoded features including inspection history, violation counts, geographic "
        "grid cells, temporal patterns, and more. Many of these features are derived from "
        "cumulative inspection histories and require the full preprocessing pipeline to compute "
        "correctly.\n\n"
        "Because the full preprocessing pipeline (encoding, feature engineering, "
        "temporal features) is not bundled with this dashboard, **live inference on arbitrary "
        "new inputs is not available**. The scenarios below use the precomputed probability "
        "data to illustrate how different risk factors relate to predicted failure probability."
    )

    if df.empty:
        st.warning("Restaurant data is unavailable for scenario analysis.")
    else:
        st.markdown("### Scenario: How does violation count relate to risk?")

        if "Violation_Count" in df.columns:
            vc_df = df.dropna(subset=["Violation_Count", "_selected_prob"]).copy()
            vc_df["Violation_Count"] = pd.to_numeric(vc_df["Violation_Count"], errors="coerce")
            vc_df = vc_df.dropna(subset=["Violation_Count"])

            vc_bins = pd.cut(vc_df["Violation_Count"], bins=[0, 2, 5, 8, 12, 100], labels=["0–2", "3–5", "6–8", "9–12", "13+"])
            vc_summary = vc_df.groupby(vc_bins, observed=True)["_selected_prob"].agg(["mean", "median", "count"]).reset_index()
            vc_summary.columns = ["Violation Count Range", "Mean Prob.", "Median Prob.", "Restaurants"]

            fig = px.bar(
                vc_summary, x="Violation Count Range", y="Mean Prob.",
                color="Mean Prob.", color_continuous_scale=[GREEN, YELLOW, RED],
                template="plotly_white", text=vc_summary["Mean Prob."].apply(lambda x: f"{x:.1%}"),
            )
            fig.update_layout(height=370, yaxis_tickformat=".0%", yaxis_title=f"Mean Est. Failure Prob. ({selected_model})")
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(vc_summary, use_container_width=True, hide_index=True)

        st.markdown("### Scenario: Risk by inspection type")

        if "latest_inspection_type" in df.columns:
            it_df = df.dropna(subset=["latest_inspection_type", "_selected_prob"]).copy()
            it_summary = (
                it_df.groupby("latest_inspection_type")["_selected_prob"]
                .agg(["mean", "median", "count"])
                .reset_index()
                .sort_values("mean", ascending=False)
            )
            it_summary.columns = ["Inspection Type", "Mean Prob.", "Median Prob.", "Count"]
            it_summary = it_summary[it_summary["Count"] >= 10]

            fig = px.bar(
                it_summary, x="Inspection Type", y="Mean Prob.",
                color="Mean Prob.", color_continuous_scale=[GREEN, YELLOW, RED],
                template="plotly_white",
            )
            fig.update_layout(height=370, yaxis_tickformat=".0%", xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Scenario: Risk by inspection age")

        if "inspection_age_band" in df.columns:
            age_df = df.dropna(subset=["inspection_age_band", "_selected_prob"]).copy()
            age_summary = (
                age_df.groupby("inspection_age_band")["_selected_prob"]
                .agg(["mean", "median", "count"])
                .reset_index()
                .sort_values("mean", ascending=False)
            )
            age_summary.columns = ["Inspection Age Band", "Mean Prob.", "Median Prob.", "Count"]

            fig = px.bar(
                age_summary, x="Inspection Age Band", y="Mean Prob.",
                color="Mean Prob.", color_continuous_scale=[GREEN, YELLOW, RED],
                template="plotly_white",
            )
            fig.update_layout(height=350, yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Scenario: Risk by risk category")

        if "Risk" in df.columns:
            risk_summary = (
                df.groupby("Risk")["_selected_prob"]
                .agg(["mean", "median", "count"])
                .reset_index()
                .sort_values("mean", ascending=False)
            )
            risk_summary.columns = ["Risk Category", "Mean Prob.", "Median Prob.", "Count"]

            fig = px.bar(
                risk_summary, x="Risk Category", y="Mean Prob.",
                color="Mean Prob.", color_continuous_scale=[GREEN, YELLOW, RED],
                template="plotly_white",
            )
            fig.update_layout(height=350, yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        st.warning(
            "⚠️ These scenarios show **observed patterns** in the precomputed predictions, "
            "not causal effects. Changing a single factor in isolation would require the full "
            "model pipeline with all 56 features correctly recomputed. "
            "True counterfactual what-if analysis requires the preprocessing pipeline "
            "and is not available in this dashboard."
        )


# ============================================================
# TAB 6: VIOLATION ANALYTICS
# ============================================================

with tabs[5]:
    st.subheader("Violation Analytics")

    if df.empty:
        st.info("Restaurant data is unavailable.")
    else:
        # Use violation_excerpt or Violations column
        viol_col = None
        for candidate in ["violation_excerpt", "Violations"]:
            if candidate in df.columns:
                viol_col = candidate
                break

        if viol_col:
            st.markdown("### Most Common Violation Numbers")

            viol_data = df[df[viol_col].notna()].copy()

            # Extract individual violation numbers
            all_violations = []
            for _, row in viol_data.iterrows():
                text = str(row[viol_col])
                # Extract patterns like "21." or "1." at start of violations
                nums = re.findall(r'(?:^|\|\s*)(\d+)\.', text)
                for n in nums:
                    all_violations.append(int(n))

            if all_violations:
                viol_series = pd.Series(all_violations, name="Violation Number")
                viol_counts = (
                    viol_series.value_counts()
                    .head(25)
                    .rename_axis("Violation #")
                    .reset_index(name="Occurrences")
                )

                fig = px.bar(
                    viol_counts, x="Violation #", y="Occurrences",
                    color="Occurrences", color_continuous_scale=[GREEN, YELLOW, RED],
                    template="plotly_white",
                )
                fig.update_layout(height=420, xaxis_type="category")
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(viol_counts, use_container_width=True, hide_index=True)

                download_dataframe(
                    viol_counts,
                    "violation_number_summary.csv",
                    "⬇ Download violation summary",
                )
            else:
                st.info("Could not extract individual violation numbers from the data.")
        else:
            st.info("No violation text column found in the data.")

        # Violation count distribution
        if "Violation_Count" in df.columns:
            st.markdown("### Violation Count Distribution")
            vc_data = df.dropna(subset=["Violation_Count"]).copy()
            vc_data["Violation_Count"] = pd.to_numeric(vc_data["Violation_Count"], errors="coerce")

            fig = px.histogram(
                vc_data, x="Violation_Count", nbins=30,
                color_discrete_sequence=[GREEN],
                template="plotly_white",
                labels={"Violation_Count": "Number of Violations"},
            )
            fig.update_layout(height=350, yaxis_title="Restaurants")
            st.plotly_chart(fig, use_container_width=True)

            # Mean risk by violation count
            st.markdown("### Mean Estimated Failure Probability by Violation Count")
            vc_risk = (
                vc_data.groupby("Violation_Count")["_selected_prob"]
                .agg(["mean", "count"])
                .reset_index()
            )
            vc_risk.columns = ["Violation Count", "Mean Est. Prob.", "Restaurants"]
            vc_risk = vc_risk[vc_risk["Restaurants"] >= 5]

            fig = px.scatter(
                vc_risk, x="Violation Count", y="Mean Est. Prob.",
                size="Restaurants", color="Mean Est. Prob.",
                color_continuous_scale=[GREEN, YELLOW, RED],
                template="plotly_white",
            )
            fig.update_layout(height=370, yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 7: SEASONALITY & GEOGRAPHY
# ============================================================

with tabs[6]:
    st.subheader("Seasonality & Geographic Patterns")

    if df.empty:
        st.info("Restaurant data is unavailable.")
    else:
        # Seasonality from test predictions (which have date info)
        if not df_test.empty and "Inspection Date" in df_test.columns:
            st.markdown("### Inspection Volume Over Time")
            test_dated = df_test.dropna(subset=["Inspection Date"]).copy()
            test_dated["Month"] = test_dated["Inspection Date"].dt.to_period("M").dt.to_timestamp()

            monthly_vol = test_dated.groupby("Month").size().reset_index(name="Inspections")
            fig = px.line(
                monthly_vol, x="Month", y="Inspections",
                markers=True, template="plotly_white",
                title="Monthly Inspection Volume (Test Set)",
            )
            fig.update_traces(line_color=GREEN, line_width=3)
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

            # Monthly failure rate
            if "actual_next_inspection_result" in test_dated.columns:
                st.markdown("### Monthly Actual Failure Rate (Test Set)")
                test_dated["_actual_fail"] = pd.to_numeric(
                    test_dated["actual_next_inspection_result"], errors="coerce"
                )
                monthly_fail = (
                    test_dated.groupby("Month")["_actual_fail"]
                    .agg(["mean", "count"])
                    .reset_index()
                )
                monthly_fail.columns = ["Month", "Failure Rate", "Count"]
                monthly_fail = monthly_fail[monthly_fail["Count"] >= 20]

                fig = px.line(
                    monthly_fail, x="Month", y="Failure Rate",
                    markers=True, template="plotly_white",
                )
                fig.update_traces(line_color=RED, line_width=3)
                fig.update_layout(height=350, yaxis_tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)

                st.caption(
                    "Failure rate shows the proportion of inspections in each month "
                    "where the next inspection resulted in a failure."
                )

        elif "latest_inspection_date" in df.columns:
            st.markdown("### Inspection Dates Distribution")
            dated = df.dropna(subset=["latest_inspection_date"]).copy()
            dated["Year"] = dated["latest_inspection_date"].dt.year

            yearly = dated.groupby("Year").size().reset_index(name="Restaurants")
            fig = px.bar(
                yearly, x="Year", y="Restaurants",
                color_discrete_sequence=[GREEN],
                template="plotly_white",
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        # Geographic analysis
        st.markdown("### Geographic Risk Patterns")

        if "Latitude" in df.columns and "Longitude" in df.columns:
            geo_df = df.dropna(subset=["Latitude", "Longitude", "_selected_prob"]).copy()
            geo_df = geo_df[
                geo_df["Latitude"].between(41.0, 43.0) &
                geo_df["Longitude"].between(-89.0, -86.0)
            ]

            # Create grid cells for geographic comparison
            geo_df["_lat_grid"] = (geo_df["Latitude"] * 10).round() / 10
            geo_df["_lon_grid"] = (geo_df["Longitude"] * 10).round() / 10
            geo_df["_geo_cell"] = geo_df["_lat_grid"].astype(str) + ", " + geo_df["_lon_grid"].astype(str)

            geo_summary = (
                geo_df.groupby("_geo_cell")["_selected_prob"]
                .agg(["mean", "count"])
                .reset_index()
                .sort_values("mean", ascending=False)
            )
            geo_summary.columns = ["Grid Cell (Lat, Lon)", "Mean Est. Prob.", "Restaurants"]
            geo_summary = geo_summary[geo_summary["Restaurants"] >= 10]

            if not geo_summary.empty:
                fig = px.bar(
                    geo_summary.head(20),
                    x="Grid Cell (Lat, Lon)",
                    y="Mean Est. Prob.",
                    color="Mean Est. Prob.",
                    color_continuous_scale=[GREEN, YELLOW, RED],
                    template="plotly_white",
                    hover_data=["Restaurants"],
                )
                fig.update_layout(
                    height=420, xaxis_tickangle=-40,
                    yaxis_tickformat=".0%",
                    title="Mean Estimated Failure Probability by Geographic Grid (0.1° cells)",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(geo_summary, use_container_width=True, hide_index=True)
            else:
                st.info("Not enough data points per geographic cell for comparison.")
        else:
            st.info("Latitude/Longitude data required for geographic analysis.")

        # Risk category distribution
        if "Risk" in df.columns:
            st.markdown("### Risk Category Distribution")
            risk_dist = df["Risk"].value_counts().reset_index()
            risk_dist.columns = ["Risk Category", "Count"]

            fig = px.pie(
                risk_dist, names="Risk Category", values="Count",
                hole=0.5, template="plotly_white",
            )
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 8: MODEL INSIGHTS
# ============================================================

with tabs[7]:
    st.subheader("Model Performance & Feature Importance")

    st.write(
        "Compare standalone Random Forest and XGBoost models on held-out test data. "
        "All metrics shown are from the outer chronological test set (n=26,772)."
    )

    if df_metrics.empty:
        st.warning("Model metrics file (model_metrics_all.csv) not found or does not contain Random Forest / XGBoost rows.")
    else:
        # Key metrics comparison
        st.markdown("### Key Performance Metrics")

        key_metrics = {
            "PR-AUC (Avg. Precision)": "Test_PR_AUC_Average_Precision",
            "ROC-AUC": "Test_ROC_AUC",
            "Brier Score": "Test_Brier_score",
            "Log Loss": "Test_log_loss",
        }

        metric_cards = st.columns(len(key_metrics))
        for card, (label, col_name) in zip(metric_cards, key_metrics.items()):
            if col_name in df_metrics.columns:
                vals = df_metrics.set_index("Model")[col_name]
                rf_val = vals.get("Random Forest", np.nan)
                xgb_val = vals.get("XGBoost", np.nan)
                if selected_model == "Random Forest":
                    display_val = rf_val
                else:
                    display_val = xgb_val
                card.metric(
                    label,
                    f"{display_val:.4f}" if pd.notna(display_val) else "—",
                    help=f"RF: {rf_val:.4f}, XGB: {xgb_val:.4f}" if pd.notna(rf_val) else "",
                )

        # Side-by-side comparison chart
        st.markdown("### Model Comparison")

        compare_metrics = [
            ("PR-AUC", "Test_PR_AUC_Average_Precision"),
            ("ROC-AUC", "Test_ROC_AUC"),
            ("Brier Score ↓", "Test_Brier_score"),
        ]

        comparison_rows = []
        for label, col in compare_metrics:
            if col in df_metrics.columns:
                for _, row in df_metrics.iterrows():
                    comparison_rows.append({
                        "Metric": label,
                        "Model": row["Model"],
                        "Value": row[col],
                    })

        if comparison_rows:
            comp_df = pd.DataFrame(comparison_rows)
            fig = px.bar(
                comp_df, x="Metric", y="Value", color="Model",
                barmode="group",
                color_discrete_map={"Random Forest": GREEN, "XGBoost": "#2980B9"},
                template="plotly_white",
                text=comp_df["Value"].apply(lambda x: f"{x:.4f}"),
            )
            fig.update_layout(height=400)
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        # Top-k failure rates
        st.markdown("### Top-K Failure Capture Rates")
        st.caption(
            "What fraction of restaurants in the top 1%, 5%, 10% of predicted risk "
            "actually failed their next inspection."
        )

        topk_metrics = {
            "Top 1% Failure Rate": "top_1pct_failure_rate",
            "Top 5% Failure Rate": "top_5pct_failure_rate",
            "Top 10% Failure Rate": "top_10pct_failure_rate",
            "Top 1% Lift": "top_1pct_lift",
            "Top 5% Lift": "top_5pct_lift",
            "Top 10% Lift": "top_10pct_lift",
        }

        topk_rows = []
        for label, col in topk_metrics.items():
            if col in df_metrics.columns:
                for _, row in df_metrics.iterrows():
                    topk_rows.append({
                        "Metric": label,
                        "Model": row["Model"],
                        "Value": row[col],
                    })

        if topk_rows:
            topk_df = pd.DataFrame(topk_rows)
            # Show failure rates
            failure_rate_df = topk_df[topk_df["Metric"].str.contains("Rate")]
            if not failure_rate_df.empty:
                fig = px.bar(
                    failure_rate_df, x="Metric", y="Value", color="Model",
                    barmode="group",
                    color_discrete_map={"Random Forest": GREEN, "XGBoost": "#2980B9"},
                    template="plotly_white",
                    text=failure_rate_df["Value"].apply(lambda x: f"{x:.1%}"),
                )
                fig.update_layout(height=380, yaxis_tickformat=".0%")
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

        # Full metrics table
        st.markdown("### Full Test Metrics Table")
        display_metric_cols = [c for c in df_metrics.columns if c != "Model"]
        # Transpose for readability
        metrics_display = df_metrics.set_index("Model")[display_metric_cols[:20]].T
        metrics_display.index.name = "Metric"
        st.dataframe(metrics_display, use_container_width=True)

        with st.expander("View all metrics (raw)"):
            st.dataframe(df_metrics, use_container_width=True)

        download_dataframe(
            df_metrics,
            "model_metrics_comparison.csv",
            "⬇ Download metrics comparison",
        )

        # Calibration curve
        if not df_calibration.empty:
            st.markdown("### Calibration Curves")
            st.caption(
                "How well do predicted probabilities match observed failure rates? "
                "Points near the diagonal line indicate good calibration."
            )

            cal_plot = df_calibration.copy()
            if "Model" in cal_plot.columns:
                cal_plot = cal_plot[cal_plot["Model"].isin(["Random Forest", "XGBoost"])]

                fig = go.Figure()
                # Perfect calibration line
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode="lines", line=dict(color="gray", dash="dash"),
                    name="Perfect calibration",
                ))

                colors = {"Random Forest": GREEN, "XGBoost": "#2980B9"}
                for model_name in ["Random Forest", "XGBoost"]:
                    mdata = cal_plot[cal_plot["Model"] == model_name]
                    if not mdata.empty and "mean_predicted_probability" in mdata.columns and "observed_failure_rate" in mdata.columns:
                        fig.add_trace(go.Scatter(
                            x=mdata["mean_predicted_probability"],
                            y=mdata["observed_failure_rate"],
                            mode="lines+markers",
                            name=model_name,
                            line=dict(color=colors.get(model_name, GREEN), width=3),
                            marker=dict(size=8),
                        ))

                fig.update_layout(
                    template="plotly_white", height=420,
                    xaxis_title="Mean Predicted Probability",
                    yaxis_title="Observed Failure Rate",
                    xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                )
                st.plotly_chart(fig, use_container_width=True)

        # Threshold comparison
        if not df_threshold.empty:
            st.markdown("### Decision Threshold Analysis")
            thresh_display = df_threshold[df_threshold["Model"].isin(["Random Forest", "XGBoost"])].copy()
            if not thresh_display.empty:
                st.dataframe(thresh_display, use_container_width=True, hide_index=True)

        # Feature importance
        st.markdown("### Feature Importance")
        st.caption(
            "These are model-native feature importances (Gini impurity for Random Forest, "
            "gain for XGBoost). They are **not** SHAP values."
        )

        fi_model = st.radio(
            "Show importance for:",
            MODEL_OPTIONS,
            index=MODEL_OPTIONS.index(selected_model),
            horizontal=True,
            key="fi_model_select",
        )

        fi_df = load_feature_importance(fi_model)

        if fi_df.empty:
            st.info(f"Feature importance data not found for {fi_model}.")
        else:
            fi_df = fi_df.sort_values("importance", ascending=False).head(25)

            fig = px.bar(
                fi_df,
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale=[GREEN, YELLOW, RED],
                template="plotly_white",
                labels={"importance": "Importance", "feature": "Feature"},
            )
            fig.update_layout(
                height=max(400, len(fi_df) * 22),
                yaxis=dict(autorange="reversed"),
                title=f"Top {len(fi_df)} Features — {fi_model}",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(fi_df, use_container_width=True, hide_index=True)

            download_dataframe(
                fi_df,
                f"feature_importance_{fi_model.lower().replace(' ', '_')}.csv",
                f"⬇ Download {fi_model} feature importance",
            )

    # Model manifest
    if manifest:
        with st.expander("Model Manifest & Metadata"):
            st.json(manifest)

    st.warning(
        "⚠️ Before operational use, verify that historical features use "
        "prior inspections only, confirm the exact prediction point, "
        "and evaluate on a chronological holdout. A model score is "
        "not an official food-safety rating."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Chicago FoodSafe · Food Safety Risk Dashboard
        · Decision-support prototype using standalone Random Forest and XGBoost models.
        Estimated risks, not official classifications.
    </div>
    """,
    unsafe_allow_html=True,
)