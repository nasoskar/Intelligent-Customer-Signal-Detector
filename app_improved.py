import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import json
import anthropic
import time

# page config
st.set_page_config(
    page_title="Intelligent Customer Signal Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #ffffff; }
    [data-testid="stMainBlockContainer"] { background: #ffffff; }
    [data-testid="stSidebar"] { background: #f8f9fa; border-right: 1px solid #e5e7eb; }
    [data-testid="stSidebar"] * { color: #374151 !important; }

    .metric-card {
        background: #ffffff;
        border: 1.5px solid #e5e7eb;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .metric-value { font-size: 2rem; font-weight: 700; margin: 0; line-height: 1.1; }
    .metric-label { font-size: 0.72rem; color: #9ca3af; text-transform: uppercase;
                    letter-spacing: 0.08em; margin-top: 4px; }
    .metric-red   { color: #ef4444; }
    .metric-amber { color: #f59e0b; }
    .metric-green { color: #10b981; }
    .metric-blue  { color: #3b82f6; }
    .metric-purple { color: #8b5cf6; }

    .section-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 10px;
    }

    .detail-card {
        background: #f9fafb;
        border: 1.5px solid #e5e7eb;
        border-radius: 10px;
        padding: 20px;
        margin-top: 8px;
    }
    .detail-mini-box {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 10px 14px;
        border-radius: 8px;
        text-align: center;
    }
    .detail-mini-val { font-size: 1.1rem; font-weight: 700; color: #1f2937; }
    .detail-mini-lbl { font-size: 0.68rem; color: #9ca3af; text-transform: uppercase; margin-top: 2px; }

    .block-label {
        font-size: 0.68rem;
        font-weight: 600;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 4px;
    }

    .sentiment-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 14px;
    }

    .review-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #fbbf24;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 4px; background: transparent; }
    .stTabs [data-baseweb="tab"] {
        background: #f3f4f6;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        color: #6b7280;
        padding: 7px 18px;
    }
    .stTabs [aria-selected="true"] {
        background: #00008B !important;
        color: #ffffff !important;
        border-color: #00008B !important;
    }

    .sidebar-title {
        font-size: 0.68rem;
        font-weight: 600;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 14px 0 6px;
    }

    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Helpers
RISK_COLOURS = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}
SENTIMENT_COLOURS = {
    "Frustrated": ("#fee2e2", "#b91c1c"),
    "Neutral":    ("#fef3c7", "#92400e"),
    "Positive":   ("#d1fae5", "#065f46")
}

CONFIDENCE_GAP_THRESHOLD = 0.15


def compute_confidence_flag(probas, gap_threshold=CONFIDENCE_GAP_THRESHOLD):
    """
    probas: 2D array from predict_proba(), shape (n_customers, n_classes)
    Returns: needs_review (bool array), confidence_gap (float array)
    A customer is flagged for human review when the model's top-2 predicted
    classes are close in probability — i.e. the model itself is not confident
    about which tier the customer belongs to.
    """
    sorted_probas = np.sort(probas, axis=1)[:, ::-1]  # descending, per row
    top_prob = sorted_probas[:, 0]
    second_prob = sorted_probas[:, 1]
    gap = (top_prob - second_prob)
    needs_review = gap < gap_threshold
    return needs_review, gap.round(2)


# ML
@st.cache_resource(show_spinner=False)
def train_model(df):
    categorical_cols = ["contract_type", "payment_method", "last_contact_channel", "product"]
    drop_cols = ["customer_id", "customer_name", "latest_support_transcript", "ground_truth_risk"]
    if "risk_encoded" in df.columns:
        drop_cols.append("risk_encoded")

    df_enc = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    y = df["ground_truth_risk"]
    X = df_enc.drop(columns=drop_cols)
    feature_cols = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=5,
        min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    fi = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False).head(10)
    return clf, feature_cols, fi


def score_customers(df, clf, feature_cols):
    categorical_cols = ["contract_type", "payment_method", "last_contact_channel", "product"]
    drop_cols = ["customer_id", "customer_name", "latest_support_transcript", "ground_truth_risk"]
    if "risk_encoded" in df.columns:
        drop_cols.append("risk_encoded")

    df_enc = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    X = df_enc.drop(columns=[c for c in drop_cols if c in df_enc.columns])
    X = X.reindex(columns=feature_cols, fill_value=0)

    preds = clf.predict(X)
    probas = clf.predict_proba(X)
    high_idx = list(clf.classes_).index("High")
    scores = (probas[:, high_idx] * 10).round(1)

    # Confidence flag — borderline cases where the model isn't sure
    needs_review, confidence_gap = compute_confidence_flag(probas)

    result = df.copy()
    result["predicted_risk"] = preds
    result["risk_score"] = scores
    result["needs_review"] = needs_review
    result["confidence_gap"] = confidence_gap
    return result


# Claude api
def analyse_transcript(row, client):
    prompt = f"""You are a customer retention analyst. Analyse the following customer signals and return ONLY a JSON object with no additional text, no markdown, no backticks.

Customer signals:
- Tenure: {row['tenure_months']} months
- Monthly charge: £{row['monthly_charge_gbp']}
- Payment delays (last 6 months): {row['payment_delays_last_6mo']}
- Complaints (last 6 months): {row['num_complaints_last_6mo']}
- Satisfaction score: {row['satisfaction_score']}/5
- NPS score: {row['nps_score']}/10
- Open tickets: {row['open_tickets']}
- Support transcript: "{row['latest_support_transcript']}"

Return this exact JSON structure:
{{
    "sentiment": "Frustrated" or "Neutral" or "Positive",
    "rationale": "2 sentence explanation of why this customer is or is not at risk",
    "recommended_action": "one specific retention action for the team"
}}"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception:
        return {
            "sentiment": "Unknown",
            "rationale": "Analysis unavailable.",
            "recommended_action": "Manual review required."
        }


# sidebar
with st.sidebar:
    st.image('Firstsource.png')
    st.markdown("---")

    st.markdown('<div class="sidebar-title">Data Source</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload customer CSV", type=["csv"], label_visibility="collapsed")

    st.markdown('<div class="sidebar-title">Anthropic API Key</div>', unsafe_allow_html=True)
    api_key = st.text_input("API Key", type="password", placeholder="sk-ant-…", label_visibility="collapsed")
    st.caption("Used to generate AI rationale for the customer you select below.")


# Main 
st.markdown("# CustomerIQ")
st.markdown('<p style="color:#6b7280;margin-top:-8px;">FirstSource\'s prototype for customer risk signals and recommended actions </p>', unsafe_allow_html=True)

if "df_result" not in st.session_state:
    st.session_state.df_result = None
if "fi" not in st.session_state:
    st.session_state.fi = None

if uploaded:
    df_raw = pd.read_csv(uploaded)
    with st.spinner("Training risk model…"):
        clf, feature_cols, fi = train_model(df_raw)
        st.session_state.fi = fi
        df_scored = score_customers(df_raw, clf, feature_cols)

    if "sentiment" not in df_scored.columns:
        df_scored["sentiment"]          = "—"
        df_scored["rationale"]          = "Select this customer in Customer Detail to run AI analysis."
        df_scored["recommended_action"] = "—"

    st.session_state.df_result = df_scored

if "ai_cache" not in st.session_state:
    st.session_state.ai_cache = {}  # customer_id -> {sentiment, rationale, recommended_action}

# tabs
tab1, tab2 = st.tabs(["  📊  Dashboard  ", "  🧠  Model Insights  "])


with tab1:

    if st.session_state.df_result is None:
        st.markdown("""
        <div style="text-align:center;padding:80px 20px;color:#4b5563;">
            <div style="font-size:3rem;margin-bottom:16px;">📂</div>
            <div style="font-size:1.1rem;font-weight:600;color:#9ca3af;">
                Upload a customer CSV to get started
            </div>
            <div style="font-size:0.85rem;margin-top:8px;">
                Use the sidebar to upload your data and optionally add your 
                Anthropic API key for AI transcript analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df = st.session_state.df_result.copy()

        # ── Customer selection + Claude call happens FIRST, so the rest of the
        #    page (table, detail card) reflects the latest data on this same run ──
        customer_ids = df["customer_id"].tolist()
        if "selected_customer" not in st.session_state or st.session_state.selected_customer not in customer_ids:
            st.session_state.selected_customer = customer_ids[0] if customer_ids else None

        selected_id = st.session_state.selected_customer

        if selected_id:
            cache = st.session_state.ai_cache
            if selected_id not in cache and api_key:
                row_for_call = df[df["customer_id"] == selected_id].iloc[0]
                with st.spinner("Asking Claude to analyse this customer…"):
                    client = anthropic.Anthropic(api_key=api_key)
                    result = analyse_transcript(row_for_call, client)
                    cache[selected_id] = result

            if selected_id in cache:
                result = cache[selected_id]
                idx = df.index[df["customer_id"] == selected_id]
                df.loc[idx, "sentiment"]          = result["sentiment"]
                df.loc[idx, "rationale"]          = result["rationale"]
                df.loc[idx, "recommended_action"] = result["recommended_action"]
                st.session_state.df_result = df

        n_total  = len(df)
        n_high   = (df["predicted_risk"] == "High").sum()
        n_med    = (df["predicted_risk"] == "Medium").sum()
        n_low    = (df["predicted_risk"] == "Low").sum()
        n_review = int(df["needs_review"].sum()) if "needs_review" in df.columns else 0

        # Metric cards
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-value metric-blue">{n_total}</div><div class="metric-label">Customers Analysed</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-value metric-red">{n_high}</div><div class="metric-label">High Risk</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="metric-value metric-amber">{n_med}</div><div class="metric-label">Medium Risk</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><div class="metric-value metric-green">{n_low}</div><div class="metric-label">Low Risk</div></div>', unsafe_allow_html=True)
        with c5:
            st.markdown(f'<div class="metric-card"><div class="metric-value metric-purple">{n_review}</div><div class="metric-label">Needs Review</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts — smaller height
        ch1, ch2 = st.columns(2)
        with ch1:
            st.markdown('<div class="section-label">Risk Score Distribution</div>', unsafe_allow_html=True)
            fig_hist = px.histogram(df, x="risk_score", nbins=20, color_discrete_sequence=["#3b82f6"])
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f9fafb",
                font_color="#374151", height=260,
                xaxis=dict(gridcolor="#e5e7eb", title="Risk Score (0–10)"),
                yaxis=dict(gridcolor="#e5e7eb", title="Customers"),
                margin=dict(l=0, r=0, t=10, b=0), showlegend=False
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with ch2:
            st.markdown('<div class="section-label">Risk Tier Breakdown</div>', unsafe_allow_html=True)
            tier_counts = df["predicted_risk"].value_counts().reset_index()
            tier_counts.columns = ["tier", "count"]
            fig_pie = px.pie(tier_counts, names="tier", values="count",
                             color="tier", color_discrete_map=RISK_COLOURS, hole=0.55)
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#374151", height=260,
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(bgcolor="rgba(0,0,0,0)")
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Customer table 
        st.markdown('<div class="section-label">Ranked Customer List</div>', unsafe_allow_html=True)

        display_cols = ["customer_id", "product", "contract_type", "tenure_months",
                        "satisfaction_score", "payment_delays_last_6mo",
                        "risk_score", "predicted_risk", "needs_review"]

        available    = [c for c in display_cols if c in df.columns]
        df_display   = df[available].reset_index(drop=True)

        # Friendlier values for the review flag before renaming
        if "needs_review" in df_display.columns:
            df_display["needs_review"] = df_display["needs_review"].map({True: "Needs Review", False: "-"})

        rename_map   = {
            "customer_id": "Customer ID", "product": "Product",
            "contract_type": "Contract", "tenure_months": "Tenure (mo)",
            "satisfaction_score": "Satisfaction Score", "payment_delays_last_6mo": "Delays",
            "risk_score": "Score", "predicted_risk": "Risk Tier", "needs_review": "Confidence"
        }
        df_display.rename(columns=rename_map, inplace=True)

        def highlight_high_risk(row):
            if "Risk Tier" in row and row["Risk Tier"] == "High":
                return ["color: red; font-weight: bold;"] * len(row)
            return [""] * len(row)

        styled_df = df_display.style.apply(highlight_high_risk, axis=1)

        st.dataframe(styled_df, use_container_width=True, height=340, hide_index=True)

        # Customer detail
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Customer Detail</div>', unsafe_allow_html=True)

        new_selected_id = st.selectbox(
            "Select a customer",
            customer_ids,
            index=customer_ids.index(selected_id) if selected_id in customer_ids else 0,
            label_visibility="collapsed",
            key="customer_selectbox"
        )
        if new_selected_id != st.session_state.selected_customer:
            st.session_state.selected_customer = new_selected_id
            st.rerun()

        if selected_id:
            row = df[df["customer_id"] == selected_id].iloc[0]
            risk_col = RISK_COLOURS.get(row["predicted_risk"], "#6b7280")

            # Header row using columns (safe, no HTML injection)
            h1, h2 = st.columns([3, 1])
            with h1:
                st.markdown(f"**{row['customer_id']}** &nbsp; <span style='color:#9ca3af;font-size:0.85rem'>{row['product']} · {row['contract_type']}</span>", unsafe_allow_html=True)
            with h2:
                st.markdown(f"<div style='text-align:right'><span style='font-size:1.8rem;font-weight:800;color:{risk_col}'>{row['risk_score']}</span><span style='color:#9ca3af;font-size:0.8rem'> /10</span></div>", unsafe_allow_html=True)

            # Mini stats grid
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f'<div class="detail-mini-box"><div class="detail-mini-val">{row["tenure_months"]}mo</div><div class="detail-mini-lbl">Tenure</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="detail-mini-box"><div class="detail-mini-val">£{row["monthly_charge_gbp"]}</div><div class="detail-mini-lbl">Monthly</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="detail-mini-box"><div class="detail-mini-val">{row["satisfaction_score"]}/5</div><div class="detail-mini-lbl">Satisfaction Score</div></div>', unsafe_allow_html=True)
            with m4:
                st.markdown(f'<div class="detail-mini-box"><div class="detail-mini-val">{row["nps_score"]}/10</div><div class="detail-mini-lbl">NPS</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Confidence / "needs review" warning — shown above sentiment when applicable
            if bool(row.get("needs_review", False)):
                gap_val = row.get("confidence_gap", None)
                gap_text = f" (confidence gap: {gap_val})" if gap_val is not None else ""
                st.markdown('<div class="block-label">Model Confidence</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div style="background:#fef3c7;border-left:3px solid #f59e0b;padding:12px 16px;'
                    f'border-radius:0 6px 6px 0;color:#92400e;font-size:0.88rem;margin-bottom:14px;">'
                    f'Borderline case: the model\'s top two risk tiers are close{gap_text}. '
                    f'Recommend manual review before taking action.</div>',
                    unsafe_allow_html=True
                )

            # Sentiment badge — only shown once we have a real value
            sentiment_val = row.get("sentiment", "—")
            if sentiment_val and sentiment_val != "—":
                bg, fg = SENTIMENT_COLOURS.get(sentiment_val, ("#f3f4f6", "#6b7280"))
                st.markdown('<div class="block-label">Sentiment</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<span class="sentiment-badge" style="background:{bg};color:{fg};">{sentiment_val}</span>',
                    unsafe_allow_html=True
                )

            # Transcript — always visible
            st.markdown('<div class="block-label">Latest Support Transcript</div>', unsafe_allow_html=True)
            st.info(f'"{row["latest_support_transcript"]}"')

            cache = st.session_state.ai_cache

            if selected_id not in cache and not api_key:
                st.warning("Enter your Anthropic API key in the sidebar to generate AI rationale for this customer.")

            if selected_id in cache:
                result = cache[selected_id]

                st.markdown('<div class="block-label">Explanation</div>', unsafe_allow_html=True)
                st.success(result["rationale"])

                st.markdown('<div class="block-label" style="margin-top:12px;">Recommended Action</div>', unsafe_allow_html=True)
                st.warning(result["recommended_action"])

with tab2:

    if st.session_state.fi is None:
        st.markdown("""
        <div style="text-align:center;padding:80px 20px;color:#9ca3af;">
            <div style="font-size:3rem;margin-bottom:16px;">🧠</div>
            <div style="font-size:1.1rem;font-weight:600;color:#6b7280;">Upload a dataset to view model insights</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df  = st.session_state.df_result.copy()
        fi  = st.session_state.fi

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-label">Top 10 Feature Importances</div>', unsafe_allow_html=True)
            fi_df = fi.reset_index()
            fi_df.columns = ["feature", "importance"]
            fi_df["feature"] = fi_df["feature"].str.replace("_", " ").str.title()
            fig_fi = px.bar(fi_df, x="importance", y="feature", orientation="h",
                            color="importance", color_continuous_scale=["#bfdbfe", "#1d4ed8"])
            fig_fi.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f9fafb",
                font_color="#374151", height=300,
                xaxis=dict(gridcolor="#e5e7eb", title="Importance"),
                yaxis=dict(gridcolor="#e5e7eb", title="", autorange="reversed"),
                margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False
            )
            st.plotly_chart(fig_fi, use_container_width=True)
            st.caption("Higher bars = features the model relied on most for risk classification.")

        with col2:
            st.markdown('<div class="section-label">Signal Correlation Matrix</div>', unsafe_allow_html=True)
            num_cols = ["tenure_months", "monthly_charge_gbp", "payment_delays_last_6mo",
                        "num_complaints_last_6mo", "open_tickets", "days_since_last_contact",
                        "satisfaction_score", "nps_score"]
            risk_map = {"Low": 0, "Medium": 1, "High": 2}
            df["risk_encoded"] = df["ground_truth_risk"].map(risk_map)
            corr   = df[num_cols + ["risk_encoded"]].corr()
            labels = [c.replace("_", " ").title() for c in corr.columns]
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr.values, x=labels, y=labels,
                colorscale="RdBu", zmid=0,
                text=np.round(corr.values, 2),
                texttemplate="%{text}", textfont={"size": 8},
                hoverongaps=False
            ))
            fig_corr.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#374151", height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(tickfont=dict(size=8)),
                yaxis=dict(tickfont=dict(size=8))
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            st.caption("Red = positive correlation, Blue = negative. Risk Encoded row shows strongest churn predictors.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Average Risk Score by Segment</div>', unsafe_allow_html=True)

        s1, s2 = st.columns(2)
        with s1:
            avg_c = df.groupby("contract_type")["risk_score"].mean().round(2).reset_index()
            fig_c = px.bar(avg_c, x="contract_type", y="risk_score",
                           color="risk_score", color_continuous_scale=["#bbf7d0","#ef4444"],
                           labels={"contract_type": "Contract", "risk_score": "Avg Score"})
            fig_c.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f9fafb",
                font_color="#374151", height=250,
                xaxis=dict(gridcolor="#e5e7eb"),
                yaxis=dict(gridcolor="#e5e7eb"),
                margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False
            )
            st.plotly_chart(fig_c, use_container_width=True)

        with s2:
            avg_p = df.groupby("product")["risk_score"].mean().round(2).reset_index()
            fig_p = px.bar(avg_p, x="product", y="risk_score",
                           color="risk_score", color_continuous_scale=["#bbf7d0","#ef4444"],
                           labels={"product": "Product", "risk_score": "Avg Score"})
            fig_p.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f9fafb",
                font_color="#374151", height=250,
                xaxis=dict(gridcolor="#e5e7eb", tickangle=-15),
                yaxis=dict(gridcolor="#e5e7eb"),
                margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False
            )
            st.plotly_chart(fig_p, use_container_width=True)
