"""
Interactive Review Analysis Dashboard — Keepa API Data
Streamlit app for exploring Amazon FBA business acquisition data.

Run: streamlit run dashboard.py
"""

import json
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, date

import openpyxl
from openpyxl.utils import column_index_from_string

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BIZ_COLORS = {
    "Wallaroo Wallets": {"primary": "#2c3e50", "accent": "#3498db"},
    "TeacherFav": {"primary": "#c0392b", "accent": "#e74c3c"},
}

PRODUCT_COLORS = [
    "#00d4aa", "#ff6b6b", "#4ecdc4", "#ffd93d", "#c084fc",
    "#ff8a5c", "#5ce1e6", "#ff5757", "#38b6ff", "#7bed9f",
]


# ─── Data Loading ───────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    with open(os.path.join(BASE_DIR, "keepa_deduped_catalog.json")) as f:
        catalog = json.load(f)

    biz_data = {}
    for biz_name in ["Wallaroo Wallets", "TeacherFav"]:
        products = sorted(
            [p for p in catalog if p["business"] == biz_name],
            key=lambda p: -p["shared_reviews"],
        )
        biz_data[biz_name] = {
            "products": products,
            "metrics": compute_metrics(products),
        }
    return biz_data


@st.cache_data
def load_pnl(biz_name):
    """Load P&L data from Excel if available. Returns dict of monthly data or None."""
    pnl_files = {
        "TeacherFav": os.path.join(BASE_DIR, "TeacherFav", "92221 - TeacherFav - P&L.xlsx"),
    }
    path = pnl_files.get(biz_name)
    if not path or not os.path.exists(path):
        return None

    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb["Summary"]

    # Read month headers from row 3
    months = []
    for col in range(2, ws.max_column + 1):
        val = ws.cell(row=3, column=col).value
        if isinstance(val, datetime):
            months.append((col, val))

    key_rows = {
        "Total Revenue": 42,
        "Total Expenses": 133,
        "COGS": 95,
        "Gross Profit": 99,
        "Net Income": 145,
    }

    data = {}
    for label, row_num in key_rows.items():
        series = {}
        for col, month_dt in months:
            val = ws.cell(row=row_num, column=col).value
            if isinstance(val, (int, float)) and val != 0:
                series[month_dt.strftime("%Y-%m")] = round(val, 2)
        data[label] = series

    # Also grab margin rows (as percentages)
    margin_rows = {
        "Net Profit Margin (12mo trailing)": 150,
        "TaCOS (Total Ad Cost of Sales)": 149,
        "COGS %": 148,
    }
    for label, row_num in margin_rows.items():
        series = {}
        for col, month_dt in months:
            val = ws.cell(row=row_num, column=col).value
            if isinstance(val, (int, float)) and val != 0:
                series[month_dt.strftime("%Y-%m")] = round(val * 100, 1)
        data[label] = series

    return data


def compute_integrity_signals(products):
    """Analyze review/rating data for signs of manipulation or organic growth."""
    signals = []
    for p in products:
        ch = p["review_count_history"]
        rh = p["rating_history"]
        name = product_name(p, short=True)
        s = {"name": name, "asin": p["representative_asin"]}

        # 1. Review count drops (Amazon purges)
        drops = []
        for i in range(1, len(ch)):
            diff = ch[i]["count"] - ch[i - 1]["count"]
            if diff < -5:  # ignore tiny fluctuations
                drops.append({
                    "date": ch[i]["date"][:7],
                    "removed": abs(diff),
                    "from_count": ch[i - 1]["count"],
                })
        s["review_drops"] = drops
        s["total_removed"] = sum(d["removed"] for d in drops)

        # 2. Growth consistency — split into quarters, check for spikes
        if len(ch) >= 8:
            quarter_len = len(ch) // 4
            quarter_growths = []
            for q in range(4):
                start_idx = q * quarter_len
                end_idx = (q + 1) * quarter_len if q < 3 else len(ch) - 1
                q_months = max(1, (datetime.fromisoformat(ch[end_idx]["date"]) - datetime.fromisoformat(ch[start_idx]["date"])).days / 30.44)
                q_growth = (ch[end_idx]["count"] - ch[start_idx]["count"]) / q_months
                quarter_growths.append(round(q_growth, 1))
            s["quarter_growths"] = quarter_growths
            avg_growth = sum(quarter_growths) / len(quarter_growths) if quarter_growths else 0
            max_growth = max(quarter_growths) if quarter_growths else 0
            s["growth_spike"] = max_growth > avg_growth * 2 and avg_growth > 0
        else:
            s["quarter_growths"] = []
            s["growth_spike"] = False

        # 3. Rating stability — standard deviation
        if len(rh) >= 4:
            ratings = [r["rating"] for r in rh]
            mean_r = sum(ratings) / len(ratings)
            variance = sum((r - mean_r) ** 2 for r in ratings) / len(ratings)
            s["rating_std"] = round(variance ** 0.5, 3)
            s["rating_stable"] = s["rating_std"] < 0.3
        else:
            s["rating_std"] = None
            s["rating_stable"] = None

        # 4. Overall verdict
        red_flags = []
        if s["total_removed"] > 100:
            red_flags.append(f"Amazon removed {s['total_removed']:,} reviews ({len(drops)} events)")
        elif s["total_removed"] > 0:
            red_flags.append(f"Minor review removals: {s['total_removed']} reviews ({len(drops)} events)")
        if s["growth_spike"]:
            red_flags.append(f"Uneven growth pattern — quarterly rates: {s['quarter_growths']} reviews/mo")
        s["red_flags"] = red_flags
        s["looks_organic"] = len(red_flags) == 0 or (s["total_removed"] <= 20 and not s["growth_spike"])

        signals.append(s)
    return signals


def compute_metrics(products):
    m = {}
    m["n_products"] = len(products)
    m["n_variations"] = sum(p["num_variations"] for p in products)

    main = products[0]  # already sorted by reviews desc
    m["main_product"] = main
    m["main_asin"] = main["representative_asin"]
    m["main_title"] = main["title"] or ""
    # Use review count from filtered history if available, else fall back to static field
    main_ch = main["review_count_history"]
    m["main_reviews"] = main_ch[-1]["count"] if main_ch else main["shared_reviews"]
    m["main_rating"] = main["rating_history"][-1]["rating"] if main["rating_history"] else None

    # Weighted avg rating (use latest rating and review count within the filtered range)
    total_w, weighted_r = 0, 0
    for p in products:
        if p["rating_history"] and p["review_count_history"]:
            r = p["rating_history"][-1]["rating"]
            w = p["review_count_history"][-1]["count"]
            weighted_r += r * w
            total_w += w
    m["weighted_avg_rating"] = weighted_r / total_w if total_w > 0 else None

    # Total reviews = sum of latest review counts in filtered range
    m["total_reviews"] = sum(
        p["review_count_history"][-1]["count"] if p["review_count_history"] else p["shared_reviews"]
        for p in products
    )

    # Rating trend for main product
    rh = main["rating_history"]
    if len(rh) >= 4:
        mid = len(rh) // 2
        early = [r["rating"] for r in rh[:mid]]
        recent = [r["rating"] for r in rh[mid:]]
        m["main_early_r"] = sum(early) / len(early)
        m["main_recent_r"] = sum(recent) / len(recent)
        m["main_r_delta"] = m["main_recent_r"] - m["main_early_r"]
        m["main_r_dir"] = "IMPROVING" if m["main_r_delta"] > 0.05 else "DECLINING" if m["main_r_delta"] < -0.05 else "STABLE"
    else:
        m["main_early_r"] = m["main_recent_r"] = m["main_r_delta"] = 0
        m["main_r_dir"] = "INSUFFICIENT DATA"

    # Review growth
    ch = main["review_count_history"]
    if len(ch) >= 2:
        first_date = datetime.fromisoformat(ch[0]["date"])
        last_date = datetime.fromisoformat(ch[-1]["date"])
        months = max(1, (last_date - first_date).days / 30.44)
        m["main_growth_per_mo"] = (ch[-1]["count"] - ch[0]["count"]) / months
        m["main_first_date"] = ch[0]["date"][:10]
        m["main_last_date"] = ch[-1]["date"][:10]

        mid_idx = len(ch) // 2
        mid_date = datetime.fromisoformat(ch[mid_idx]["date"])
        early_months = max(1, (mid_date - first_date).days / 30.44)
        recent_months = max(1, (last_date - mid_date).days / 30.44)
        m["early_growth_mo"] = (ch[mid_idx]["count"] - ch[0]["count"]) / early_months
        m["recent_growth_mo"] = (ch[-1]["count"] - ch[mid_idx]["count"]) / recent_months
        m["growth_dir"] = "ACCELERATING" if m["recent_growth_mo"] > m["early_growth_mo"] * 1.1 else "DECELERATING" if m["recent_growth_mo"] < m["early_growth_mo"] * 0.9 else "STEADY"
    else:
        m["main_growth_per_mo"] = 0
        m["main_first_date"] = m["main_last_date"] = "?"
        m["early_growth_mo"] = m["recent_growth_mo"] = 0
        m["growth_dir"] = "UNKNOWN"

    return m


def get_date_range(biz_data):
    """Find the min/max dates across all products in all businesses."""
    all_dates = []
    for biz_name in biz_data:
        for p in biz_data[biz_name]["products"]:
            for r in p["rating_history"]:
                all_dates.append(r["date"][:10])
            for c in p["review_count_history"]:
                all_dates.append(c["date"][:10])
    if not all_dates:
        return date(2020, 1, 1), date.today()
    return (
        datetime.fromisoformat(min(all_dates)).date(),
        datetime.fromisoformat(max(all_dates)).date(),
    )


def filter_products(products, start_date, end_date):
    """Return a deep-copy of products with time-series data filtered to [start, end]."""
    start_str = start_date.isoformat()
    end_str = end_date.isoformat()
    filtered = []
    for p in products:
        fp = dict(p)
        fp["rating_history"] = [
            r for r in p["rating_history"] if start_str <= r["date"][:10] <= end_str
        ]
        fp["review_count_history"] = [
            c for c in p["review_count_history"] if start_str <= c["date"][:10] <= end_str
        ]
        if p.get("monthly_avg_rank"):
            fp["monthly_avg_rank"] = {
                m: v for m, v in p["monthly_avg_rank"].items()
                if start_str[:7] <= m <= end_str[:7]
            }
        filtered.append(fp)
    return filtered


def amazon_link(asin):
    return f"https://www.amazon.com/dp/{asin}"


def keepa_link(asin):
    return f"https://keepa.com/#!product/1-{asin}"


# Short, distinctive names for sidebar and compact displays
PRODUCT_SHORT_NAMES = {
    "B01LYPPPUU": "Phone Wallet (main)",
    "B0BQ3RCZMQ": "MagSafe Wallet",
    "B07YS5CZYB": "Card Holder",
    "B0C2W24JY8": "Wristlet",
    "B08L879DLS": "Sand Timer 3-Pack (main)",
    "B09GF1VSBH": "Toothbrush Timer 2-Pack",
    "B09G6HHP2X": "Sand Timer 6-Pack",
    "B09Q9CBWNB": "Sand Timer 6-Pack Colorful",
    "B09VR2W9TV": "Hourglass 60min Large",
    "B09GP2HDCF": "Sand Timer 60min Black",
}


def product_name(product, short=False):
    """Return product display name. short=True uses compact sidebar names."""
    asin = product["representative_asin"]
    if short:
        return PRODUCT_SHORT_NAMES.get(asin, product["title"] or "Unknown")
    return product["title"] or "Unknown"


def short_title(title, max_len=80):
    if not title:
        return "Unknown"
    return title[:max_len] + "..." if len(title) > max_len else title


# ─── Chart Helpers ──────────────────────────────────────────────────────────

def make_rating_chart(products, title, colors=None):
    fig = go.Figure()
    for i, p in enumerate(products):
        rh = p["rating_history"]
        if not rh:
            continue
        dates = [r["date"] for r in rh]
        ratings = [r["rating"] for r in rh]
        color = (colors or PRODUCT_COLORS)[i % len(colors or PRODUCT_COLORS)]
        fig.add_trace(go.Scatter(
            x=dates, y=ratings, mode="lines+markers",
            name=f"{product_name(p, short=True)} ({p['shared_reviews']:,} rev)",
            line=dict(width=2, color=color),
            marker=dict(size=4),
        ))

    fig.add_hline(y=4.0, line_dash="dot", line_color="green", opacity=0.3,
                  annotation_text="4.0", annotation_position="left")
    layout_kwargs = dict(
        title=dict(text=f"{title}<br><sup>Point-in-time snapshot — each dot is the product's star rating on that date. Source: Keepa API (keepa.com)</sup>"),
        yaxis_title="Rating",
        yaxis=dict(range=[2.5, 5.1], dtick=0.5),
        height=450,
        legend=dict(font=dict(size=10)),
        margin=dict(l=60, r=20, t=70, b=40),
    )
    if len(products) > 1:
        layout_kwargs["legend"] = dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5, font=dict(size=10))
        layout_kwargs["margin"] = dict(l=60, r=20, t=70, b=80)
        layout_kwargs["height"] = 480
    fig.update_layout(**layout_kwargs)
    return fig


def make_review_count_chart(products, title, colors=None):
    fig = go.Figure()
    for i, p in enumerate(products):
        ch = p["review_count_history"]
        if not ch:
            continue
        dates = [c["date"] for c in ch]
        counts = [c["count"] for c in ch]
        color = (colors or PRODUCT_COLORS)[i % len(colors or PRODUCT_COLORS)]
        fig.add_trace(go.Scatter(
            x=dates, y=counts, mode="lines",
            name=f"{product_name(p, short=True)} ({counts[-1]:,} current)",
            line=dict(width=2.5, color=color),
        ))

    layout_kwargs = dict(
        title=dict(
            text=f"{title}<br><sup>Cumulative running total — shows all reviews to date, not new reviews per period. Drops indicate reviews removed by Amazon. Source: Keepa API (keepa.com)</sup>",
        ),
        yaxis_title="Cumulative Review Count",
        height=450,
        yaxis=dict(rangemode="tozero"),
        legend=dict(font=dict(size=10)),
        margin=dict(l=60, r=20, t=70, b=40),
    )
    if len(products) > 1:
        layout_kwargs["legend"] = dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5, font=dict(size=10))
        layout_kwargs["margin"] = dict(l=60, r=20, t=70, b=80)
        layout_kwargs["height"] = 480
    fig.update_layout(**layout_kwargs)
    return fig


def make_combined_chart(products, metric_type, title, colors=None):
    """Create a combined (summed) chart across all products."""
    # Collect all dates and sum values
    if metric_type == "reviews":
        all_series = []
        for p in products:
            ch = p["review_count_history"]
            if ch:
                all_series.append(pd.Series(
                    {c["date"][:10]: c["count"] for c in ch}
                ))
        if not all_series:
            return go.Figure()

        # Resample to monthly, forward-fill, then sum
        combined = pd.DataFrame(all_series).T
        combined.index = pd.to_datetime(combined.index)
        combined = combined.sort_index().ffill().fillna(0)
        monthly = combined.resample("MS").last().ffill()
        totals = monthly.sum(axis=1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=totals.index, y=totals.values, mode="lines",
            name="All Products Combined",
            line=dict(width=3, color="#00d4aa"),
            fill="tozeroy", fillcolor="rgba(0,212,170,0.15)",
        ))
        fig.update_layout(
            title=dict(text=f"{title}<br><sup>Source: Keepa API (keepa.com)</sup>"),
            yaxis_title="Total Review Count (All Products)",
            height=400,
            yaxis=dict(rangemode="tozero"),
            margin=dict(l=60, r=20, t=50, b=40),
        )
        return fig

    elif metric_type == "rating":
        # Weighted average rating over time
        all_dates = set()
        for p in products:
            for r in p["rating_history"]:
                all_dates.add(r["date"][:10])
        all_dates = sorted(all_dates)

        if not all_dates:
            return go.Figure()

        # Build per-product rating series, forward-fill
        product_ratings = {}
        product_weights = {}
        for p in products:
            rh = {r["date"][:10]: r["rating"] for r in p["rating_history"]}
            ch = {c["date"][:10]: c["count"] for c in p["review_count_history"]}
            product_ratings[p["representative_asin"]] = rh
            product_weights[p["representative_asin"]] = ch

        dates_out = []
        ratings_out = []

        last_ratings = {asin: None for asin in product_ratings}
        last_weights = {asin: 1 for asin in product_ratings}

        for d in all_dates:
            for asin in product_ratings:
                if d in product_ratings[asin]:
                    last_ratings[asin] = product_ratings[asin][d]
                if d in product_weights.get(asin, {}):
                    last_weights[asin] = product_weights[asin][d]

            total_w = 0
            weighted_sum = 0
            for asin in product_ratings:
                if last_ratings[asin] is not None:
                    w = last_weights[asin]
                    weighted_sum += last_ratings[asin] * w
                    total_w += w

            if total_w > 0:
                dates_out.append(d)
                ratings_out.append(weighted_sum / total_w)

        # Downsample to monthly for readability
        df = pd.DataFrame({"date": pd.to_datetime(dates_out), "rating": ratings_out})
        df = df.set_index("date").resample("MS").mean().dropna()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df["rating"], mode="lines+markers",
            name="Weighted Avg Rating (All Products)",
            line=dict(width=3, color="#00d4aa"),
            marker=dict(size=5),
        ))
        fig.add_hline(y=4.0, line_dash="dot", line_color="green", opacity=0.3)
        fig.update_layout(
            title=dict(text=f"{title}<br><sup>Source: Keepa API (keepa.com)</sup>"),
            yaxis_title="Weighted Avg Rating",
            yaxis=dict(range=[2.5, 5.1], dtick=0.5),
            height=400,
            margin=dict(l=60, r=20, t=50, b=40),
        )
        return fig


def make_sales_rank_chart(products, title):
    fig = go.Figure()
    for i, p in enumerate(products):
        rank_data = p.get("monthly_avg_rank", {})
        if not rank_data or len(rank_data) < 2:
            continue
        months = sorted(rank_data.keys())
        dates = [datetime.strptime(m, "%Y-%m") for m in months]
        ranks = [rank_data[m] for m in months]
        fig.add_trace(go.Scatter(
            x=dates, y=ranks, mode="lines",
            name=product_name(p, short=True),
            line=dict(width=1.8, color=PRODUCT_COLORS[i % len(PRODUCT_COLORS)]),
        ))

    # Add reference bands for sales rank context
    rank_bands = [
        (0, 1_000, "Top seller", "rgba(0,212,170,0.12)"),
        (1_000, 10_000, "Strong seller", "rgba(56,182,255,0.10)"),
        (10_000, 50_000, "Moderate seller", "rgba(255,217,61,0.08)"),
        (50_000, 200_000, "Low volume", "rgba(255,107,107,0.08)"),
        (200_000, 1_000_000, "Very low volume / niche", "rgba(192,132,252,0.08)"),
    ]
    for y0, y1, label, color in rank_bands:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0)

    fig.update_layout(
        title=dict(text=f"{title}<br><sup>Lower rank = more sales. Rank #1 is the best-selling product in the category.</sup>"),
        yaxis_title="Amazon Sales Rank",
        yaxis=dict(autorange="reversed"),
        height=450,
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5, font=dict(size=10)),
        margin=dict(l=60, r=20, t=60, b=80),
    )
    return fig


# ─── Pages ──────────────────────────────────────────────────────────────────

def page_executive_summary(biz_data):
    st.header("Executive Summary")
    st.caption("Based on Keepa API data | All products deduplicated by parent ASIN | Filtered by selected date range")

    col1, col2 = st.columns(2)

    for col, biz_name in zip([col1, col2], ["Wallaroo Wallets", "TeacherFav"]):
        m = biz_data[biz_name]["metrics"]
        with col:
            st.subheader(biz_name)
            st.metric("Total Reviews", f"{m['total_reviews']:,}")
            st.metric("Weighted Avg Rating", f"{m['weighted_avg_rating']:.1f}" if m["weighted_avg_rating"] else "?")
            st.metric("Unique Products", f"{m['n_products']} ({m['n_variations']} variants)")
            st.metric("Main Product Reviews", f"{m['main_reviews']:,}")
            st.metric("Main Product Rating", f"{m['main_rating']:.1f}" if m["main_rating"] else "?")
            st.metric("Review Growth (reviews/mo)", f"{m['main_growth_per_mo']:.1f}")

    st.divider()

    # Comparison table
    w = biz_data["Wallaroo Wallets"]["metrics"]
    t = biz_data["TeacherFav"]["metrics"]

    table_data = {
        "Metric": [
            "Unique Products (Variants)",
            "Total Reviews (All Products)",
            "Weighted Avg Rating",
            "Main Product",
            "Main Product Rating",
            "Main Product Reviews",
            "Rating Trend (Main)",
            "Review Growth (Main, reviews/mo)",
            "Review Growth Direction",
            "Data Range (Main)",
            "Data Source",
        ],
        "Wallaroo Wallets": [
            f"{w['n_products']} ({w['n_variations']})",
            f"{w['total_reviews']:,}",
            f"{w['weighted_avg_rating']:.1f}" if w["weighted_avg_rating"] else "?",
            short_title(w["main_title"], 50),
            f"{w['main_rating']:.1f}" if w["main_rating"] else "?",
            f"{w['main_reviews']:,}",
            f"{w['main_r_dir']} ({w['main_early_r']:.1f} -> {w['main_recent_r']:.1f})",
            f"{w['main_growth_per_mo']:.1f} reviews/mo",
            f"{w['growth_dir']} (early {w['early_growth_mo']:.1f} -> recent {w['recent_growth_mo']:.1f} reviews/mo)",
            f"{w['main_first_date']} to {w['main_last_date']}",
            "Keepa API (complete)",
        ],
        "TeacherFav": [
            f"{t['n_products']} ({t['n_variations']})",
            f"{t['total_reviews']:,}",
            f"{t['weighted_avg_rating']:.1f}" if t["weighted_avg_rating"] else "?",
            short_title(t["main_title"], 50),
            f"{t['main_rating']:.1f}" if t["main_rating"] else "?",
            f"{t['main_reviews']:,}",
            f"{t['main_r_dir']} ({t['main_early_r']:.1f} -> {t['main_recent_r']:.1f})",
            f"{t['main_growth_per_mo']:.1f} reviews/mo",
            f"{t['growth_dir']} (early {t['early_growth_mo']:.1f} -> recent {t['recent_growth_mo']:.1f} reviews/mo)",
            f"{t['main_first_date']} to {t['main_last_date']}",
            "Keepa API (complete)",
        ],
    }
    st.dataframe(pd.DataFrame(table_data).set_index("Metric"), use_container_width=True)

    with st.expander("Metric Definitions, Review Integrity & Date Range Impact", expanded=False):
        for biz_name in ["Wallaroo Wallets", "TeacherFav"]:
            m = biz_data[biz_name]["metrics"]
            integrity = compute_integrity_signals(biz_data[biz_name]["products"])
            flagged = [s for s in integrity if not s["looks_organic"]]

            st.markdown(f"**{biz_name}**")
            st.markdown(
                f"Data covers {m['main_first_date']} to {m['main_last_date']} (main product). "
                f"Review growth is {m['main_growth_per_mo']:.1f} reviews/mo — "
                f"{'this is strong velocity for the category.' if m['main_growth_per_mo'] > 30 else 'moderate velocity.'}"
            )
            if not flagged:
                st.markdown(f"No manipulation signals detected across {len(integrity)} products. Review growth looks organic.")
            else:
                for s in flagged:
                    st.markdown(f"- **{s['name']}**: {'; '.join(s['red_flags'])}")
            st.markdown("")

        st.markdown("""
**Reading the comparison table:**
- **Weighted Avg Rating** — products with more reviews have more weight. A 4.6 product with 7,000 reviews dominates over a 3.4 product with 16 reviews.
- **Rating Trend** — STABLE/IMPROVING/DECLINING compares first half vs second half of the selected period. The numbers in parentheses show the actual average shift.
- **Review Growth Direction** — ACCELERATING means the product is gaining reviews faster recently; DECELERATING means it's slowing down.
- **Data Range** — the period covered by Keepa data within your selected date range filter.

**Date range:** All values on this page reflect your sidebar date range filter. Change it to zoom into recent performance or see the full historical picture.
""")


def page_business_analysis(biz_data, biz_name):
    products = biz_data[biz_name]["products"]
    metrics = biz_data[biz_name]["metrics"]
    main = products[0]

    st.header(f"{biz_name} — Review Analysis")
    st.caption("Filtered by selected date range")

    # KPI row
    cols = st.columns(5)
    cols[0].metric("Products", f"{metrics['n_products']} ({metrics['n_variations']} var)")
    cols[1].metric("Total Reviews", f"{metrics['total_reviews']:,}")
    cols[2].metric("Avg Rating", f"{metrics['weighted_avg_rating']:.1f}" if metrics["weighted_avg_rating"] else "?")
    cols[3].metric("Review Growth (reviews/mo)", f"{metrics['main_growth_per_mo']:.1f}")
    cols[4].metric("Rating Trend", metrics["main_r_dir"])

    st.divider()

    # Review integrity + context
    integrity = compute_integrity_signals(products)
    main_sig = integrity[0]  # main product (sorted by reviews desc)

    with st.expander("Metric Definitions, Review Integrity & Date Range Impact", expanded=False):
        st.markdown(f"""
**You're looking at** a summary of {biz_name}'s Amazon review data from Keepa, filtered to your selected date range.

**Reading the numbers:**
- **Products** — number of unique products (deduplicated by parent ASIN). The number in parentheses is total color/style variants sharing those review pools.
- **Total Reviews** — sum of review counts across all products at the end of the selected period.
- **Avg Rating** — weighted average rating across all products, weighted by review count (so a product with 7,000 reviews matters more than one with 100).
- **Review Growth** — how many new reviews the main product gains per month on average over the selected period. Higher = more customers buying.
- **Rating Trend** — compares the average rating in the first half vs second half of the selected period. STABLE means quality hasn't changed; DECLINING means recent ratings are lower.

**Review Integrity Assessment**
""")

        for sig in integrity:
            if sig["looks_organic"] and not sig["red_flags"]:
                st.markdown(f"- **{sig['name']}**: Reviews look organic. Steady growth, no Amazon purges, stable ratings{' (std dev: ' + str(sig['rating_std']) + ')' if sig['rating_std'] is not None else ''}.")
            elif sig["looks_organic"]:
                st.markdown(f"- **{sig['name']}**: Mostly organic. {'; '.join(sig['red_flags'])}.")
            else:
                st.markdown(f"- **{sig['name']}**: **Review flags detected.** {'; '.join(sig['red_flags'])}.")

        flagged = [s for s in integrity if not s["looks_organic"]]
        if not flagged:
            st.success("Overall: No significant manipulation signals detected across the portfolio. Review growth patterns are consistent and Amazon has not performed major review purges.")
        else:
            st.warning(f"Overall: {len(flagged)} product(s) have review flags worth investigating. See details above.")

        st.markdown(f"""
**How the date range affects these values:**
Changing the date range in the sidebar recalculates everything on this page. Narrowing to the last 1-2 years isolates recent performance — useful for spotting if a once-strong product is now declining. The full range gives the historical average, which smooths out short-term noise but can hide recent trends.
""")

    tab1, tab2, tab3, tab4 = st.tabs(["Main Product", "Portfolio", "All Products", "Product Table"])

    # ─── 1. Main Product Only ───
    with tab1:
        st.subheader(f"Main Product: {short_title(main['title'])}")
        st.caption(
            f"ASIN: [{main['representative_asin']}]({amazon_link(main['representative_asin'])}) | "
            f"[Keepa]({keepa_link(main['representative_asin'])}) | "
            f"{main['shared_reviews']:,} reviews | {main['num_variations']} variants"
        )

        # Data range for this product
        main_ch = main["review_count_history"]
        main_rh = main["rating_history"]
        if main_ch:
            st.caption(
                f"Data range: {main_ch[0]['date'][:7]} to {main_ch[-1]['date'][:7]} "
                f"({len(main_ch)} review observations, {len(main_rh)} rating observations)"
            )
        if main_sig["red_flags"]:
            st.caption(f"Review flags: {'; '.join(main_sig['red_flags'])}")
        else:
            st.caption("Review integrity: No manipulation signals detected — steady growth, no Amazon purges.")

        fig = make_rating_chart(
            [main],
            f"Rating History — {product_name(main, short=True)}",
            colors=["#00d4aa"],
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = make_review_count_chart(
            [main],
            f"Review Count Growth — {product_name(main, short=True)}",
            colors=["#00d4aa"],
        )
        st.plotly_chart(fig, use_container_width=True)

        # Reviews per month velocity chart
        if len(main_ch) >= 4:
            monthly_data = {}
            for c in main_ch:
                ym = c["date"][:7]
                monthly_data[ym] = c["count"]
            months_sorted = sorted(monthly_data.keys())
            velocity_months = []
            velocity_values = []
            for i in range(1, len(months_sorted)):
                prev_m, curr_m = months_sorted[i - 1], months_sorted[i]
                prev_dt = datetime.strptime(prev_m, "%Y-%m")
                curr_dt = datetime.strptime(curr_m, "%Y-%m")
                days_between = max(1, (curr_dt - prev_dt).days)
                new_reviews = monthly_data[curr_m] - monthly_data[prev_m]
                per_month = new_reviews / (days_between / 30.44)
                velocity_months.append(curr_dt)
                velocity_values.append(round(per_month, 1))

            fig_vel = go.Figure()
            fig_vel.add_trace(go.Bar(
                x=velocity_months,
                y=velocity_values,
                marker_color=["#00d4aa" if v >= 0 else "#ff6b6b" for v in velocity_values],
            ))
            fig_vel.update_layout(
                title=dict(text=f"New Reviews per Month — {product_name(main, short=True)}<br><sup>Rate of new reviews added each month (not cumulative). Negative bars = Amazon removed reviews that month.</sup>"),
                yaxis_title="New Reviews / Month",
                height=400,
                margin=dict(l=60, r=20, t=70, b=40),
            )
            st.plotly_chart(fig_vel, use_container_width=True)

    # ─── 2. All Products Combined ───
    with tab2:
        st.subheader("All Products Combined (Portfolio Performance)")

        fig = make_combined_chart(
            products, "rating",
            f"{biz_name} — Weighted Avg Rating (All Products Combined)",
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = make_combined_chart(
            products, "reviews",
            f"{biz_name} — Total Reviews (All Products Combined)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ─── 3. All Individual Products ───
    with tab3:
        st.subheader("All Individual Products (Comprehensive Breakdown)")

        fig = make_rating_chart(products, f"{biz_name} — Rating History (All Products)")
        st.plotly_chart(fig, use_container_width=True)

        fig = make_review_count_chart(products, f"{biz_name} — Review Count Growth (All Products)")
        st.plotly_chart(fig, use_container_width=True)

        # Sales rank
        fig = make_sales_rank_chart(products, f"{biz_name} — Monthly Avg Sales Rank")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Sales rank shows relative sales position within the product's Amazon category. A rank of 1,000 means it outsells ~99.9% of products in that category. Chart bands: green < 1K (top seller), blue 1K\u201310K (strong), yellow 10K\u201350K (moderate), red 50K\u2013200K (low volume), purple > 200K (niche).")

    # ─── Product Breakdown Table ───
    with tab4:
        st.subheader("Product Breakdown Table")

        rows = []
        for p in products:
            rh = p["rating_history"]
            ch = p["review_count_history"]
            rating = f"{rh[-1]['rating']:.1f}" if rh else "?"

            if len(ch) >= 2:
                first_date = datetime.fromisoformat(ch[0]["date"])
                last_date = datetime.fromisoformat(ch[-1]["date"])
                months = max(1, (last_date - first_date).days / 30.44)
                growth = (ch[-1]["count"] - ch[0]["count"]) / months
                growth_str = f"{growth:.1f}"
                since = ch[0]["date"][:7]
            else:
                growth_str = "?"
                since = "?"

            if len(rh) >= 4:
                mid = len(rh) // 2
                early = sum(r["rating"] for r in rh[:mid]) / mid
                recent = sum(r["rating"] for r in rh[mid:]) / (len(rh) - mid)
                trend = "STABLE" if abs(recent - early) <= 0.05 else ("UP" if recent > early else "DOWN")
            else:
                trend = "?"

            rows.append({
                "ASIN": p["representative_asin"],
                "Product": product_name(p, short=True),
                "Amazon": amazon_link(p["representative_asin"]),
                "Keepa": keepa_link(p["representative_asin"]),
                "Rating": rating,
                "Reviews": f"{p['shared_reviews']:,}",
                "Variants": p["num_variations"],
                "Reviews/mo": growth_str,
                "Rating Trend": trend,
                "Data Range": f"{since} to {ch[-1]['date'][:7]}" if len(ch) >= 2 else since,
            })

        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            column_config={
                "Amazon": st.column_config.LinkColumn("Amazon", display_text="View"),
                "Keepa": st.column_config.LinkColumn("Keepa", display_text="View"),
            },
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("ℹ️ Column Definitions"):
            st.markdown("""
| Column | Definition |
|--------|-----------|
| **ASIN** | Amazon Standard Identification Number — unique product identifier |
| **Product** | Short display name for the product |
| **Amazon / Keepa** | Links to the product's Amazon listing and Keepa tracking page |
| **Rating** | Most recent star rating from Keepa data (last data point in rating history). Keepa stores ratings as integers (e.g. 43 = 4.3 stars), divided by 10 for display. Shows "?" if no rating data exists. |
| **Reviews** | Total review count shared across all variants of this product (from `shared_reviews`). This is the review count at the most recent Keepa data point. |
| **Variants** | Number of color/style variations sharing the same review pool (from `num_variations`) |
| **Reviews/mo** | Average new reviews per month = `(last_count - first_count) / months_between`. Calculated over the full data range. Shows "?" if fewer than 2 data points. |
| **Rating Trend** | Compares average rating in the first half vs second half of rating history. **UP** = recent avg > early avg by >0.05, **DOWN** = recent avg < early avg by >0.05, **STABLE** = within ±0.05. Shows "?" if fewer than 4 rating data points. |
| **Data Range** | Date range of Keepa review count data (first to last observation), formatted as YYYY-MM. |

---

**Example:** A product with 10 rating samples \\[4.5, 4.5, 4.4, 4.4, 4.3, 4.2, 4.2, 4.1, 4.1, 4.0\\]:
- **Rating** = 4.0 (last data point)
- **Rating Trend** = early half avg (4.5+4.5+4.4+4.4+4.3)/5 = 4.42, recent half avg (4.2+4.2+4.1+4.1+4.0)/5 = 4.12 → difference is −0.30 → **DOWN**
- If first review count was 100 on 2023-01 and last was 220 on 2024-01 (12 months): **Reviews/mo** = (220−100)/12 = 10.0

---

**How the date range affects this table:**
All values are computed from data within your selected date range. Rating uses the last data point in range. Reviews/mo divides review growth across the months in range. Rating Trend splits the in-range data in half. Narrowing the range means fewer data points — trends become less reliable but more recent. If a product has no data in the selected range, values show "?".
""")


def page_evaluation(biz_data, biz_name):
    products = biz_data[biz_name]["products"]
    m = biz_data[biz_name]["metrics"]

    st.header(f"{biz_name} — Acquisition Evaluation")
    st.caption("Based on Keepa API data, filtered by selected date range. Verify financials independently.")

    main = products[0]
    main_ch = main["review_count_history"]
    if main_ch:
        st.caption(
            f"Showing data from {main_ch[0]['date'][:7]} to {main_ch[-1]['date'][:7]} "
            f"(main product: {product_name(main, short=True)}). "
            f"Adjust the date range in the sidebar to change the evaluation window."
        )

    if biz_name == "Wallaroo Wallets":
        st.markdown(
            "Listing: [Flippa](https://flippa.com/12260386-amazon-s-favorite-leather-phone-wallet-since-2016-"
            "hundreds-of-thousands-of-customers-over-7-000-reviews-for-the-hero-sku-with-an-average-4-6-star-rating)"
        )
    else:
        st.markdown(
            "Listing: [Empire Flippers #92221](https://app.empireflippers.com/listing/unlocked/92221)"
        )

    if biz_name == "Wallaroo Wallets":
        verdict = "CAUTIOUSLY POSITIVE"
        verdict_emoji = "🟢"
        pros = [
            f"10-year brand, 4.6 stars across {m['main_reviews']:,} reviews — exceptional longevity",
            f"Rating STABLE at 4.4-4.6 over full history — no quality degradation",
            f"Main product growing at {m['main_growth_per_mo']:.0f} new reviews/month consistently",
            f"{m['n_variations']} color variants from just {m['n_products']} products = efficient SKU strategy",
            "Ultra-low unit cost ($1.65-1.81) = massive tariff buffer",
            "Category leader in phone wallets — strong organic search positioning",
        ]
        cons = [
            f"Only {m['n_products']} unique products — high concentration in Phone Wallet ({m['main_reviews']:,}/{m['total_reviews']:,} reviews)",
            "MagSafe & Wristlet trending DOWN (4.5->4.1, 4.7->4.2) — newer products struggling",
            "No P&L data (Flippa listing) — cannot verify financial claims",
            "Competitive market: phone wallets have low barriers to entry",
            f"Card Holder & Wristlet have low volume ({159+114} reviews combined) — limited traction",
        ]
        questions = [
            "What are the actual revenue and profit numbers? Need P&L or bank statements",
            "Why are newer products (MagSafe, Wristlet) rated lower? Quality or expectations issue?",
            "What is the product cost breakdown including tariffs?",
            "What % of revenue comes from the main Phone Wallet vs other products?",
            "Is there a supplier dependency risk? Backup manufacturers?",
        ]
        summary = (
            f"The main Phone Wallet has a rock-solid 4.6 rating across {m['main_reviews']:,} reviews with stable "
            f"trends — a 10-year track record of consistent quality. However, the business is essentially a "
            f"one-product company — the Phone Wallet accounts for {m['main_reviews']/m['total_reviews']*100:.0f}% "
            f"of all reviews. Newer products (MagSafe at 4.2, Wristlet at 4.2) show declining ratings. The core "
            f"product is strong but diversification is weak. CRITICAL: No P&L available — cannot make a financial "
            f"assessment. Proceed only with verified financials."
        )
    else:
        verdict = "MIXED — STRONG CORE, WEAK TAILS"
        verdict_emoji = "🟡"
        pros = [
            f"Core Sand Timer line at 4.4 stars with {products[0]['shared_reviews']:,} reviews — market leader",
            f"Toothbrush Timer at 4.6 stars ({products[1]['shared_reviews']:,} reviews) — STRONGER than initially reported",
            f"16 color variants of main timer = strong shelf presence and color choice moat",
            "P&L verified: $46K/mo revenue, 10% margin via Empire Flippers",
            f"Review growth at {m['main_growth_per_mo']:.0f} new reviews/month on main product — healthy velocity",
            "Two strong products (Sand Timer 4.4, Toothbrush Timer 4.6) — diversified within niche",
        ]
        cons = [
            f"Rating trend slightly DOWN on main product ({m['main_early_r']:.1f} -> {m['main_recent_r']:.1f})",
            f"60-Minute Timer at 3.4 stars (16 reviews) — worst performing product",
            "10% margin is thin — vulnerable to cost increases or ad spend pressure",
            f"Review growth {m['growth_dir'].lower()} — early {m['early_growth_mo']:.0f} reviews/mo vs recent {m['recent_growth_mo']:.0f} reviews/mo",
            f"Dashboard: FAILS age (<5yr) and multiple (>30x). 2 FAIL criteria on acquisition checklist",
        ]
        questions = [
            "Can the 60-Minute Timer (3.4 stars) be discontinued or redesigned?",
            "Why is margin only 10% on $46K/mo revenue? Where are costs going?",
            "Can TaCOS / Total Ad Cost of Sales (12.9%) be reduced without losing rank?",
            "What is the defect/return rate for the Sand Timer line?",
            "Would seller accept $100K-120K (19-23x) given the 2 FAIL criteria?",
        ]
        summary = (
            f"The Toothbrush Timer at 4.6 stars is the strongest product in the portfolio. The core Sand Timer "
            f"line at 4.4 stars with {products[0]['shared_reviews']:,} reviews is solid. The only weak spot is the "
            f"60-Minute Timer at 3.4 stars with just 16 reviews. Overall product quality is strong across the "
            f"catalog. However, financial concerns remain: 10% margin is thin, and the listing fails 2 acquisition "
            f"criteria (age <5yr, multiple >30x). Worth pursuing at a lower price."
        )

    st.markdown(f"### {verdict_emoji} Verdict: **{verdict}**")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Pros")
        for p in pros:
            st.markdown(f"- {p}")
    with col2:
        st.markdown("#### Cons")
        for c in cons:
            st.markdown(f"- {c}")

    st.markdown("#### Key Questions for Seller")
    for q in questions:
        st.markdown(f"- {q}")

    st.divider()

    st.markdown("#### AI Assessment")
    st.info(summary)

    # Wallaroo seller-reported financials (no P&L available, only summary table)
    if biz_name == "Wallaroo Wallets":
        st.divider()
        st.markdown("#### Seller-Reported Financials")
        st.warning(
            "⚠️ **Unverified data** — The seller provided this summary but has not shared a P&L, "
            "bank statements, or Amazon payout reports. Verify independently before relying on these numbers."
        )
        wallaroo_fin = pd.DataFrame({
            "": ["Gross Product Sales", "Net Revenue", "Total Amazon Fees", "FBA Reimbursements"],
            "2021 (Sep-Dec)": ["$225,514", "$214,864", "($78,328)", "$1,426"],
            "2022": ["$859,439", "$828,268", "($319,738)", "$8,780"],
            "2023": ["$738,150", "$696,466", "($288,554)", "$13,124"],
            "2024": ["$469,138", "$446,495", "($178,623)", "$5,753"],
        })
        st.dataframe(wallaroo_fin, use_container_width=True, hide_index=True)
        st.markdown(
            "**Notable trends:**\n"
            "- Revenue peaked in 2022 ($859K gross) and has declined since: "
            "-14% in 2023, -36% in 2024\n"
            "- Amazon fees consistently ~38-41% of net revenue\n"
            "- 2021 is partial year (Sep-Dec only) — annualized would be ~$676K gross\n"
            "- No COGS or profit data provided — margin is unknown"
        )

    # P&L Financial Data (if available)
    pnl_raw = load_pnl(biz_name)
    if pnl_raw:
        # Filter P&L to selected date range
        start_ym = st.session_state.get("date_start", date(2020, 1, 1))
        end_ym = st.session_state.get("date_end", date.today())
        if isinstance(start_ym, date):
            start_ym = start_ym.strftime("%Y-%m")
        if isinstance(end_ym, date):
            end_ym = end_ym.strftime("%Y-%m")
        pnl = {}
        for key, series in pnl_raw.items():
            pnl[key] = {m: v for m, v in series.items() if start_ym <= m <= end_ym}

        st.divider()
        st.markdown("#### Financial Data (Empire Flippers P&L)")

        # Revenue & Net Income chart
        rev = pnl.get("Total Revenue", {})
        net = pnl.get("Net Income", {})
        if rev:
            months_sorted = sorted(rev.keys())
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[datetime.strptime(m, "%Y-%m") for m in months_sorted],
                y=[rev[m] for m in months_sorted],
                name="Revenue",
                marker_color="#3498db",
            ))
            if net:
                fig.add_trace(go.Bar(
                    x=[datetime.strptime(m, "%Y-%m") for m in months_sorted if m in net],
                    y=[net[m] for m in months_sorted if m in net],
                    name="Net Income",
                    marker_color="#2ecc71",
                ))
            fig.update_layout(
                title=dict(text="Monthly Revenue & Net Income<br><sup>Source: Empire Flippers P&L (Listing #92221)</sup>"),
                yaxis_title="USD",
                height=400,
                barmode="group",
                margin=dict(l=60, r=20, t=50, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Margin chart
        margin = pnl.get("Net Profit Margin (12mo trailing)", {})
        tacos = pnl.get("TaCOS (Total Ad Cost of Sales)", {})
        if margin:
            months_sorted = sorted(margin.keys())
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[datetime.strptime(m, "%Y-%m") for m in months_sorted],
                y=[margin[m] for m in months_sorted],
                name="Net Profit Margin (12mo trailing)",
                mode="lines+markers",
                line=dict(width=2, color="#2ecc71"),
            ))
            if tacos:
                fig.add_trace(go.Scatter(
                    x=[datetime.strptime(m, "%Y-%m") for m in months_sorted if m in tacos],
                    y=[tacos[m] for m in months_sorted if m in tacos],
                    name="TaCOS (Total Ad Cost of Sales)",
                    mode="lines+markers",
                    line=dict(width=2, color="#e74c3c"),
                ))
            fig.update_layout(
                title=dict(text="Margins & Ad Efficiency<br><sup>Source: Empire Flippers P&L (Listing #92221)</sup>"),
                yaxis_title="%",
                height=350,
                margin=dict(l=60, r=20, t=50, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.caption("Source: Empire Flippers verified P&L (Listing #92221)")


def _period_summary(products, label, cutoff_iso):
    """Render a summary block for a given time period cutoff."""
    st.markdown(f"**{label}**")
    rows = []
    for p in products:
        rh = [r for r in p["rating_history"] if r["date"][:10] >= cutoff_iso]
        ch = [c for c in p["review_count_history"] if c["date"][:10] >= cutoff_iso]
        if not rh and not ch:
            continue

        avg_r = sum(r["rating"] for r in rh) / len(rh) if rh else None
        reviews_added = (ch[-1]["count"] - ch[0]["count"]) if len(ch) >= 2 else None
        months_span = ((datetime.fromisoformat(ch[-1]["date"]) - datetime.fromisoformat(ch[0]["date"])).days / 30.44) if len(ch) >= 2 else None
        growth_rate = reviews_added / months_span if reviews_added is not None and months_span and months_span > 0 else None

        rows.append({
            "Product": product_name(p, short=True),
            "Avg Rating": f"{avg_r:.2f}" if avg_r else "\u2014",
            "Reviews Added": f"{reviews_added:,}" if reviews_added is not None else "\u2014",
            "Growth (rev/mo)": f"{growth_rate:.1f}" if growth_rate else "\u2014",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def page_yearly_breakdown(biz_data, biz_name):
    """Static yearly breakdown showing rating and review trends year over year.
    This page always uses full historical data — it is NOT affected by the date range filter."""
    products = biz_data[biz_name]["products"]
    st.header(f"{biz_name} — Yearly Breakdown")
    st.caption("Source: Keepa API (keepa.com) | Static summary — not affected by date range filter")

    # Product overview table
    overview_rows = []
    for p in products:
        ch = p["review_count_history"]
        overview_rows.append({
            "Product": product_name(p, short=True),
            "ASIN": p["representative_asin"],
            "Reviews": f"{p['shared_reviews']:,}",
            "Variants": p["num_variations"],
            "Data Range": f"{ch[0]['date'][:7]} to {ch[-1]['date'][:7]}" if ch else "No data",
        })
    st.dataframe(pd.DataFrame(overview_rows), use_container_width=True, hide_index=True)

    # Filter to products with data
    products_with_data = [p for p in products if p["rating_history"] or p["review_count_history"]]
    if not products_with_data:
        st.info("No products with rating or review data.")
        return

    product_tabs = st.tabs([product_name(p, short=True) for p in products_with_data])

    for tab, p in zip(product_tabs, products_with_data):
      with tab:
        rh = p["rating_history"]
        ch = p["review_count_history"]
        if not rh and not ch:
            continue

        st.subheader(product_name(p, short=True))
        st.caption(
            f"ASIN: [{p['representative_asin']}]({amazon_link(p['representative_asin'])}) | "
            f"[Keepa]({keepa_link(p['representative_asin'])})"
        )
        if ch:
            range_text = (
                f"Review data: {ch[0]['date'][:7]} to {ch[-1]['date'][:7]} "
                f"({len(ch)} observations)"
            )
            if rh:
                range_text += (
                    f" | Rating data: {rh[0]['date'][:7]} to {rh[-1]['date'][:7]} "
                    f"({len(rh)} observations)"
                )
            else:
                range_text += " | No rating data"
            st.caption(range_text)

        # Group data by year
        rating_by_year = {}
        for r in rh:
            year = r["date"][:4]
            rating_by_year.setdefault(year, []).append(r["rating"])

        review_by_year = {}
        for c in ch:
            year = c["date"][:4]
            review_by_year.setdefault(year, []).append(c["count"])

        years = sorted(set(list(rating_by_year.keys()) + list(review_by_year.keys())))

        # Build year-over-year table
        rows = []
        prev_end_count = None
        last_known_avg = None
        last_known_min = None
        last_known_max = None
        for year in years:
            ratings = rating_by_year.get(year, [])
            counts = review_by_year.get(year, [])

            avg_rating = sum(ratings) / len(ratings) if ratings else None
            if avg_rating is not None:
                last_known_avg = avg_rating
                last_known_min = min(ratings)
                last_known_max = max(ratings)

            end_count = counts[-1] if counts else None
            start_count = counts[0] if counts else None
            reviews_added = (end_count - start_count) if end_count is not None and start_count is not None else None

            # Amazon removed reviews between end of prev year and start of this year
            if prev_end_count is not None and start_count is not None:
                gap = start_count - prev_end_count
                removed_str = f"{gap:,}" if gap < 0 else "0"
            else:
                removed_str = "No prior year"

            row = {
                "Year": year,
                "Avg Rating": f"{avg_rating:.2f}" if avg_rating else (f"~{last_known_avg:.2f} (carried forward)" if last_known_avg else "No data"),
                "Rating Min": f"{min(ratings):.1f}" if ratings else (f"~{last_known_min:.1f}" if last_known_min else "No data"),
                "Rating Max": f"{max(ratings):.1f}" if ratings else (f"~{last_known_max:.1f}" if last_known_max else "No data"),
                "Reviews (Start of Year)": f"{start_count:,}" if start_count is not None else "No data",
                "Reviews (End of Year)": f"{end_count:,}" if end_count is not None else "No data",
                "Net Reviews Added": f"{reviews_added:,}" if reviews_added is not None else "No data",
                "Reviews Removed Between Years": removed_str,
            }
            rows.append(row)
            prev_end_count = end_count

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.caption(
            '"Reviews Removed Between Years" = Reviews (Start of Year) minus Reviews (End of Previous Year). '
            "A negative number means Amazon removed reviews between those dates."
        )

        # Yearly charts
        chart_years = [row["Year"] for row in rows]

        # 1. Avg Rating by Year (bar chart)
        chart_ratings = []
        for row in rows:
            val = row["Avg Rating"]
            if val.startswith("~"):
                chart_ratings.append(float(val.split("~")[1].split(" ")[0]))
            elif val == "No data":
                chart_ratings.append(None)
            else:
                chart_ratings.append(float(val))

        if any(r is not None for r in chart_ratings):
            fig_r = go.Figure()
            fig_r.add_trace(go.Bar(
                x=chart_years,
                y=chart_ratings,
                marker_color=["#00d4aa" if r is not None and r >= 4.0 else "#ff6b6b" if r is not None else "#666" for r in chart_ratings],
                text=[f"{r:.2f}" if r is not None else "" for r in chart_ratings],
                textposition="outside",
            ))
            fig_r.add_hline(y=4.0, line_dash="dot", line_color="green", opacity=0.4)
            fig_r.update_layout(
                title=dict(text=f"Average Rating by Year — {product_name(p, short=True)}"),
                yaxis=dict(range=[2.5, 5.2], dtick=0.5, title="Avg Rating"),
                xaxis_title="Year",
                height=350,
                margin=dict(l=60, r=20, t=50, b=40),
                showlegend=False,
            )
            st.plotly_chart(fig_r, use_container_width=True)

        # 2. Net Reviews Added by Year (bar chart)
        chart_reviews = []
        for row in rows:
            val = row["Net Reviews Added"]
            if val == "No data":
                chart_reviews.append(None)
            else:
                chart_reviews.append(int(val.replace(",", "")))

        if any(r is not None for r in chart_reviews):
            fig_rev = go.Figure()
            fig_rev.add_trace(go.Bar(
                x=chart_years,
                y=[r if r is not None else 0 for r in chart_reviews],
                marker_color=["#3498db" if r is not None and r >= 0 else "#ff6b6b" if r is not None else "#666" for r in chart_reviews],
                text=[f"{r:,}" if r is not None else "" for r in chart_reviews],
                textposition="outside",
            ))
            fig_rev.update_layout(
                title=dict(text=f"Net Reviews Added by Year — {product_name(p, short=True)}"),
                yaxis_title="Reviews Added",
                xaxis_title="Year",
                height=350,
                margin=dict(l=60, r=20, t=50, b=40),
                showlegend=False,
            )
            st.plotly_chart(fig_rev, use_container_width=True)

        # Build carried-forward avg ratings for YoY comparison
        carried_avg_by_year = {}
        last_avg_for_yoy = None
        for year in years:
            ratings = rating_by_year.get(year, [])
            if ratings:
                last_avg_for_yoy = sum(ratings) / len(ratings)
            carried_avg_by_year[year] = last_avg_for_yoy

        # Year-over-year comparison
        if len(years) >= 2:
            st.markdown("**Year-over-Year Rating Trend** (avg rating change between years)")
            comparisons = []
            for i in range(1, len(years)):
                y1, y2 = years[i - 1], years[i]
                avg1 = carried_avg_by_year.get(y1)
                avg2 = carried_avg_by_year.get(y2)
                if avg1 is not None and avg2 is not None:
                    r1_actual = bool(rating_by_year.get(y1, []))
                    r2_actual = bool(rating_by_year.get(y2, []))
                    note = " (using carried-forward rating)" if not r1_actual or not r2_actual else ""
                    delta = avg2 - avg1
                    direction = "UP" if delta > 0.05 else "DOWN" if delta < -0.05 else "STABLE"
                    comparisons.append(
                        f"- {y1} to {y2}: avg rating {avg1:.2f} to {avg2:.2f} "
                        f"(rating change: {delta:+.2f}, trend: {direction}){note}"
                    )
                else:
                    missing = y1 if avg1 is None else y2
                    comparisons.append(f"- {y1} to {y2}: No rating data for {missing}")
            st.markdown("\n".join(comparisons))

        st.divider()

    # Period Summaries
    today = date.today()
    _period_summary(
        products,
        "Last 2 Years Summary",
        today.replace(year=today.year - 2).isoformat(),
    )
    st.markdown("")
    _period_summary(
        products,
        "Last 1 Year Summary",
        today.replace(year=today.year - 1).isoformat(),
    )
    st.markdown("")
    six_mo_year = today.year if today.month > 6 else today.year - 1
    six_mo_month = today.month - 6 if today.month > 6 else today.month + 6
    _period_summary(
        products,
        "Last 6 Months Summary",
        date(six_mo_year, six_mo_month, min(today.day, 28)).isoformat(),
    )


WALLAROO_CONVERSATION = """
**Dan Kim** — Thursday, 16 April 2:11 PM
> Accepted NDA and has been allowed access to view listing.
> Dan Kim is a premium buyer and pre-qualified by Flippa.

---

**Lazlo Cocheba** — Thursday, 16 April 2:12 PM
> Hi Dan,
>
> Thanks for signing the NDA. If you have any questions as you review the materials, feel free to reach out at any time. When you have a moment, it would be helpful to understand a bit about your professional background, how you typically fund acquisitions, and any initial questions you may have. If it's easier to discuss anything live, you can book time directly on my calendar here: https://meetings.hubspot.com/lazlo-cocheba
>
> Best,
> Lazlo

---

**Dan Kim** — Friday, 17 April 10:02 PM
> Hi Lazlo,
>
> Thank you for putting this listing together, the brand history and review base caught my attention, and I'd like to take a closer look before deciding whether to move forward.
>
> Would you be open to sharing a longer-term P&L, ideally going back to 2020 or earlier if available? I'd like to understand the full revenue and expense picture across those years.
>
> Looking forward to hearing from you.
>
> Best,
> Dan

---

**Lazlo Cocheba** — Saturday, 18 April 2:59 AM
> Hi Dan,
>
> We don't have a long-term P&L available at this stage, as that's typically shared post-LOI. That said, I've included a screenshot with high-level performance.
>
> Let me know if you have any questions. Happy to walk through the deal in more detail, feel free to grab time on my calendar.
>
> Best,
> Lazlo

---

**Dan Kim** — Tuesday, 21 April 5:16 PM
> Hi Lazlo,
>
> Thank you for sharing that. I'm interested in moving forward, if the deal comes in at or below a 24-month profit multiple, I'd be happy to submit a Letter of Intent.
> Could you confirm where things stand on that basis?
>
> Best,
> Dan

---

**Lazlo Cocheba** — Wednesday, 22 April 2:17 AM
> Hi Dan,
>
> We're open to exploring this. Could you submit an LOI with a preliminary offer? It would also be helpful to include a brief rationale for your valuation.
>
> If there's alignment, we're happy to move forward and provide any and all information requested.
>
> Best,
> Lazlo
"""


def page_seller_conversation():
    st.header("Wallaroo Wallets — Seller Conversation")
    st.caption("Ongoing conversation with Lazlo Cocheba via Flippa")
    st.markdown(
        "Listing: [Flippa](https://flippa.com/12260386-amazon-s-favorite-leather-phone-wallet-since-2016-"
        "hundreds-of-thousands-of-customers-over-7-000-reviews-for-the-hero-sku-with-an-average-4-6-star-rating)"
    )
    st.divider()
    st.markdown(WALLAROO_CONVERSATION)

    st.divider()
    st.markdown("#### Status")
    st.info(
        "Seller has asked Dan to submit an LOI with a preliminary offer and valuation rationale. "
        "P&L will be shared post-LOI. Dan's target: 24-month profit multiple or below."
    )


def page_methodology():
    st.header("Methodology & Definitions")
    st.caption("How metrics are calculated and data is sourced")

    st.markdown("""
### Data Source: Keepa API (keepa.com)

All data comes from [Keepa](https://keepa.com), which tracks Amazon product data continuously.
Keepa provides **complete, accurate** historical data for ratings and review counts.

- **Keepa Product API** with `rating=1` parameter returns csv indices 16 (rating) and 17 (review count)
- **Keepa Seller API** with `storefront=1` discovers all ASINs for a seller
- **Parent ASIN mapping** deduplicates product variations that share review pools

---

### Why We Track Historical Trends

A product with 4.6★ and 8,000 reviews looks great on the surface — but aggregate numbers can hide serious problems. This is why trend analysis is essential for acquisition due diligence:

- **Aggregate ratings are sticky/lagging** — A product with 8,000 reviews at 4.6★ could have quality problems for 6 months and the average barely moves. Recent trend catches this.
- **Detects product/seller changes** — Common Amazon pattern: product builds reputation, then seller cheapens materials or listing gets hijacked. Old reviews mask the change.
- **Spots review manipulation** — Sudden spikes in review volume (especially 5★ clusters) indicate paid reviews or giveaway campaigns. Healthy products accumulate reviews smoothly.
- **Distinguishes "old hit" from "current hit"** — Two products both at 4.5★/5,000 reviews look identical, but one might have gotten 90% of reviews 3 years ago and barely sells now.
- **Review removals are a signal** — Amazon mass-purging reviews (visible as drops in review count charts) indicates past manipulation. This is valuable due diligence data.

---

### Key Metrics

#### Weighted Average Rating
$$
\\text{Weighted Avg Rating} = \\frac{\\sum_{i=1}^{n} (r_i \\times w_i)}{\\sum_{i=1}^{n} w_i}
$$

Where:
- $r_i$ = latest rating for product $i$
- $w_i$ = total review count for product $i$ (used as weight)
- $n$ = number of unique products

This gives more weight to products with more reviews, providing a fairer overall rating.

**Example:** Product A has 4.6 stars (1,000 reviews), Product B has 4.2 stars (200 reviews).
Weighted avg = (4.6 × 1,000 + 4.2 × 200) / (1,000 + 200) = 5,440 / 1,200 = **4.53**
Note: Product A's rating dominates because it has 5× more reviews — this prevents a low-volume product from unfairly dragging the average.

---

#### Review Growth Rate (per month)
$$
\\text{Growth/mo} = \\frac{C_{\\text{last}} - C_{\\text{first}}}{\\text{months}}
$$

Where:
- $C_{\\text{first}}$ = review count at first data point
- $C_{\\text{last}}$ = review count at last data point
- $\\text{months} = \\frac{\\text{days between first and last}}{30.44}$

**Example:** A product starts with 500 reviews in Jan 2022 and reaches 2,000 reviews by Jan 2024.
Growth = (2,000 - 500) / 24 months = **62.5 reviews/month**

---

#### Rating Trend Direction
Compares the average rating in the **first half** vs **second half** of the rating history:

$$
\\text{early\\_avg} = \\frac{1}{\\lfloor n/2 \\rfloor} \\sum_{i=1}^{\\lfloor n/2 \\rfloor} r_i \\qquad
\\text{recent\\_avg} = \\frac{1}{n - \\lfloor n/2 \\rfloor} \\sum_{i=\\lfloor n/2 \\rfloor+1}^{n} r_i
$$

$$
\\Delta = \\text{recent\\_avg} - \\text{early\\_avg}
$$

| Condition | Label |
|-----------|-------|
| $\\Delta > 0.05$ | IMPROVING |
| $\\Delta < -0.05$ | DECLINING |
| $|\\Delta| \\leq 0.05$ | STABLE |

**Example:** Rating history: [4.5, 4.4, 4.3, 4.6, 4.7, 4.8]
First half avg = (4.5 + 4.4 + 4.3) / 3 = 4.40
Second half avg = (4.6 + 4.7 + 4.8) / 3 = 4.70
Delta = +0.30 → **IMPROVING** (exceeds +0.05 threshold)

---

#### Growth Direction
Compares monthly review growth in the **first half** vs **second half** of the review count history:

$$
\\text{early\\_growth} = \\frac{C_{\\text{mid}} - C_{\\text{first}}}{\\text{months}_{\\text{first half}}} \\qquad
\\text{recent\\_growth} = \\frac{C_{\\text{last}} - C_{\\text{mid}}}{\\text{months}_{\\text{second half}}}
$$

| Condition | Label |
|-----------|-------|
| recent > early × 1.1 | ACCELERATING |
| recent < early × 0.9 | DECELERATING |
| otherwise | STEADY |

---

#### Parent ASIN Deduplication
Amazon product variations (colors, sizes) share a single review pool under a **parent ASIN**.
For example, Wallaroo's Phone Wallet has 14 color variants, all sharing ~7,107 reviews.

We use Keepa's `parentAsin` field to group child ASINs, then keep only one representative
per parent, avoiding double-counting reviews.

---

#### Sales Rank
Monthly average of Keepa's sales rank tracking (csv index 3). Lower rank = more sales.
The rank is category-specific (e.g., "Cell Phones & Accessories" for Wallaroo).

---

#### Review Count Drops
Review count charts show the raw cumulative total as reported by Keepa. Drops in the chart
indicate reviews removed by Amazon — this is normal Amazon behavior:
- **Small drops (1-9 reviews)**: Amazon removing individual policy-violating reviews
- **Large drops (100+)**: Amazon mass-purging reviews (e.g., suspected incentivized or fake reviews)

These drops are real data, not glitches, and represent valuable due diligence information.

---

### Combined Charts

**Combined Rating**: Weighted average of all products' ratings at each time point,
weighted by each product's review count at that time.

**Combined Reviews**: Sum of all products' review counts at each time point.
Missing data points are forward-filled before summing.

---

### Data Quality Notes
- Keepa data is sourced directly from Amazon's API — it represents the **true** rating and review count at each point in time
- Rating values from Keepa are integers (e.g., 46 = 4.6 stars), divided by 10 for display
- Keepa timestamps are in "Keepa minutes" (minutes since 2011-01-01 00:00 UTC)
- Review count drops in charts reflect real Amazon review removals, not data errors

---

### Limitations
- **No verified financials for Wallaroo** — Flippa listing does not include a P&L. Revenue/profit claims are unverified.
- **TeacherFav P&L is seller-reported** — Empire Flippers notes that product landed costs are submitted by the seller and not verified.
- **Review counts ≠ sales** — Review count correlates with but does not equal units sold. Typical review rates are 1-5% of purchases.
- **Sales rank is relative, not absolute** — Rank depends on category size and competition. A rank of 5,000 in "Cell Phones & Accessories" means different volume than 5,000 in "Toys & Games".
- **Rating data has gaps** — Keepa samples ratings less frequently than review counts. Some product/year combinations have no rating samples even though Keepa is still tracking review counts. The yearly breakdown carries forward the last known rating where possible (marked "carried forward" in the table).
- **No competitor analysis** — Dashboard only shows these products, not competing products in the same categories.
- **Date range filter scope** — The date range filter affects Executive Summary, Business Analysis, and Evaluation pages. Yearly Breakdown and Seller Conversation use full historical data regardless of the filter.

---

### Formula Verification
These formulas use standard e-commerce analytics approaches:
- **Weighted average** is the standard method for combining ratings across products of different sizes. An unweighted average would let a product with 16 reviews have equal influence to one with 7,107 reviews.
- **Linear growth rate** (reviews/month) provides a simple, interpretable measure. More complex models (exponential, polynomial) would overfit to noise in review data.
- **Half-split comparison** for trend direction is robust against outliers — comparing halves rather than individual data points avoids flagging temporary fluctuations as trends.
- **Review count as weight** for combined ratings reflects each product's market presence — a product customers actually buy should have proportionally more influence on the portfolio rating.
""")


# ─── Main App ───────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Amazon FBA Review Analysis",
        page_icon="📊",
        layout="wide",
    )

    st.title("Amazon FBA Review Analysis Dashboard")
    st.caption("Wallaroo Wallets & TeacherFav | Keepa API Data | April 2026")

    biz_data = load_data()

    # Sidebar navigation
    page = st.sidebar.radio(
        "Navigation",
        [
            "Executive Summary",
            "Wallaroo Wallets",
            "TeacherFav",
            "Wallaroo — Evaluation",
            "TeacherFav — Evaluation",
            "Wallaroo — Yearly Breakdown",
            "TeacherFav — Yearly Breakdown",
            "Wallaroo — Seller Conversation",
            "Methodology & Definitions",
        ],
    )

    # Date range filter
    st.sidebar.divider()
    min_date, max_date = get_date_range(biz_data)
    st.sidebar.markdown("**Date Range Filter**")
    st.sidebar.caption("Affects: Executive Summary, Business Analysis, Evaluation pages. Does NOT affect: Yearly Breakdown, Seller Conversation, Methodology.")
    start_date = st.sidebar.date_input("Start", value=min_date, min_value=min_date, max_value=max_date, key="date_start")
    end_date = st.sidebar.date_input("End", value=max_date, min_value=min_date, max_value=max_date, key="date_end")

    if start_date >= end_date:
        st.sidebar.error("Start date must be before end date.")
        return

    # Show per-product data availability as reference
    st.sidebar.divider()
    st.sidebar.markdown("**Data Availability by Product**")
    for biz_name_ref in ["Wallaroo Wallets", "TeacherFav"]:
        st.sidebar.markdown(f"*{biz_name_ref}*")
        for p in biz_data[biz_name_ref]["products"]:
            all_dates = (
                [r["date"][:10] for r in p["rating_history"]]
                + [c["date"][:10] for c in p["review_count_history"]]
            )
            name = product_name(p, short=True)
            if all_dates:
                p_min = min(all_dates)[:7]
                p_max = max(all_dates)[:7]
                st.sidebar.caption(f"{name}: {p_min} to {p_max}")
            else:
                st.sidebar.caption(f"{name}: No data")

    # Build filtered data
    filtered_data = {}
    for biz_name in biz_data:
        fp = filter_products(biz_data[biz_name]["products"], start_date, end_date)
        filtered_data[biz_name] = {
            "products": fp,
            "metrics": compute_metrics(fp),
        }

    if page == "Executive Summary":
        page_executive_summary(filtered_data)
    elif page == "Wallaroo Wallets":
        page_business_analysis(filtered_data, "Wallaroo Wallets")
    elif page == "TeacherFav":
        page_business_analysis(filtered_data, "TeacherFav")
    elif page == "Wallaroo — Evaluation":
        page_evaluation(filtered_data, "Wallaroo Wallets")
    elif page == "TeacherFav — Evaluation":
        page_evaluation(filtered_data, "TeacherFav")
    elif page == "Wallaroo — Yearly Breakdown":
        page_yearly_breakdown(biz_data, "Wallaroo Wallets")
    elif page == "TeacherFav — Yearly Breakdown":
        page_yearly_breakdown(biz_data, "TeacherFav")
    elif page == "Wallaroo — Seller Conversation":
        page_seller_conversation()
    elif page == "Methodology & Definitions":
        page_methodology()


if __name__ == "__main__":
    main()
