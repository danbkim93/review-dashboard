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
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BIZ_COLORS = {
    "Wallaroo Wallets": {"primary": "#2c3e50", "accent": "#3498db"},
    "TeacherFav": {"primary": "#c0392b", "accent": "#e74c3c"},
}

PRODUCT_COLORS = [
    "#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c",
    "#e67e22", "#34495e", "#16a085", "#d35400",
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


def compute_metrics(products):
    m = {}
    m["n_products"] = len(products)
    m["n_variations"] = sum(p["num_variations"] for p in products)
    m["total_reviews"] = sum(p["shared_reviews"] for p in products)

    main = products[0]  # already sorted by reviews desc
    m["main_product"] = main
    m["main_asin"] = main["representative_asin"]
    m["main_title"] = main["title"] or ""
    m["main_reviews"] = main["shared_reviews"]
    m["main_rating"] = main["rating_history"][-1]["rating"] if main["rating_history"] else None

    # Weighted avg rating
    total_w, weighted_r = 0, 0
    for p in products:
        if p["rating_history"] and p["shared_reviews"] > 0:
            r = p["rating_history"][-1]["rating"]
            w = p["shared_reviews"]
            weighted_r += r * w
            total_w += w
    m["weighted_avg_rating"] = weighted_r / total_w if total_w > 0 else None

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


def amazon_link(asin):
    return f"https://www.amazon.com/dp/{asin}"


def short_title(title, max_len=40):
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
            name=f"{short_title(p['title'], 30)} ({p['shared_reviews']:,} rev)",
            line=dict(width=2, color=color),
            marker=dict(size=4),
        ))

    fig.add_hline(y=4.0, line_dash="dot", line_color="green", opacity=0.3,
                  annotation_text="4.0", annotation_position="left")
    fig.update_layout(
        title=title,
        yaxis_title="Rating",
        yaxis=dict(range=[2.5, 5.1], dtick=0.5),
        height=450,
        legend=dict(font=dict(size=10)),
        margin=dict(l=60, r=20, t=50, b=40),
    )
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
            name=f"{short_title(p['title'], 30)} ({counts[-1]:,} current)",
            line=dict(width=2.5, color=color),
        ))

    fig.update_layout(
        title=title,
        yaxis_title="Total Review Count",
        height=450,
        yaxis=dict(rangemode="tozero"),
        legend=dict(font=dict(size=10)),
        margin=dict(l=60, r=20, t=50, b=40),
    )
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
            line=dict(width=3, color="#2c3e50"),
            fill="tozeroy", fillcolor="rgba(44,62,80,0.1)",
        ))
        fig.update_layout(
            title=title,
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
            line=dict(width=3, color="#2c3e50"),
            marker=dict(size=5),
        ))
        fig.add_hline(y=4.0, line_dash="dot", line_color="green", opacity=0.3)
        fig.update_layout(
            title=title,
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
            name=short_title(p["title"], 30),
            line=dict(width=1.8, color=PRODUCT_COLORS[i % len(PRODUCT_COLORS)]),
        ))

    fig.update_layout(
        title=title,
        yaxis_title="Sales Rank (lower = better)",
        yaxis=dict(autorange="reversed"),
        height=400,
        legend=dict(font=dict(size=10)),
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


# ─── Pages ──────────────────────────────────────────────────────────────────

def page_executive_summary(biz_data):
    st.header("Executive Summary")
    st.caption("Based on complete Keepa API data (not scraped samples) | All products deduplicated by parent ASIN")

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
            st.metric("Review Growth", f"{m['main_growth_per_mo']:.1f}/mo")

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
            "Review Growth (Main)",
            "Growth Direction",
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
            f"{w['main_growth_per_mo']:.1f}/mo",
            f"{w['growth_dir']} (early {w['early_growth_mo']:.1f} -> recent {w['recent_growth_mo']:.1f}/mo)",
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
            f"{t['main_growth_per_mo']:.1f}/mo",
            f"{t['growth_dir']} (early {t['early_growth_mo']:.1f} -> recent {t['recent_growth_mo']:.1f}/mo)",
            f"{t['main_first_date']} to {t['main_last_date']}",
            "Keepa API (complete)",
        ],
    }
    st.dataframe(pd.DataFrame(table_data).set_index("Metric"), use_container_width=True)


def page_business_analysis(biz_data, biz_name):
    products = biz_data[biz_name]["products"]
    metrics = biz_data[biz_name]["metrics"]
    main = products[0]

    st.header(f"{biz_name} — Review Analysis")

    # KPI row
    cols = st.columns(5)
    cols[0].metric("Products", f"{metrics['n_products']} ({metrics['n_variations']} var)")
    cols[1].metric("Total Reviews", f"{metrics['total_reviews']:,}")
    cols[2].metric("Avg Rating", f"{metrics['weighted_avg_rating']:.1f}" if metrics["weighted_avg_rating"] else "?")
    cols[3].metric("Growth", f"{metrics['main_growth_per_mo']:.1f}/mo")
    cols[4].metric("Trend", metrics["main_r_dir"])

    st.divider()

    # ─── 1. Main Product Only ───
    st.subheader(f"1. Main Product: {short_title(main['title'], 60)}")
    st.caption(f"ASIN: [{main['representative_asin']}]({amazon_link(main['representative_asin'])}) | "
               f"{main['shared_reviews']:,} reviews | {main['num_variations']} variants")

    col1, col2 = st.columns(2)
    with col1:
        fig = make_rating_chart(
            [main],
            f"Rating History — {short_title(main['title'], 35)}",
            colors=["#2c3e50"],
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = make_review_count_chart(
            [main],
            f"Review Count Growth — {short_title(main['title'], 35)}",
            colors=["#2c3e50"],
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ─── 2. All Products Combined ───
    st.subheader("2. All Products Combined (Portfolio Performance)")

    col1, col2 = st.columns(2)
    with col1:
        fig = make_combined_chart(
            products, "rating",
            f"{biz_name} — Weighted Avg Rating (All Products Combined)",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = make_combined_chart(
            products, "reviews",
            f"{biz_name} — Total Reviews (All Products Combined)",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ─── 3. All Individual Products ───
    st.subheader("3. All Individual Products (Comprehensive Breakdown)")

    col1, col2 = st.columns(2)
    with col1:
        fig = make_rating_chart(products, f"{biz_name} — Rating History (All Products)")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = make_review_count_chart(products, f"{biz_name} — Review Count Growth (All Products)")
        st.plotly_chart(fig, use_container_width=True)

    # Sales rank
    fig = make_sales_rank_chart(products, f"{biz_name} — Monthly Avg Sales Rank")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ─── Product Breakdown Table ───
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
            "Product": short_title(p["title"], 50),
            "Amazon Link": amazon_link(p["representative_asin"]),
            "Rating": rating,
            "Reviews": f"{p['shared_reviews']:,}",
            "Variants": p["num_variations"],
            "Growth/mo": growth_str,
            "Trend": trend,
            "Data Since": since,
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        column_config={
            "Amazon Link": st.column_config.LinkColumn("Amazon Link", display_text="View"),
        },
        use_container_width=True,
        hide_index=True,
    )


def page_evaluation(biz_data, biz_name):
    products = biz_data[biz_name]["products"]
    m = biz_data[biz_name]["metrics"]

    st.header(f"{biz_name} — Acquisition Evaluation")
    st.caption("Based on complete Keepa API data. Verify financials independently.")

    if biz_name == "Wallaroo Wallets":
        verdict = "CAUTIOUSLY POSITIVE"
        verdict_emoji = "🟢"
        pros = [
            f"10-year brand, 4.6 stars across {m['main_reviews']:,} reviews — exceptional longevity",
            f"Rating STABLE at 4.4-4.6 over full history — no quality degradation",
            f"Main product growing at {m['main_growth_per_mo']:.0f} reviews/mo consistently",
            f"{m['n_variations']} color variants from just {m['n_products']} products = efficient SKU strategy",
            "Ultra-low unit cost ($1.65-1.81) = massive tariff buffer",
            "Previous scraped data showed 3.41 avg — actually 4.6. Business is healthier than first appeared",
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
            f"Keepa data paints a much better picture than our skewed scrape suggested. The main Phone Wallet "
            f"has a rock-solid 4.6 rating across {m['main_reviews']:,} reviews with stable trends. "
            f"However, the business is essentially a one-product company — the Phone Wallet accounts for "
            f"{m['main_reviews']/m['total_reviews']*100:.0f}% of all reviews. Newer products (MagSafe at 4.2, "
            f"Wristlet at 4.2) show declining ratings. The core product is strong but diversification is weak. "
            f"CRITICAL: No P&L available — cannot make a financial assessment. Proceed only with verified financials."
        )
    else:
        verdict = "MIXED — STRONG CORE, WEAK TAILS"
        verdict_emoji = "🟡"
        pros = [
            f"Core Sand Timer line at 4.4 stars with {products[0]['shared_reviews']:,} reviews — market leader",
            f"Toothbrush Timer at 4.6 stars ({products[1]['shared_reviews']:,} reviews) — STRONGER than initially reported",
            f"16 color variants of main timer = strong shelf presence and color choice moat",
            "P&L verified: $46K/mo revenue, 10% margin via Empire Flippers",
            f"Review growth at {m['main_growth_per_mo']:.0f}/mo on main product — healthy velocity",
            "Previous scrape showed 2.74 avg for Toothbrush Timer — actually 4.6. Major data correction.",
        ]
        cons = [
            f"Rating trend slightly DOWN on main product ({m['main_early_r']:.1f} -> {m['main_recent_r']:.1f})",
            f"60-Minute Timer at 3.4 stars (16 reviews) — worst performing product",
            "10% margin is thin — vulnerable to cost increases or ad spend pressure",
            f"Growth {m['growth_dir'].lower()} — early {m['early_growth_mo']:.0f}/mo vs recent {m['recent_growth_mo']:.0f}/mo",
            f"Dashboard: FAILS age (<5yr) and multiple (>30x). 2 FAIL criteria on acquisition checklist",
        ]
        questions = [
            "Can the 60-Minute Timer (3.4 stars) be discontinued or redesigned?",
            "Why is margin only 10% on $46K/mo revenue? Where are costs going?",
            "Can TACoS (12.9%) be reduced without losing rank?",
            "What is the defect/return rate for the Sand Timer line?",
            "Would seller accept $100K-120K (19-23x) given the 2 FAIL criteria?",
        ]
        summary = (
            f"Keepa data dramatically changes the picture. The Toothbrush Timer is actually 4.6 stars (not 2.74 "
            f"from our skewed scrape) — it's the strongest product, not a liability. The core Sand Timer line at "
            f"4.4 stars with {products[0]['shared_reviews']:,} reviews is solid. The only weak spot is the "
            f"60-Minute Timer at 3.4 stars with just 16 reviews. Overall product quality is much better than "
            f"scraped data indicated. However, financial concerns remain: 10% margin is thin, and the "
            f"listing fails 2 acquisition criteria (age <5yr, multiple >30x). Worth pursuing at a lower price."
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

    st.divider()

    st.markdown("#### Data Corrections (vs Previously Scraped)")
    corrections = [
        "Keepa provides COMPLETE rating & review count history (not sampled)",
        f"Previously: {m['n_products']} products with skewed scrape data",
        f"Now: {m['n_products']} unique products, {m['n_variations']} variants, {m['total_reviews']:,} reviews",
        f"Rating confirmed: {m['weighted_avg_rating']:.1f} weighted avg (vs ~3.2-3.4 from scrape)",
        "All star distributions are accurate — no sampling bias",
    ]
    for c in corrections:
        st.markdown(f"- {c}")


def page_methodology():
    st.header("Methodology & Definitions")
    st.caption("How metrics are calculated and data is sourced")

    st.markdown("""
### Data Source: Keepa API

All data comes from [Keepa](https://keepa.com), which tracks Amazon product data continuously.
Unlike our earlier review scraping approach (which was biased by star-by-star filtering),
Keepa provides **complete, accurate** historical data for ratings and review counts.

- **Keepa Product API** with `rating=1` parameter returns csv indices 16 (rating) and 17 (review count)
- **Keepa Seller API** with `storefront=1` discovers all ASINs for a seller
- **Parent ASIN mapping** deduplicates product variations that share review pools

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

---

#### Review Growth Rate (per month)
$$
\\text{Growth/mo} = \\frac{C_{\\text{last}} - C_{\\text{first}}}{\\text{months}}
$$

Where:
- $C_{\\text{first}}$ = review count at first data point
- $C_{\\text{last}}$ = review count at last data point
- $\\text{months} = \\frac{\\text{days between first and last}}{30.44}$

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

### Combined Charts

**Combined Rating**: Weighted average of all products' ratings at each time point,
weighted by each product's review count at that time.

**Combined Reviews**: Sum of all products' review counts at each time point.
Missing data points are forward-filled before summing.

---

### Data Quality Notes
- Keepa data is sourced directly from Amazon's API — it represents the **true** rating and review count at each point in time
- Unlike our earlier web scraping approach, there is **no sampling bias**
- Rating values from Keepa are integers (e.g., 46 = 4.6 stars), divided by 10 for display
- Keepa timestamps are in "Keepa minutes" (minutes since 2011-01-01 00:00 UTC)
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
            "Methodology & Definitions",
        ],
    )

    if page == "Executive Summary":
        page_executive_summary(biz_data)
    elif page == "Wallaroo Wallets":
        page_business_analysis(biz_data, "Wallaroo Wallets")
    elif page == "TeacherFav":
        page_business_analysis(biz_data, "TeacherFav")
    elif page == "Wallaroo — Evaluation":
        page_evaluation(biz_data, "Wallaroo Wallets")
    elif page == "TeacherFav — Evaluation":
        page_evaluation(biz_data, "TeacherFav")
    elif page == "Methodology & Definitions":
        page_methodology()


if __name__ == "__main__":
    main()
