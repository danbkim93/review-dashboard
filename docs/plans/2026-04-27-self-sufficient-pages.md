# Self-Sufficient Pages: Contextual Explanations & Review Integrity

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make each dashboard page self-sufficient — readers understand what they're seeing, whether reviews are legit, and how the date range affects values, without navigating to other pages.

**Architecture:** Add a `compute_integrity_signals()` helper that analyzes review/rating data for manipulation signals, then surface concise context blocks (as `st.expander` or inline markdown) on each page. No new dependencies.

**Tech Stack:** Streamlit, Python, existing Keepa data structures

**File:** `/Users/dankim/Desktop/empireflipper_business_v2/dashboard.py`

---

### Task 1: Add `compute_integrity_signals()` helper function

**Files:**
- Modify: `dashboard.py:107` (insert before `compute_metrics`)

**Step 1: Add the helper function**

Insert before the `compute_metrics` function (line 107). This function analyzes per-product data and returns a structured integrity assessment.

```python
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
            s["growth_spike"] = max_growth > avg_growth * 2 and avg_growth > 0  # any quarter 2x the average
        else:
            s["quarter_growths"] = []
            s["growth_spike"] = False

        # 3. Rating stability — standard deviation
        if len(rh) >= 4:
            ratings = [r["rating"] for r in rh]
            mean_r = sum(ratings) / len(ratings)
            variance = sum((r - mean_r) ** 2 for r in ratings) / len(ratings)
            s["rating_std"] = round(variance ** 0.5, 3)
            s["rating_stable"] = s["rating_std"] < 0.3  # low variance = stable
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
```

**Step 2: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('dashboard.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add dashboard.py
git commit -m "feat: add compute_integrity_signals() helper for review legitimacy analysis"
```

---

### Task 2: Business Analysis page — KPI context block + integrity assessment

**Files:**
- Modify: `dashboard.py` — inside `page_business_analysis`, after the KPI row (after line 548 `st.divider()`) and before the tabs (line 550)

**Step 1: Compute integrity signals and add context block**

After the `st.divider()` on line 548, before the `tab1, tab2, ...` line, insert:

```python
    # Review integrity + context
    integrity = compute_integrity_signals(products)
    main_sig = integrity[0]  # main product (sorted by reviews desc)

    with st.expander("What do these numbers mean? Are the reviews legit?", expanded=False):
        st.markdown(f"""
**You're looking at** a summary of {biz_name}'s Amazon review data from Keepa, filtered to your selected date range.

**Reading the numbers:**
- **Products** — number of unique products (deduplicated by parent ASIN). The number in parentheses is total color/style variants sharing those review pools.
- **Total Reviews** — sum of review counts across all products at the end of the selected period.
- **Avg Rating** — weighted average rating across all products, weighted by review count (so a product with 7,000 reviews matters more than one with 100).
- **Review Growth** — how many new reviews the main product gains per month on average over the selected period. Higher = more customers buying.
- **Rating Trend** — compares the average rating in the first half vs second half of the selected period. STABLE means quality hasn't changed; DECLINING means recent ratings are lower.

**Are the reviews legitimate?**
""")

        for sig in integrity:
            if sig["looks_organic"] and not sig["red_flags"]:
                st.markdown(f"- **{sig['name']}**: Reviews look organic. Steady growth, no Amazon purges, stable ratings{' (std dev: ' + str(sig['rating_std']) + ')' if sig['rating_std'] is not None else ''}.")
            elif sig["looks_organic"]:
                st.markdown(f"- **{sig['name']}**: Mostly organic. {'; '.join(sig['red_flags'])}.")
            else:
                st.markdown(f"- **{sig['name']}**: **Review flags detected.** {'; '.join(sig['red_flags'])}.")

        # Overall read
        flagged = [s for s in integrity if not s["looks_organic"]]
        if not flagged:
            st.success("Overall: No significant manipulation signals detected across the portfolio. Review growth patterns are consistent and Amazon has not performed major review purges.")
        else:
            st.warning(f"Overall: {len(flagged)} product(s) have review flags worth investigating. See details above.")

        st.markdown(f"""
**How the date range affects these values:**
Changing the date range in the sidebar recalculates everything on this page. Narrowing to the last 1-2 years isolates recent performance — useful for spotting if a once-strong product is now declining. The full range gives the historical average, which smooths out short-term noise but can hide recent trends.
""")
```

**Step 2: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('dashboard.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add dashboard.py
git commit -m "feat: add KPI context block with review integrity assessment to Business Analysis page"
```

---

### Task 3: Business Analysis > Main Product tab — show data range + integrity note

**Files:**
- Modify: `dashboard.py` — inside `page_business_analysis`, in the `with tab1:` block, after the `st.caption(...)` line (after line 558)

**Step 1: Add data range and integrity note**

After the `st.caption(...)` line that shows ASIN/Keepa/reviews/variants (line 558), insert:

```python
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
```

Note: `main_sig` was computed in Task 2 and is accessible here since it's in the same function scope.

**Step 2: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('dashboard.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add dashboard.py
git commit -m "feat: add data range and integrity note to Main Product tab"
```

---

### Task 4: Business Analysis > Product Table tab — add date range impact note

**Files:**
- Modify: `dashboard.py` — inside the Product Table tab's existing `st.expander("Column Definitions")`, append to the markdown

**Step 1: Add date range note to existing expander**

At the end of the existing Column Definitions expander markdown (before the closing `"""`), append:

```
---

**How the date range affects this table:**
All values are computed from data within your selected date range. Rating uses the last data point in range. Reviews/mo divides review growth across the months in range. Rating Trend splits the in-range data in half. Narrowing the range means fewer data points — trends become less reliable but more recent. If a product has no data in the selected range, values show "?".
```

**Step 2: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('dashboard.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add dashboard.py
git commit -m "feat: add date range impact note to Product Table column definitions"
```

---

### Task 5: Executive Summary page — add context expander with integrity read

**Files:**
- Modify: `dashboard.py` — inside `page_executive_summary`, after the comparison table `st.dataframe(...)` (after line 529)

**Step 1: Add context expander**

After the `st.dataframe(...)` call on line 529, insert:

```python
    with st.expander("What do these numbers mean? Are the reviews legit?", expanded=False):
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
```

**Step 2: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('dashboard.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add dashboard.py
git commit -m "feat: add context and integrity summary to Executive Summary page"
```

---

### Task 6: Evaluation page — show active date range prominently

**Files:**
- Modify: `dashboard.py` — inside `page_evaluation`, after the `st.caption(...)` on line 687

**Step 1: Add date range context**

After `st.caption("Based on Keepa API data, filtered by selected date range. Verify financials independently.")` on line 687, insert:

```python
    main = products[0]
    main_ch = main["review_count_history"]
    if main_ch:
        st.caption(
            f"Showing data from {main_ch[0]['date'][:7]} to {main_ch[-1]['date'][:7]} "
            f"(main product: {product_name(main, short=True)}). "
            f"Adjust the date range in the sidebar to change the evaluation window."
        )
```

Note: `main` is not defined until later in the current code (it's used as `products[0]` in the metrics). This is safe because `products` is already available at line 683.

**Step 2: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('dashboard.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add dashboard.py
git commit -m "feat: show active date range on Evaluation page"
```

---

### Task 7: Yearly Breakdown > product tab headers — show data range

**Files:**
- Modify: `dashboard.py` — inside `page_yearly_breakdown`, in each product tab, after the `st.caption(...)` showing ASIN/Keepa (after line 951)

**Step 1: Add data range to product tabs**

After the `st.caption(...)` on line 951, insert:

```python
        if ch:
            st.caption(
                f"Review data: {ch[0]['date'][:7]} to {ch[-1]['date'][:7]} "
                f"({len(ch)} observations) | "
                f"Rating data: {rh[0]['date'][:7]} to {rh[-1]['date'][:7]} "
                f"({len(rh)} observations)" if rh else
                f"Review data: {ch[0]['date'][:7]} to {ch[-1]['date'][:7]} "
                f"({len(ch)} observations) | No rating data"
            )
```

Note: `rh` and `ch` are defined on lines 943-944 in the existing code, so they're in scope.

**Step 2: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('dashboard.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add dashboard.py
git commit -m "feat: show data range in Yearly Breakdown product tabs"
```

---

### Task 8: Final verification

**Step 1: Full syntax check**

Run: `python3 -c "import ast; ast.parse(open('dashboard.py').read()); print('OK')"`
Expected: `OK`

**Step 2: Visual verification**

Run: `streamlit run dashboard.py`

Check each page:
- Executive Summary → expander with integrity read + metric explanations
- Wallaroo Wallets → KPI context block, Main Product tab has data range, Product Table has date range note
- TeacherFav → same as above
- Evaluation pages → date range shown at top
- Yearly Breakdown → each product tab shows data range

**Step 3: Final commit**

```bash
git add dashboard.py
git commit -m "feat: self-sufficient pages with contextual explanations and review integrity signals"
```
