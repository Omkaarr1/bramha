import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from models import (
    create_journal,
    create_journal_batch,
    get_conn,
    get_schema,
    get_table_df,
    init_db,
    insert_facts,
    list_batches,
    list_journal_batches,
    list_journals,
    load_facts,
    upsert_file,
)
from services import ask_sheet_ai, tidy_excel_to_long

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Accounting Information",
    layout="wide",
    page_icon="📊",
)

# ------------------------------------------------------------
# Simple hard-coded auth
# ------------------------------------------------------------
VALID_USERS = {
    "admin": "admin123",
    "analyst": "tb2025",
}


def ensure_authenticated() -> None:
    """
    Show a login page until the user is authenticated.
    This is called ONLY on the main Analytics page (mode=analytics),
    so child pages (chat / journal_create) won't ask for login again.
    """
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
        st.session_state["current_user"] = None

    if st.session_state["authenticated"]:
        # Optional small user badge in the sidebar
        with st.sidebar:
            st.markdown("### 🔐 Session")
            st.write(f"Logged in as: **{st.session_state['current_user']}**")
            if st.button("Logout"):
                st.session_state["authenticated"] = False
                st.session_state["current_user"] = None
                st.rerun()
        return

    # Not authenticated → show login page and stop execution
    st.title("🔐 Login to Accounting Information App")
    # st.caption(
    #     "Demo login with hard-coded credentials. "
    #     "Example: `admin / admin123` or `analyst / tb2025`."
    # )

    col1, col2 = st.columns([2, 1])
    with col1:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        login_clicked = st.button("Login")

        if login_clicked:
            if username in VALID_USERS and VALID_USERS[username] == password:
                st.session_state["authenticated"] = True
                st.session_state["current_user"] = username
                st.success("✅ Login successful. Redirecting...")
                st.rerun()
            else:
                st.error("❌ Invalid username or password. Please try again.")

    # with col2:
    #     st.markdown("#### Test Credentials")
    #     st.code("Username: admin\nPassword: admin123", language="text")
    #     st.code("Username: analyst\nPassword: tb2025", language="text")

    # Stop rest of the app from running until authenticated
    st.stop()


def get_query_params():
    """
    Use new st.query_params API directly.
    Returns dict[str, list[str]] for compatibility with old code.
    """
    qp = st.query_params
    return {k: [v] if isinstance(v, str) else v for k, v in qp.items()}


# ------------------------------------------------------------
# Chat page (full-screen, new tab)
# ------------------------------------------------------------
def render_chat_page(conn, batch_from_query: str | None):
    st.title("💬 Sheet AI Assistant")
    st.caption(
        "This is a **full-screen chat window**. It uses the same data batches you "
        "uploaded in the analytics tab."
    )

    batches = list_batches(conn)
    if batches.empty:
        st.info("⬆️ Upload a file in the main Analytics view first, then open chat again.")
        return

    all_ids = batches["id"].tolist()
    initial_idx = 0
    if batch_from_query is not None:
        try:
            q_id = int(batch_from_query)
            if q_id in all_ids:
                initial_idx = all_ids.index(q_id)
        except ValueError:
            pass

    batch_id = st.selectbox(
        "Choose batch (upload) to chat about",
        options=all_ids,
        index=initial_idx,
        format_func=lambda i: f"#{i} — "
        f"{batches.set_index('id').loc[i, 'filename']} — "
        f"{batches.set_index('id').loc[i, 'uploaded_at']}",
    )

    data_df = load_facts(conn, batch_id)
    if data_df.empty:
        st.warning("No numeric facts found for this batch yet.")
        return

    st.markdown(
        f"> Currently chatting about **batch #{batch_id}**. "
        "Change the selection above to switch context."
    )

    st.markdown(
        '<a href="./" target="_self">⬅️ Back to analytics view</a>',
        unsafe_allow_html=True,
    )

    history_key = f"chat_history_{batch_id}"
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    for msg in st.session_state[history_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask a question about this dataset...")
    if user_q:
        st.session_state[history_key].append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing the uploaded sheet..."):
                answer, out_of_context = ask_sheet_ai(user_q, data_df)
                if out_of_context:
                    msg = (
                        "❌ This question can't be answered from the uploaded sheet. "
                        "Please ask something related to its line items, years, or amounts."
                    )
                    st.error(msg)
                    st.session_state[history_key].append(
                        {"role": "assistant", "content": msg}
                    )
                else:
                    st.markdown(answer)
                    st.session_state[history_key].append(
                        {"role": "assistant", "content": answer}
                    )


# ------------------------------------------------------------
# Journal Creator (new tab)
# ------------------------------------------------------------
def render_journal_create_page(conn):
    st.title("Oracle Journal Creator")
    st.caption("Create and submit journal entries similar to Oracle ERP journals.")

    st.markdown(
        """
        <div style="display:flex;gap:0.75rem;margin-bottom:1.5rem;">
          <div style="padding:0.4rem 0.9rem;border-radius:999px;background:#e6f0ff;color:#2850a7;font-weight:600;">
            1. Journal Setup
          </div>
          <div style="padding:0.4rem 0.9rem;border-radius:999px;background:#f5f7fb;color:#6b7280;font-weight:500;">
            2. Journal Lines
          </div>
          <div style="padding:0.4rem 0.9rem;border-radius:999px;background:#f5f7fb;color:#6b7280;font-weight:500;">
            3. Review & Submit
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    existing_batches = list_journal_batches(conn)

    # ---------------------- Journal Batch section ---------------------- #
    st.subheader("Journal Batch")
    st.caption("Configure batch settings")

    c1, c2 = st.columns([2, 1])
    with c1:
        template_options = (
            ["No batches available"]
            if existing_batches.empty
            else ["Do not use template"]
            + [
                f"#{row.id} — {row.batch_name}"
                for _, row in existing_batches.iterrows()
            ]
        )
        # st.selectbox(
        #     "Use Existing Batch as Template",
        #     options=template_options,
        #     index=0,
        #     help="This demo does not auto-copy values, it's for reference.",
        #     key="jb_template",
        # )

        batch_name = st.text_input("Journal Batch Name *", "", key="jb_name")
        batch_description = st.text_area(
            "Batch Description",
            "",
            height=80,
            key="jb_description",
        )
        balance_type = st.selectbox(
            "Balance Type",
            options=["Actual", "Budget", "Encumbrance"],
            index=0,
            key="jb_balance_type",
        )

    with c2:
        st.markdown("**Batch Status**")
        st.write("Source: Manual")
        st.write("Approval: Not required")
        st.write("Funds: Not attempted")
        st.write("Status: Unposted")
        st.write("Complete: Incomplete")
        st.markdown("---")
        st.markdown("**Attachments**")
        st.write("No attachments (not implemented in this demo).")

    st.markdown("---")

    # ---------------------- Journal section ---------------------- #
    st.subheader("Journal")
    st.caption("Configure journal details and settings")

    c3, c4 = st.columns(2)
    with c3:
        journal_name = st.text_input("Journal Name *", "", key="j_name")
        journal_desc = st.text_area(
            "Journal Description",
            "",
            height=80,
            key="j_description",
        )
    with c4:
        accounting_date = st.date_input("Accounting Date *", key="j_accounting_date")
        category = st.selectbox(
            "Category *",
            options=["General", "Accrual", "Adjustment", "Reclassification"],
            index=0,
            key="j_category",
        )

    c5, c6, c7 = st.columns(3)
    with c5:
        ledger = st.selectbox(
            "Ledger *",
            options=[
                "No ledgers available",
                "Primary Ledger",
                "US Ledger",
                "IN Ledger",
            ],
            index=1,
            key="j_ledger",
        )
    with c6:
        accounting_period = st.selectbox(
            "Accounting Period *",
            options=[
                "No periods available",
                "Jan-2025",
                "Feb-2025",
                "Mar-2025",
                "FY-2025",
            ],
            index=1,
            key="j_period",
        )
    with c7:
        legal_entity = st.selectbox(
            "Legal Entity *",
            options=[
                "No entities available",
                "Global Corp India Pvt Ltd",
                "Global Corp US Inc",
            ],
            index=1,
            key="j_legal_entity",
        )

    # Currency & conversion
    st.markdown("#### Currency & Conversion")
    c8, c9, c10 = st.columns(3)
    with c8:
        currency = st.selectbox(
            "Currency",
            options=["INR", "USD", "EUR", "GBP"],
            index=0,
            key="j_currency",
        )
        conversion_rate_type = st.selectbox(
            "Conversion Rate Type",
            options=["Corporate", "Spot", "User"],
            index=0,
            key="j_conv_type",
        )
    with c9:
        conversion_date = st.date_input(
            "Conversion Date",
            value=accounting_date,
            key="j_conv_date",
        )
    with c10:
        conversion_rate = st.number_input(
            "Conversion Rate",
            min_value=0.0001,
            value=1.0,
            step=0.0001,
            format="%.4f",
            key="j_conv_rate",
        )
        inverse_rate = st.number_input(
            "Inverse Rate",
            min_value=0.0001,
            value=1.0,
            step=0.0001,
            format="%.4f",
            key="j_inverse_rate",
        )

    st.markdown("---")
    save_clicked = st.button("Save Journal")

    if save_clicked:
        errors = []
        if not batch_name.strip():
            errors.append("Journal Batch Name is required.")
        if not journal_name.strip():
            errors.append("Journal Name is required.")
        if ledger == "No ledgers available":
            errors.append("Please choose a valid Ledger.")
        if accounting_period == "No periods available":
            errors.append("Please choose a valid Accounting Period.")
        if legal_entity == "No entities available":
            errors.append("Please choose a valid Legal Entity.")

        if errors:
            st.error("Please fix the following before saving:")
            for e in errors:
                st.write(f"- {e}")
            return

        batch_id = create_journal_batch(
            conn,
            batch_name=batch_name.strip(),
            description=batch_description.strip(),
            balance_type=balance_type,
        )
        journal_id = create_journal(
            conn,
            batch_id=batch_id,
            journal_name=journal_name.strip(),
            description=journal_desc.strip(),
            ledger=ledger,
            accounting_period=accounting_period,
            legal_entity=legal_entity,
            accounting_date=accounting_date.isoformat(),
            category=category,
            currency=currency,
            conversion_date=conversion_date.isoformat() if conversion_date else None,
            conversion_rate_type=conversion_rate_type,
            conversion_rate=float(conversion_rate),
            inverse_rate=float(inverse_rate),
        )

        st.success(
            f"✅ Journal #{journal_id} has been created inside batch #{batch_id} "
            "and stored in the database."
        )
        st.info(
            "You can view the stored data at the bottom of the main Analytics page "
            "inside **Database Explorer → Journal Batches / Journals** tabs."
        )


# ------------------------------------------------------------
# Analytics home page
# ------------------------------------------------------------
def render_analytics_page(conn):
    st.title("📊 Accounting Information")
    st.caption(
        "Upload a Trial Balance / BS sheet, explore analytics, and chat with an AI that knows only your sheet."
    )

    # Upload
    with st.container():
        st.markdown("### 📂 Upload Excel")
        uploaded = st.file_uploader(
            "Upload Balance Sheet / Trial Balance",
            type=["xlsx", "xls"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            content = uploaded.read()
            with st.spinner("📥 Ingesting and normalizing your data…"):
                file_id, is_new = upsert_file(conn, uploaded.name, content)
                df_long, _ = tidy_excel_to_long(content, sheet_name="BS")
                insert_facts(conn, file_id, df_long)
            st.success(
                f"✅ {'Ingested' if is_new else 'Already in DB'}: "
                f"{uploaded.name} (batch #{file_id})"
            )
            with st.expander("🔎 Preview normalized data (current upload)", expanded=False):
                st.dataframe(df_long.head(30), use_container_width=True)

    batches = list_batches(conn)
    if batches.empty:
        st.info("⬆️ Upload a file to begin.")
        st.stop()

    col_b1, col_b2 = st.columns([2, 1])
    with col_b1:
        batch_label = st.selectbox(
            "Choose batch (upload) to analyze",
            options=batches["id"].tolist(),
            format_func=lambda i: f"#{i} — "
            f"{batches.set_index('id').loc[i, 'filename']} — "
            f"{batches.set_index('id').loc[i, 'uploaded_at']}",
        )
    with col_b2:
        st.caption("AI model: `gpt-4.1-nano`")

    # ---- Guard against empty data / missing years ----
    data = load_facts(conn, batch_label)
    if data.empty or "year" not in data.columns:
        st.warning(
            "No numeric facts found for this batch yet. "
            "Please upload a valid Balance Sheet / Trial Balance file for this batch."
        )
        st.stop()
    # -------------------------------------------------------

    years = sorted(data["year"].unique().tolist())
    min_y, max_y = min(years), max(years)

    with st.sidebar:
        st.header("🔎 Filters")
        year_range = st.slider(
            "Year range",
            min_value=min_y,
            max_value=max_y,
            value=(max(min_y, max_y - 4), max_y),
            step=1,
        )
        search = st.text_input("Search line item (contains)", "")
        top_n = st.number_input(
            "Top N for some charts", min_value=5, max_value=50, value=10, step=1
        )
        st.caption("Tip: Use search to narrow charts and tables in real time.")

    mask = data["year"].between(year_range[0], year_range[1])
    if search.strip():
        mask &= data["line_item"].str.contains(search.strip(), case=False, na=False)
    data_f = data.loc[mask].copy()

    pivot = (
        data_f.pivot_table(
            index="line_item", columns="year", values="amount", aggfunc="sum"
        )
        .fillna(0.0)
    )
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    latest_years = sorted(pivot.columns.tolist())
    yoy_col = None
    if len(latest_years) >= 2:
        yoy_col = f"YoY_{latest_years[-2]}→{latest_years[-1]}"
        pivot[yoy_col] = (
            (pivot[latest_years[-1]] - pivot[latest_years[-2]])
            / pivot[latest_years[-2]].replace(0, np.nan)
        ) * 100

    def sum_year(y: int) -> float:
        return float(data_f.loc[data_f["year"] == y, "amount"].sum())

    total_latest = sum_year(max_y)
    total_prev = sum_year(max_y - 1) if (max_y - 1) in years else None
    growth_pct = (
        None
        if total_prev in (None, 0)
        else (total_latest - total_prev) / total_prev * 100
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        f"Total {max_y}",
        f"₹{total_latest:,.0f}",
        delta=None if growth_pct is None else f"{growth_pct:.2f}% vs {max_y-1}",
    )
    k2.metric("Items (filtered)", f"{pivot.shape[0]}")
    k3.metric("Years in view", f"{year_range[0]}–{year_range[1]}")
    if yoy_col is not None and pivot[yoy_col].notna().any():
        k4.metric("Avg YoY of items", f"{pivot[yoy_col].mean():.2f}%")
    else:
        k4.metric("Avg YoY of items", "—")

    # Narrative
    st.subheader("🤖 Narrative Highlights")
    bullets = []
    this_year_totals = (
        data_f[data_f["year"] == year_range[1]]
        .groupby("line_item", as_index=False)["amount"]
        .sum()
        .sort_values("amount", ascending=False)
    )
    if not this_year_totals.empty:
        top_item = this_year_totals.iloc[0]["line_item"]
        top_amt = this_year_totals.iloc[0]["amount"]
        bullets.append(
            f"**Largest item in {year_range[1]}**: {top_item} (₹{top_amt:,.0f})."
        )
    if growth_pct is not None:
        direction = "↑" if growth_pct >= 0 else "↓"
        bullets.append(
            f"**Portfolio change {max_y-1}→{max_y}**: {direction} "
            f"{growth_pct:.2f}% (₹{total_prev:,.0f} → ₹{total_latest:,.0f})."
        )
    if yoy_col is not None and pivot[yoy_col].notna().any():
        hi = pivot[yoy_col].idxmax()
        lo = pivot[yoy_col].idxmin()
        bullets.append(f"**Fastest YoY grower**: {hi} ({pivot.loc[hi, yoy_col]:.2f}%).")
        bullets.append(
            f"**Largest YoY decline**: {lo} ({pivot.loc[lo, yoy_col]:.2f}%)."
        )

    if bullets:
        st.markdown("\n\n".join(f"- {b}" for b in bullets))
    else:
        st.info("Adjust filters to generate insights.")

    # Actions: open Chat & Create Journal in new tabs
    st.markdown("---")
    st.subheader("🚀 Actions")

    st.markdown(
        """
        <style>
        .primary-launch-button {
            display: inline-block;
            padding: 0.6rem 1.3rem;
            border-radius: 0.6rem;
            background: #4b8df8;
            color: white !important;
            font-weight: 600;
            text-decoration: none;
            border: none;
            margin-right: 0.75rem;
        }
        .primary-launch-button:hover {
            background: #3b75d1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    chat_url = f"?mode=chat&batch_id={batch_label}"
    journal_url = "?mode=journal_create"

    st.markdown(
        f"""
        <a href="{chat_url}" target="_blank" class="primary-launch-button">
            💬 Open full-screen AI chat for batch #{batch_label}
        </a>
        <a href="{journal_url}" target="_blank" class="primary-launch-button">
            📘 Create Journal
        </a>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Chat opens a **new browser tab/window** linked to this batch. "
        "Create Journal opens a **new Oracle-style journal page** and saves "
        "data into the same SQLite database."
    )

    # Visual analytics tabs
    st.markdown("---")
    st.subheader("📊 Visual Analytics")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🎞️ Animated Top-N",
            "📈 Trends",
            "🪴 Composition",
            "📊 YoY Movers",
            "🧠 Correlation & Downloads",
        ]
    )

    with tab1:
        st.markdown("#### 🎞️ Animated Top-N Line Items Over Years")
        anim_df = (
            data_f.groupby(["year", "line_item"], as_index=False)["amount"]
            .sum()
            .copy()
        )
        anim_df["rank"] = anim_df.groupby("year")["amount"].rank(
            method="first", ascending=False
        )
        anim_df_top = (
            anim_df.sort_values(["year", "amount"], ascending=[True, False])
            .groupby("year")
            .head(int(top_n))
        )
        if not anim_df_top.empty:
            fig_anim = px.bar(
                anim_df_top,
                x="amount",
                y="line_item",
                color="line_item",
                animation_frame="year",
                orientation="h",
                range_x=[0, anim_df_top["amount"].max() * 1.1],
                title=f"Top {top_n} items by year",
            )
            fig_anim.update_layout(showlegend=False, height=560)
            st.plotly_chart(fig_anim, use_container_width=True)
        else:
            st.info("No data for the selected filters.")

    with tab2:
        st.markdown("#### 📈 Trendlines for Selected Items")
        latest_totals = (
            data_f[data_f["year"] == max_y]
            .groupby("line_item", as_index=False)["amount"]
            .sum()
            .sort_values("amount", ascending=False)
            .head(5)["line_item"]
            .tolist()
        )
        sel_items = st.multiselect(
            "Choose items",
            options=sorted(pivot.index.tolist()),
            default=latest_totals[:5],
        )
        if sel_items:
            trend_df = (
                data_f[data_f["line_item"].isin(sel_items)]
                .groupby(["year", "line_item"], as_index=False)["amount"]
                .sum()
            )
            fig_trend = px.line(
                trend_df,
                x="year",
                y="amount",
                color="line_item",
                markers=True,
                title="Trends",
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Select at least one line item to view trends.")

    with tab3:
        st.markdown(f"#### 🪴 Composition Treemap ({year_range[1]})")
        comp_df = (
            data_f[data_f["year"] == year_range[1]]
            .groupby("line_item", as_index=False)["amount"]
            .sum()
        )
        if not comp_df.empty:
            fig_tree = px.treemap(
                comp_df,
                path=["line_item"],
                values="amount",
                color="amount",
                color_continuous_scale="Blues",
                title=f"Composition in {year_range[1]}",
            )
            st.plotly_chart(fig_tree, use_container_width=True)
        else:
            st.info("No data for treemap in the selected year.")

    with tab4:
        if yoy_col is not None and pivot[yoy_col].notna().any():
            st.markdown(
                f"#### 📊 YoY Growth Leaders & Laggards ({latest_years[-2]}→{latest_years[-1]})"
            )
            yoy_sorted = pivot[[yoy_col]].dropna().sort_values(yoy_col, ascending=False)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Top growers**")
                st.dataframe(
                    yoy_sorted.head(top_n).rename(columns={yoy_col: "YoY %"}),
                    use_container_width=True,
                )
            with c2:
                st.markdown("**Top decliners**")
                st.dataframe(
                    yoy_sorted.tail(top_n).rename(columns={yoy_col: "YoY %"}),
                    use_container_width=True,
                )
            fig_yoy = px.bar(
                yoy_sorted.head(min(top_n, 20))
                .reset_index()
                .rename(columns={"index": "line_item"}),
                x=yoy_col,
                y="line_item",
                orientation="h",
                color=yoy_col,
                color_continuous_scale="RdYlGn",
                title="Top YoY %",
            )
            fig_yoy.update_layout(showlegend=False, height=520)
            st.plotly_chart(fig_yoy, use_container_width=True)
        else:
            st.markdown("#### 📊 YoY Growth")
            st.info("Not enough year-over-year data to compute growth.")

    with tab5:
        st.markdown("#### 🧠 Correlation of Years (by item totals)")
        if pivot.shape[1] >= 2:
            only_year_cols = [c for c in pivot.columns if isinstance(c, (int, np.integer))]
            if len(only_year_cols) >= 2:
                cor = pivot.loc[:, only_year_cols].corr()
                fig_cor = px.imshow(
                    cor, text_auto=".2f", aspect="auto", title="Correlation Heatmap"
                )
                st.plotly_chart(fig_cor, use_container_width=True)
            else:
                st.info("Need at least two numeric year columns for correlation.")
        else:
            st.info("No sufficient columns for correlation.")

        st.markdown("#### ⬇️ Downloads")
        clean_export = (
            data_f.rename(
                columns={"line_item": "Line Item", "year": "Year", "amount": "Amount"}
            )
            .sort_values(["Year", "Line Item"])
        )
        st.download_button(
            "Download current view (CSV)",
            clean_export.to_csv(index=False).encode("utf-8"),
            file_name="balance_view.csv",
            mime="text/csv",
        )
        wide_export = pivot.reset_index().rename(columns={"line_item": "Line Item"})
        st.download_button(
            "Download pivot (CSV)",
            wide_export.to_csv(index=False).encode("utf-8"),
            file_name="balance_pivot.csv",
            mime="text/csv",
        )

    # Database Explorer
    st.markdown("---")
    st.header("🗄️ Database Explorer")

    t1, t2, t3, t4, t5 = st.tabs(
        ["📁 Files", "🏷️ Line Items", "📊 Facts", "📦 Journal Batches", "📘 Journals"]
    )

    # Files
    with t1:
        st.caption("Upload batches deduplicated by SHA-256 hash")
        df_files = get_table_df(conn, "files")
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.metric("Rows", len(df_files))
        with c2:
            st.metric(
                "Unique filenames",
                df_files["filename"].nunique() if not df_files.empty else 0,
            )
        with c3:
            st.metric(
                "Last upload (UTC)",
                df_files["uploaded_at"].max() if not df_files.empty else "—",
            )
        st.markdown("**Schema**")
        st.dataframe(
            get_schema(conn, "files"),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("**Data**")
        if df_files.empty:
            st.info("No rows yet.")
        else:
            page_size = st.number_input(
                "Rows per page", 5, 100, 10, step=5, key="files_page_size"
            )
            max_page = max(1, int(np.ceil(len(df_files) / page_size)))
            page = st.number_input(
                "Page", 1, max_page, 1, key="files_page"
            )
            start = (page - 1) * page_size
            st.dataframe(
                df_files.iloc[start : start + page_size], use_container_width=True
            )
            st.download_button(
                "Download files (CSV)",
                df_files.to_csv(index=False).encode("utf-8"),
                "files.csv",
                "text/csv",
            )

    # Line items
    with t2:
        st.caption("Dimension table of unique line item names")
        df_items = get_table_df(conn, "line_items")
        c1, c2 = st.columns([1, 3])
        with c1:
            st.metric("Rows", len(df_items))
        with c2:
            st.metric(
                "Distinct names",
                df_items["name"].nunique() if not df_items.empty else 0,
            )
        st.markdown("**Schema**")
        st.dataframe(
            get_schema(conn, "line_items"),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("**Data (searchable)**")
        if df_items.empty:
            st.info("No rows yet.")
        else:
            q = st.text_input("Search name (contains)", "", key="li_search")
            dfv = df_items.copy()
            if q.strip():
                dfv = dfv[dfv["name"].str.contains(q.strip(), case=False, na=False)]
            page_size = st.number_input(
                "Rows per page", 5, 200, 15, step=5, key="li_page_size"
            )
            max_page = max(1, int(np.ceil(len(dfv) / page_size)))
            page = st.number_input(
                "Page", 1, max_page, 1, key="li_page"
            )
            start = (page - 1) * page_size
            st.dataframe(dfv.iloc[start : start + page_size], use_container_width=True)
            st.download_button(
                "Download line_items (CSV)",
                dfv.to_csv(index=False).encode("utf-8"),
                "line_items.csv",
                "text/csv",
            )

    # Facts
    with t3:
        st.caption("Normalized numeric facts per (file_id, line_item_id, year)")
        df_facts = get_table_df(conn, "facts")
        st.markdown("**Schema**")
        st.dataframe(
            get_schema(conn, "facts"),
            use_container_width=True,
            hide_index=True,
        )

        if df_facts.empty:
            st.info("No rows yet.")
        else:
            st.markdown("**Readable view (joined)**")
            q = """
            SELECT f.id, f.file_id, fi.filename, li.name AS line_item, f.year, f.amount
            FROM facts f
            JOIN files fi ON fi.id = f.file_id
            JOIN line_items li ON li.id = f.line_item_id
            ORDER BY f.file_id DESC, f.year DESC, f.amount DESC
            """
            df_join = pd.read_sql_query(q, conn)

            cc1, cc2, cc3, cc4 = st.columns(4)
            with cc1:
                sel_file = st.selectbox(
                    "Filter by file_id",
                    options=["All"]
                    + sorted(df_join["file_id"].unique().tolist(), reverse=True),
                )
            with cc2:
                years_all = sorted(df_join["year"].unique().tolist())
                sel_years = st.multiselect(
                    "Years", options=years_all, default=years_all
                )
            with cc3:
                name_q = st.text_input("Line item (contains)", "")
            with cc4:
                page_size = st.number_input(
                    "Rows per page", 10, 500, 25, step=5, key="facts_page_size"
                )

            dfv = df_join.copy()
            if sel_file != "All":
                dfv = dfv[dfv["file_id"] == sel_file]
            if sel_years:
                dfv = dfv[dfv["year"].isin(sel_years)]
            if name_q.strip():
                dfv = dfv[
                    dfv["line_item"].str.contains(name_q.strip(), case=False, na=False)
                ]

            max_page = max(1, int(np.ceil(len(dfv) / page_size)))
            page = st.number_input(
                "Page", 1, max_page, 1, key="facts_page"
            )
            start = (page - 1) * page_size
            st.dataframe(dfv.iloc[start : start + page_size], use_container_width=True)
            st.download_button(
                "Download facts (CSV)",
                dfv.to_csv(index=False).encode("utf-8"),
                "facts.csv",
                "text/csv",
            )

    # Journal batches
    with t4:
        st.caption("Header information for created journal batches")
        df_jb = list_journal_batches(conn)
        st.markdown("**Schema**")
        st.dataframe(
            get_schema(conn, "journal_batches"),
            use_container_width=True,
            hide_index=True,
        )
        if df_jb.empty:
            st.info("No journal batches yet. Use **Create Journal** to add one.")
        else:
            page_size = st.number_input(
                "Rows per page", 5, 200, 10, step=5, key="jb_page_size"
            )
            max_page = max(1, int(np.ceil(len(df_jb) / page_size)))
            page = st.number_input(
                "Page", 1, max_page, 1, key="jb_page"
            )
            start = (page - 1) * page_size
            st.dataframe(df_jb.iloc[start : start + page_size], use_container_width=True)
            st.download_button(
                "Download journal_batches (CSV)",
                df_jb.to_csv(index=False).encode("utf-8"),
                "journal_batches.csv",
                "text/csv",
            )

    # Journals
    with t5:
        st.caption("Journal headers created under batches")
        df_j = list_journals(conn)
        st.markdown("**Schema**")
        st.dataframe(
            get_schema(conn, "journals"),
            use_container_width=True,
            hide_index=True,
        )
        if df_j.empty:
            st.info("No journals yet. Use **Create Journal** from the home page.")
        else:
            page_size = st.number_input(
                "Rows per page", 5, 200, 10, step=5, key="j_page_size"
            )
            max_page = max(1, int(np.ceil(len(df_j) / page_size)))
            page = st.number_input(
                "Page", 1, max_page, 1, key="j_page"
            )
            start = (page - 1) * page_size
            st.dataframe(df_j.iloc[start : start + page_size], use_container_width=True)
            st.download_button(
                "Download journals (CSV)",
                df_j.to_csv(index=False).encode("utf-8"),
                "journals.csv",
                "text/csv",
            )


# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------
def main():
    conn = get_conn()
    init_db(conn)

    params = get_query_params()
    mode = params.get("mode", ["analytics"])[0]
    batch_from_query = params.get("batch_id", [None])[0]

    # 🔐 Login is enforced ONLY for the main analytics view
    if mode == "chat":
        render_chat_page(conn, batch_from_query)
    elif mode == "journal_create":
        render_journal_create_page(conn)
    else:  # analytics or any other fallback
        ensure_authenticated()
        render_analytics_page(conn)


if __name__ == "__main__":
    main()
