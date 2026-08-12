import glob
import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import joblib
from PIL import Image
from google import genai
from datetime import date, timedelta
from rag_chain import ask_question
client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)

# PAGE CONFIG
apple_icon = Image.open("apple_icon.png")
st.set_page_config(
    page_title="Apple Retail Price Intelligence",
    page_icon=apple_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

CATEGORY_COLORS = {
    "iPhone": "#0071e3",
    "Mac": "#8e8e93",
    "iPad": "#34c759",
    "Watch": "#ff375f",
}

#Data Loading
DATA_DIR = "data" if os.path.isdir("data") else "."
FILE_PATTERN = "apple_products_pricing*.csv"


@st.cache_data(show_spinner="Loading dataset...")
def load_data(data_dir: str, pattern: str):
    paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not paths:
        return None, []

    frames = []
    for p in paths:
        frame = pd.read_csv(p)
        frame["__source_file"] = os.path.basename(p)
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True, sort=False)
    # Drop exact-duplicate rows that can appear if files overlap in date range
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "__source_file"])

    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["Savings_INR"] = df["Launch_Price_INR"] - df["Current_Price_INR"]
    df["Has_Sale_Event"] = df["Sale_Event"].notna()
    df = df.sort_values("Date").reset_index(drop=True)
    return df, [os.path.basename(p) for p in paths]


df_raw, loaded_files = load_data(DATA_DIR, FILE_PATTERN)

if df_raw is None:
    st.error(
        f"No files matching `{FILE_PATTERN}` were found in `{os.path.abspath(DATA_DIR)}`. "
        "Place the dataset CSV(s) in that folder (or a `data/` subfolder) and rerun the app."
    )
    st.stop()

# #Load the model for prediction
# @st.cache_resource
# def load_model():
#     return joblib.load("xgb_price_model.joblib")

# model = load_model()


# Slicers
st.sidebar.title("🔎 Filters")

min_date, max_date = df_raw["Date"].min(), df_raw["Date"].max()
date_range = st.sidebar.date_input(
    "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)
if len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    start_date, end_date = min_date, max_date

categories = st.sidebar.multiselect(
    "Product Category", sorted(df_raw["Product_Category"].unique()),
    default=sorted(df_raw["Product_Category"].unique()),
)
platforms = st.sidebar.multiselect(
    "Platform", sorted(df_raw["Platform"].unique()),
    default=sorted(df_raw["Platform"].unique()),
)
conditions = st.sidebar.multiselect(
    "Condition", sorted(df_raw["Condition"].unique()),
    default=sorted(df_raw["Condition"].unique()),
)
models_available = sorted(df_raw[df_raw["Product_Category"].isin(categories)]["Model_Name"].unique())
models = st.sidebar.multiselect("Model (optional, leave empty = all)", models_available, default=[])

df = df_raw[
    (df_raw["Date"] >= start_date)
    & (df_raw["Date"] <= end_date)
    & (df_raw["Product_Category"].isin(categories))
    & (df_raw["Platform"].isin(platforms))
    & (df_raw["Condition"].isin(conditions))
]
if models:
    df = df[df["Model_Name"].isin(models)]

if df.empty:
    st.warning("No data matches the current filters. Widen your selection.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.caption(f"Rows in view: **{len(df):,}** / {len(df_raw):,}")


#header

col1, col2 = st.columns([1, 8])

with col1:
    st.image(apple_icon, width=70)

with col2:
    st.markdown(
        """
        <h1 style="
            margin:0;
            padding:0;
            line-height:1;
        ">
        Apple Product Intelligence Center
        </h1>
        """,
        unsafe_allow_html=True
    )

st.caption(
    f"Based on Apple product retail price"
    f"Daily price/stock/rating tracking across Amazon & Flipkart · "
    f"{df_raw['Date'].min().date()} → {df_raw['Date'].max().date()}"
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Avg Current Price", f"₹{df['Current_Price_INR'].mean():,.0f}")
k2.metric("Avg Discount", f"{df['Discount_Pct'].mean():.1f}%")
k3.metric("Avg Rating", f"{df['Rating'].mean():.2f} ★")
oos_rate = (df["Stock_Status"] == "Out of Stock").mean() * 100
k4.metric("Out-of-Stock Rate", f"{oos_rate:.1f}%")
k5.metric("Total Reviews Tracked", f"{df['Reviews_Count'].sum():,.0f}")

st.markdown("---")

#Tabs
tab_overview, tab_pricing, tab_platform, tab_sales, tab_stock, tab_ratings, tab_trends, tab_prediction, tab_ai, tab_price_assistance = st.tabs(
    ["📊 Overview", "💰 Pricing & Discounts", "🏪 Platform Comparison",
     "🎉 Sale Events", "📦 Inventory", "⭐ Ratings", "📈 Trends & Lifecycle","🤖 Price Prediction","💬 AI Assistant","🤖 Apple Pricing Assistant"]
)

#tab_overview
with tab_overview:
    st.subheader("Key Metrics")

    #KPI
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Unique Models", f"{df['Model_Name'].nunique()}")
    o2.metric("Categories", f"{df['Product_Category'].nunique()}")
    o3.metric("Platforms", f"{df['Platform'].nunique()}")
    o4.metric("Days Covered", f"{(df['Date'].max() - df['Date'].min()).days:,}")

    #price & value metrics
    price_by_cat = df.groupby("Product_Category")["Current_Price_INR"].mean()
    most_expensive_cat = price_by_cat.idxmax()
    cheapest_cat = price_by_cat.idxmin()
    avg_savings = df["Savings_INR"].mean()
    best_rated_model = df.groupby("Model_Name")["Rating"].mean().idxmax()

    o5, o6, o7, o8 = st.columns(4)
    o5.metric("Priciest Category (avg)", most_expensive_cat, f"₹{price_by_cat.max():,.0f}")
    o6.metric("Cheapest Category (avg)", cheapest_cat, f"₹{price_by_cat.min():,.0f}")
    o7.metric("Avg ₹ Saved vs. Launch", f"₹{avg_savings:,.0f}")
    o8.metric("Top-Rated Model", best_rated_model,
              f"{df.groupby('Model_Name')['Rating'].mean().max():.2f} ★")

    st.markdown("---")

    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Listings by Category")
        cat_counts = df["Product_Category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig = px.pie(cat_counts, names="Category", values="Count", hole=0.45,
                     color="Category", color_discrete_map=CATEGORY_COLORS)
        st.plotly_chart(fig, width='stretch')

    with c2:
        st.subheader("Avg Current Price by Category")
        price_cat = df.groupby("Product_Category")["Current_Price_INR"].mean().sort_values().reset_index()
        fig = px.bar(price_cat, x="Current_Price_INR", y="Product_Category", orientation="h",
                     color="Product_Category", color_discrete_map=CATEGORY_COLORS,
                     labels={"Current_Price_INR": "Avg Current Price (₹)", "Product_Category": ""})
        st.plotly_chart(fig, width='stretch')

    st.subheader("New vs. Refurbished — Price Gap")
    cond_price = df.groupby(["Product_Category", "Condition"])["Current_Price_INR"].mean().reset_index()
    fig = px.bar(cond_price, x="Product_Category", y="Current_Price_INR", color="Condition",
                 barmode="group", labels={"Current_Price_INR": "Avg Current Price (₹)"})
    st.plotly_chart(fig, width='stretch')

    st.info(
        f"**Insight:** Refurbished units sell for an average of "
        f"**₹{df[df.Condition=='New']['Current_Price_INR'].mean() - df[df.Condition=='Renewed/Refurbished']['Current_Price_INR'].mean():,.0f} less** "
        f"than New units across the filtered data."
    )

#tab_pricing(Pricing and Discount)
with tab_pricing:
    st.subheader("Discount Depth by Category & Condition")
    fig = px.box(df, x="Product_Category", y="Discount_Pct", color="Condition",
                 points=False, labels={"Discount_Pct": "Discount (%)"})
    st.plotly_chart(fig, width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Deepest-Discounted Models (avg)")
        top_disc = (df.groupby("Model_Name")["Discount_Pct"].mean()
                    .sort_values(ascending=False).head(10).reset_index())
        fig = px.bar(top_disc, x="Discount_Pct", y="Model_Name", orientation="h",
                     labels={"Discount_Pct": "Avg Discount (%)", "Model_Name": ""})
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width='stretch')

    with c2:
        st.subheader("Least-Discounted Models (avg)")
        low_disc = (df.groupby("Model_Name")["Discount_Pct"].mean()
                    .sort_values(ascending=True).head(10).reset_index())
        fig = px.bar(low_disc, x="Discount_Pct", y="Model_Name", orientation="h",
                     labels={"Discount_Pct": "Avg Discount (%)", "Model_Name": ""},
                     color_discrete_sequence=["#ff9f0a"])
        fig.update_layout(yaxis={"categoryorder": "total descending"})
        st.plotly_chart(fig, width='stretch')

    st.subheader("Discount % vs. Rating (Does deeper discount hurt perceived quality?)")
    sample = df.sample(min(4000, len(df)), random_state=42)
    try:
        import statsmodels.api  # noqa: F401 - only used to check availability for the trendline
        fig = px.scatter(sample, x="Discount_Pct", y="Rating", color="Product_Category",
                          opacity=0.4, color_discrete_map=CATEGORY_COLORS,
                          trendline="ols", trendline_scope="overall")
    except ImportError:
        st.caption("ℹ️ Install `statsmodels` (`pip install statsmodels`) to show the trendline.")
        fig = px.scatter(sample, x="Discount_Pct", y="Rating", color="Product_Category",
                          opacity=0.4, color_discrete_map=CATEGORY_COLORS)
    st.plotly_chart(fig, width='stretch')

    corr = df["Discount_Pct"].corr(df["Rating"])
    st.info(f"**Insight:** Correlation between discount % and rating is **{corr:.3f}** "
            f"({'weak positive' if corr > 0.05 else 'weak negative' if corr < -0.05 else 'negligible'} relationship) — "
            f"discounting doesn't strongly move perceived product quality here.")


#Platform comparision
with tab_platform:
    st.subheader("Amazon vs. Flipkart — Average Price by Category")
    plat_price = df.groupby(["Product_Category", "Platform"])["Current_Price_INR"].mean().reset_index()
    fig = px.bar(plat_price, x="Product_Category", y="Current_Price_INR", color="Platform",
                 barmode="group", labels={"Current_Price_INR": "Avg Current Price (₹)"})
    st.plotly_chart(fig, width='stretch')

    st.subheader("Avg Discount % by Platform")
    plat_disc = df.groupby("Platform")["Discount_Pct"].mean().reset_index()
    fig = px.bar(plat_disc, x="Platform", y="Discount_Pct", color="Platform",
                labels={"Discount_Pct": "Avg Discount (%)"})
    st.plotly_chart(fig, width='stretch')

    

    st.subheader("Avg Rating by Platform & Category")
    plat_rating = df.groupby(["Product_Category", "Platform"])["Rating"].mean().reset_index()
    fig = px.bar(plat_rating, x="Product_Category", y="Rating", color="Platform", barmode="group")
    fig.update_yaxes(range=[3.5, 5])
    st.plotly_chart(fig, width='stretch')

    cheaper = plat_price.pivot(index="Product_Category", columns="Platform", values="Current_Price_INR")
    if {"Amazon", "Flipkart"}.issubset(cheaper.columns):
        cheaper["diff"] = cheaper["Amazon"] - cheaper["Flipkart"]
        cheaper_platform = "Flipkart" if cheaper["diff"].mean() > 0 else "Amazon"
        st.info(f"**Insight:** On average, **{cheaper_platform}** offers lower prices across the filtered categories "
                f"(avg gap: **₹{abs(cheaper['diff'].mean()):,.0f}**).")

#Sale Event
with tab_sales:
    sale_summary = df.groupby("Has_Sale_Event")["Discount_Pct"].mean().reset_index()
    sale_summary["Has_Sale_Event"] = sale_summary["Has_Sale_Event"].map({True: "Sale Event", False: "Regular Day"})

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Avg Discount: Sale Event vs. Regular Day")
        fig = px.bar(sale_summary, x="Has_Sale_Event", y="Discount_Pct",
                     color="Has_Sale_Event", labels={"Discount_Pct": "Avg Discount (%)", "Has_Sale_Event": ""})
        st.plotly_chart(fig, width='stretch')

    with c2:
        st.subheader("Discount by Sale Event Type")
        event_disc = df[df["Sale_Event"].notna()].groupby("Sale_Event")["Discount_Pct"].mean().sort_values().reset_index()
        fig = px.bar(event_disc, x="Discount_Pct", y="Sale_Event", orientation="h",
                     labels={"Discount_Pct": "Avg Discount (%)", "Sale_Event": ""})
        st.plotly_chart(fig, width='stretch')

    st.subheader("Which Category Benefits Most from Sale Events?")
    event_cat = (df[df["Sale_Event"].notna()]
                 .groupby("Product_Category")["Discount_Pct"].mean()
                 .reset_index().sort_values("Discount_Pct", ascending=False))
    fig = px.bar(event_cat, x="Product_Category", y="Discount_Pct", color="Product_Category",
                 color_discrete_map=CATEGORY_COLORS, labels={"Discount_Pct": "Avg Discount During Sales (%)"})
    st.plotly_chart(fig, width='stretch')

    reg = sale_summary.loc[sale_summary.Has_Sale_Event == "Regular Day", "Discount_Pct"].values
    sale = sale_summary.loc[sale_summary.Has_Sale_Event == "Sale Event", "Discount_Pct"].values
    if len(reg) and len(sale):
        lift = sale[0] - reg[0]
        st.info(f"**Insight:** Discounts run **{lift:.1f} percentage points deeper** on tagged sale-event days "
                f"vs. regular days.")

#Inventory
with tab_stock:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Stock Status Distribution")
        stock_counts = df["Stock_Status"].value_counts().reset_index()
        stock_counts.columns = ["Stock_Status", "Count"]
        fig = px.pie(stock_counts, names="Stock_Status", values="Count", hole=0.45)
        st.plotly_chart(fig, width='stretch')

    with c2:
        st.subheader("Out-of-Stock Rate by Category")
        oos = df.groupby("Product_Category").apply(
            lambda x: (x["Stock_Status"] == "Out of Stock").mean() * 100
        ).reset_index(name="OOS_Rate")
        fig = px.bar(oos, x="Product_Category", y="OOS_Rate", color="Product_Category",
                     color_discrete_map=CATEGORY_COLORS, labels={"OOS_Rate": "Out-of-Stock Rate (%)"})
        st.plotly_chart(fig, width='stretch')

    st.subheader("Out-of-Stock Rate Over Time")
    monthly_oos = df.groupby("Month").apply(
        lambda x: (x["Stock_Status"] == "Out of Stock").mean() * 100
    ).reset_index(name="OOS_Rate")
    fig = px.line(monthly_oos, x="Month", y="OOS_Rate", markers=True,
                  labels={"OOS_Rate": "Out-of-Stock Rate (%)"})
    st.plotly_chart(fig, width='stretch')

    st.subheader("Discount Level by Stock Status (does clearance drive discounts?)")
    fig = px.box(df, x="Stock_Status", y="Discount_Pct", color="Stock_Status")
    st.plotly_chart(fig, width='stretch')

#Raiting
with tab_ratings:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Avg Rating by Category")
        rating_cat = df.groupby("Product_Category")["Rating"].mean().sort_values().reset_index()
        fig = px.bar(rating_cat, x="Rating", y="Product_Category", orientation="h",
                     color="Product_Category", color_discrete_map=CATEGORY_COLORS)
        fig.update_xaxes(range=[3.5, 5])
        st.plotly_chart(fig, width='stretch')

    with c2:
        st.subheader("Rating: New vs. Refurbished")
        rating_cond = df.groupby("Condition")["Rating"].mean().reset_index()
        fig = px.bar(rating_cond, x="Condition", y="Rating", color="Condition")
        fig.update_yaxes(range=[3.5, 5])
        st.plotly_chart(fig, width='stretch')

    st.subheader("Rating Distribution")
    fig = px.histogram(df, x="Rating", color="Product_Category", nbins=25, barmode="overlay", opacity=0.6,
                        color_discrete_map=CATEGORY_COLORS)
    st.plotly_chart(fig, width='stretch')

    st.subheader("Most-Reviewed Models")
    top_reviewed = df.groupby("Model_Name")["Reviews_Count"].sum().sort_values(ascending=False).head(10).reset_index()
    fig = px.bar(top_reviewed, x="Reviews_Count", y="Model_Name", orientation="h",
                 labels={"Reviews_Count": "Total Reviews"})
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, width='stretch')

#trends and lifecycle
with tab_trends:
    st.subheader("Average Price Over Time by Category")
    trend = df.groupby(["Month", "Product_Category"])["Current_Price_INR"].mean().reset_index()
    fig = px.line(trend, x="Month", y="Current_Price_INR", color="Product_Category",
                  color_discrete_map=CATEGORY_COLORS, labels={"Current_Price_INR": "Avg Price (₹)"})
    st.plotly_chart(fig, width='stretch')

    st.subheader("Price Trajectory for a Selected Model")
    model_choice = st.selectbox("Choose a model", sorted(df["Model_Name"].unique()))
    model_df = df[df["Model_Name"] == model_choice].groupby("Date")["Current_Price_INR"].mean().reset_index()
    fig = px.line(model_df, x="Date", y="Current_Price_INR",
                  labels={"Current_Price_INR": "Avg Price (₹)"}, markers=False)
    launch_price = df[df["Model_Name"] == model_choice]["Launch_Price_INR"].iloc[0]
    fig.add_hline(y=launch_price, line_dash="dash", line_color="gray",
                  annotation_text=f"Launch price ₹{launch_price}")
    st.plotly_chart(fig, width='stretch')

    st.subheader("Discount % Trend Over Time")
    disc_trend = df.groupby("Month")["Discount_Pct"].mean().reset_index()
    fig = px.area(disc_trend, x="Month", y="Discount_Pct", labels={"Discount_Pct": "Avg Discount (%)"})
    st.plotly_chart(fig, width='stretch')

    first_price = model_df["Current_Price_INR"].iloc[0]
    last_price = model_df["Current_Price_INR"].iloc[-1]
    pct_change = (last_price - first_price) / first_price * 100
    st.info(f"**Insight:** *{model_choice}* moved from **₹{first_price:,.0f}** to **₹{last_price:,.0f}** "
            f"over the tracked period ({pct_change:+.1f}%).")


#Prediction tab
with tab_prediction:
       
    @st.cache_resource
    def load_model():
        return joblib.load("xgb_price_model_corrected.joblib")


    @st.cache_data
    def load_lookup():
        with open("model_lookup.json") as f:
            data = json.load(f)
        lookup = {row["Model_Name"]: row for row in data}
        return lookup


    model = load_model()
    lookup = load_lookup()

    PLATFORMS = ["Amazon", "Flipkart"]
    CONDITIONS = ["New", "Renewed/Refurbished"]
    STOCK_STATUSES = ["In Stock", "Low Stock", "Out of Stock"]
    CATEGORIES = sorted({v["category"] for v in lookup.values()})

    # --------------------------------------------------------------------------
    # Header
    # --------------------------------------------------------------------------
    
    col1, col2 = st.columns([1, 8])

    with col1:
        st.image(apple_icon, width=70)

    with col2:
        st.markdown(
                """
                <h1 style="
                    margin:0;
                    padding:0;
                    line-height:1;
                ">
                Apple Product Price Predictor
                </h1>
                """,
                unsafe_allow_html=True
            )
    st.caption(
        "Predict the market price of any Apple product, on any day since its "
        "launch — including future-dated forecasts for recently launched "
        "products (e.g. price in 2028, 2030, ...)."
    )

    st.divider()

    #Product Selection
    st.subheader("1️⃣ Choose the product")

    col1, col2 = st.columns(2)

    with col1:
        category = st.selectbox("Product Category", CATEGORIES)

    models_in_category = sorted(
        [name for name, v in lookup.items() if v["category"] == category]
    )

    with col2:
        model_name = st.selectbox("Model", models_in_category)

    info = lookup[model_name]

    # --------------------------------------------------------------------------
    # Step 2 — listing details
    # --------------------------------------------------------------------------
    st.subheader("2️⃣ Listing details")

    col1, col2, col3 = st.columns(3)
    with col1:
        platform = st.selectbox("Platform", PLATFORMS)
    with col2:
        condition = st.selectbox("Condition", CONDITIONS)
    with col3:
        stock_status = st.selectbox("Stock Status", STOCK_STATUSES)

    col1, col2, col3 = st.columns(3)
    with col1:
        launch_price_inr = st.number_input(
            "Launch Price (₹ INR)",
            min_value=1000,
            max_value=500000,
            value=int(info["launch_price_inr"]),
            step=1000,
            help="Official launch price of this model. Auto-filled, editable "
            "for what-if scenarios.",
        )
    with col2:
        rating = st.slider(
            "Rating", min_value=3.0, max_value=5.0, value=float(info["avg_rating"]), step=0.1
        )
    with col3:
        reviews_count = st.number_input(
            "Reviews Count",
            min_value=0,
            max_value=20000,
            value=int(info["avg_reviews"]),
            step=10,
        )

    # --------------------------------------------------------------------------
    # Step 3 — launch date & target (prediction) date
    # --------------------------------------------------------------------------
    st.subheader("3️⃣ Launch date & the date you want a price for")

    col1, col2 = st.columns(2)
    with col1:
        launch_date = st.date_input(
            "Launch Date",
            value=date.fromisoformat(info["launch_date"]),
            min_value=date(2015, 1, 1),
            max_value=date(2035, 12, 31),
            help="When this product launched. Set a 2026 date for a brand-new "
            "product, then pick any future target date below.",
        )
    with col2:
        target_date = st.date_input(
            "Predict price on this date",
            value=date.today() + timedelta(days=30),
            min_value=launch_date,
            max_value=date(2040, 12, 31),
            help="Can be far in the future (2028, 2030, ...) to see the "
            "model's long-range forecast.",
        )

    days_since_launch = (target_date - launch_date).days

    st.info(
        f"📅 **Days since launch:** {days_since_launch} days  "
        f"(from {launch_date.strftime('%d %b %Y')} to {target_date.strftime('%d %b %Y')})"
    )

    max_training_days = 2141
    if days_since_launch > max_training_days:
        st.warning(
            f"⚠️ This is {days_since_launch - max_training_days} days beyond the "
            "range the model was trained on (~5.9 years). Long-range "
            "extrapolations like this are a rough trend estimate, not a "
            "precise forecast — accuracy decreases the further out you go."
        )

    # --------------------------------------------------------------------------
    # Build input row
    # --------------------------------------------------------------------------
    def build_row(days, plat, stock):
        return pd.DataFrame(
            [
                {
                    "Days_Since_Launch": days,
                    "Platform": plat,
                    "Product_Category": category,
                    "Model_Name": model_name,
                    "Condition": condition,
                    "Launch_Price_INR": launch_price_inr,
                    "Stock_Status": stock,
                    "Rating": rating,
                    "Reviews_Count": reviews_count,
                    "Year": (launch_date + timedelta(days=days)).year,
                    "Month": (launch_date + timedelta(days=days)).month,
                }
            ]
        )


    st.divider()

    # --------------------------------------------------------------------------
    # Predict button
    # --------------------------------------------------------------------------
    predict_clicked = st.button("🔮 Predict Price", type="primary", width='stretch')

    if predict_clicked:
        row = build_row(days_since_launch, platform, stock_status)
        predicted_price = float(model.predict(row)[0])
        predicted_price = max(predicted_price, 0)

        st.subheader("Result")
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Price (₹ INR)", f"₹{predicted_price:,.0f}")
        delta_pct = (predicted_price - launch_price_inr) / launch_price_inr * 100
        c2.metric(
            "vs Launch Price",
            f"₹{launch_price_inr:,.0f}",
            f"{delta_pct:+.1f}%",
        )
        c3.metric("Predicted USD (approx, @83 ₹/$)", f"${predicted_price/83:,.0f}")

        st.success(
            f"On **{target_date.strftime('%d %b %Y')}** ({days_since_launch} days after "
            f"launch), the **{model_name}** ({condition}, {stock_status}) is predicted "
            f"to cost **₹{predicted_price:,.0f}** on **{platform}**."
        )

        # ----------------------------------------------------------------
        # Trend chart: price trajectory from launch to target date (+buffer)
        # ----------------------------------------------------------------
        st.subheader("📈 Price trend over time")

        horizon = max(days_since_launch + 365, 730)
        horizon = min(horizon, 3650)  # cap at ~10 years for chart sanity
        step = max(horizon // 120, 1)
        day_points = list(range(0, horizon + 1, step))
        if days_since_launch not in day_points:
            day_points.append(days_since_launch)
            day_points.sort()

        trend_rows = pd.concat(
            [build_row(d, platform, stock_status) for d in day_points], ignore_index=True
        )
        trend_prices = model.predict(trend_rows)
        trend_dates = [launch_date + timedelta(days=d) for d in day_points]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=trend_dates,
                y=trend_prices,
                mode="lines",
                name="Predicted price",
                line=dict(color="#0071e3", width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[target_date],
                y=[predicted_price],
                mode="markers",
                name="Your selected date",
                marker=dict(color="#ff3b30", size=12, symbol="star"),
            )
        )
        fig.add_hline(
            y=launch_price_inr,
            line_dash="dash",
            line_color="gray",
            annotation_text="Launch price",
            annotation_position="top left",
        )
        if days_since_launch > max_training_days:
            fig.add_vline(
                x=launch_date + timedelta(days=max_training_days),
                line_dash="dot",
                line_color="orange",
                annotation_text="Training data ends here →",
                annotation_position="top right",
            )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Predicted Price (₹ INR)",
            height=450,
            margin=dict(t=20, b=20),
            hovermode="x unified",
        )
        st.plotly_chart(fig, width='stretch')

    else:
        st.info("👆 Fill in the details above and click **Predict Price** to see the result.")
# ChatBot
    # with tab_ai:
        
    #     st.title("🤖 AI Assistant")

    #     st.markdown(
    #         """
    #         Ask questions about:

    #         - Apple products
    #         - Pricing trends
    #         - Discounts
    #         - Ratings
    #         - Platform comparison
    #         - Dataset insights
    #         """
    #     )

    #     if "messages" not in st.session_state:
    #         st.session_state.messages = []

    #     # Display chat history
    #     for message in st.session_state.messages:

    #         with st.chat_message(message["role"]):
    #             st.markdown(message["content"])

    #     prompt = st.chat_input(
    #         "Ask anything about the dashboard..."
    #     )

    #     if prompt:

    #         # Store user message
    #         st.session_state.messages.append(
    #             {
    #                 "role": "user",
    #                 "content": prompt
    #             }
    #         )

    #         with st.chat_message("user"):
    #             st.markdown(prompt)

    #         with st.chat_message("assistant"):

    #             with st.spinner("🍎 Thinking..."):

    #                 try:

    #                     # Optional: provide dashboard context
    #                     context = f"""
    #                     You are an Apple Market Intelligence Assistant.

    #                     Dataset Information:

    #                     Total Records: {len(df)}

    #                     User Question:
    #                     {prompt}

    #                     Answer professionally.
    #                     """

    #                     response = client.models.generate_content(
    #                         model="gemini-3.6-flash",
    #                         contents=context
    #                     )

    #                     answer = response.text

    #                     st.markdown(answer)

    #                     st.session_state.messages.append(
    #                         {
    #                             "role": "assistant",
    #                             "content": answer
    #                         }
    #                     )

    #                 except Exception as e:

    #                     st.error(f"Error: {e}")
                    
    with tab_ai:

        st.title("🤖 AI Assistant")

        st.markdown(
            """
            Ask questions about:

            - Apple products
            - Pricing trends
            - Discounts
            - Ratings
            - Platform comparison
            - Dataset insights
            """
        )

        # Separate history for AI Assistant
        if "chatbot_messages" not in st.session_state:
            st.session_state.chatbot_messages = []

        # Display chat history
        for message in st.session_state.chatbot_messages:

            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Clear Chat Button
        col1, col2 = st.columns([1, 5])

        with col1:
            if st.button("🗑️ Clear AI Chat"):
                st.session_state.chatbot_messages = []
                st.rerun()

        prompt = st.chat_input(
            "Ask anything about the dashboard...",
            key="ai_chat_input"
        )

        if prompt:

            # Store User Message
            st.session_state.chatbot_messages.append(
                {
                    "role": "user",
                    "content": prompt
                }
            )

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):

                with st.spinner(" Thinking..."):

                    try:

                        context = f"""
                        You are an Apple Market Intelligence Assistant.

                        # Dataset Information:
                        # Total Records: {len(df)}

                        User Question:
                        {prompt}

                        Answer professionally and concisely.
                        """

                        response = client.models.generate_content(
                            model="gemini-3.6-flash",
                            contents=context
                        )

                        answer = response.text

                        st.markdown(answer)

                        # Store Assistant Message
                        st.session_state.chatbot_messages.append(
                            {
                                "role": "assistant",
                                "content": answer
                            }
                        )

                    except Exception as e:

                        st.error(f"Error: {e}")


    # with tab_price_assistance:
        
    #     st.title("Apple Pricing Assistant")

    #     st.caption(
    #         "Ask questions about Apple product pricing, discounts, ratings, reviews, and sale events."
    #     )

    #     if "messages" not in st.session_state:
    #         st.session_state.messages = []

    #     # Display chat history
    #     for message in st.session_state.messages:

    #         with st.chat_message(message["role"]):
    #             st.markdown(message["content"])

    #     question = st.chat_input(
    #         "Ask about Apple products..."
    #     )

    #     if question:

    #         # User Message
    #         st.session_state.messages.append(
    #             {
    #                 "role": "user",
    #                 "content": question
    #             }
    #         )

    #         with st.chat_message("user"):
    #             st.markdown(question)

    #         # Assistant Message
    #         with st.chat_message("assistant"):

    #             with st.spinner("🔍 Searching Apple Pricing Database..."):

    #                 result = ask_question(question)

    #                 answer = result["result"]

    #                 # Short clean response
    #                 st.markdown("### 📌 Quick Answer")
    #                 st.markdown(answer)

    #                 # Sources
    #                 with st.expander("📄 View Retrieved Sources"):

    #                     for i, doc in enumerate(
    #                         result["source_documents"],
    #                         start=1
    #                     ):

    #                         st.markdown(
    #                             f"**Source {i}**"
    #                         )

    #                         st.info(
    #                             doc.page_content[:500]
    #                         )

    #         st.session_state.messages.append(
    #             {
    #                 "role": "assistant",
    #                 "content": answer
    #             }
    #         )

    with tab_price_assistance:

        st.title("🔍 Apple Data Explorer")

        st.caption(
            "Ask questions about Apple product pricing, discounts, ratings, reviews, and sale events."
        )

        # Separate chat history for RAG Assistant
        if "rag_messages" not in st.session_state:
            st.session_state.rag_messages = []

        # Display chat history
        for message in st.session_state.rag_messages:

            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Clear Chat Button
        col1, col2 = st.columns([1, 5])

        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.rag_messages = []
                st.rerun()

        question = st.chat_input(
            "Ask about Apple products..."
        )

        if question:

            # Save User Message
            st.session_state.rag_messages.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):

                with st.spinner("🔍 Searching Apple Pricing Database..."):

                    result = ask_question(question)

                    answer = result["result"]

                    st.markdown(answer)

                    # Optional Sources
                    with st.expander("📄 Retrieved Sources"):

                        for i, doc in enumerate(
                            result["source_documents"],
                            start=1
                        ):

                            st.markdown(f"**Source {i}**")

                            st.caption(
                                doc.page_content[:300] + "..."
                            )

            # Save Assistant Message
            st.session_state.rag_messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )