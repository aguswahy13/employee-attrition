import os
import pandas as pd
import streamlit as st
import plotly.express as px

# 1) Determine the folder in which this script resides
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# 2) Build the CSV path by joining that folder with "data-clean.csv"
DATA_PATH = os.path.join(BASE_DIR, "data-clean.csv")

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

# 3) Now load the data from the correct location
df = load_data(DATA_PATH)

# 3. Compute Metrics
total_employees = len(df)
# Assuming 'Attrition' column is 1 for leaving, 0 for staying
total_leavers = int(df["Attrition"].sum())
attrition_rate = (total_leavers / total_employees) * 100

# 4. Title and Metric Cards
st.title("🌟 Employee Attrition Dashboard")
st.markdown(
    """
    This dashboard helps HR and managers monitor key factors influencing attrition. 
    Use the controls below to explore different demographic, role, and satisfaction metrics.
    """
)

col1, col2, col3 = st.columns(3)
col1.metric(label="Total Employees", value=f"{total_employees:,}")
col2.metric(label="Total Leavers", value=f"{total_leavers:,}")
col3.metric(label="Attrition Rate", value=f"{attrition_rate:.2f}%")

st.markdown("---")

# 5. Sidebar Filters
st.sidebar.header("🔍 Filters and Settings")

# 5a. Choose categorical feature for grouping
categorical_cols = [
    "BusinessTravel", "Department", "EducationField", "Gender", 
    "JobRole", "MaritalStatus", "OverTime"
]
cat_feature = st.sidebar.selectbox(
    "Select Categorical Feature", 
    options=categorical_cols, 
    index=0
)

# 5b. Choose numeric feature for distribution
numeric_cols = [
    "Age", "DailyRate", "DistanceFromHome", "MonthlyIncome", "JobSatisfaction", 
    "EnvironmentSatisfaction", "YearsAtCompany", "YearsInCurrentRole", 
    "YearsSinceLastPromotion"
]
num_feature = st.sidebar.selectbox(
    "Select Numeric Feature", 
    options=numeric_cols, 
    index=0
)

# 5c. Attrition Filter (to compare leavers vs total)
attr_filter = st.sidebar.multiselect(
    "Filter by Attrition Status",
    options=[0, 1],
    format_func=lambda x: "Stayed" if x == 0 else "Left",
    default=[0, 1]
)

# 6. Filter Data Based on Sidebar
df_filtered = df[df["Attrition"].isin(attr_filter)]

# 7. Categorical Grouped Bar Chart: Attrition Rate by Category
st.subheader(f"Attrition Rate by {cat_feature}")

# Compute counts by category & attrition
grouped = (
    df_filtered
    .groupby([cat_feature, "Attrition"])
    .size()
    .reset_index(name="Count")
)

# Pivot so that Attrition=1 and Attrition=0 become separate columns
pivot = grouped.pivot(index=cat_feature, columns="Attrition", values="Count").fillna(0)
pivot["Attrition Rate (%)"] = pivot[1] / (pivot[0] + pivot[1]) * 100
pivot = pivot.sort_values("Attrition Rate (%)", ascending=False).reset_index()

fig_cat = px.bar(
    pivot,
    x=cat_feature,
    y="Attrition Rate (%)",
    text=pivot["Attrition Rate (%)"].apply(lambda x: f"{x:.1f}%"),
    labels={"Attrition Rate (%)": "Attrition Rate (%)"},
    title=f"Attrition Rate by {cat_feature}"
)
fig_cat.update_traces(textposition="outside")
st.plotly_chart(fig_cat, use_container_width=True)

st.markdown("---")

# 8. Numeric Distribution: Compare feature distribution for Leavers vs Stay
st.subheader(f"Distribution of {num_feature} by Attrition Status")

fig_num = px.histogram(
    df_filtered,
    x=num_feature,
    color=df_filtered["Attrition"].map({0: "Stayed", 1: "Left"}),
    barmode="overlay",
    nbins=30,
    labels={"color": "Attrition Status"},
    title=f"{num_feature} Distribution: Stayed vs Left"
)
fig_num.update_layout(legend_title_text="Attrition Status")
st.plotly_chart(fig_num, use_container_width=True)

st.markdown("---")

# 9. Box Plot: Numeric Feature vs Attrition
st.subheader(f"Box Plot of {num_feature} by Attrition Status")

fig_box = px.box(
    df_filtered,
    x="Attrition",
    y=num_feature,
    labels={"Attrition": "Attrition Status", num_feature: num_feature},
    title=f"{num_feature} by Attrition"
)
fig_box.update_xaxes(
    tickmode="array",
    tickvals=[0, 1],
    ticktext=["Stayed", "Left"]
)
st.plotly_chart(fig_box, use_container_width=True)

st.markdown("---")

# 10. Correlation Heatmap (Optional)
if st.sidebar.checkbox("Show Correlation Heatmap", value=False):
    st.subheader("Correlation Matrix of Numeric Features")
    corr = df[numeric_cols + ["Attrition"]].corr()
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Agus Wahyudi | © 2025")
