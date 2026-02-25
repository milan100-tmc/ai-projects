import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="Sales Intelligence Dashboard", page_icon="📈", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/sales_data.csv"))

df = load_data()

st.title("📈 Sales Intelligence Dashboard")
st.caption("Real-time sales analytics with AI-powered insights")

# Filters
col1, col2, col3 = st.columns(3)
with col1:
    regions = st.multiselect("Region", df['region'].unique(), default=df['region'].unique())
with col2:
    products = st.multiselect("Product", df['product'].unique(), default=df['product'].unique())
with col3:
    months = st.multiselect("Month", sorted(df['month'].unique()), default=sorted(df['month'].unique()))

filtered = df[df['region'].isin(regions) & df['product'].isin(products) & df['month'].isin(months)]

# KPIs
st.divider()
k1, k2, k3, k4 = st.columns(4)
total_revenue = filtered['revenue'].sum()
total_units = filtered['units_sold'].sum()
total_target = filtered['target'].sum()
attainment = (total_revenue / total_target * 100) if total_target > 0 else 0

k1.metric("Total Revenue", f"€{total_revenue:,.0f}")
k2.metric("Units Sold", f"{total_units:,}")
k3.metric("Target", f"€{total_target:,.0f}")
k4.metric("Target Attainment", f"{attainment:.1f}%", delta=f"{attainment-100:.1f}%")

st.divider()

# Charts
col1, col2 = st.columns(2)
with col1:
    rev_by_month = filtered.groupby('month')['revenue'].sum().reset_index()
    fig1 = px.line(rev_by_month, x='month', y='revenue', title='Revenue Over Time', markers=True)
    fig1.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    rev_by_region = filtered.groupby('region')['revenue'].sum().reset_index()
    fig2 = px.bar(rev_by_region, x='region', y='revenue', title='Revenue by Region', color='region')
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    rev_by_product = filtered.groupby('product')['revenue'].sum().reset_index()
    fig3 = px.pie(rev_by_product, values='revenue', names='product', title='Revenue by Product')
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    region_month = filtered.groupby(['month', 'region'])['revenue'].sum().reset_index()
    fig4 = px.line(region_month, x='month', y='revenue', color='region', title='Regional Performance Over Time', markers=True)
    fig4.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

# AI Analyst
st.subheader("🤖 AI Analyst")
col1, col2 = st.columns([1, 3])
with col1:
    analyze_btn = st.button("🔍 Analyse this data", type="primary")

if analyze_btn:
    summary = filtered.groupby('region').agg(
        revenue=('revenue', 'sum'),
        target=('target', 'sum'),
        units=('units_sold', 'sum')
    ).reset_index()
    summary['attainment'] = (summary['revenue'] / summary['target'] * 100).round(1)

    monthly = filtered.groupby('month')['revenue'].sum().reset_index()
    best_month = monthly.loc[monthly['revenue'].idxmax(), 'month']
    worst_month = monthly.loc[monthly['revenue'].idxmin(), 'month']

    data_summary = f"""
    Sales Data Summary:
    - Total Revenue: €{total_revenue:,.0f}
    - Target Attainment: {attainment:.1f}%
    - Best Month: {best_month}
    - Worst Month: {worst_month}
    - Regional Performance:
    {summary.to_string()}
    """

    with st.spinner("Analysing your data..."):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a senior business analyst. Provide sharp, actionable insights from sales data. Be specific, identify patterns, flag risks, suggest actions. Use bullet points."},
                {"role": "user", "content": f"Analyse this sales data and give me the 5 most important insights a sales director needs to know:\n{data_summary}"}
            ]
        )
    st.markdown(response.choices[0].message.content)

st.divider()
if "dash_messages" not in st.session_state:
    st.session_state.dash_messages = []

st.subheader("💬 Ask the data anything")
for message in st.session_state.dash_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("e.g. Why is South underperforming? Which product should we push?"):
    st.session_state.dash_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    data_context = f"Sales data summary:\n{filtered.groupby(['region','product'])['revenue'].sum().reset_index().to_string()}"
    history = st.session_state.dash_messages[-6:]
    messages = [
        {"role": "system", "content": f"You are a senior business analyst. Answer questions about this sales data with specific numbers and actionable recommendations:\n{data_context}"}
    ] + history

    with st.chat_message("assistant"):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )
        reply = response.choices[0].message.content
        st.markdown(reply)
    st.session_state.dash_messages.append({"role": "assistant", "content": reply})
