import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="Supply Chain Analytics", page_icon="🚚", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/supply_chain_data.csv"))

df = load_data()

st.title("🚚 Supply Chain Performance Analytics")
st.caption("Supplier performance, delivery reliability and stockout risk analysis")

col1, col2 = st.columns(2)
with col1:
    suppliers = st.multiselect("Supplier", df['supplier'].unique(), default=list(df['supplier'].unique()))
with col2:
    categories = st.multiselect("Category", df['category'].unique(), default=list(df['category'].unique()))

filtered = df[df['supplier'].isin(suppliers) & df['category'].isin(categories)]

st.divider()
k1, k2, k3, k4 = st.columns(4)
avg_delivery = filtered['delivery_rate'].mean()
avg_lead = filtered['lead_time_days'].mean()
total_stockouts = filtered['stockout_incident'].sum()
total_value = filtered['order_value'].sum()

k1.metric("Avg Delivery Rate", f"{avg_delivery:.1f}%", delta=f"{avg_delivery-92:.1f}% vs target")
k2.metric("Avg Lead Time", f"{avg_lead:.1f} days")
k3.metric("Stockout Incidents", f"{total_stockouts}")
k4.metric("Total Order Value", f"€{total_value:,.0f}")

st.divider()

col1, col2 = st.columns(2)
with col1:
    sup_perf = filtered.groupby('supplier').agg(
        delivery_rate=('delivery_rate', 'mean'),
        quality_score=('quality_score', 'mean'),
        stockouts=('stockout_incident', 'sum')
    ).reset_index().round(1)
    fig1 = px.bar(sup_perf, x='supplier', y='delivery_rate',
                  title='Delivery Rate by Supplier',
                  color='delivery_rate',
                  color_continuous_scale='RdYlGn',
                  range_color=[60, 100])
    fig1.add_hline(y=90, line_dash="dash", line_color="red", annotation_text="Target 90%")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    monthly = filtered.groupby(['month', 'supplier'])['delivery_rate'].mean().reset_index()
    fig2 = px.line(monthly, x='month', y='delivery_rate', color='supplier',
                   title='Delivery Rate Trend by Supplier', markers=True)
    fig2.add_hline(y=90, line_dash="dash", line_color="red", annotation_text="Target")
    fig2.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    lead_time = filtered.groupby('supplier')['lead_time_days'].mean().reset_index()
    fig3 = px.bar(lead_time, x='supplier', y='lead_time_days',
                  title='Average Lead Time by Supplier (days)',
                  color='lead_time_days',
                  color_continuous_scale='RdYlGn_r')
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    stockouts = filtered.groupby('supplier')['stockout_incident'].sum().reset_index()
    fig4 = px.bar(stockouts, x='supplier', y='stockout_incident',
                  title='Stockout Incidents by Supplier',
                  color='stockout_incident',
                  color_continuous_scale='RdYlGn_r')
    st.plotly_chart(fig4, use_container_width=True)

st.divider()
st.subheader("🤖 AI Supply Chain Analyst")
if st.button("🔍 Analyse Supplier Performance", type="primary"):
    sup_summary = filtered.groupby('supplier').agg(
        avg_delivery=('delivery_rate', 'mean'),
        avg_lead_time=('lead_time_days', 'mean'),
        total_stockouts=('stockout_incident', 'sum'),
        avg_quality=('quality_score', 'mean'),
        total_value=('order_value', 'sum')
    ).reset_index().round(2)

    monthly_issues = filtered[filtered['delivery_rate'] < 75].groupby(
        ['supplier', 'month'])['delivery_rate'].mean().reset_index()

    with st.spinner("Analysing supply chain data..."):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a senior supply chain analyst. Provide sharp, specific insights with clear recommendations. Use bullet points."},
                {"role": "user", "content": f"""Analyse this supplier performance data and give me the 5 most critical insights:

Supplier Summary:
{sup_summary.to_string()}

Critical Incidents (delivery rate below 75%):
{monthly_issues.to_string()}"""}
            ]
        ).choices[0].message.content
    st.markdown(response)

st.divider()
st.subheader("💬 Ask the supply chain data")
if "sc_messages" not in st.session_state:
    st.session_state.sc_messages = []

for message in st.session_state.sc_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("e.g. Which supplier is highest risk?"):
    st.session_state.sc_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    data_context = f"Supply chain data:\n{filtered.groupby('supplier').agg({'delivery_rate': 'mean', 'lead_time_days': 'mean', 'stockout_incident': 'sum', 'order_value': 'sum'}).round(2).to_string()}"

    with st.chat_message("assistant"):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": f"You are a senior supply chain analyst:\n{data_context}"},
                {"role": "user", "content": prompt}
            ]
        ).choices[0].message.content
        st.markdown(response)

    st.session_state.sc_messages.append({"role": "assistant", "content": response})
