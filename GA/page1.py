import streamlit as st

def page1():
    # Header outside the bordered div
    st.markdown("<h1 style='text-align: center; color: black;'>Future Horizons: Predicting Malaysia's Export Trends with ARIMA</h1>", unsafe_allow_html=True)
    
    # Styled markdown content
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:wght@400;700&display=swap');
    .custom-body {
        font-family: 'EB Garamond', serif;
        font-size: 20px;
        color: black;  
    }
    .custom-h2 {
        font-family: 'EB Garamond', serif;
        font-size: 26px;
        color: black;  
    }
    .custom-h3 {
        font-family: 'EB Garamond', serif;
        font-size: 22px;
        color: black;  
    }
    .custom-div {
        background-color: #EEEEEE;
        border: 1px solid #DCDCDC;
        padding: 20px;
        border-radius: 10px;
    }
    </style>

    <div class="custom-div">
    <div class="custom-body">
    <h2 class="custom-h2">Overview of Malaysia Exports</h2>

    Malaysia is one of the world's largest exporters, with a diverse range of products being shipped across the globe. The country has a well-developed export sector that plays a significant role in its economy. Key export commodities include:

    - <b>Electronics and Electrical Products</b>: Malaysia is a major hub for the production and export of semiconductors, integrated circuits, and other electronic components.
    - <b>Palm Oil and Palm-Based Products</b>: As one of the world's leading producers of palm oil, Malaysia exports a significant amount of palm oil and related products.
    - <b>Petroleum and Petrochemical Products</b>: The country has vast reserves of oil and natural gas, contributing to its exports in the form of crude oil, LNG, and various petrochemical products.
    - <b>Rubber Products</b>: Malaysia is known for its high-quality rubber and rubber-based products, including gloves, tires, and other industrial products.
    - <b>Chemical Products</b>: A range of chemicals and chemical products are exported from Malaysia, including industrial chemicals and pharmaceuticals.
    - <b>Manufactured Goods</b>: This includes machinery, equipment, and other industrial goods that are in high demand globally.
    - <b>Agricultural Products</b>: Apart from palm oil, Malaysia exports other agricultural products such as cocoa, pepper, and tropical fruits.

    <h3 class="custom-h3">Importance of Exports to Malaysia's Economy</h3>

    The export sector is a critical component of Malaysia's economy, contributing significantly to GDP growth, employment, and foreign exchange earnings. The country's strategic location, well-developed infrastructure, and favorable trade policies have made it an attractive destination for international trade.

    <h3 class="custom-h3">Analyzing and Forecasting Export Data</h3>

    Understanding export trends and forecasting future exports is vital for businesses, policymakers, and researchers. By analyzing historical export data and using forecasting models like ARIMA, user can gain valuable insights into future export performance, helping stakeholders make informed decisions.

    In this application, it provides tools to analyze and forecast Malaysia's export data, allowing users to explore trends and make predictions based on historical data.
    </div>
    </div>
    """, unsafe_allow_html=True)


