import streamlit as st
import pandas as pd
import plotly.express as px

def page2():
    # Load dataset
    df = pd.read_csv('GA/MalaysiaExports.csv')

    # Header and styles
    st.markdown("<h1 style='text-align: center; color: black;'>Future Horizons: Predicting Malaysia's Export Trends with ARIMA</h1>", unsafe_allow_html=True)
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:wght@400;700&display=swap');
    body {
        font-family: 'EB Garamond', serif;
        font-size: 20px;     
    }
    h2 {
        font-family: 'EB Garamond', serif;
        font-size: 26px;
    }
    h3 {
        font-family: 'EB Garamond', serif;
        font-size: 22px;
    }
    div.st-emotion-cache-ocqkz7.e1f1d6gn5 {
        background-color: #EEEEEE;
        border: 1px solid #DCDCDC;
        padding: 20px 20px 20px 70px;
        padding: 5px 5px 5px 10px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Dataset information
    st.subheader("Dataset Information:")
    st.write("Data sources: [ekonomi.gov.my](https://www.ekonomi.gov.my/ms/statistik-sosioekonomi/statistik-ekonomi/dagangan-luar)")
    st.write("This dataset contains the following columns:")
    st.write(df.columns.tolist())

    # Quick Statistics
    st.subheader("Quick Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    # Data Preview
    st.subheader("Data Preview")
    st.write(df.head())

    if st.checkbox("Show full dataset"):
        st.write(df)

    # Data exploration
    st.subheader("Explore the Data")
    columns_to_display = st.multiselect("Select columns to display", df.columns.tolist(), default=df.columns.tolist())
    st.dataframe(df[columns_to_display].head())

    # Distribution of Export Categories
    st.subheader("Overview: Distribution of Export Categories")
    year = st.selectbox("Select Year", df['Year'].unique())

    exclude_columns = ['Year', 'Total Exports']
    data_for_year = df[df['Year'] == year].drop(columns=exclude_columns, errors='ignore').iloc[0]

    fig = px.pie(
        data_for_year,
        names=data_for_year.index,
        values=data_for_year.values,
        title=f"Export Distribution for Year {year}",
        labels={name: name for name in data_for_year.index}
    )

    fig.update_traces(textinfo='none', hovertemplate='%{label}: %{percent:.1%}')
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.5,
            xanchor="center",
            x=0.5
        )
    )

    st.plotly_chart(fig)

# Run the app
if __name__ == "__main__":
    page2()
