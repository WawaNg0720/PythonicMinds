import streamlit as st
from streamlit_option_menu import option_menu
import page1 as p1
import page2 as p2
import page3 as p3

# Set page configuration with title and icon
st.set_page_config(
    page_title="Malaysia Exports Analysis",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
    )

st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)

#set the background_image
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://thumbs.dreamstime.com/b/global-business-logistics-import-export-white-background-container-cargo-freight-ship-transport-concept-193223299.jpg");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;  
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)

def main():
    st.logo("PM.png")
    
    # Use streamlit-option-menu for navigation
    with st.sidebar:
        selection = option_menu(
            "Main Menu", ["Overview", "Dataset", "ARIMA Model"],
            icons=['house', 'table', 'graph-up-arrow'],
            menu_icon="cast", default_index=0
        )
    
    st.sidebar.image("m3.png")

    if selection == "Overview":
        p1.page1()
    elif selection == "Dataset":
        p2.page2()
    elif selection == "ARIMA Model":
        p3.page3()

if __name__ == "__main__":
    main()

#PythonicMinds
#WaiChuan123
#naiseman (Wei Xin)
#Tanyiqi0416
#WawaNg0720
#Tan294188
