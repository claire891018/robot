import streamlit as st

st.set_page_config(
    page_title="Robot Demos",
    page_icon="https://api.dicebear.com/9.x/thumbs/svg?seed=Chase",
    layout="wide",
)
def render_header():
    icon = "https://api.dicebear.com/9.x/thumbs/svg?seed=Chase"
    st.markdown(
        f'''
        <h2 style="display:flex;align-items:center;gap:.5rem;">
        <img src="{icon}" width="28" height="28"
            style="border-radius:20%; display:block;" />
        Robot Demos
        </h2>
        ''',
        unsafe_allow_html=True,
    )
    
    st.markdown("""
    ## ** ⬅︎ 選擇左側分頁開始 **
    """)

if __name__ == "__main__":
    render_header()