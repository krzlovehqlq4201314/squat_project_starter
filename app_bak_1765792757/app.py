import streamlit as st
st.set_page_config(page_title="Squat Checker", layout="wide")  # 必须第一行

from app.utils.i18n import t, language_selector

language_selector(location="sidebar", key="lang_radio_home")
st.title(t("home_title"))
st.caption(t("home_tip"))
