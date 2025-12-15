import streamlit as st
st.set_page_config(page_title="Squat Checker", layout="wide")

from app.utils.i18n import t, language_selector  # 复用你已有的 i18n
language_selector(location="sidebar", key="lang_radio_global")

st.title(t("home_title") if callable(t) else "Home")
st.info(t("home_tip") if callable(t) else "选择左侧页面开始使用。")
