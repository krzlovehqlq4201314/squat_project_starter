import streamlit as st
st.set_page_config(page_title="Squat Checker", layout="wide")

from app.i18n import t, language_selector

language_selector("sidebar", key="lang_home")

st.title(t("home_title"))
st.write(t("home_tip"))
