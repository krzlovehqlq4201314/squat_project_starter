import streamlit as st

# 翻译表（示例，可按需再增补）
TRANSLATIONS = {
    "zh": {
        "app_name": "Squat Checker",
        "lang_label": "语言 / Language",
        "home_title": "首页 / Home",
        "upload_title": "上传图片 评估",
        "webcam_title": "本地摄像头 稳定版",
    },
    "en": {
        "app_name": "Squat Checker",
        "lang_label": "Language",
        "home_title": "Home",
        "upload_title": "Upload & Evaluate",
        "webcam_title": "Webcam (Stable)",
    },
    "ko": {
        "app_name": "스쿼트 체커",
        "lang_label": "언어",
        "home_title": "홈",
        "upload_title": "이미지 업로드 · 평가",
        "webcam_title": "웹캠 (안정판)",
    },
}

LANG_OPTIONS = [("zh", "中文"), ("en", "English"), ("ko", "한국어")]

def t(key: str, **kwargs) -> str:
    """按会话语言取文案；找不到就回退 key。"""
    lang = st.session_state.get("lang", "zh")
    text = TRANSLATIONS.get(lang, {}).get(key, key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except Exception:
            pass
    return text

def language_selector(location: str = "sidebar", key: str = "lang_radio_global") -> str:
    """渲染语言选择器（需要在 set_page_config 之后调用；不会在 import 时自动执行）"""
    container = st.sidebar if location == "sidebar" else st
    if "lang" not in st.session_state:
        st.session_state["lang"] = "zh"

    codes = [c for c, _ in LANG_OPTIONS]
    labels = {c: n for c, n in LANG_OPTIONS}
    idx = codes.index(st.session_state["lang"]) if st.session_state["lang"] in codes else 0

    choice = container.radio(
        t("lang_label"),
        codes,
        index=idx,
        key=key,
        format_func=lambda c: labels.get(c, c),
    )
    st.session_state["lang"] = choice
    return choice
