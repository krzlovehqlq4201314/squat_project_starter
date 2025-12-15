import streamlit as st

# 三语文案（先给常用项，后续你按 key 增加即可）
TRANSLATIONS = {
    "zh": {
        "lang_label": "语言 / Language",
        "home_title": "首页 / Home",
        "home_tip": "在左侧选择：上传图片 评估 或 本地摄像头。",
        "nav_home": "首页 / Home",
        "nav_upload": "上传图片 评估",
        "nav_cam": "本地摄像头 稳定版",
    },
    "en": {
        "lang_label": "Language",
        "home_title": "Home",
        "home_tip": "Choose a page on the left: Upload/Evaluate or Webcam.",
        "nav_home": "Home",
        "nav_upload": "Upload & Evaluate",
        "nav_cam": "Webcam (stable)",
    },
    "ko": {
        "lang_label": "언어",
        "home_title": "홈",
        "home_tip": "왼쪽에서 페이지를 선택하세요: 이미지 업로드/평가 또는 로컬 웹캠.",
        "nav_home": "홈",
        "nav_upload": "이미지 업로드 · 평가",
        "nav_cam": "로컬 웹캠(안정판)",
    },
}

LABELS = {"zh": "中文", "en": "English", "ko": "한국어"}

def t(trans: dict, key: str, **kwargs) -> str:
    s = trans.get(key, key)
    if isinstance(s, str):
        try:
            return s.format(**kwargs)
        except Exception:
            return s
    return key

def select_language(container=None, key: str = "lang_radio_global"):
    """
    只在 app/app.py 调一次！不要在各页面重复创建。
    """
    container = container or st.sidebar
    current = st.session_state.get("lang", "zh")
    options = list(TRANSLATIONS.keys())
    idx = options.index(current) if current in options else 0

    lang = container.radio(
        TRANSLATIONS[current]["lang_label"] if current in TRANSLATIONS else "Language",
        options=options,
        index=idx,
        format_func=lambda k: LABELS.get(k, k),
        key=key,
    )
    st.session_state["lang"] = lang
    trans = TRANSLATIONS[lang]
    st.session_state["i18n"] = trans
    return lang, trans
