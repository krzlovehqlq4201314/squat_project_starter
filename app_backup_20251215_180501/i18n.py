from __future__ import annotations
import streamlit as st

LANGS = {"zh": "中文", "en": "English", "ko": "한국어"}

STRINGS = {
    "zh": {
        "lang_title": "语言 / Language",
        "home_title": "首页 / Home",
        "home_tip": "在左侧选择：上传图片 评估 或 本地摄像头 稳定版。",
        "menu_home": "Home",
        "menu_upload": "上传图片 评估",
        "menu_webcam": "本地摄像头 稳定版",
        "upload_title": "上传图片评估",
        "webcam_title": "本地摄像头 - Squat Checker",
    },
    "en": {
        "lang_title": "Language",
        "home_title": "Home",
        "home_tip": "Choose a page on the left: Upload or Webcam.",
        "menu_home": "Home",
        "menu_upload": "Upload",
        "menu_webcam": "Webcam (stable)",
        "upload_title": "Upload & Evaluate",
        "webcam_title": "Webcam - Squat Checker",
    },
    "ko": {
        "lang_title": "언어",
        "home_title": "홈",
        "home_tip": "왼쪽에서 페이지를 선택하세요: 업로드 또는 웹캠.",
        "menu_home": "Home",
        "menu_upload": "이미지 업로드",
        "menu_webcam": "로컬 웹캠(안정판)",
        "upload_title": "이미지 업로드/평가",
        "webcam_title": "로컬 웹캠 - Squat Checker",
    },
}

def _fmt_lang(code: str) -> str:
    return LANGS.get(code, code)

def get_lang(default: str = "zh") -> str:
    return st.session_state.get("lang", default)

def set_lang(code: str) -> None:
    st.session_state["lang"] = code

def t(key: str, **kwargs) -> str:
    lang = get_lang()
    text = STRINGS.get(lang, {}).get(key, key)
    try:
        return text.format(**kwargs)
    except Exception:
        return text

def language_selector(location: str = "sidebar", *, key: str = "lang_radio_global") -> str:
    container = st.sidebar if location == "sidebar" else st
    opts = list(LANGS.keys())
    cur = get_lang()
    try:
        idx = opts.index(cur)
    except ValueError:
        idx = 0
    choice = container.radio(
        t("lang_title"),
        options=opts,
        index=idx,
        format_func=_fmt_lang,
        key=key,
    )
    set_lang(choice)
    return choice
