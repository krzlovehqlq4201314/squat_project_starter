from __future__ import annotations
import streamlit as st

TRANSLATIONS = {
    "zh": {
        "app_name": "Squat Checker",
        "lang_label": "语言 / Language",
        "home_title": "首页 / Home",
        "home_tip": "左侧切换语言；选择上方页面进入“上传图片 评估”或“本地摄像头·稳定版”。",
        "upload_title": "上传图片·评估",
        "upload_tip": "在此上传或替换一张 JPG/PNG。",
        "upload_prompt": "点击上传或拖拽一张 JPG/PNG",
        "upload_ok": "图片已载入。",
        "webcam_title": "本地摄像头·稳定版",
        "webcam_tip": "此页保留摄像头入口说明。若需采集/识别，请先安装 OpenCV：pip install opencv-python。",
    },
    "en": {
        "app_name": "Squat Checker",
        "lang_label": "Language",
        "home_title": "Home",
        "home_tip": "Use the sidebar to switch language; open 'Upload & Evaluate' or 'Webcam (stable)'.",
        "upload_title": "Upload & Evaluate",
        "upload_tip": "Upload a JPG/PNG image here.",
        "upload_prompt": "Drop or select a JPG/PNG",
        "upload_ok": "Image loaded.",
        "webcam_title": "Webcam · Stable",
        "webcam_tip": "Install OpenCV first: pip install opencv-python.",
    },
    "ko": {
        "app_name": "Squat Checker",
        "lang_label": "언어 / Language",
        "home_title": "홈",
        "home_tip": "사이드바에서 언어를 변경하고 상단 페이지로 이동하세요.",
        "upload_title": "이미지 업로드 · 평가",
        "upload_tip": "JPG/PNG 이미지를 업로드하세요.",
        "upload_prompt": "JPG/PNG 파일을 선택 또는 드래그",
        "upload_ok": "이미지가 로드되었습니다.",
        "webcam_title": "로컬 웹캠 · 안정판",
        "webcam_tip": "OpenCV 설치 후 사용: pip install opencv-python.",
    },
}

def _get_lang() -> str:
    return st.session_state.get("lang", "zh")

def t(key: str, lang: str | None = None) -> str:
    lang = lang or _get_lang()
    return TRANSLATIONS.get(lang, TRANSLATIONS["zh"]).get(key, key)

def language_selector(location: str = "sidebar", key: str = "lang_radio") -> str:
    container = st.sidebar if location == "sidebar" else st
    container.markdown(f"**{t('lang_label', 'zh')}** / **{t('lang_label', 'en')}**")
    names = {"zh": "中文", "en": "English", "ko": "한국어"}
    default = st.session_state.get("lang", "zh")
    choice_name = container.radio(
        label="",
        options=list(names.values()),
        index=["zh","en","ko"].index(default),
        key=key,
    )
    inv = {v:k for k,v in names.items()}
    st.session_state["lang"] = inv.get(choice_name, "zh")
    return st.session_state["lang"]
