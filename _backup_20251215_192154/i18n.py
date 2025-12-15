import streamlit as st

TRANSLATIONS = {
    "zh": {
        "app_name": "Squat Checker",
        "lang_label": "语言 / Language",
        "home_title": "首页 / Home",
        "home_tip": "在左侧选择：上传图片 评估 或 本地摄像头 稳定版。",
        "upload_title": "上传图片 · 评估",
        "upload_tip": "在此上传一张 JPG/PNG，页面将展示热力图与关键线（示例）。",
        "upload_prompt": "拖拽或选择一张 JPG/PNG",
        "upload_ok": "图片上传成功",
        "webcam_title": "本地摄像头 · 稳定版",
        "webcam_tip": "此页面仅保留识别流程的占位入口，后续可接入你的 OpenCV/推理逻辑。",
    },
    "en": {
        "app_name": "Squat Checker",
        "lang_label": "Language",
        "home_title": "Home",
        "home_tip": "Choose in sidebar: Upload & Evaluate, or Local Webcam.",
        "upload_title": "Upload · Evaluate",
        "upload_tip": "Upload a JPG/PNG. The page will show heatmap & key lines (demo).",
        "upload_prompt": "Drag or select a JPG/PNG image",
        "upload_ok": "Image uploaded",
        "webcam_title": "Local Webcam · Stable",
        "webcam_tip": "Placeholder for webcam pipeline; plug in your OpenCV/inference later.",
    },
    "ko": {
        "app_name": "Squat Checker",
        "lang_label": "언어 / Language",
        "home_title": "홈",
        "home_tip": "왼쪽에서 업로드/평가 또는 로컬 웹캠을 선택하세요.",
        "upload_title": "이미지 업로드 · 평가",
        "upload_tip": "JPG/PNG 이미지를 업로드하면 열지도와 키라인(데모)을 표시합니다.",
        "upload_prompt": "JPG/PNG 이미지를 선택 또는 드래그",
        "upload_ok": "업로드 완료",
        "webcam_title": "로컬 웹캠 · 안정판",
        "webcam_tip": "웹캠 파이프라인용 자리 표시자입니다. 나중에 OpenCV/추론을 연결하세요.",
    },
}

LANG_CHOICES = [("zh", "中文"), ("en", "English"), ("ko", "한국어")]

def _get_lang() -> str:
    return st.session_state.get("lang", "zh")

def t(key: str) -> str:
    lang = _get_lang()
    return TRANSLATIONS.get(lang, {}).get(key, key)

def language_selector(location: str = "sidebar", key: str | None = None) -> str:
    """只渲染一个语言选择控件；不在导入阶段做任何 st.*。"""
    container = st.sidebar if location == "sidebar" else st
    labels = [label for _, label in LANG_CHOICES]
    code2label = dict(LANG_CHOICES)
    label2code = {v: k for k, v in LANG_CHOICES}

    current_code = _get_lang()
    try:
        idx = [c for c, _ in LANG_CHOICES].index(current_code)
    except ValueError:
        idx = 0

    choice = container.radio(t("lang_label"), labels, index=idx, key=key)
    st.session_state["lang"] = label2code.get(choice, "zh")
    return st.session_state["lang"]
