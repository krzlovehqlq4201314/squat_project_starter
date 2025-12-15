import streamlit as st

# 可切换的语言
LANGS = {"zh": "中文", "en": "English", "ko": "한국어"}

# 文案（先放一些必要键；找不到键时会原样返回 key）
TEXTS = {
    "home_title": {"zh": "首页", "en": "Home", "ko": "홈"},
    "home_tip": {
        "zh": "在左侧选择：上传图片 或 本地摄像头。",
        "en": "Choose on the left: Upload image or Local camera.",
        "ko": "왼쪽에서 업로드 또는 로컬 카메라를 선택하세요.",
    },
    # 你可以按需继续往这里补 key
}

def _ensure_lang():
    if "lang" not in st.session_state:
        st.session_state["lang"] = "zh"
    return st.session_state["lang"]

def t(key: str, **kwargs) -> str:
    """翻译函数：t('home_title')。没有就回退到 key 本身。"""
    lang = _ensure_lang()
    s = TEXTS.get(key, {})
    if isinstance(s, dict):
        s = s.get(lang) or s.get("en") or key
    else:
        s = str(s)
    try:
        return s.format(**kwargs) if kwargs else s
    except Exception:
        return s

def language_selector(location: str = "sidebar", key: str = "lang_radio"):
    """渲染语言切换组件；不会在 import 时执行任何 UI。"""
    _ensure_lang()
    container = st.sidebar if location == "sidebar" else st
    opts = list(LANGS.keys())
    idx = opts.index(st.session_state["lang"]) if st.session_state["lang"] in opts else 0
    choice = container.radio("语言 / Language", opts, index=idx,
                             format_func=lambda k: LANGS[k], key=key)
    st.session_state["lang"] = choice
    return t  # 方便随手赋值：_ = language_selector(...)
