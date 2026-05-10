import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import deepl
import streamlit as st
from streamlit_javascript import st_javascript

# ── Language data ────────────────────────────────────────────────────────────

SOURCE_LANGUAGES = [
    {"code": None,    "name": "Auto-detect"},
    {"code": "AR",    "name": "Arabic"},
    {"code": "BG",    "name": "Bulgarian"},
    {"code": "CS",    "name": "Czech"},
    {"code": "DA",    "name": "Danish"},
    {"code": "DE",    "name": "German"},
    {"code": "EL",    "name": "Greek"},
    {"code": "EN",    "name": "English"},
    {"code": "ES",    "name": "Spanish"},
    {"code": "ET",    "name": "Estonian"},
    {"code": "FI",    "name": "Finnish"},
    {"code": "FR",    "name": "French"},
    {"code": "HU",    "name": "Hungarian"},
    {"code": "ID",    "name": "Indonesian"},
    {"code": "IT",    "name": "Italian"},
    {"code": "JA",    "name": "Japanese"},
    {"code": "KO",    "name": "Korean"},
    {"code": "LT",    "name": "Lithuanian"},
    {"code": "LV",    "name": "Latvian"},
    {"code": "NB",    "name": "Norwegian"},
    {"code": "NL",    "name": "Dutch"},
    {"code": "PL",    "name": "Polish"},
    {"code": "PT",    "name": "Portuguese"},
    {"code": "RO",    "name": "Romanian"},
    {"code": "RU",    "name": "Russian"},
    {"code": "SK",    "name": "Slovak"},
    {"code": "SL",    "name": "Slovenian"},
    {"code": "SV",    "name": "Swedish"},
    {"code": "TR",    "name": "Turkish"},
    {"code": "UK",    "name": "Ukrainian"},
    {"code": "ZH",    "name": "Chinese"},
]

TARGET_LANGUAGES = [
    {"code": "AR",    "name": "Arabic"},
    {"code": "BG",    "name": "Bulgarian"},
    {"code": "CS",    "name": "Czech"},
    {"code": "DA",    "name": "Danish"},
    {"code": "DE",    "name": "German"},
    {"code": "EL",    "name": "Greek"},
    {"code": "EN-GB", "name": "English (UK)"},
    {"code": "EN-US", "name": "English (US)"},
    {"code": "ES",    "name": "Spanish"},
    {"code": "ET",    "name": "Estonian"},
    {"code": "FI",    "name": "Finnish"},
    {"code": "FR",    "name": "French"},
    {"code": "HU",    "name": "Hungarian"},
    {"code": "ID",    "name": "Indonesian"},
    {"code": "IT",    "name": "Italian"},
    {"code": "JA",    "name": "Japanese"},
    {"code": "KO",    "name": "Korean"},
    {"code": "LT",    "name": "Lithuanian"},
    {"code": "LV",    "name": "Latvian"},
    {"code": "NB",    "name": "Norwegian"},
    {"code": "NL",    "name": "Dutch"},
    {"code": "PL",    "name": "Polish"},
    {"code": "PT-BR", "name": "Portuguese (BR)"},
    {"code": "PT-PT", "name": "Portuguese (PT)"},
    {"code": "RO",    "name": "Romanian"},
    {"code": "RU",    "name": "Russian"},
    {"code": "SK",    "name": "Slovak"},
    {"code": "SL",    "name": "Slovenian"},
    {"code": "SV",    "name": "Swedish"},
    {"code": "TR",    "name": "Turkish"},
    {"code": "UK",    "name": "Ukrainian"},
    {"code": "ZH",    "name": "Chinese (Simplified)"},
]

# ── Translation logic ────────────────────────────────────────────────────────

def flatten_json(obj, prefix=""):
    result = {}
    for key, val in obj.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict):
            result.update(flatten_json(val, full_key))
        else:
            result[full_key] = val
    return result


def unflatten_json(flat):
    result = {}
    for key, val in flat.items():
        parts = key.split(".")
        cur = result
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = val
    return result


def translate_all(json_text, source_lang, target_langs, api_key, concurrency=5, on_progress=None):
    source = json.loads(json_text)
    flat = flatten_json(source)
    string_keys = [k for k, v in flat.items() if isinstance(v, str)]

    tasks = [(lang, key, flat[key]) for lang in target_langs for key in string_keys]
    total = len(tasks)
    completed = 0

    results = {lang: {} for lang in target_langs}
    translator = deepl.Translator(api_key)

    def do_translate(lang, key, text):
        attempt = 0
        while True:
            try:
                result = translator.translate_text(
                    text,
                    source_lang=source_lang or None,
                    target_lang=lang,
                )
                return lang, key, result.text
            except Exception:
                attempt += 1
                delay = min(1 * 2 ** attempt, 30)
                if on_progress:
                    on_progress(completed, total, key, retrying=True)
                time.sleep(delay)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(do_translate, lang, key, text): (lang, key) for lang, key, text in tasks}
        for future in as_completed(futures):
            lang, key, translated = future.result()
            results[lang][key] = translated
            completed += 1
            if on_progress:
                on_progress(completed, total, key, retrying=False)

    output = {}
    for lang in target_langs:
        merged = dict(flat)
        merged.update(results[lang])
        output[lang] = unflatten_json(merged)

    return output

# ── Streamlit app ────────────────────────────────────────────────────────────

st.set_page_config(page_title="JSON Translator", page_icon="🌐", layout="wide")


def ls_get(key: str, default: str = "") -> str:
    val = st_javascript(f"localStorage.getItem('{key}') || ''")
    return val if isinstance(val, str) else default


def ls_set(key: str, value: str):
    escaped = value.replace("'", "\\'")
    st_javascript(f"localStorage.setItem('{key}', '{escaped}')")


if "settings_loaded" not in st.session_state:
    st.session_state.api_key = ls_get("jt_apiKey")
    raw_concurrency = ls_get("jt_concurrency", "5")
    try:
        st.session_state.concurrency = int(raw_concurrency)
    except ValueError:
        st.session_state.concurrency = 5
    st.session_state.settings_loaded = True

with st.sidebar:
    st.title("⚙️ Settings")
    st.caption("Nothing is stored on the server — your API key lives only in your browser's localStorage.")

    api_key_input = st.text_input(
        "DeepL API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="Enter your DeepL API key",
    )
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        ls_set("jt_apiKey", api_key_input)

    concurrency_input = st.slider(
        "Concurrency",
        min_value=1,
        max_value=20,
        value=st.session_state.concurrency,
        help="Number of parallel translation requests",
    )
    if concurrency_input != st.session_state.concurrency:
        st.session_state.concurrency = concurrency_input
        ls_set("jt_concurrency", str(concurrency_input))

st.title("🌐 JSON Translator")

col_input, col_langs = st.columns([1, 1])

with col_input:
    st.subheader("JSON Input")
    source_idx = st.selectbox(
        "Source language",
        range(len(SOURCE_LANGUAGES)),
        format_func=lambda i: SOURCE_LANGUAGES[i]["name"],
        index=0,
    )
    selected_source = SOURCE_LANGUAGES[source_idx]["code"]

    json_text = st.text_area(
        "Paste your JSON here",
        height=350,
        placeholder='{\n  "hello": "Hello",\n  "goodbye": "Goodbye"\n}',
    )

with col_langs:
    st.subheader("Target Languages")
    col_a, col_b = st.columns(2)
    select_all = col_a.button("Select All")
    clear_all = col_b.button("Clear All")

    if "target_langs" not in st.session_state:
        st.session_state.target_langs = set()

    if select_all:
        st.session_state.target_langs = {lang["code"] for lang in TARGET_LANGUAGES}
    if clear_all:
        st.session_state.target_langs = set()

    lang_cols = st.columns(3)
    for i, lang in enumerate(TARGET_LANGUAGES):
        checked = lang_cols[i % 3].checkbox(
            lang["name"],
            value=lang["code"] in st.session_state.target_langs,
            key=f"lang_{lang['code']}",
        )
        if checked:
            st.session_state.target_langs.add(lang["code"])
        else:
            st.session_state.target_langs.discard(lang["code"])

st.divider()

can_translate = (
    bool(json_text.strip())
    and bool(st.session_state.api_key)
    and bool(st.session_state.target_langs)
)

if not st.session_state.api_key:
    st.info("Enter your DeepL API key in the sidebar to get started.")

if st.button("🚀 Translate", disabled=not can_translate, type="primary"):
    try:
        json.loads(json_text)
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
        st.stop()

    target_langs = sorted(st.session_state.target_langs)
    progress_bar = st.progress(0, text="Starting translation…")
    status_text = st.empty()

    def on_progress(completed, total, current_key, retrying):
        pct = completed / total if total else 0
        msg = f"Translating `{current_key}`…"
        if retrying:
            msg += " (retrying)"
        progress_bar.progress(pct, text=msg)
        status_text.caption(f"{completed}/{total} keys translated")

    try:
        results = translate_all(
            json_text=json_text,
            source_lang=selected_source,
            target_langs=target_langs,
            api_key=st.session_state.api_key,
            concurrency=st.session_state.concurrency,
            on_progress=on_progress,
        )
        progress_bar.progress(1.0, text="Done!")
        status_text.success(f"Translated into {len(target_langs)} language(s).")
        st.session_state.results = results
    except Exception as e:
        progress_bar.empty()
        status_text.error(f"Translation failed: {e}")

if "results" in st.session_state and st.session_state.results:
    results = st.session_state.results
    st.subheader("Results")

    if len(results) > 1:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for lang, data in results.items():
                zf.writestr(f"{lang}.json", json.dumps(data, ensure_ascii=False, indent=2))
        st.download_button(
            "⬇️ Download all as ZIP",
            data=zip_buf.getvalue(),
            file_name="translations.zip",
            mime="application/zip",
        )

    for lang, data in results.items():
        with st.expander(f"{lang}"):
            json_str = json.dumps(data, ensure_ascii=False, indent=2)
            st.code(json_str, language="json")
            st.download_button(
                f"⬇️ Download {lang}.json",
                data=json_str.encode("utf-8"),
                file_name=f"{lang}.json",
                mime="application/json",
                key=f"dl_{lang}",
            )
