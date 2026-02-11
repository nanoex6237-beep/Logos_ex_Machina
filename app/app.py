import json
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

DB_PATH = os.path.join("data", "lsj.db")
GREEK_WORD_RE = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]+")
LSJ_SNIPPET_LEN = 600
MODEL_NAME = "gpt-4o"

CREDIT_TEXT = (
    "LSJLogeion data: please credit Perseus Tufts and Helma Dik/Logeion. "
    "Issues welcome."
)


@dataclass
class LemmaItem:
    surface: str
    lemma: str
    beta_code: str


def load_env() -> None:
    load_dotenv()


def get_api_key() -> Optional[str]:
    return os.getenv("OPENAI_API_KEY")


@st.cache_resource
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def extract_tokens(text: str) -> List[str]:
    return GREEK_WORD_RE.findall(text)


def split_targets(text: str) -> List[str]:
    if not text.strip():
        return []
    return [t for t in re.split(r"\s+", text.strip()) if t]


def response_text(response) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text
    # Fallback: try to extract from output items
    parts = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "message":
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) == "output_text":
                    parts.append(getattr(content, "text", ""))
    return "\n".join(parts).strip()


def openai_json_response(client: OpenAI, model: str, system: str, user: str, schema: dict) -> dict:
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": schema,
                "strict": True,
            },
        },
    )
    text = response_text(response)
    return json.loads(text)


def lemmatize_and_betacode(
    client: OpenAI, model: str, tokens: List[str]
) -> List[LemmaItem]:
    if not tokens:
        return []

    uniq = list(dict.fromkeys(tokens))
    if len(uniq) > 50:
        uniq = uniq[:50]

    system = (
        "You are a classical Greek linguistics assistant. "
        "Given Greek word forms, return the lemma (dictionary headword) "
        "and the corresponding Beta Code for that lemma. "
        "Use standard Perseus/LSJ Beta Code conventions."
    )
    user = (
        "Return lemma and beta_code for each surface form. "
        "If unsure, provide your best guess.\n\n"
        f"Surface forms: {', '.join(uniq)}"
    )

    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "surface": {"type": "string"},
                        "lemma": {"type": "string"},
                        "beta_code": {"type": "string"},
                    },
                    "required": ["surface", "lemma", "beta_code"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }

    try:
        data = openai_json_response(client, model, system, user, schema)
    except Exception:
        return []

    items = []
    for item in data.get("items", []):
        items.append(
            LemmaItem(
                surface=item.get("surface", ""),
                lemma=item.get("lemma", ""),
                beta_code=item.get("beta_code", ""),
            )
        )
    return items


def beta_lookup(conn: sqlite3.Connection, betacodes: List[str]) -> Dict[str, Dict[str, str]]:
    betacodes = [b for b in betacodes if b]
    if not betacodes:
        return {}

    uniq = list(dict.fromkeys(betacodes))
    placeholders = ",".join(["?"] * len(uniq))
    query = (
        f"SELECT headword, beta_code, substr(full_content, 1, {LSJ_SNIPPET_LEN}) AS snippet "
        f"FROM dictionary WHERE beta_code IN ({placeholders})"
    )
    cur = conn.cursor()
    cur.execute(query, uniq)
    rows = cur.fetchall()

    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        out[row["beta_code"]] = {
            "headword": row["headword"],
            "beta_code": row["beta_code"],
            "snippet": row["snippet"],
        }
    return out


def build_lsj_context(
    client: OpenAI,
    model: str,
    text: str,
    targets: List[str],
) -> Tuple[List[LemmaItem], Dict[str, Dict[str, str]]]:
    tokens = targets if targets else extract_tokens(text)
    lemma_items = lemmatize_and_betacode(client, model, tokens)
    betacodes = [li.beta_code for li in lemma_items]
    conn = get_db()
    lsj = beta_lookup(conn, betacodes)
    return lemma_items, lsj


def format_lsj_context(lsj: Dict[str, Dict[str, str]]) -> str:
    if not lsj:
        return "(No LSJ entries found for the selected words.)"
    lines = ["LSJ entries (by beta_code):"]
    for beta, entry in lsj.items():
        lines.append(
            f"- {beta} :: {entry['headword']} :: {entry['snippet']}"
        )
    return "\n".join(lines)


def analyze_text(
    client: OpenAI,
    model: str,
    mode: str,
    text: str,
    targets: List[str],
    target_lang: str,
) -> str:
    lemma_items, lsj = build_lsj_context(client, model, text, targets)
    lsj_context = format_lsj_context(lsj)

    if mode == "translate":
        system = (
            "You are a classical Greek translator. "
            "Use the LSJ entries as preferred lexical guidance." 
        )
        user = (
            f"Translate the following Greek text into {target_lang}.\n\n"
            f"Text:\n{text}\n\n"
            f"LSJ context:\n{lsj_context}"
        )
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response_text(response)

    if mode == "morphology":
        system = (
            "You analyze Greek morphology. "
            "Return a table-like JSON array with columns: "
            "surface, lemma, beta_code, part_of_speech, inflection, gloss. "
            "Use LSJ context when helpful."
        )
        user = (
            f"Analyze the following Greek text. Target language for gloss: {target_lang}.\n\n"
            f"Text:\n{text}\n\n"
            f"If specific target words are provided, focus only on them.\n"
            f"Targets: {', '.join(targets) if targets else '(none)'}\n\n"
            f"LSJ context:\n{lsj_context}"
        )
        schema = {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "surface": {"type": "string"},
                            "lemma": {"type": "string"},
                            "beta_code": {"type": "string"},
                            "part_of_speech": {"type": "string"},
                            "inflection": {"type": "string"},
                            "gloss": {"type": "string"},
                        },
                        "required": [
                            "surface",
                            "lemma",
                            "beta_code",
                            "part_of_speech",
                            "inflection",
                            "gloss",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["rows"],
            "additionalProperties": False,
        }
        data = openai_json_response(client, model, system, user, schema)
        return json.dumps(data, ensure_ascii=False)

    if mode == "grammar":
        system = (
            "You analyze Greek syntax (SVO and grammar). "
            "Return JSON with keys: structures (array) and notes (string). "
            "Each structure item should have: subject, verb, object, extras."
        )
        user = (
            f"Analyze syntax for the following Greek text. Explain in {target_lang}.\n\n"
            f"Text:\n{text}\n\n"
            f"If specific target words are provided, focus only on them.\n"
            f"Targets: {', '.join(targets) if targets else '(none)'}\n\n"
            f"LSJ context:\n{lsj_context}"
        )
        schema = {
            "type": "object",
            "properties": {
                "structures": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "verb": {"type": "string"},
                            "object": {"type": "string"},
                            "extras": {"type": "string"},
                        },
                        "required": ["subject", "verb", "object", "extras"],
                        "additionalProperties": False,
                    },
                },
                "notes": {"type": "string"},
            },
            "required": ["structures", "notes"],
            "additionalProperties": False,
        }
        data = openai_json_response(client, model, system, user, schema)
        return json.dumps(data, ensure_ascii=False)

    return ""


def render_footer() -> None:
    st.markdown(
        f"<div style='margin-top:2rem; font-size:0.85em; color:#666;'>{CREDIT_TEXT}</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Logos ex Machina", layout="wide")
    st.title("Logos ex Machina")
    st.caption("Greek reading assistant (translation, morphology, syntax)")

    load_env()

    api_key = get_api_key()
    if not api_key:
        st.error("OPENAI_API_KEY is not set. Please add it to .env or your environment.")
        st.stop()

    model = MODEL_NAME
    st.caption(f"Model: {model}")

    text = st.text_area("Greek text", height=200, placeholder="Paste Greek text here")
    targets_input = st.text_area(
        "Target words (optional, space/newline separated)", height=80
    )
    targets = split_targets(targets_input)

    target_lang = st.selectbox("Translation / Explanation language", ["Japanese", "English"])

    client = OpenAI()

    tab_translate, tab_morph, tab_grammar = st.tabs(["翻訳", "活用", "文法"])

    with tab_translate:
        if st.button("Translate", use_container_width=True):
            if not text.strip():
                st.error("Please enter Greek text.")
            else:
                with st.spinner("Translating..."):
                    result = analyze_text(
                        client, model, "translate", text, targets, target_lang
                    )
                st.markdown(result)

    with tab_morph:
        if st.button("Analyze Morphology", use_container_width=True):
            if not text.strip():
                st.error("Please enter Greek text.")
            else:
                with st.spinner("Analyzing morphology..."):
                    result = analyze_text(
                        client, model, "morphology", text, targets, target_lang
                    )
                try:
                    data = json.loads(result)
                    st.table(data.get("rows", []))
                except Exception:
                    st.markdown(result)

    with tab_grammar:
        if st.button("Analyze Grammar", use_container_width=True):
            if not text.strip():
                st.error("Please enter Greek text.")
            else:
                with st.spinner("Analyzing grammar..."):
                    result = analyze_text(
                        client, model, "grammar", text, targets, target_lang
                    )
                try:
                    data = json.loads(result)
                    st.table(data.get("structures", []))
                    st.markdown(data.get("notes", ""))
                except Exception:
                    st.markdown(result)

    render_footer()


if __name__ == "__main__":
    main()
