import csv
import io
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components
from cltk import NLP
from dotenv import load_dotenv
from openai import OpenAI

DB_PATH = os.path.join("data", "lsj.db")
GREEK_WORD_RE = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]+")
LSJ_SNIPPET_LEN = 600
MODEL_NAME = "gpt-4o"

CREDIT_TEXT = (
    "using LSJ Data provided by Perseus Tufts and edited by Helma Dik/Logeion.. "
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
def get_cltk_nlp() -> NLP:
    return NLP(language_code="grc")


@st.cache_resource
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def extract_tokens(text: str) -> List[str]:
    return GREEK_WORD_RE.findall(text)


def split_targets(text: str) -> List[str]:
    if not text.strip():
        return []
    return [t for t in re.split(r"\s+", text.strip()) if t]


CLAUSE_BOUNDARY_TOKENS = {
    "ὅτι",
    "ἵνα",
    "ὡς",
    "ἐπεί",
    "ὅτε",
    "εἰ",
    "ὅς",
    "ἥ",
    "ὅ",
}

CLAUSE_SPLIT_PUNCT = re.compile(r"[,;·;]")


def split_clauses(text: str) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return []

    parts = [p.strip() for p in CLAUSE_SPLIT_PUNCT.split(cleaned) if p.strip()]
    clauses: List[str] = []
    for part in parts:
        tokens = part.split(" ")
        start = 0
        for idx, tok in enumerate(tokens):
            if idx == 0:
                continue
            if tok in CLAUSE_BOUNDARY_TOKENS:
                chunk = " ".join(tokens[start:idx]).strip()
                if chunk:
                    clauses.append(chunk)
                start = idx
        tail = " ".join(tokens[start:]).strip()
        if tail:
            clauses.append(tail)
    return [c for c in clauses if c]

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


def rows_to_csv_bytes(rows: List[Dict[str, str]], headers: List[str]) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=headers,
        extrasaction="ignore",
        quoting=csv.QUOTE_ALL,
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue().encode("utf-8-sig")


def render_copy_button(text: str, label: str = "Copy") -> None:
    if not text:
        return
    safe = json.dumps(text)
    html = f"""
    <button id="copy-btn" style="padding:6px 12px; border:1px solid #ccc; background:#f7f7f7; cursor:pointer;">
      {label}
    </button>
    <script>
      const btn = document.getElementById("copy-btn");
      const text = {safe};
      btn.addEventListener("click", async () => {{
        try {{
          await navigator.clipboard.writeText(text);
          btn.textContent = "Copied";
          setTimeout(() => btn.textContent = "{label}", 1200);
        }} catch (e) {{
          btn.textContent = "Failed";
          setTimeout(() => btn.textContent = "{label}", 1200);
        }}
      }});
    </script>
    """
    components.html(html, height=40)


def rows_to_csv(rows: List[Dict[str, str]], headers: List[str]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()

def openai_json_response(client: OpenAI, model: str, system: str, user: str, schema: dict) -> dict:
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "response",
                "schema": schema,
                "strict": True,
            }
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

    nlp = get_cltk_nlp()
    doc = nlp.analyze(" ".join(uniq))

    lemmas_by_surface: Dict[str, List[str]] = {t: [] for t in uniq}
    for word in getattr(doc, "words", []) or []:
        surface = (
            getattr(word, "string", None)
            or getattr(word, "form", None)
            or getattr(word, "text", None)
        )
        if not surface or surface not in lemmas_by_surface:
            continue
        candidates: List[str] = []
        lemma = getattr(word, "lemma", None)
        if lemma:
            candidates.append(str(lemma))
        lemmas = getattr(word, "lemmas", None)
        if lemmas:
            for lval in lemmas:
                if lval:
                    candidates.append(str(lval))
        seen = set()
        for cand in candidates:
            if cand not in seen:
                lemmas_by_surface[surface].append(cand)
                seen.add(cand)

    all_lemmas = [
        lemma
        for lemma_list in lemmas_by_surface.values()
        for lemma in lemma_list
        if lemma
    ]
    conn = get_db()
    beta_by_lemma = beta_lookup_by_headword(conn, all_lemmas)

    items: List[LemmaItem] = []
    for surface in uniq:
        lemmas = lemmas_by_surface.get(surface) or []
        if not lemmas:
            items.append(LemmaItem(surface=surface, lemma="未解析", beta_code=""))
            continue
        for lemma in lemmas:
            items.append(
                LemmaItem(
                    surface=surface,
                    lemma=lemma,
                    beta_code=beta_by_lemma.get(lemma, ""),
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


def beta_lookup_by_headword(
    conn: sqlite3.Connection, headwords: List[str]
) -> Dict[str, str]:
    headwords = [h for h in headwords if h]
    if not headwords:
        return {}

    uniq = list(dict.fromkeys(headwords))
    placeholders = ",".join(["?"] * len(uniq))
    query = (
        f"SELECT headword, beta_code FROM dictionary WHERE headword IN ({placeholders})"
    )
    cur = conn.cursor()
    cur.execute(query, uniq)
    rows = cur.fetchall()

    out: Dict[str, str] = {}
    for row in rows:
        headword = row["headword"]
        if headword not in out:
            out[headword] = row["beta_code"]
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
            "Use the LSJ entries as preferred lexical guidance. "
            "Reflect tense and aspect naturally in the translation. "
            "Present participles should convey ongoing or continuous action. "
            "Aorist should convey completed action. "
            "Perfect should convey a resulting state. "
            "Imperfect should convey ongoing or repeated past action."
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
        segment_system = (
            "You are given candidate clause segments. "
            "Do not merge or reorder candidates and do not create text outside candidates. "
            "Return JSON with key: clauses (array). "
            "Each clause item must have: text."
        )
        segment_schema = {
            "type": "object",
            "properties": {
                "clauses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["clauses"],
            "additionalProperties": False,
        }

        analyze_system = (
            "You analyze Greek syntax (SVO and grammar). "
            "Analyze only the provided clause and do not split it further. "
            "Return JSON with keys: id, text, structures, notes. "
            "Each structure item should have: subject, verb, object, extras. "
            "The object must be the direct object of the verb; do not include indirect objects. "
            "Make the notes detailed and explain grammar thoroughly. "
            "Include mood/tense/voice, case usage, clause type, and key particles where relevant. "
            "Use clause types such as: 主節, 従属節, 関係節, 条件節, 目的節, 時間節, "
            "分詞節, 独立絶対属格. "
            "The id must be a clause type plus number (e.g., 従属節 1, 主節 1)."
        )
        analyze_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "text": {"type": "string"},
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
            "required": ["id", "text", "structures", "notes"],
            "additionalProperties": False,
        }

        candidates = split_clauses(text) or [text]
        candidate_lines = "\n".join(
            [f"{i}. {c}" for i, c in enumerate(candidates, start=1)]
        )
        segment_user = (
            f"Original text:\n{text}\n\n"
            f"Candidate segments (in order):\n{candidate_lines}\n\n"
            f"Rules: Use only these candidates as-is. "
            f"Do not merge, do not reorder, do not omit content."
        )
        segment_data = openai_json_response(
            client, model, segment_system, segment_user, segment_schema
        )
        clause_texts = [c.get("text", "") for c in segment_data.get("clauses", []) if c]
        if not clause_texts:
            clause_texts = [text]

        results = []
        for idx, clause_text in enumerate(clause_texts, start=1):
            analyze_user = (
                f"Analyze syntax for the following Greek clause. Explain in {target_lang}.\n\n"
                f"Clause:\n{clause_text}\n\n"
                f"If specific target words are provided, focus only on them.\n"
                f"Targets: {', '.join(targets) if targets else '(none)'}\n\n"
                f"LSJ context:\n{lsj_context}\n\n"
                f"Use clause number {idx} in the id."
            )
            data = openai_json_response(
                client, model, analyze_system, analyze_user, analyze_schema
            )
            results.append(data)

        return json.dumps({"clauses": results}, ensure_ascii=False)

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
                st.session_state["translate_result"] = result

        translate_result = st.session_state.get("translate_result")
        if translate_result:
            st.markdown(translate_result)
            render_copy_button(translate_result, label="Copy Translation")

    with tab_morph:
        if st.button("Analyze Morphology", use_container_width=True):
            if not text.strip():
                st.error("Please enter Greek text.")
            else:
                with st.spinner("Analyzing morphology..."):
                    result = analyze_text(
                        client, model, "morphology", text, targets, target_lang
                    )
                st.session_state["morph_result"] = result

        morph_result = st.session_state.get("morph_result")
        if morph_result:
            try:
                data = json.loads(morph_result)
                rows = data.get("rows", [])
                st.table(rows)
                if rows:
                    headers = [
                        "surface",
                        "lemma",
                        "beta_code",
                        "part_of_speech",
                        "inflection",
                        "gloss",
                    ]
                    csv_bytes = rows_to_csv_bytes(rows, headers)
                    st.download_button(
                        "Download CSV",
                        csv_bytes,
                        file_name="morphology.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            except Exception:
                st.markdown(morph_result)

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
                    clauses = data.get("clauses")
                    if isinstance(clauses, list):
                        for idx, clause in enumerate(clauses):
                            clause_text = clause.get("text", "")
                            if clause_text:
                                st.markdown(clause_text)
                            st.table(clause.get("structures", []))
                            notes = clause.get("notes", "")
                            if notes:
                                st.markdown(notes)
                            if idx != len(clauses) - 1:
                                st.divider()
                    else:
                        st.table(data.get("structures", []))
                        st.markdown(data.get("notes", ""))
                except Exception:
                    st.markdown(result)

    render_footer()


if __name__ == "__main__":
    main()




