import glob
import os
import sqlite3
import sys
from lxml import etree

DATA_GLOB = os.path.join("data", "LSJLogeion-master", "*.xml")
DB_PATH = os.path.join("data", "lsj.db")
BATCH_SIZE = 1000


def local_name(tag: str) -> str:
    if tag is None:
        return ""
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def clean_text(text: str) -> str:
    return " ".join(text.split())


def init_db(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE dictionary (
            id INTEGER PRIMARY KEY,
            headword TEXT,
            beta_code TEXT,
            full_content TEXT
        )
        """
    )
    cur.execute("CREATE INDEX idx_dictionary_headword ON dictionary(headword)")
    cur.execute("CREATE INDEX idx_dictionary_beta_code ON dictionary(beta_code)")
    conn.commit()


def process_file(conn: sqlite3.Connection, path: str, total_counter: int) -> int:
    cur = conn.cursor()
    batch = []

    print(f"Processing {os.path.basename(path)}...", flush=True)

    context = etree.iterparse(path, events=("end",), recover=True, huge_tree=True)
    for _event, elem in context:
        if local_name(elem.tag) != "div2":
            continue
        if elem.get("type") != "main":
            continue

        headword = ""
        for child in elem:
            if local_name(child.tag) == "head":
                headword = clean_text("".join(child.itertext()))
                break

        if not headword:
            headword = "UNKNOWN"
            beta = elem.get("key") or ""
            print(
                f"WARNING: missing headword in {os.path.basename(path)} (key={beta})",
                file=sys.stderr,
                flush=True,
            )
        else:
            beta = elem.get("key") or ""

        full_content = clean_text("".join(elem.itertext()))

        batch.append((headword, beta, full_content))
        total_counter += 1

        if total_counter % BATCH_SIZE == 0:
            cur.executemany(
                "INSERT INTO dictionary (headword, beta_code, full_content) VALUES (?, ?, ?)",
                batch,
            )
            conn.commit()
            batch.clear()
            print(f"Processed entries: {total_counter}", flush=True)

        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    if batch:
        cur.executemany(
            "INSERT INTO dictionary (headword, beta_code, full_content) VALUES (?, ?, ?)",
            batch,
        )
        conn.commit()
        batch.clear()

    return total_counter


def main() -> int:
    xml_files = sorted(glob.glob(DATA_GLOB))
    if not xml_files:
        print(f"No XML files found for pattern: {DATA_GLOB}", file=sys.stderr)
        return 1

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    try:
        init_db(conn)
        total = 0
        for path in xml_files:
            total = process_file(conn, path, total)
    finally:
        conn.close()

    print(f"Done. Total entries: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
