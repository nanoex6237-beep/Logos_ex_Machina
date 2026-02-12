import glob
import os
import sqlite3
import sys
import xml.etree.ElementTree as ET

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


def to_roman(num: int) -> str:
    vals = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    out = []
    for v, sym in vals:
        while num >= v:
            out.append(sym)
            num -= v
    return "".join(out) if out else "I"


def label_for_counters(counters: list[int]) -> str:
    parts = []
    for idx, val in enumerate(counters):
        if idx == 0:
            parts.append(to_roman(val))
        elif idx == 1:
            parts.append(chr(ord("A") + val - 1))
        elif idx == 2:
            parts.append(chr(ord("a") + val - 1))
        else:
            parts.append(str(val))
    return ".".join(parts)


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
    cur.execute(
        """
        CREATE TABLE senses (
            id INTEGER PRIMARY KEY,
            entry_id INTEGER,
            sense_id TEXT,
            level INTEGER,
            label TEXT,
            content TEXT
        )
        """
    )
    cur.execute("CREATE INDEX idx_dictionary_headword ON dictionary(headword)")
    cur.execute("CREATE INDEX idx_dictionary_beta_code ON dictionary(beta_code)")
    cur.execute("CREATE INDEX idx_senses_entry_id ON senses(entry_id)")
    cur.execute("CREATE INDEX idx_senses_label ON senses(label)")
    conn.commit()


def process_file(conn: sqlite3.Connection, path: str, total_counter: int) -> int:
    cur = conn.cursor()
    sense_batch = []

    print(f"Processing {os.path.basename(path)}...", flush=True)

    context = ET.iterparse(path, events=("end",))
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

        cur.execute(
            "INSERT INTO dictionary (headword, beta_code, full_content) VALUES (?, ?, ?)",
            (headword, beta, full_content),
        )
        entry_id = cur.lastrowid

        counters: list[int] = []
        for sense in elem.findall(".//sense"):
            level_raw = sense.get("level") or "1"
            try:
                level = max(1, int(level_raw))
            except ValueError:
                level = 1

            if len(counters) < level:
                counters.extend([0] * (level - len(counters)))
            elif len(counters) > level:
                counters = counters[:level]

            counters[level - 1] += 1
            label = label_for_counters(counters)

            sense_text = clean_text("".join(sense.itertext()))
            sense_batch.append(
                (entry_id, sense.get("id") or "", level, label, sense_text)
            )

        total_counter += 1

        if total_counter % BATCH_SIZE == 0:
            if sense_batch:
                cur.executemany(
                    "INSERT INTO senses (entry_id, sense_id, level, label, content) VALUES (?, ?, ?, ?, ?)",
                    sense_batch,
                )
            conn.commit()
            sense_batch.clear()
            print(f"Processed entries: {total_counter}", flush=True)

        elem.clear()

    if sense_batch:
        cur.executemany(
            "INSERT INTO senses (entry_id, sense_id, level, label, content) VALUES (?, ?, ?, ?, ?)",
            sense_batch,
        )
        conn.commit()
        sense_batch.clear()

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
