# models.py
import hashlib
import sqlite3
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

DB_PATH = "./balance.db"


# ----------------- Connection & schema ----------------- #
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """
    Create all tables if they don't exist.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_hash TEXT NOT NULL UNIQUE,
            uploaded_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS line_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER NOT NULL,
            line_item_id INTEGER NOT NULL,
            year INTEGER NOT NULL,
            amount REAL NOT NULL,
            UNIQUE(file_id, line_item_id, year),
            FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE,
            FOREIGN KEY(line_item_id) REFERENCES line_items(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS journal_batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_name TEXT NOT NULL,
            description TEXT,
            balance_type TEXT,
            source TEXT DEFAULT 'Manual',
            approval_status TEXT DEFAULT 'Not required',
            funds_status TEXT DEFAULT 'Not attempted',
            status TEXT DEFAULT 'Unposted',
            completeness TEXT DEFAULT 'Incomplete',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS journals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id INTEGER NOT NULL,
            journal_name TEXT NOT NULL,
            description TEXT,
            ledger TEXT,
            accounting_period TEXT,
            legal_entity TEXT,
            accounting_date TEXT,
            category TEXT,
            currency TEXT,
            conversion_date TEXT,
            conversion_rate_type TEXT,
            conversion_rate REAL,
            inverse_rate REAL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(batch_id) REFERENCES journal_batches(id) ON DELETE CASCADE
        );
        """
    )
    conn.commit()


# ----------------- Generic helpers ----------------- #
def get_table_df(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def get_schema(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    return pd.read_sql_query(f"PRAGMA table_info({table})", conn)


# ----------------- TB/BS ingestion ----------------- #
def upsert_file(conn: sqlite3.Connection, filename: str, content: bytes) -> tuple[int, bool]:
    file_hash = hashlib.sha256(content).hexdigest()
    cur = conn.cursor()
    cur.execute("SELECT id FROM files WHERE file_hash = ?", (file_hash,))
    row = cur.fetchone()
    if row:
        return row[0], False
    cur.execute(
        "INSERT INTO files(filename, file_hash, uploaded_at) VALUES (?, ?, ?)",
        (filename, file_hash, datetime.utcnow().isoformat()),
    )
    conn.commit()
    return cur.lastrowid, True


def get_or_create_line_ids(conn: sqlite3.Connection, names: List[str]) -> dict:
    names = list(dict.fromkeys([str(n).strip() for n in names if str(n).strip()]))
    cur = conn.cursor()
    if not names:
        return {}
    q_marks = ",".join("?" for _ in names)
    cur.execute(f"SELECT id, name FROM line_items WHERE name IN ({q_marks})", names)
    have = {name: lid for lid, name in cur.fetchall()}
    missing = [nm for nm in names if nm not in have]
    if missing:
        cur.executemany(
            "INSERT OR IGNORE INTO line_items(name) VALUES (?)",
            [(m,) for m in missing],
        )
        conn.commit()
        cur.execute(f"SELECT id, name FROM line_items WHERE name IN ({q_marks})", names)
        have = {name: lid for lid, name in cur.fetchall()}
    return have


def insert_facts(conn: sqlite3.Connection, file_id: int, df_long: pd.DataFrame) -> None:
    names = df_long["Line Item"].dropna().astype(str).tolist()
    name_to_id = get_or_create_line_ids(conn, names)
    payload = []
    for _, row in df_long.iterrows():
        if pd.isna(row["Amount"]):
            continue
        li = row["Line Item"]
        if li not in name_to_id:
            continue
        payload.append(
            (file_id, name_to_id[li], int(row["Year"]), float(row["Amount"]))
        )
    cur = conn.cursor()
    cur.executemany(
        "INSERT OR IGNORE INTO facts(file_id, line_item_id, year, amount) "
        "VALUES (?, ?, ?, ?)",
        payload,
    )
    conn.commit()


def list_batches(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT id, filename, uploaded_at FROM files ORDER BY id DESC",
        conn,
    )


def load_facts(conn: sqlite3.Connection, file_id: int) -> pd.DataFrame:
    q = """
    SELECT li.name AS line_item, f.year, f.amount
    FROM facts f
    JOIN line_items li ON li.id = f.line_item_id
    WHERE f.file_id = ?
    """
    return pd.read_sql_query(q, conn, params=(file_id,))


# ----------------- Journals (Oracle-style) ----------------- #
def create_journal_batch(
    conn: sqlite3.Connection,
    *,
    batch_name: str,
    description: str,
    balance_type: str,
) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO journal_batches
            (batch_name, description, balance_type, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (batch_name, description, balance_type, datetime.utcnow().isoformat()),
    )
    conn.commit()
    return cur.lastrowid


def create_journal(
    conn: sqlite3.Connection,
    *,
    batch_id: int,
    journal_name: str,
    description: str,
    ledger: str,
    accounting_period: str,
    legal_entity: str,
    accounting_date: str,
    category: str,
    currency: str,
    conversion_date: Optional[str],
    conversion_rate_type: str,
    conversion_rate: float,
    inverse_rate: float,
) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO journals (
            batch_id,
            journal_name,
            description,
            ledger,
            accounting_period,
            legal_entity,
            accounting_date,
            category,
            currency,
            conversion_date,
            conversion_rate_type,
            conversion_rate,
            inverse_rate,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            batch_id,
            journal_name,
            description,
            ledger,
            accounting_period,
            legal_entity,
            accounting_date,
            category,
            currency,
            conversion_date,
            conversion_rate_type,
            conversion_rate,
            inverse_rate,
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    return cur.lastrowid


def list_journal_batches(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM journal_batches ORDER BY id DESC",
        conn,
    )


def list_journals(conn: sqlite3.Connection) -> pd.DataFrame:
    q = """
    SELECT j.*, b.batch_name
    FROM journals j
    JOIN journal_batches b ON b.id = j.batch_id
    ORDER BY j.id DESC
    """
    return pd.read_sql_query(q, conn)
