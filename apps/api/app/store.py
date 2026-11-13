"""로컬 저장소(SQLite). 1단계 빠른 시험용이며, 나중에 Supabase(pgvector)로 바꿔 끼운다."""
import os
import sqlite3

import numpy as np

DB_PATH = os.environ.get("APP_DB", "data/app.db")


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    return c


def init() -> None:
    with _conn() as c:
        c.executescript(
            """
            create table if not exists documents (
              id integer primary key autoincrement,
              title text not null,
              chunk_count int not null default 0,
              created_at text not null default (datetime('now'))
            );
            create table if not exists chunks (
              id integer primary key autoincrement,
              document_id integer not null references documents(id) on delete cascade,
              idx int not null,
              content text not null,
              embedding blob not null
            );
            """
        )


def add_document(title: str, chunks: list[str], embeddings: list[list[float]]) -> int:
    with _conn() as c:
        doc_id = c.execute(
            "insert into documents(title, chunk_count) values (?, ?)", (title, len(chunks))
        ).lastrowid
        c.executemany(
            "insert into chunks(document_id, idx, content, embedding) values (?, ?, ?, ?)",
            [
                (doc_id, i, text, np.asarray(emb, dtype=np.float32).tobytes())
                for i, (text, emb) in enumerate(zip(chunks, embeddings))
            ],
        )
        return doc_id


def list_documents() -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "select id, title, chunk_count, created_at from documents order by id desc"
        ).fetchall()
        return [dict(r) for r in rows]


def delete_document(doc_id: int) -> bool:
    with _conn() as c:
        return c.execute("delete from documents where id = ?", (doc_id,)).rowcount > 0


def search(query_embedding: list[float], k: int = 5) -> list[dict]:
    """질문과 가장 비슷한 조각 k개를 코사인 유사도로 찾는다."""
    with _conn() as c:
        rows = c.execute(
            "select c.id, c.document_id, c.idx, c.content, c.embedding, d.title "
            "from chunks c join documents d on d.id = c.document_id"
        ).fetchall()
    if not rows:
        return []
    mat = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    q = np.asarray(query_embedding, dtype=np.float32)
    sims = (mat @ q) / (np.linalg.norm(mat, axis=1) * np.linalg.norm(q) + 1e-9)
    top = np.argsort(-sims)[:k]
    return [
        {
            "document_id": rows[i]["document_id"],
            "title": rows[i]["title"],
            "content": rows[i]["content"],
            "score": float(sims[i]),
        }
        for i in top
    ]
