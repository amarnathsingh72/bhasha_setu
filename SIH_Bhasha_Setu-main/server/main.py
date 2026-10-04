"""
Local Offline FastAPI Backend for Bhasha Setu
Connects to local PostgreSQL (tribal_translation_db) or falls back seamlessly
to local SQLite (translations.db) for 100% offline reliability.
Provides offline JSON APIs for the React frontend service layer.
"""

import os
import sys
import sqlite3
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure UTF-8 output
reconfigure_stdout = getattr(sys.stdout, "reconfigure", None)
if callable(reconfigure_stdout):
    reconfigure_stdout(encoding="utf-8")

# Load environment
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
env_path = BASE_DIR / '.env'
load_dotenv(dotenv_path=env_path)

DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = int(os.getenv('DB_PORT', '5432'))
DB_NAME = os.getenv('DB_NAME', 'tribal_translation_db')
DB_USER = os.getenv('DB_USER', 'tribal_app')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')
PORT = int(os.getenv('PORT', '5000'))

SQLITE_DB_PATH = BASE_DIR / 'translations.db'

# Connection Pool & Fallback Flag
db_pool = None
pg_available = None

def check_pg_available() -> bool:
    global pg_available, db_pool
    if pg_available is not None:
        return pg_available
    try:
        import psycopg2
        from psycopg2 import pool
        db_pool = pool.SimpleConnectionPool(
            1, 10,
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        conn = db_pool.getconn()
        db_pool.putconn(conn)
        pg_available = True
        print(f"✅ Connected to PostgreSQL ({DB_NAME})")
    except Exception as e:
        print(f"ℹ️ PostgreSQL not available ({e}). Using local SQLite database {SQLITE_DB_PATH.name}.")
        pg_available = False
        db_pool = None
    return pg_available

def get_sqlite_conn():
    conn = sqlite3.connect(str(SQLITE_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

app = FastAPI(
    title="Bhasha Setu Offline Translation Engine",
    description="Local offline PostgreSQL / SQLite bridge for tribal multilingual translations.",
    version="1.0.0"
)

# CORS: Allow all local origins, private networks, and web dev ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
        headers={"Access-Control-Allow-Origin": "*"}
    )

@app.get("/")
def root():
    return {
        "name": "Bhasha Setu Offline Translation Engine",
        "status": "online",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/api/health"
    }

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return JSONResponse(status_code=204, content=None)


# Neural ASR Engine Router (Santali IndicConformer + Faster-Whisper)
try:
    from server.api.asr_routes import router as asr_api_router
    app.include_router(asr_api_router, prefix="/api/asr", tags=["Neural ASR"])
    print("✅ Mounted Neural ASR router at /api/asr")
except Exception as asr_err:
    print(f"⚠️ ASR router mount warning: {asr_err}")

# Video Subtitle Engine Router
try:
    from server.api.video_routes import router as video_api_router
    app.include_router(video_api_router, prefix="/api/video", tags=["Video Subtitles"])
    print("✅ Mounted Video Subtitles router at /api/video")
except Exception as video_err:
    print(f"⚠️ Video Subtitles router mount warning: {video_err}")

# Hindi -> Mundari Neural & Dataset Translation Engine
try:
    from server.translation import translate_hindi_to_mundari
    print("✅ Mounted Hindi -> Mundari Translation Engine (Level 1: Dataset, Level 2: mT5 Fallback)")
except Exception as t_err:
    print(f"⚠️ Hindi -> Mundari translation engine mount warning: {t_err}")
    translate_hindi_to_mundari = None

# Language code normalization
LANG_MAP = {
    'eng': 1, 'en': 1, 'english': 1,
    'hin': 2, 'hi': 2, 'hindi': 2,
    'sat': 3, 'santali': 3,
    'hoc': 4, 'ho': 4,
    'unr': 5, 'mundari': 5, 'mun': 5
}

LANG_COL_MAP = {
    'eng': 'english', 'en': 'english', 'english': 'english',
    'hin': 'hindi', 'hi': 'hindi', 'hindi': 'hindi',
    'sat': 'santali', 'santali': 'santali',
    'hoc': 'ho', 'ho': 'ho',
    'unr': 'mundari', 'mundari': 'mundari', 'mun': 'mundari'
}

class TranslateRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

class HindiToMundariRequest(BaseModel):
    text: str
    domain: Optional[str] = "General"

class HindiToMundariResponse(BaseModel):
    hindi: str
    mundari: str
    mundari_roman: str
    source: str
    confidence: str
    status: str = "success"
    warning: Optional[str] = None
    error: Optional[str] = None

class TranslationRowResponse(BaseModel):
    id: int
    english: str
    hindi: str
    santali: str
    santali_roman: Optional[str] = None
    ho: Optional[str] = None
    mundari: Optional[str] = None
    mundari_roman: Optional[str] = None
    category: str
    verified: str = "Yes"

@app.get("/api/health")
def health_check():
    """Health check endpoint to verify backend and database connection."""
    if not check_pg_available():
        conn = get_sqlite_conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM translations;")
            total_rows = cur.fetchone()[0]
            cur.close()
            return {
                "status": "online",
                "mode": "offline-sqlite",
                "database": "translations.db",
                "total_translation_sets": total_rows,
                "total_translation_texts": total_rows
            }
        finally:
            conn.close()

    p = db_pool
    conn = p.getconn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM translation_sets;")
        r_sets = cur.fetchone()
        total_sets = r_sets[0] if r_sets else 0
        cur.execute("SELECT COUNT(*) FROM translation_texts;")
        r_texts = cur.fetchone()
        total_texts = r_texts[0] if r_texts else 0
        cur.close()
        return {
            "status": "online",
            "mode": "offline-local",
            "database": DB_NAME,
            "total_translation_sets": total_sets,
            "total_translation_texts": total_texts
        }
    finally:
        p.putconn(conn)

@app.get("/api/stats")
def get_stats():
    """Returns database stats formatted for SqliteStats interface."""
    if not check_pg_available():
        conn = get_sqlite_conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM translations;")
            total_rows = cur.fetchone()[0]
            cur.execute("SELECT COUNT(DISTINCT category) FROM translations WHERE category IS NOT NULL;")
            cat_count = cur.fetchone()[0]
            cur.close()
            return {
                "isReady": True,
                "totalRows": total_rows,
                "categoriesCount": cat_count,
                "loadError": None
            }
        except Exception as e:
            return {
                "isReady": False,
                "totalRows": 0,
                "categoriesCount": 0,
                "loadError": str(e)
            }
        finally:
            conn.close()

    p = db_pool
    conn = p.getconn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM translation_sets;")
        r_total = cur.fetchone()
        total_rows = r_total[0] if r_total else 0

        cur.execute("SELECT COUNT(DISTINCT category_id) FROM translation_sets WHERE category_id IS NOT NULL;")
        r_cat = cur.fetchone()
        cat_count = r_cat[0] if r_cat else 0
        cur.close()

        return {
            "isReady": True,
            "totalRows": total_rows,
            "categoriesCount": cat_count,
            "loadError": None
        }
    except Exception as e:
        return {
            "isReady": False,
            "totalRows": 0,
            "categoriesCount": 0,
            "loadError": str(e)
        }
    finally:
        p.putconn(conn)

@app.get("/api/categories")
def get_categories():
    """Get categories and count of sentences per category."""
    if not check_pg_available():
        conn = get_sqlite_conn()
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT category, COUNT(id) as count
                FROM translations
                WHERE category IS NOT NULL
                GROUP BY category
                ORDER BY count DESC;
            """)
            rows = cur.fetchall()
            cur.close()
            return [{"category": r["category"], "count": r["count"]} for r in rows]
        finally:
            conn.close()

    p = db_pool
    conn = p.getconn()
    try:
        cur = conn.cursor()
        query = """
            SELECT c.category_name, COUNT(ts.translation_set_id) as count
            FROM categories c
            JOIN translation_sets ts ON c.category_id = ts.category_id
            GROUP BY c.category_name
            ORDER BY count DESC;
        """
        cur.execute(query)
        rows = cur.fetchall()
        cur.close()
        return [{"category": r[0], "count": r[1]} for r in rows]
    finally:
        p.putconn(conn)

@app.get("/api/search", response_model=List[TranslationRowResponse])
def search_sentences(
    keyword: str = Query("", description="Keyword to search across languages"),
    category: str = Query("All", description="Category filter"),
    limit: int = Query(30, description="Max rows to return")
):
    """Search parallel sentences with optional category filter."""
    clean_keyword = keyword.strip().lower()

    if not check_pg_available():
        conn = get_sqlite_conn()
        try:
            cur = conn.cursor()
            query = """
                SELECT id, english, hindi, santali, santali_roman, ho, mundari, COALESCE(mundari_roman, '') as mundari_roman, category, verified
                FROM translations
            """
            conditions = []
            params = []

            if clean_keyword:
                conditions.append("""(
                    LOWER(english) LIKE ? OR
                    LOWER(hindi) LIKE ? OR
                    LOWER(santali) LIKE ? OR
                    LOWER(COALESCE(santali_roman, '')) LIKE ? OR
                    LOWER(COALESCE(mundari, '')) LIKE ? OR
                    LOWER(COALESCE(mundari_roman, '')) LIKE ?
                )""")
                pattern = f"%{clean_keyword}%"
                params.extend([pattern, pattern, pattern, pattern, pattern, pattern])

            if category and category != "All":
                conditions.append("category = ?")
                params.append(category)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY id ASC LIMIT ?"
            params.append(limit)

            cur.execute(query, params)
            rows = cur.fetchall()
            cur.close()

            results = []
            for r in rows:
                results.append({
                    "id": r["id"],
                    "english": r["english"] or "",
                    "hindi": r["hindi"] or "",
                    "santali": r["santali"] or "",
                    "santali_roman": r["santali_roman"],
                    "ho": r["ho"],
                    "mundari": r["mundari"],
                    "mundari_roman": r["mundari_roman"],
                    "category": r["category"] or "General",
                    "verified": r["verified"] or "Yes"
                })
            return results
        finally:
            conn.close()

    p = db_pool
    conn = p.getconn()
    try:
        cur = conn.cursor()
        query = """
            SELECT 
                ts.translation_set_id as id,
                COALESCE(en.text_content, '') as english,
                COALESCE(hi.text_content, '') as hindi,
                COALESCE(sat.text_content, '') as santali,
                sat.pronunciation as santali_roman,
                COALESCE(ho.text_content, '') as ho,
                COALESCE(unr.text_content, '') as mundari,
                COALESCE(c.category_name, 'General') as category,
                CASE WHEN ts.verified THEN 'Yes' ELSE 'No' END as verified
            FROM translation_sets ts
            LEFT JOIN categories c ON ts.category_id = c.category_id
            LEFT JOIN translation_texts en ON en.translation_set_id = ts.translation_set_id AND en.language_id = 1
            LEFT JOIN translation_texts hi ON hi.translation_set_id = ts.translation_set_id AND hi.language_id = 2
            LEFT JOIN translation_texts sat ON sat.translation_set_id = ts.translation_set_id AND sat.language_id = 3
            LEFT JOIN translation_texts ho ON ho.translation_set_id = ts.translation_set_id AND ho.language_id = 4
            LEFT JOIN translation_texts unr ON unr.translation_set_id = ts.translation_set_id AND unr.language_id = 5
        """
        conditions = []
        params: List[object] = []

        if clean_keyword:
            conditions.append("""(
                LOWER(en.text_content) LIKE %s OR
                LOWER(hi.text_content) LIKE %s OR
                LOWER(sat.text_content) LIKE %s OR
                LOWER(COALESCE(sat.pronunciation, '')) LIKE %s
            )""")
            pattern = f"%{clean_keyword}%"
            params.extend([pattern, pattern, pattern, pattern])

        if category and category != "All":
            conditions.append("c.category_name = %s")
            params.append(category)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY ts.translation_set_id ASC LIMIT %s;"
        params.append(limit)

        cur.execute(query, params)
        rows = cur.fetchall()
        cur.close()

        results = []
        for r in rows:
            results.append({
                "id": r[0],
                "english": r[1],
                "hindi": r[2],
                "santali": r[3],
                "santali_roman": r[4],
                "ho": r[5],
                "mundari": r[6],
                "category": r[7],
                "verified": r[8]
            })
        return results
    finally:
        p.putconn(conn)

# In-memory translation cache (Phase 11)
_translation_memory_cache = {}

@app.post("/api/translate")
def translate_text(req: TranslateRequest):
    """
    Direct SQL parallel query for exact or fuzzy match across all 5 supported languages.
    Fronted by in-memory cache for sub-millisecond repeated lookups.
    """
    clean_text = req.text.strip()
    if not clean_text:
        return {"target_text": "", "roman": None, "confidence": 0, "row": None}

    src_lang = req.source_lang.lower()
    target_lang = req.target_lang.lower()
    src_lang_id = LANG_MAP.get(src_lang, 1)
    target_lang_id = LANG_MAP.get(target_lang, 3)
    lower = clean_text.lower().replace('?', '').replace('!', '').replace('.', '').replace(',', '').strip()

    cache_key = f"{src_lang_id}:{target_lang_id}:{lower}"
    if cache_key in _translation_memory_cache:
        return _translation_memory_cache[cache_key]

    src_col = LANG_COL_MAP.get(src_lang, 'english')
    target_col = LANG_COL_MAP.get(target_lang, 'santali')

    # Level 1 (Dataset) & Level 2 (mT5 Fallback) Hindi -> Mundari Delegation
    if src_col == 'hindi' and target_col == 'mundari' and translate_hindi_to_mundari is not None:
        h2m_res = translate_hindi_to_mundari(clean_text)
        if h2m_res and h2m_res.get("status") == "success" and h2m_res.get("mundari"):
            target_text = h2m_res["mundari"]
            roman = h2m_res.get("mundari_roman")
            is_ds = "dataset" in h2m_res.get("source", "")
            res = {
                "target_text": target_text,
                "roman": roman,
                "confidence": 0.99 if is_ds else 0.70,
                "source": h2m_res.get("source"),
                "row": {
                    "id": 0,
                    "english": "",
                    "hindi": clean_text,
                    "santali": "",
                    "santali_roman": None,
                    "ho": None,
                    "mundari": target_text,
                    "mundari_roman": roman,
                    "category": "General",
                    "verified": "Yes" if is_ds else "Experimental"
                }
            }
            if len(_translation_memory_cache) < 5000:
                _translation_memory_cache[cache_key] = res
            return res

    if not check_pg_available():
        conn = get_sqlite_conn()
        try:
            cur = conn.cursor()

            # 1. Exact match
            query = f"""
                SELECT id, english, hindi, santali, santali_roman, ho, mundari, COALESCE(mundari_roman, '') as mundari_roman, category, verified,
                       {target_col} as target_text,
                       CASE 
                           WHEN '{target_col}' = 'mundari' THEN COALESCE(mundari_roman, '')
                           ELSE COALESCE(santali_roman, '')
                       END as target_roman
                FROM translations
                WHERE LOWER(TRIM({src_col})) = ?
                   OR LOWER(TRIM(COALESCE(santali_roman, ''))) = ?
                   OR LOWER(TRIM(COALESCE(mundari_roman, ''))) = ?
                LIMIT 1;
            """
            cur.execute(query, (lower, lower, lower))
            row = cur.fetchone()

            confidence = 0.99
            if not row:
                # 2. Fuzzy match
                query_fuzzy = f"""
                    SELECT id, english, hindi, santali, santali_roman, ho, mundari, COALESCE(mundari_roman, '') as mundari_roman, category, verified,
                           {target_col} as target_text,
                           CASE 
                               WHEN '{target_col}' = 'mundari' THEN COALESCE(mundari_roman, '')
                               ELSE COALESCE(santali_roman, '')
                           END as target_roman
                    FROM translations
                    WHERE LOWER({src_col}) LIKE ?
                    LIMIT 1;
                """
                cur.execute(query_fuzzy, (f"%{lower}%",))
                row = cur.fetchone()
                confidence = 0.95

            if row and row["target_text"]:
                target_text = row["target_text"]
                roman = row["target_roman"] if row["target_roman"] else None

                res_row = {
                    "id": row["id"], "english": row["english"] or "", "hindi": row["hindi"] or "",
                    "santali": row["santali"] or "", "santali_roman": row["santali_roman"],
                    "ho": row["ho"], "mundari": row["mundari"], "mundari_roman": row["mundari_roman"],
                    "category": row["category"] or "General",
                    "verified": row["verified"] or "Yes"
                }
                cur.close()
                res = {
                    "target_text": target_text,
                    "roman": roman,
                    "confidence": confidence,
                    "row": res_row
                }
                if len(_translation_memory_cache) < 5000:
                    _translation_memory_cache[cache_key] = res
                return res

            cur.close()
            # If Hindi -> Mundari and no SQLite row found, use translation_service
            if src_col == 'hindi' and target_col == 'mundari' and translate_hindi_to_mundari is not None:
                h2m_res = translate_hindi_to_mundari(clean_text)
                if h2m_res and h2m_res.get("status") == "success" and h2m_res.get("mundari"):
                    target_text = h2m_res["mundari"]
                    roman = h2m_res["mundari_roman"]
                    is_ds = "dataset" in h2m_res.get("source", "")
                    res = {
                        "target_text": target_text,
                        "roman": roman,
                        "confidence": 0.99 if is_ds else 0.70,
                        "source": h2m_res.get("source"),
                        "row": {
                            "id": 0, "english": "", "hindi": clean_text,
                            "santali": "", "santali_roman": None, "ho": None,
                            "mundari": target_text, "mundari_roman": roman,
                            "category": "General",
                            "verified": "Yes" if is_ds else "Experimental"
                        }
                    }
                    if len(_translation_memory_cache) < 5000:
                        _translation_memory_cache[cache_key] = res
                    return res

            return {"target_text": None, "roman": None, "confidence": 0, "row": None}
        finally:
            conn.close()

    p = db_pool
    conn = p.getconn()
    try:
        cur = conn.cursor()

        # 1. Exact Match
        exact_query = """
            SELECT 
                ts.translation_set_id as id,
                COALESCE(en.text_content, '') as english,
                COALESCE(hi.text_content, '') as hindi,
                COALESCE(sat.text_content, '') as santali,
                sat.pronunciation as santali_roman,
                COALESCE(ho.text_content, '') as ho,
                COALESCE(unr.text_content, '') as mundari,
                COALESCE(c.category_name, 'General') as category,
                CASE WHEN ts.verified THEN 'Yes' ELSE 'No' END as verified,
                target.text_content as target_text,
                target.pronunciation as target_roman
            FROM translation_sets ts
            JOIN translation_texts src ON src.translation_set_id = ts.translation_set_id AND src.language_id = %s
            LEFT JOIN translation_texts target ON target.translation_set_id = ts.translation_set_id AND target.language_id = %s
            LEFT JOIN categories c ON ts.category_id = c.category_id
            LEFT JOIN translation_texts en ON en.translation_set_id = ts.translation_set_id AND en.language_id = 1
            LEFT JOIN translation_texts hi ON hi.translation_set_id = ts.translation_set_id AND hi.language_id = 2
            LEFT JOIN translation_texts sat ON sat.translation_set_id = ts.translation_set_id AND sat.language_id = 3
            LEFT JOIN translation_texts ho ON ho.translation_set_id = ts.translation_set_id AND ho.language_id = 4
            LEFT JOIN translation_texts unr ON unr.translation_set_id = ts.translation_set_id AND unr.language_id = 5
            WHERE LOWER(TRIM(src.text_content)) = %s 
               OR LOWER(TRIM(COALESCE(src.pronunciation, ''))) = %s
            LIMIT 1;
        """
        cur.execute(exact_query, (src_lang_id, target_lang_id, lower, lower))
        row = cur.fetchone()

        if row and row[9]:
            target_text = row[9]
            roman = row[10] or (row[4] if target_lang == 'sat' else None)
            if target_lang in ('sat', 'hoc', 'unr') and roman and '(' not in target_text:
                target_text = f"{target_text} ({roman})"

            res_row = {
                "id": row[0], "english": row[1], "hindi": row[2],
                "santali": row[3], "santali_roman": row[4], "ho": row[5],
                "mundari": row[6], "category": row[7], "verified": row[8]
            }
            cur.close()
            res = {
                "target_text": target_text,
                "roman": roman,
                "confidence": 0.99,
                "row": res_row
            }
            if len(_translation_memory_cache) < 5000:
                _translation_memory_cache[cache_key] = res
            return res

        # 2. Fuzzy LIKE Match
        fuzzy_query = """
            SELECT 
                ts.translation_set_id as id,
                COALESCE(en.text_content, '') as english,
                COALESCE(hi.text_content, '') as hindi,
                COALESCE(sat.text_content, '') as santali,
                sat.pronunciation as santali_roman,
                COALESCE(ho.text_content, '') as ho,
                COALESCE(unr.text_content, '') as mundari,
                COALESCE(c.category_name, 'General') as category,
                CASE WHEN ts.verified THEN 'Yes' ELSE 'No' END as verified,
                target.text_content as target_text,
                target.pronunciation as target_roman
            FROM translation_sets ts
            JOIN translation_texts src ON src.translation_set_id = ts.translation_set_id AND src.language_id = %s
            LEFT JOIN translation_texts target ON target.translation_set_id = ts.translation_set_id AND target.language_id = %s
            LEFT JOIN categories c ON ts.category_id = c.category_id
            LEFT JOIN translation_texts en ON en.translation_set_id = ts.translation_set_id AND en.language_id = 1
            LEFT JOIN translation_texts hi ON hi.translation_set_id = ts.translation_set_id AND hi.language_id = 2
            LEFT JOIN translation_texts sat ON sat.translation_set_id = ts.translation_set_id AND sat.language_id = 3
            LEFT JOIN translation_texts ho ON ho.translation_set_id = ts.translation_set_id AND ho.language_id = 4
            LEFT JOIN translation_texts unr ON unr.translation_set_id = ts.translation_set_id AND unr.language_id = 5
            WHERE LOWER(src.text_content) LIKE %s
            LIMIT 1;
        """
        pattern = f"%{lower}%"
        cur.execute(fuzzy_query, (src_lang_id, target_lang_id, pattern))
        row = cur.fetchone()

        if row and row[9]:
            target_text = row[9]
            roman = row[10] or (row[4] if target_lang == 'sat' else None)
            if target_lang in ('sat', 'hoc', 'unr') and roman and '(' not in target_text:
                target_text = f"{target_text} ({roman})"

            res_row = {
                "id": row[0], "english": row[1], "hindi": row[2],
                "santali": row[3], "santali_roman": row[4], "ho": row[5],
                "mundari": row[6], "category": row[7], "verified": row[8]
            }
            cur.close()
            res = {
                "target_text": target_text,
                "roman": roman,
                "confidence": 0.95,
                "row": res_row
            }
            if len(_translation_memory_cache) < 5000:
                _translation_memory_cache[cache_key] = res
            return res

        cur.close()
        # If Hindi -> Mundari and no PostgreSQL row found, use translation_service
        if src_col == 'hindi' and target_col == 'mundari' and translate_hindi_to_mundari is not None:
            h2m_res = translate_hindi_to_mundari(clean_text)
            if h2m_res and h2m_res.get("status") == "success" and h2m_res.get("mundari"):
                target_text = h2m_res["mundari"]
                roman = h2m_res["mundari_roman"]
                is_ds = "dataset" in h2m_res.get("source", "")
                res = {
                    "target_text": target_text,
                    "roman": roman,
                    "confidence": 0.99 if is_ds else 0.70,
                    "source": h2m_res.get("source"),
                    "row": {
                        "id": 0, "english": "", "hindi": clean_text,
                        "santali": "", "santali_roman": None, "ho": None,
                        "mundari": target_text, "mundari_roman": roman,
                        "category": "General",
                        "verified": "Yes" if is_ds else "Experimental"
                    }
                }
                if len(_translation_memory_cache) < 5000:
                    _translation_memory_cache[cache_key] = res
                return res

        return {"target_text": None, "roman": None, "confidence": 0, "row": None}

    finally:
        p.putconn(conn)

@app.post("/api/translate/hindi-to-mundari", response_model=HindiToMundariResponse)
def api_translate_hindi_to_mundari(req: HindiToMundariRequest):
    """
    Dedicated hierarchical Hindi -> Mundari translation endpoint:
    - Level 1: Local Verified Mundari Dataset (O(1) exact, normalized, high-confidence fuzzy)
    - Level 2: mT5 Neural Fallback (google/mt5-small or local fine-tuned checkpoint)
    Strictly isolates Mundari from Santali.
    """
    if translate_hindi_to_mundari is None:
        raise HTTPException(status_code=503, detail="Hindi -> Mundari translation engine is unavailable.")
    
    result = translate_hindi_to_mundari(req.text)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("error", "Invalid input"))
    
    return result

if __name__ == '__main__':
    import uvicorn
    print(f"🚀 Starting Bhasha Setu Local Backend on http://127.0.0.1:{PORT}...")
    uvicorn.run("main:app", host="127.0.0.1", port=PORT, reload=False)
