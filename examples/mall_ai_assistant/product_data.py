"""商品数据层 —— MySQL（生产）+ SQLite（开发回退）"""
import os
import sqlite3
from typing import List, Dict

MYSQL_HOST = os.environ.get("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.environ.get("MYSQL_PORT", "3306"))
MYSQL_USER = os.environ.get("MYSQL_USER", "root")
MYSQL_PWD = os.environ.get("MYSQL_PWD", "20041123")
MYSQL_DB = os.environ.get("MYSQL_DB", "isdp-db")

DB_PATH = os.path.join(os.path.dirname(__file__), "cloudemall.db")


def search_products(query: str, limit: int = 5) -> List[Dict]:
    # 1) 尝试 MySQL
    result = _search_mysql(query, limit)
    if result:
        return result

    # 2) 回退 SQLite
    return _search_sqlite(query, limit)


def _search_mysql(query: str, limit: int) -> List[Dict]:
    try:
        import pymysql
        conn = pymysql.connect(host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER,
                               password=MYSQL_PWD, database=MYSQL_DB,
                               charset="utf8mb4", connect_timeout=3)
        cur = conn.cursor()
        like = f"%{query}%"
        cur.execute("""
            SELECT p.product_id, p.product_name, p.product_description,
                   p.price, p.stock, p.image_url,
                   c1.category_name, c2.category_name, c3.category_name
            FROM pos_product p
            LEFT JOIN pos_category c3 ON p.product_category_id = c3.category_id
            LEFT JOIN pos_category c2 ON c3.parent_id = c2.category_id
            LEFT JOIN pos_category c1 ON c2.parent_id = c1.category_id
            WHERE p.product_name LIKE %s OR p.product_description LIKE %s
               OR c3.category_name LIKE %s OR c2.category_name LIKE %s OR c1.category_name LIKE %s
            LIMIT %s
        """, (like, like, like, like, like, limit))
        rows = cur.fetchall()
        conn.close()
        if rows:
            return [_row_to_dict(r) for r in rows]
    except Exception:
        pass
    return []


def _search_sqlite(query: str, limit: int) -> List[Dict]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    q = query.lower()

    cur.execute("""
        SELECT p.product_id, p.product_name, p.product_description,
               p.price, p.stock, p.image_url,
               c1.category_name, c2.category_name, c3.category_name
        FROM pos_product p
        LEFT JOIN pos_category c3 ON p.product_category_id = c3.category_id
        LEFT JOIN pos_category c2 ON c3.parent_id = c2.category_id
        LEFT JOIN pos_category c1 ON c2.parent_id = c1.category_id
        WHERE LOWER(p.product_name) LIKE ? OR LOWER(p.product_description) LIKE ?
           OR LOWER(c3.category_name) LIKE ? OR LOWER(c2.category_name) LIKE ?
           OR LOWER(c1.category_name) LIKE ?
        LIMIT ?
    """, (f"%{q}%", f"%{q}%", f"%{q}%", f"%{q}%", f"%{q}%", limit))
    rows = cur.fetchall()

    if rows:
        conn.close()
        return [_row_to_dict(r) for r in rows]

    # Fallback: 中文 2-gram 匹配
    cur.execute("""
        SELECT p.product_id, p.product_name, p.product_description,
               p.price, p.stock, p.image_url,
               c1.category_name, c2.category_name, c3.category_name
        FROM pos_product p
        LEFT JOIN pos_category c3 ON p.product_category_id = c3.category_id
        LEFT JOIN pos_category c2 ON c3.parent_id = c2.category_id
        LEFT JOIN pos_category c1 ON c2.parent_id = c1.category_id
    """)
    all_rows = cur.fetchall()
    conn.close()

    chunks = [q[i:i+2] for i in range(len(q)-1)]
    scored = []
    for r in all_rows:
        text = f"{r[1]} {r[2]} {r[6] or ''} {r[7] or ''} {r[8] or ''}".lower()
        s = sum(1 for c in chunks if c in text)
        if s > 0:
            scored.append((s, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [_row_to_dict(r) for _, r in scored[:limit]]


def _row_to_dict(r) -> Dict:
    return {
        "id": str(r[0]), "name": r[1], "desc": r[2] or "",
        "price": float(r[3]), "stock": r[4] or 0, "image": r[5] or "",
        "category": f"{r[6] or ''}>{r[7] or ''}>{r[8] or ''}".strip(">"),
    }
