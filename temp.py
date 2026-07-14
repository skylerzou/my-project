import sqlite3

with sqlite3.connect('log.db') as conn:
    print(conn.cursor().execute("SELECT * FROM model_status WHERE id=1").fetchone())