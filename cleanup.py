import sqlite3

with sqlite3.connect('log.db') as conn:
    conn.execute("DELETE FROM predictions")
    conn.commit()