import sqlite3

with sqlite3.connect("log.db") as conn:
        cursor = conn.cursor()
        cursor.execute("""SELECT * FROM predictions""")
        print(cursor.fetchall())