import sqlite3
import datetime

def check_status():
    with sqlite3.connect('log.db') as conn:
        return conn.cursor().execute("SELECT status FROM model_status WHERE id=1").fetchone()[0]

def apply_window(results, start, end, exclude_from_quorum={'season', 'mnth'}):
    is_stale = any(r['drifted'] for r in results if r['feature'] not in exclude_from_quorum)
    if not is_stale:
        return
    with sqlite3.connect('log.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM model_status WHERE id=1")
        current_status = cursor.fetchone()[0]
        if current_status == 'fresh':
            cursor.execute(
                "UPDATE model_status SET status = ?, triggering_window_start = ?, triggering_window_end = ?, updated_at = ? WHERE id = 1",
                ('stale', start, end, datetime.datetime.now().isoformat())
            )
            conn.commit()

# reset to known state
with sqlite3.connect('log.db') as conn:
    conn.cursor().execute(
        "UPDATE model_status SET status='fresh', triggering_window_start=NULL, triggering_window_end=NULL, updated_at=NULL WHERE id=1"
    )
    conn.commit()

apply_window([{'feature': 'temp', 'drifted': False}], 1, 170)
assert check_status() == 'fresh', "should still be fresh"

apply_window([{'feature': 'temp', 'drifted': True}], 171, 340)
assert check_status() == 'stale', "should now be stale"

apply_window([{'feature': 'temp', 'drifted': False}], 341, 510)
assert check_status() == 'stale', "should still be stale — persist until cleared"

print("Half A smoke test passed.")