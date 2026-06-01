import os
import psycopg2
from datetime import datetime
from collections import defaultdict
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DIRECT_URL") or os.getenv("DATABASE_URL")

OPEN_HOUR, CLOSE_HOUR = 7, 19  # jam operasional lab

def fetch_attendance(conn):
    with conn.cursor() as cur:
        cur.execute('SELECT "time" FROM "Attendance"')
        return [r[0] for r in cur.fetchall()]

def build_samples(times):
    counts = defaultdict(int)
    for t in times:
        if not isinstance(t, datetime):
            t = datetime.fromisoformat(str(t))
        day = (t.weekday() + 1) % 7   # 0=Minggu .. 6=Sabtu
        counts[(t.date(), day, t.hour)] += 1
    return counts

def train_and_predict(counts):
    if not counts:
        return None
    X = np.array([[day, hour] for (_, day, hour) in counts])
    y = np.array(list(counts.values()))
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    results = []
    for day in range(7):
        for hour in range(OPEN_HOUR, CLOSE_HOUR + 1):
            pred = float(model.predict(np.array([[day, hour]]))[0])
            results.append((day, hour, round(pred, 1)))
    return results

def classify(results):
    max_pred = max((r[2] for r in results), default=0)
    out = []
    for day, hour, pred in results:
        if max_pred <= 0:
            level = "sepi"
        else:
            ratio = pred / max_pred
            level = "ramai" if ratio >= 0.66 else "sedang" if ratio >= 0.33 else "sepi"
        out.append((day, hour, pred, level))
    return out

def save_predictions(conn, rows):
    with conn.cursor() as cur:
        cur.execute('DELETE FROM "Prediction"')
        for day, hour, pred, level in rows:
            cur.execute(
                'INSERT INTO "Prediction" (day, hour, predicted_count, level, updated_at) '
                'VALUES (%s, %s, %s, %s, NOW())',
                (day, hour, pred, level),
            )
    conn.commit()

def main():
    if not DB_URL:
        print("ERROR: DATABASE_URL/DIRECT_URL belum di-set di .env")
        return
    conn = psycopg2.connect(DB_URL)
    try:
        times = fetch_attendance(conn)
        print(f"Total {len(times)} record absensi.")
        if len(times) < 5:
            print("⚠️  Data masih sedikit — prediksi belum akurat, tapi tetap dihitung.")
        counts = build_samples(times)
        results = train_and_predict(counts)
        if results is None:
            print("Tidak ada data untuk diprediksi.")
            return
        save_predictions(conn, classify(results))
        print(f"✅ {len(results)} slot prediksi tersimpan ke tabel Prediction.")
    finally:
        conn.close()

if __name__ == "__main__":
    main()