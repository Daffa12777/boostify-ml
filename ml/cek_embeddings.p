import pickle
import numpy as np

with open("models/embeddings.pkl", "rb") as f:
    db = pickle.load(f)

with open("models/labels.pkl", "rb") as f:
    labels = pickle.load(f)

print(f"Orang terdaftar: {labels}")
print(f"Jumlah orang   : {len(db)}")
for nama, emb in db.items():
    print(f"  {nama} -> shape: {emb.shape} | norm: {np.linalg.norm(emb):.3f}")
