"""
register_users.py — Daftarin Semua User Dataset ML ke Backend Boostify
========================================================================
Jalankan SATU KALI untuk mendaftarkan semua orang yang ada di dataset ML
ke tabel Assisstant. Setelah ini, mereka bisa:
  1. Absen via face recognition (ML → /api/uploadfromml)
  2. Login ke dashboard web (boostify-fe.vercel.app)

Cara pakai:
    python register_users.py
"""

import requests

# ── Konfigurasi ──────────────────────────────────────────────────
API_REGISTER = "https://boostify-back-end.vercel.app/api/auth/register"

# Format: nama → (assisstant_code 3 huruf, password awal)
# Password awal sebaiknya diganti user setelah login pertama.
USERS = {
    "daffa" : ("FDR", "daffa123"),
    "dirgi" : ("DRG", "dirgi123"),
    "daffaf": ("DYR", "daffaf123"),
    # tambah orang baru di sini:
    # "nama_baru": ("KODE", "password_awal"),
}


def register_user(nama: str, kode: str, password: str) -> bool:
    """Register satu user ke backend."""
    payload = {
        "name"           : nama,
        "assisstant_code": kode,
        "password"       : password,
    }

    try:
        response = requests.post(
            API_REGISTER,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )

        if response.status_code == 200:
            data = response.json().get("regis", {})
            if data.get("status"):
                print(f"  ✅ {nama:10s} ({kode}) — terdaftar")
                return True
            else:
                print(f"  ⚠️  {nama:10s} ({kode}) — {data.get('message')}")
                return False
        else:
            # Kemungkinan sudah pernah didaftarkan (unique constraint)
            error_msg = response.json().get("error", "Unknown error")
            print(f"  ⚠️  {nama:10s} ({kode}) — {error_msg}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"  ❌ {nama:10s} — gagal konek: {e}")
        return False


def main():
    print("🚀 Mulai daftarin user ke backend Boostify...\n")
    print(f"   Total user: {len(USERS)}\n")

    sukses = 0
    gagal = 0

    for nama, (kode, password) in USERS.items():
        if register_user(nama, kode, password):
            sukses += 1
        else:
            gagal += 1

    print(f"\n{'─' * 50}")
    print(f"  Sukses: {sukses} | Gagal/skip: {gagal}")
    print(f"{'─' * 50}\n")

    if sukses > 0:
        print("✅ User berhasil terdaftar.")
        print("   Mereka sudah bisa:")
        print("   • Absen via face recognition (ML)")
        print("   • Login di https://boostify-fe.vercel.app")
        print("\n   Login pakai assisstant_code (bukan nama) + password.\n")


if __name__ == "__main__":
    main()