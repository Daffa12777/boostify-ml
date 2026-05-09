"""
utils/logger.py — Setup Logger Boostify ML
Fix: Support UTF-8 + emoji di Windows PowerShell
"""

import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LOG_FILE, LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    """
    Buat logger dengan format yang konsisten.
    Semua log ditulis ke console DAN ke file log.
    Support emoji dan karakter UTF-8 di Windows.
    """
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))

    if logger.handlers:
        return logger  # hindari duplikasi handler

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ── Handler ke console (fix UTF-8 Windows) ──
    try:
        # Windows: paksa stdout pakai UTF-8
        console_stream = open(
            sys.stdout.fileno(),
            mode='w',
            encoding='utf-8',
            buffering=1,
            closefd=False
        )
        ch = logging.StreamHandler(console_stream)
    except Exception:
        # Fallback kalau gagal (Linux/Mac tidak butuh ini)
        ch = logging.StreamHandler(sys.stdout)

    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # ── Handler ke file (UTF-8) ──
    fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger