"""
config.py
=========
Configuración central de ZAI. Rutas, modelos, parámetros.

Variables de entorno reconocidas:
  ZAI_DATASET      — ruta al DataSet.xlsx
  ZAI_DICCIONARIO  — ruta al Diccionario.xlsx
  ZAI_OLLAMA_URL   — URL de Ollama (default: http://localhost:11434)
  ZAI_OLLAMA_MODEL — modelo Ollama a usar
  ZAI_CLAUDE_MODEL — modelo Claude a usar
  ZAI_LOG_LEVEL    — nivel de log (DEBUG/INFO/WARNING, default: INFO)
"""

from pathlib import Path
import json
import logging
import logging.handlers
import os

# ── Rutas base ──────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent          # Correct/
DATA_DIR    = ROOT / "data"

DATASET_PATH     = Path(os.environ.get("ZAI_DATASET",     str(DATA_DIR / "DataSet.xlsx")))
DICCIONARIO_PATH = Path(os.environ.get("ZAI_DICCIONARIO", str(DATA_DIR / "Diccionario 2026.xlsx")))
PDF_FONÉTICA     = DATA_DIR / "APUNTES DE ZAPOTECO ACTUAL.pdf"
PDF_VOCAB        = DATA_DIR / "DICCIONARIO CORREGIDO 14.pdf"

CONFIG_FILE  = Path.home() / ".zai_config.json"
MODELO_DIR   = DATA_DIR / "modelo_zapoteco"

# ── Nombres de hojas del DataSet ─────────────────────────────────────
HOJA_ZAPOTECO      = "transcripciones"
HOJA_ESPANOL       = "traducciones"
HOJA_PALABRAS      = "palabras_nuevas"

# ── Modelos (sobrescribibles con env vars) ────────────────────────────
CLAUDE_MODEL  = os.environ.get("ZAI_CLAUDE_MODEL", "claude-opus-4-6")
OLLAMA_MODEL  = os.environ.get("ZAI_OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_URL    = os.environ.get("ZAI_OLLAMA_URL",   "http://localhost:11434")

# ── Parámetros de corrección ─────────────────────────────────────────
AUTOGUARDADO_INTERVALO = 50   # filas


# ── Logging ──────────────────────────────────────────────────────────

def configurar_logging() -> None:
    """Configura el sistema de logging para toda la app."""
    log_dir = Path.home() / ".zai"
    log_dir.mkdir(exist_ok=True)

    nivel_str = os.environ.get("ZAI_LOG_LEVEL", "INFO").upper()
    nivel = getattr(logging, nivel_str, logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Archivo rotativo (5 MB × 3 backups)
    fh = logging.handlers.RotatingFileHandler(
        log_dir / "zai.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setFormatter(fmt)

    # Consola
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    root = logging.getLogger("zai")
    root.setLevel(nivel)
    if not root.handlers:
        root.addHandler(fh)
        root.addHandler(ch)


# ── Configuración persistente ────────────────────────────────────────

def cargar_config() -> dict:
    """Lee configuración persistente del usuario."""
    defaults = {
        "dataset":     str(DATASET_PATH),
        "diccionario": str(DICCIONARIO_PATH),
        "ollama_model": OLLAMA_MODEL,
        "claude_model": CLAUDE_MODEL,
        "modo_inicio":  "corrector",
    }
    if CONFIG_FILE.exists():
        try:
            saved = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                defaults.update(saved)
        except json.JSONDecodeError:
            logging.getLogger("zai.config").warning(
                "Archivo de configuración corrupto, usando valores por defecto"
            )
    return defaults


def guardar_config(cfg: dict) -> None:
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
