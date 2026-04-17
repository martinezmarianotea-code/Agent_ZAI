"""
agent.py
========
Núcleo de IA de ZAI: usa Ollama 100% local, sin APIs de pago.

Modelos:
  - OLLAMA_MODEL_MAIN  (qwen3:4b)         — correcciones, chat, traducciones
  - OLLAMA_MODEL_FAST  (qwen2.5-coder:7b) — clasificación batch rápida
"""

from __future__ import annotations
import json
import logging
import threading
from typing import Callable, Optional
import requests

from zai.config import OLLAMA_MODEL, OLLAMA_MODEL_MAIN, OLLAMA_URL
from zai.context import ContextoLinguistico

logger = logging.getLogger(__name__)


# ── Agente principal ─────────────────────────────────────────────────

class ZAIAgent:
    """Agente 100% local usando Ollama."""

    def __init__(self, contexto: ContextoLinguistico):
        self.ctx = contexto
        self._ollama_ok = self._verificar_ollama()
        self._system_corrector = contexto.system_prompt()
        self._system_chat      = contexto.system_prompt_chat()

    # ── Verificación ─────────────────────────────────────────────────

    def _verificar_ollama(self) -> bool:
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    @property
    def ollama_disponible(self) -> bool:
        return self._ollama_ok

    # ════════════════════════════════════════════════════════════════
    #  OLLAMA — Corrector de Transcripciones (streaming)
    # ════════════════════════════════════════════════════════════════

    def corregir_streaming(
        self,
        zapoteco:  str,
        espanol:   str,
        n:         int,
        total:     int,
        historial: list[dict],
        on_chunk:  Callable[[str], None],
        on_done:   Callable[[str], None],
    ) -> None:
        """Corrección en streaming via Ollama."""
        def _run():
            if not self._ollama_ok:
                msg = "[Ollama no disponible — ejecuta: ollama serve]"
                on_chunk(msg)
                on_done(msg)
                return

            prompt_usuario = _prompt_correccion(zapoteco, espanol, n, total)
            mensajes = (
                [{"role": "system", "content": self._system_corrector}]
                + historial
                + [{"role": "user", "content": prompt_usuario}]
            )
            _stream_ollama(OLLAMA_MODEL_MAIN, mensajes, on_chunk, on_done)

        threading.Thread(target=_run, daemon=True).start()

    # ════════════════════════════════════════════════════════════════
    #  OLLAMA — Chat (streaming)
    # ════════════════════════════════════════════════════════════════

    def chat_streaming(
        self,
        mensaje:   str,
        historial: list[dict],
        on_chunk:  Callable[[str], None],
        on_done:   Callable[[str], None],
    ) -> None:
        """Chat libre sobre zapoteco. Streaming."""
        def _run():
            if not self._ollama_ok:
                msg = "[Ollama no disponible — ejecuta: ollama serve]"
                on_chunk(msg)
                on_done(msg)
                return

            mensajes = (
                [{"role": "system", "content": self._system_chat}]
                + historial
                + [{"role": "user", "content": mensaje}]
            )
            _stream_ollama(OLLAMA_MODEL_MAIN, mensajes, on_chunk, on_done)

        threading.Thread(target=_run, daemon=True).start()

    # ════════════════════════════════════════════════════════════════
    #  OLLAMA — Pre-análisis rápido (síncrono)
    # ════════════════════════════════════════════════════════════════

    def preanalizar(self, zapoteco: str, espanol: str) -> dict:
        """Análisis rápido antes de la corrección principal."""
        if not self._ollama_ok:
            return {}

        prompt = f"""\
Analiza este texto en Zapoteco del Istmo y su traducción española.
ZAPOTECO: {zapoteco}
ESPAÑOL:  {espanol}

Responde SOLO con JSON (sin markdown, sin explicación):
{{
  "palabras_espanol": ["palabras que parecen español"],
  "palabras_dudosas": ["palabras con posible error"],
  "observacion": "problema principal en una línea"
}}"""

        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 200},
                },
                timeout=30,
            )
            content = _extraer_contenido_ollama(r)
            if not content:
                return {}
            inicio = content.find("{")
            fin    = content.rfind("}") + 1
            if inicio >= 0 and fin > inicio:
                return json.loads(content[inicio:fin])
        except requests.Timeout:
            logger.warning("Ollama timeout en preanalizar")
        except requests.ConnectionError:
            logger.warning("Ollama no disponible en preanalizar")
            self._ollama_ok = False
        except json.JSONDecodeError as e:
            logger.warning("JSON inválido de Ollama en preanalizar: %s", e)
        except Exception:
            logger.exception("Error inesperado en preanalizar")
        return {}

    # ════════════════════════════════════════════════════════════════
    #  OLLAMA — Extracción batch de palabras nuevas
    # ════════════════════════════════════════════════════════════════

    def extraer_palabras_nuevas_batch(
        self,
        tokens: list[str],
        diccionario: set[str],
        on_progreso: Optional[Callable[[int, int], None]] = None,
    ) -> list[dict]:
        """Clasifica tokens desconocidos en lotes usando Ollama."""
        if not self._ollama_ok:
            return []

        desconocidos = [t for t in tokens if t.lower() not in diccionario and len(t) > 2]
        if not desconocidos:
            return []

        LOTE = 30
        resultados = []

        for i in range(0, len(desconocidos), LOTE):
            lote = desconocidos[i:i + LOTE]
            if on_progreso:
                on_progreso(i, len(desconocidos))

            prompt = f"""\
Estas palabras fueron encontradas en transcripciones de Zapoteco del Istmo.
Clasifica cada una. Responde SOLO con JSON array (sin markdown):
[
  {{"palabra": "...", "tipo": "zapoteca|española|nombre_propio|desconocida", "posible_significado": "..."}}
]
Palabras: {json.dumps(lote, ensure_ascii=False)}"""

            try:
                r = requests.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model": OLLAMA_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "options": {"temperature": 0.1, "num_predict": 600},
                    },
                    timeout=60,
                )
                content = _extraer_contenido_ollama(r)
                if content:
                    inicio = content.find("[")
                    fin    = content.rfind("]") + 1
                    if inicio >= 0 and fin > inicio:
                        resultados.extend(json.loads(content[inicio:fin]))
            except requests.Timeout:
                logger.warning("Ollama timeout en lote %d-%d", i, i + LOTE)
            except requests.ConnectionError:
                logger.warning("Ollama no disponible durante extracción batch")
                self._ollama_ok = False
                break
            except json.JSONDecodeError as e:
                logger.warning("JSON inválido de Ollama en lote %d: %s", i, e)
            except Exception:
                logger.exception("Error inesperado en extraer_palabras_nuevas_batch")

        return resultados

    # ════════════════════════════════════════════════════════════════
    #  OLLAMA — Sugerir traducciones (streaming)
    # ════════════════════════════════════════════════════════════════

    def sugerir_traduccion(
        self,
        palabras: list[str],
        on_chunk: Callable[[str], None],
        on_done:  Callable[[str], None],
    ) -> None:
        """Sugiere traducciones para palabras nuevas. Streaming."""
        def _run():
            if not self._ollama_ok:
                msg = "[Ollama no disponible]"
                on_chunk(msg)
                on_done(msg)
                return

            lista = "\n".join(f"- {p}" for p in palabras[:40])
            prompt = (
                f"Las siguientes palabras fueron encontradas en transcripciones "
                f"de Zapoteco del Istmo y NO están en el diccionario.\n"
                f"Para cada una, sugiere su posible significado en español "
                f"(morfología, similitud con palabras conocidas, contexto).\n"
                f"Formato: `palabra → significado posible (certeza: alta/media/baja)`\n\n"
                f"{lista}"
            )
            mensajes = [
                {"role": "system", "content": self._system_corrector},
                {"role": "user",   "content": prompt},
            ]
            _stream_ollama(OLLAMA_MODEL_MAIN, mensajes, on_chunk, on_done)

        threading.Thread(target=_run, daemon=True).start()

    # ════════════════════════════════════════════════════════════════
    #  OLLAMA — Consulta rápida (síncrona)
    # ════════════════════════════════════════════════════════════════

    def consulta_rapida(self, pregunta: str) -> str:
        """Respuesta rápida con Ollama para sugerencias inline."""
        if not self._ollama_ok:
            return ""
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": pregunta}],
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 150},
                },
                timeout=15,
            )
            return _extraer_contenido_ollama(r)
        except requests.Timeout:
            logger.warning("Ollama timeout en consulta_rapida")
        except requests.ConnectionError:
            logger.warning("Ollama no disponible en consulta_rapida")
            self._ollama_ok = False
        except Exception:
            logger.exception("Error inesperado en consulta_rapida")
        return ""


# ── Helpers internos ─────────────────────────────────────────────────

def _stream_ollama(
    model: str,
    mensajes: list[dict],
    on_chunk: Callable[[str], None],
    on_done:  Callable[[str], None],
) -> None:
    """Llama a /api/chat con stream=True y envía chunks al caller."""
    completo: list[str] = []
    try:
        with requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": mensajes,
                "stream": True,
                "options": {"temperature": 0.3, "num_predict": 2048},
            },
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for linea in resp.iter_lines():
                if not linea:
                    continue
                try:
                    data = json.loads(linea)
                except json.JSONDecodeError:
                    continue
                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    completo.append(chunk)
                    on_chunk(chunk)
                if data.get("done"):
                    break
        on_done("".join(completo))
    except requests.Timeout:
        msg = "\n[Ollama tardó demasiado — intenta con un modelo más pequeño]\n"
        on_chunk(msg)
        on_done(msg)
    except requests.ConnectionError:
        msg = "\n[Ollama no disponible — ejecuta: ollama serve]\n"
        on_chunk(msg)
        on_done(msg)
    except Exception:
        logger.exception("Error en _stream_ollama")
        on_chunk("\n[Error al comunicarse con Ollama]\n")
        on_done("")


def _extraer_contenido_ollama(response: requests.Response) -> str:
    """Extrae el texto de una respuesta Ollama no-streaming."""
    try:
        data = response.json()
    except json.JSONDecodeError:
        logger.warning("Respuesta de Ollama no es JSON válido")
        return ""
    if "message" in data:
        return str(data["message"].get("content", "")).strip()
    if "response" in data:
        return str(data["response"]).strip()
    logger.warning("Formato de respuesta Ollama desconocido: %s", list(data.keys()))
    return ""


# ── Helpers de prompt ────────────────────────────────────────────────

def _prompt_correccion(zapoteco: str, espanol: str, n: int, total: int) -> str:
    esp_linea = f"ESPAÑOL (traducción): {espanol}" if espanol else "ESPAÑOL: (sin traducción disponible)"
    return (
        f"FILA {n + 1} de {total}\n\n"
        f"ZAPOTECO (transcripción): {zapoteco}\n"
        f"{esp_linea}\n\n"
        "Analiza y corrige esta transcripción según las instrucciones."
    )
