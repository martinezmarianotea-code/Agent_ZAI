"""
agent.py
========
Núcleo de IA de ZAI: orquesta Claude (Anthropic) + Ollama (local) en paralelo.

Estrategia híbrida:
  - Ollama (qwen2.5-coder:7b, local, gratis, rápido):
      * Pre-análisis rápido: detecta palabras en español, errores obvios
      * Extracción masiva de palabras nuevas
      * Búsquedas rápidas sin costo
  - Claude claude-opus-4-6 (streaming, thinking adaptivo):
      * Correcciones profundas con contexto español
      * Modo chat conversacional
      * Análisis lingüístico complejo
"""

from __future__ import annotations
import json
import logging
import threading
import time
from typing import Callable, Optional
import requests

import anthropic
from anthropic import RateLimitError, APIConnectionError, APIStatusError

from zai.config import CLAUDE_MODEL, OLLAMA_MODEL, OLLAMA_URL
from zai.context import ContextoLinguistico

logger = logging.getLogger(__name__)


# ── Agente principal ─────────────────────────────────────────────────

class ZAIAgent:
    """
    Agente híbrido Claude + Ollama.
    Todos los métodos de Claude usan streaming — el caller recibe
    un iterador de chunks de texto.
    """

    def __init__(self, contexto: ContextoLinguistico):
        self.ctx = contexto
        self._claude = anthropic.Anthropic()
        self._ollama_ok = self._verificar_ollama()

        # System prompt completo (estable → prompt caching)
        self._system_corrector = contexto.system_prompt()
        self._system_chat      = contexto.system_prompt_chat()

    # ── Verificación de Ollama ───────────────────────────────────────

    def _verificar_ollama(self) -> bool:
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    @property
    def ollama_disponible(self) -> bool:
        return self._ollama_ok

    # ════════════════════════════════════════════════════════════════
    #  CLAUDE — Corrector de Transcripciones (streaming)
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
        """
        Lanza la corrección de una fila en un thread separado.
        Llama on_chunk(texto) por cada token recibido.
        Llama on_done(texto_completo) al terminar.
        """
        def _run():
            completo = []
            prompt_usuario = _prompt_correccion(zapoteco, espanol, n, total)
            mensajes = historial + [{"role": "user", "content": prompt_usuario}]

            for intento in range(3):
                try:
                    with self._claude.messages.stream(
                        model=CLAUDE_MODEL,
                        max_tokens=2048,
                        thinking={"type": "adaptive"},
                        system=[{
                            "type": "text",
                            "text": self._system_corrector,
                            "cache_control": {"type": "ephemeral"},
                        }],
                        messages=mensajes,
                    ) as stream:
                        for text in stream.text_stream:
                            completo.append(text)
                            on_chunk(text)
                    on_done("".join(completo))
                    return
                except RateLimitError:
                    espera = 2 ** intento
                    logger.warning("Claude rate limit, reintentando en %ss (intento %d/3)", espera, intento + 1)
                    on_chunk(f"\n[Límite de tasa alcanzado, esperando {espera}s…]\n")
                    time.sleep(espera)
                    completo.clear()
                except APIConnectionError as e:
                    logger.error("Error de conexión con Claude: %s", e)
                    msg = "\n[Sin conexión con Claude — verifica tu red o la API key]\n"
                    on_chunk(msg)
                    on_done(msg)
                    return
                except APIStatusError as e:
                    logger.error("Claude API error %s: %s", e.status_code, e.message)
                    msg = f"\n[Error de API ({e.status_code}): {e.message}]\n"
                    on_chunk(msg)
                    on_done(msg)
                    return
                except Exception as e:
                    logger.exception("Error inesperado en corregir_streaming")
                    msg = f"\n[Error inesperado: {e}]\n"
                    on_chunk(msg)
                    on_done(msg)
                    return

            msg = "\n[No se pudo conectar con Claude tras 3 intentos]\n"
            on_chunk(msg)
            on_done(msg)

        threading.Thread(target=_run, daemon=True).start()

    # ════════════════════════════════════════════════════════════════
    #  CLAUDE — Chat (streaming)
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
            completo = []
            mensajes = historial + [{"role": "user", "content": mensaje}]
            for intento in range(3):
                try:
                    with self._claude.messages.stream(
                        model=CLAUDE_MODEL,
                        max_tokens=1024,
                        system=[{
                            "type": "text",
                            "text": self._system_chat,
                            "cache_control": {"type": "ephemeral"},
                        }],
                        messages=mensajes,
                    ) as stream:
                        for text in stream.text_stream:
                            completo.append(text)
                            on_chunk(text)
                    on_done("".join(completo))
                    return
                except RateLimitError:
                    espera = 2 ** intento
                    logger.warning("Claude rate limit en chat, reintentando en %ss", espera)
                    time.sleep(espera)
                    completo.clear()
                except APIConnectionError as e:
                    logger.error("Error de conexión con Claude (chat): %s", e)
                    msg = "[Sin conexión con Claude]"
                    on_chunk(msg)
                    on_done(msg)
                    return
                except Exception as e:
                    logger.exception("Error inesperado en chat_streaming")
                    msg = f"[Error: {e}]"
                    on_chunk(msg)
                    on_done(msg)
                    return

            on_done("[No se pudo conectar con Claude tras 3 intentos]")

        threading.Thread(target=_run, daemon=True).start()

    # ════════════════════════════════════════════════════════════════
    #  OLLAMA — Pre-análisis rápido (síncrono, local)
    # ════════════════════════════════════════════════════════════════

    def preanalizar(self, zapoteco: str, espanol: str) -> dict:
        """
        Análisis rápido con Ollama antes de enviar a Claude.
        Retorna dict con: palabras_sospechosas, en_espanol, observaciones.
        Falla silenciosamente si Ollama no está disponible.
        """
        if not self._ollama_ok:
            return {}

        prompt = f"""\
Analiza este texto en Zapoteco del Istmo y su traducción española.
ZAPOTECO: {zapoteco}
ESPAÑOL:  {espanol}

Responde SOLO con JSON (sin markdown, sin explicación):
{{
  "palabras_espanol": ["lista", "de", "palabras", "que", "parecen", "español"],
  "palabras_dudosas": ["palabras", "que", "podrían", "tener", "error"],
  "observacion": "una sola línea con el problema principal si lo hay"
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
                timeout=10,
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

    def extraer_palabras_nuevas_batch(
        self,
        tokens: list[str],
        diccionario: set[str],
        on_progreso: Optional[Callable[[int, int], None]] = None,
    ) -> list[dict]:
        """
        Usa Ollama para analizar tokens desconocidos en lotes.
        Retorna lista de {zapoteco, espanol, es_nueva}.
        """
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
                    timeout=20,
                )
                content = _extraer_contenido_ollama(r)
                if content:
                    inicio = content.find("[")
                    fin    = content.rfind("]") + 1
                    if inicio >= 0 and fin > inicio:
                        items = json.loads(content[inicio:fin])
                        resultados.extend(items)
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
    #  CLAUDE — Generar traducciones para palabras nuevas
    # ════════════════════════════════════════════════════════════════

    def sugerir_traduccion(
        self,
        palabras: list[str],
        on_chunk: Callable[[str], None],
        on_done:  Callable[[str], None],
    ) -> None:
        """Pide a Claude que sugiera traducciones para palabras nuevas."""
        def _run():
            completo = []
            lista = "\n".join(f"- {p}" for p in palabras[:40])
            prompt = (
                f"Las siguientes palabras fueron encontradas en transcripciones "
                f"de Zapoteco del Istmo y NO están en el diccionario.\n"
                f"Para cada una, sugiere su posible significado en español "
                f"(basándote en similitud con palabras conocidas, morfología, o contexto).\n"
                f"Formato: `palabra → significado posible (nivel de certeza: alto/medio/bajo)`\n\n"
                f"{lista}"
            )
            try:
                with self._claude.messages.stream(
                    model=CLAUDE_MODEL,
                    max_tokens=1500,
                    system=[{
                        "type": "text",
                        "text": self._system_corrector,
                        "cache_control": {"type": "ephemeral"},
                    }],
                    messages=[{"role": "user", "content": prompt}],
                ) as stream:
                    for text in stream.text_stream:
                        completo.append(text)
                        on_chunk(text)
                on_done("".join(completo))
            except RateLimitError:
                logger.warning("Claude rate limit en sugerir_traduccion")
                msg = "[Límite de tasa alcanzado — intenta en unos segundos]"
                on_chunk(msg)
                on_done(msg)
            except APIConnectionError as e:
                logger.error("Sin conexión con Claude en sugerir_traduccion: %s", e)
                msg = "[Sin conexión con Claude]"
                on_chunk(msg)
                on_done(msg)
            except Exception as e:
                logger.exception("Error inesperado en sugerir_traduccion")
                msg = f"[Error: {e}]"
                on_chunk(msg)
                on_done(msg)

        threading.Thread(target=_run, daemon=True).start()

    # ════════════════════════════════════════════════════════════════
    #  OLLAMA — Consulta rápida (síncrona, para la GUI)
    # ════════════════════════════════════════════════════════════════

    def consulta_rapida(self, pregunta: str) -> str:
        """Respuesta rápida con Ollama (para sugerencias inline, completado, etc.)."""
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
                timeout=8,
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

def _extraer_contenido_ollama(response: requests.Response) -> str:
    """Extrae el texto de respuesta de Ollama de forma robusta."""
    try:
        data = response.json()
    except json.JSONDecodeError:
        logger.warning("Respuesta de Ollama no es JSON válido")
        return ""
    # Formato /api/chat: {"message": {"content": "..."}}
    if "message" in data:
        return str(data["message"].get("content", "")).strip()
    # Formato /api/generate: {"response": "..."}
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
