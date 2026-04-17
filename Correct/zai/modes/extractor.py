"""
modes/extractor.py
==================
Extrae palabras nuevas del dataset usando pipeline híbrido:
  1. Tokenización + dedup (Python puro, instantáneo)
  2. Pre-filtro rápido: Ollama clasifica tokens en lotes
  3. Claude sugiere traducciones para las candidatas
"""

from __future__ import annotations
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional

from zai.config import OLLAMA_MODEL
from zai.excel import GestorDataset, PalabraNueva
from zai.agent import ZAIAgent
from zai.context import ContextoLinguistico
from zai.modelo.inferencia import ModeloZapoteco


# ── Token helpers ────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[\w'\u00C0-\u00FF]+", re.UNICODE)

def _tokenizar(texto: str) -> list[str]:
    return _TOKEN_RE.findall(texto.lower())

def _es_valido(tok: str) -> bool:
    if len(tok) < 2:
        return False
    if tok.isdigit():
        return False
    if not any(c.isalpha() for c in tok):
        return False
    return True


# ── Resultado ────────────────────────────────────────────────────────

@dataclass
class ResultadoExtraccion:
    total_tokens:    int
    tokens_unicos:   int
    en_diccionario:  int
    candidatos:      list[str]             # tokens no encontrados en dic
    clasificados:    list[dict]            # resultado de Ollama
    palabras_nuevas: list[PalabraNueva]    # aprobadas para agregar al dic
    sugerencias_ia:  str = ""              # respuesta de Claude con traducciones


# ── Modo Extractor ───────────────────────────────────────────────────

class ModoExtractor:

    def __init__(
        self,
        dataset:   GestorDataset,
        agente:    ZAIAgent,
        contexto:  ContextoLinguistico,
    ):
        self.ds      = dataset
        self.agente  = agente
        self.ctx     = contexto
        self.resultado: Optional[ResultadoExtraccion] = None
        # NanoGPT local (si ya fue entrenado)
        self.nano = ModeloZapoteco.cargar()

    # ── Pipeline ─────────────────────────────────────────────────────

    def analizar_dataset(
        self,
        on_progreso: Callable[[str], None],
    ) -> ResultadoExtraccion:
        """
        Fase 1: extracción + clasificación local (síncrona).
        Llama on_progreso(mensaje) en cada paso.
        """
        on_progreso("Tokenizando transcripciones…")

        # Recolectar todos los tokens
        todos: list[str] = []
        for fila in self.ds.filas:
            todos.extend(_tokenizar(fila.texto_vigente))

        total_tokens  = len(todos)
        frecuencias   = Counter(todos)
        unicos        = [t for t, _ in frecuencias.most_common() if _es_valido(t)]
        tokens_unicos = len(unicos)

        on_progreso(f"{total_tokens:,} tokens → {tokens_unicos:,} únicos")

        # Separar: en diccionario vs candidatos
        en_dic = sum(1 for t in unicos if self.ctx.esta_en_diccionario(t))
        candidatos = [t for t in unicos if not self.ctx.esta_en_diccionario(t)]

        on_progreso(f"En diccionario: {en_dic:,}  |  Candidatos nuevos: {len(candidatos):,}")

        # Fase 2: Clasificación (NanoGPT > Ollama > heurística)
        clasificados: list[dict] = []

        if self.nano.disponible and candidatos:
            on_progreso(f"Clasificando con NanoGPT zapoteco ({self.nano.info()})…")
            clasificados = self.nano.clasificar_batch(candidatos)
            on_progreso(f"  NanoGPT procesó {len(candidatos):,} tokens")

        elif self.agente.ollama_disponible and candidatos:
            on_progreso(f"Clasificando con Ollama ({OLLAMA_MODEL})…")
            clasificados = self.agente.extraer_palabras_nuevas_batch(
                tokens     = candidatos,
                diccionario= self.ctx._dic_set,
                on_progreso= lambda i, t: on_progreso(f"Clasificando… {i}/{t}"),
            )

        # Filtrar: solo los que dijeron "zapoteca"
        if clasificados:
            palabras_nuevas_raw = [
                d["palabra"]
                for d in clasificados
                if d.get("tipo") == "zapoteca" and d.get("palabra")
            ]
        else:
            # Sin modelos locales: heurística básica
            on_progreso("Usando filtro heurístico (NanoGPT no entrenado aún)…")
            palabras_nuevas_raw = _filtro_heuristico(candidatos)

        # Construir objetos PalabraNueva (sin traducción todavía)
        palabras_nuevas = [
            PalabraNueva(zapoteco=p, espanol="", fuente="extracción")
            for p in palabras_nuevas_raw
        ]

        on_progreso(f"Palabras nuevas candidatas: {len(palabras_nuevas):,}")

        self.resultado = ResultadoExtraccion(
            total_tokens   = total_tokens,
            tokens_unicos  = tokens_unicos,
            en_diccionario = en_dic,
            candidatos     = candidatos,
            clasificados   = clasificados,
            palabras_nuevas= palabras_nuevas,
        )
        return self.resultado

    def pedir_traducciones_claude(
        self,
        palabras: list[str],
        on_chunk: Callable[[str], None],
        on_done:  Callable[[str], None],
    ) -> None:
        """Fase 3 (opcional): Claude sugiere traducciones. Streaming."""
        self.agente.sugerir_traduccion(palabras, on_chunk, on_done)

    def guardar_seleccionadas(self, palabras: list[PalabraNueva]) -> int:
        return self.ds.guardar_batch_palabras(palabras)


# ── Filtro heurístico (fallback sin Ollama) ──────────────────────────

_PATRONES_ZAPOTECO = re.compile(
    r"(dx|tx|ch|zh|rr|xh|xp|x[aeiou]|'|inni|ube|endá|ani|ulu|iza"
    r"|[áéíóú]|[aeiou]{2})",
    re.IGNORECASE,
)

_SUFIJOS_ESPANOL = re.compile(
    r"(ción|mente|able|ible|ivo|ismo|ista|ador|ando|ando|ado|era|ero)$",
    re.IGNORECASE,
)

_ESPANOL_COMUN = {
    "que", "de", "la", "el", "en", "y", "es", "un", "una", "los", "las",
    "me", "te", "se", "con", "por", "para", "no", "si", "le", "ya",
    "como", "pero", "muy", "más", "bien", "todo", "esto", "eso",
}

def _filtro_heuristico(tokens: list[str]) -> list[str]:
    resultado = []
    for tok in tokens:
        if tok in _ESPANOL_COMUN:
            continue
        if _SUFIJOS_ESPANOL.search(tok):
            continue
        if not _PATRONES_ZAPOTECO.search(tok):
            continue
        resultado.append(tok)
    return resultado


