"""
context.py
==========
Construye el contexto lingüístico para el agente:
  - Extrae texto de los PDFs (reglas fonéticas + vocabulario histórico)
  - Carga el diccionario completo
  - Genera el system prompt con prompt caching
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import openpyxl

try:
    from pypdf import PdfReader
    _HAS_PYPDF = True
except ImportError:
    _HAS_PYPDF = False


# ── Extracción de PDFs ───────────────────────────────────────────────

def _leer_pdf(ruta: Path, max_paginas: int = 30) -> str:
    if not _HAS_PYPDF or not ruta.exists():
        return ""
    try:
        reader = PdfReader(str(ruta))
        partes = []
        for i, page in enumerate(reader.pages):
            if i >= max_paginas:
                break
            texto = page.extract_text()
            if texto:
                partes.append(texto)
        return "\n".join(partes)
    except Exception:
        return ""


# ── Diccionario ──────────────────────────────────────────────────────

def _cargar_diccionario(ruta: Path) -> list[tuple[str, str]]:
    if not ruta.exists():
        return []
    wb = openpyxl.load_workbook(str(ruta), read_only=True)
    ws = wb.active
    entradas = []
    for fila in ws.iter_rows(min_row=1, values_only=True):
        zap = str(fila[0]).strip() if fila[0] else ""
        esp = str(fila[1]).strip() if len(fila) > 1 and fila[1] else ""
        if zap and esp and zap != "ZAPOTECO":
            entradas.append((zap, esp))
    wb.close()
    return entradas


# ── Context Manager ──────────────────────────────────────────────────

class ContextoLinguistico:
    """
    Carga y expone el contexto lingüístico completo.
    Construye el system prompt para Ollama.
    """

    def __init__(
        self,
        pdf_fonetica: Path,
        pdf_vocab:    Path,
        diccionario:  Path,
    ):
        self._pdf_fonetica = pdf_fonetica
        self._pdf_vocab    = pdf_vocab
        self._dic_path     = diccionario

        self.texto_fonetica: str = ""
        self.texto_vocab:    str = ""
        self.diccionario:    list[tuple[str, str]] = []
        self._dic_set:       set[str] = set()

    def cargar(self) -> "ContextoLinguistico":
        """Carga todos los recursos. Retorna self para encadenamiento."""
        print("[Contexto] Leyendo reglas fonéticas...", end=" ", flush=True)
        self.texto_fonetica = _leer_pdf(self._pdf_fonetica, max_paginas=25)
        print(f"{len(self.texto_fonetica):,} chars")

        print("[Contexto] Leyendo vocabulario histórico...", end=" ", flush=True)
        self.texto_vocab = _leer_pdf(self._pdf_vocab, max_paginas=20)
        print(f"{len(self.texto_vocab):,} chars")

        print("[Contexto] Cargando diccionario...", end=" ", flush=True)
        self.diccionario = _cargar_diccionario(self._dic_path)
        self._dic_set = {e[0].lower() for e in self.diccionario}
        print(f"{len(self.diccionario):,} entradas")
        return self

    # ── Consultas rápidas ────────────────────────────────────────────

    def esta_en_diccionario(self, palabra: str) -> bool:
        return palabra.lower().strip() in self._dic_set

    def buscar(self, palabra: str) -> Optional[str]:
        palabra_l = palabra.lower().strip()
        for zap, esp in self.diccionario:
            if zap.lower() == palabra_l:
                return esp
        return None

    def dic_como_texto(self, limite: int = 9000) -> str:
        """Diccionario como texto compacto para el prompt."""
        lineas = [f"{z} = {e}" for z, e in self.diccionario[:limite]]
        return "\n".join(lineas)

    # ── System Prompt ────────────────────────────────────────────────

    def system_prompt(self) -> str:
        """
        System prompt completo para Claude.
        Incluye reglas fonéticas, vocabulario histórico y diccionario.
        Diseñado para prompt caching: contenido estable al inicio.
        """
        dic_txt = self.dic_como_texto()

        secciones = [
            _IDENTIDAD,
            "# REGLAS FONÉTICAS Y ORTOGRÁFICAS\n" + (self.texto_fonetica or _REGLAS_BASICAS),
        ]

        if self.texto_vocab:
            secciones.append("# VOCABULARIO HISTÓRICO (referencia)\n" + self.texto_vocab[:4000])

        secciones.append(
            f"# DICCIONARIO ZAPOTECO-ESPAÑOL ({len(self.diccionario):,} entradas)\n"
            + dic_txt
        )

        secciones.append(_INSTRUCCIONES_CORRECTOR)

        return "\n\n" + "\n\n---\n\n".join(secciones) + "\n"

    def system_prompt_chat(self) -> str:
        """System prompt para el modo chat (más conversacional)."""
        dic_txt = self.dic_como_texto()
        return (
            _IDENTIDAD + "\n\n"
            + "# REGLAS FONÉTICAS\n" + (self.texto_fonetica or _REGLAS_BASICAS) + "\n\n"
            + f"# DICCIONARIO ({len(self.diccionario):,} entradas)\n" + dic_txt + "\n\n"
            + _INSTRUCCIONES_CHAT
        )


# ── Textos constantes ────────────────────────────────────────────────

_IDENTIDAD = """\
# IDENTIDAD
Eres ZAI, un agente lingüístico especializado en el Zapoteco del Istmo (Diidxazá).
Tienes conocimiento profundo de:
- Fonología y ortografía del zapoteco del Istmo
- El diccionario completo Zapoteco-Español
- Vocabulario histórico y arcaico (fuente: Ricardo Santos Velázquez, 2024)
- Errores comunes de transcriptores al escribir zapoteco

Tu personalidad: amigable, paciente, pedagógico. Explicas el "por qué" de cada corrección.
Usas español para comunicarte con el usuario, pero citas las palabras zapotecas correctamente."""

_REGLAS_BASICAS = """\
Alfabeto actual: a b c ch d dx e g gu gü h hui j l ll m mb n nc nd ng nn ñ o p q qu r rr s t u x xh z y

TIPOS DE VOCALES:
1. Vocales sencillas (a,e,i,o,u): pronunciación normal con cuerdas vocales abiertas
2. Vocales cortadas (a',e',i',o',u'): vocal cortada con apóstrofo DESPUÉS de la vocal
3. Vocales quebradas (aa,ee,ii,oo,uu): dos vocales idénticas, sonido más largo/amplio

CONSONANTES:
- Suaves: b, d, dx, g/gu/gü, j, x, z, n, l, r, y
- Fuertes: p, t, ch, c/qu, xh, s, nn, ll, rr, ñ, m, nd, hu, ng, mb, nc

Errores comunes:
- Omitir el apóstrofo en vocales cortadas (laa vs la')
- Confundir x / xh / s
- Confundir d / t, b / p al inicio de sílaba acentuada
- Omitir tildes en vocales tónicas (á, é, í, ó, ú)
- Escribir consonantes en castellano donde corresponde la zapoteca"""

_INSTRUCCIONES_CORRECTOR = """\
# MODO CORRECTOR — INSTRUCCIONES
Recibirás filas del dataset con:
  ZAPOTECO: texto transcrito (puede tener errores)
  ESPAÑOL:  traducción al español (fuente de verdad sobre el significado)

Tu proceso:
1. Lee la traducción española para entender QUÉ se quiso decir
2. Verifica cada palabra zapoteca contra el diccionario
3. Aplica reglas fonéticas para detectar errores ortográficos
4. Sugiere correcciones específicas con explicación breve
5. Si una palabra no está en el diccionario, indica si parece zapoteca o española
6. Sé conversacional: haz UNA pregunta a la vez si hay ambigüedad

Formato de respuesta:
- Empieza con una observación general (1 línea)
- Lista de correcciones sugeridas (una por línea, con el "por qué")
- Pregunta de confirmación si hay dudas
- Termina con la versión corregida completa entre ``` ```

Ejemplo:
"La transcripción tiene 2 ajustes. La traducción ('dame fuerza') me ayuda a confirmar las palabras.
- 'cuni' → 'cuní' (falta tilde; en el diccionario: cuní = dame)
- 'guendanabani' → 'guendanabáni' (vocal tónica en -bá-)
```cuní guendanabáni```"
"""

_INSTRUCCIONES_CHAT = """\
# MODO CHAT — INSTRUCCIONES
Responde preguntas sobre el Zapoteco del Istmo de forma conversacional.
Puedes:
- Traducir palabras o frases (zapoteco ↔ español)
- Explicar reglas de pronunciación y ortografía
- Buscar palabras en el diccionario
- Explicar diferencias entre palabras similares
- Dar contexto histórico o cultural
Sé amigable y entusiasta. Si el usuario pregunta algo que no está en el diccionario,
indícalo honestamente y ofrece la palabra más cercana."""
