"""
modelo/tokenizador.py
=====================
Tokenizador de caracteres para el NanoGPT zapoteco.

Usa tokenización carácter-a-carácter (igual que el primer enfoque
del notebook de la Práctica 2): mapea cada carácter único del corpus
a un entero. Es O(1) por carácter — instantáneo para cualquier corpus.

Para Zapoteco es ideal: lo que nos interesan son los patrones de
caracteres (dígrafos xh/dx/ch, vocales con tono ', tildes á/é/í/ó/ú).
"""

from __future__ import annotations
import json
from pathlib import Path


class CharTokenizer:
    """
    Tokenizador carácter-a-carácter.
    Construye el vocabulario a partir del corpus de entrenamiento.
    """

    def __init__(self) -> None:
        self.stoi: dict[str, int] = {}   # char → id
        self.itos: dict[int, str] = {}   # id  → char
        self._unk = 0                    # id para caracteres desconocidos

    # ── Entrenamiento ────────────────────────────────────────────────

    def train(self, texto: str) -> None:
        chars = sorted(set(texto))
        # id 0 = <UNK>
        self.itos = {0: "<UNK>"}
        self.stoi = {"<UNK>": 0}
        for i, c in enumerate(chars, start=1):
            self.stoi[c] = i
            self.itos[i] = c

    # ── Codificación / Decodificación ────────────────────────────────

    def encode(self, texto: str) -> list[int]:
        return [self.stoi.get(c, self._unk) for c in texto]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos.get(i, "?") for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    # ── Persistencia ─────────────────────────────────────────────────

    def guardar(self, ruta: Path) -> None:
        datos = {"stoi": self.stoi, "itos": {str(k): v for k, v in self.itos.items()}}
        ruta.write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def cargar(cls, ruta: Path) -> "CharTokenizer":
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        tok = cls()
        tok.stoi = datos["stoi"]
        tok.itos = {int(k): v for k, v in datos["itos"].items()}
        return tok
