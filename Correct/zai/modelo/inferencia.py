"""
modelo/inferencia.py
====================
Carga el NanoGPT entrenado y expone utilidades de inferencia.

API principal:
    modelo = ModeloZapoteco.cargar()
    modelo.disponible          → bool (False si no hay pesos guardados)
    modelo.es_zapoteca("xhiga")   → True / False
    modelo.puntuar("xhiga")       → float (perplejidad, menor = más zapoteco)
    modelo.top_zapotecas(tokens)  → lista ordenada de (token, puntaje)
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

from zai.modelo.entrenador import RUTA_PESOS, RUTA_TOKENIZADOR, RUTA_CONFIG


class ModeloZapoteco:
    """
    Wrapper de inferencia del NanoGPT zapoteco.
    Se carga desde disco; si los archivos no existen, está en modo "no disponible"
    y todas las llamadas devuelven valores neutros (el código que lo usa decide el fallback).
    """

    def __init__(self) -> None:
        self._modelo   = None
        self._tok      = None
        self._device   = "cpu"
        self.disponible = False
        self._umbral_perp: float = 200.0   # por encima → no es zapoteco

    # ── Carga ────────────────────────────────────────────────────────

    @classmethod
    def cargar(cls) -> "ModeloZapoteco":
        inst = cls()
        if not (RUTA_PESOS.exists() and RUTA_TOKENIZADOR.exists() and RUTA_CONFIG.exists()):
            return inst   # disponible = False
        try:
            import torch
            from zai.modelo.arquitectura import NanoGPT

            cfg    = torch.load(str(RUTA_CONFIG), map_location="cpu",
                                weights_only=False)
            modelo = NanoGPT(cfg)
            modelo.load_state_dict(
                torch.load(str(RUTA_PESOS), map_location="cpu",
                           weights_only=True)
            )
            modelo.eval()

            from zai.modelo.tokenizador import CharTokenizer
            inst._modelo    = modelo
            inst._tok       = CharTokenizer.cargar(RUTA_TOKENIZADOR)
            inst.disponible = True

            # Calibrar umbral con texto conocido vs texto extraño
            inst._calibrar()
        except Exception as e:
            print(f"[ModeloZapoteco] No se pudo cargar: {e}")
        return inst

    def _calibrar(self) -> None:
        """
        Estima un umbral razonable para distinguir zapoteco vs no-zapoteco.
        Usa palabras conocidas del vocabulario.
        """
        if not self.disponible:
            return
        # palabras típicamente zapotecas
        zap_test = ["xhiga", "guendaranaxhii", "binnizá", "diidxazá",
                    "ni", "ru", "bi", "za", "ca", "la"]
        # palabras claramente no-zapotecas
        esp_test = ["perro", "casa", "computadora", "internet",
                    "the", "and", "hello", "world"]

        perp_zap = [self.puntuar(w) for w in zap_test if len(w) >= 2]
        perp_esp = [self.puntuar(w) for w in esp_test if len(w) >= 2]

        if perp_zap and perp_esp:
            media_zap = sum(perp_zap) / len(perp_zap)
            media_esp = sum(perp_esp) / len(perp_esp)
            # umbral = punto medio entre los dos grupos
            self._umbral_perp = (media_zap + media_esp) / 2
        elif perp_zap:
            self._umbral_perp = (sum(perp_zap) / len(perp_zap)) * 3

    # ── Inferencia ───────────────────────────────────────────────────

    def puntuar(self, texto: str) -> float:
        """
        Devuelve la perplejidad del texto bajo el modelo.
        Texto zapoteco → perplejidad baja.
        Texto extraño  → perplejidad alta.
        """
        if not self.disponible or not texto.strip():
            return float("inf")
        ids = self._tok.encode(texto.strip().lower())
        return self._modelo.perplejidad(ids, device=self._device)

    def es_zapoteca(self, palabra: str, umbral: Optional[float] = None) -> bool:
        """
        Clasifica si una palabra parece zapoteca según el modelo.
        """
        if not self.disponible:
            return False
        perp = self.puntuar(palabra)
        lim  = umbral if umbral is not None else self._umbral_perp
        return perp < lim

    def top_zapotecas(
        self,
        tokens: list[str],
        top_n:  int = 50,
    ) -> list[tuple[str, float]]:
        """
        Dado un listado de tokens candidatos, devuelve los top_n más zapotecos
        según el modelo, ordenados de menor a mayor perplejidad.
        """
        if not self.disponible:
            return [(t, 0.0) for t in tokens[:top_n]]

        puntuados = [(t, self.puntuar(t)) for t in tokens]
        puntuados.sort(key=lambda x: x[1])
        return puntuados[:top_n]

    def clasificar_batch(
        self,
        tokens: list[str],
    ) -> list[dict]:
        """
        Clasifica una lista de tokens en 'zapoteca' / 'otro'.
        Devuelve lista de dicts compatibles con el resultado de Ollama:
            [{"palabra": "xhiga", "tipo": "zapoteca", "puntaje": 45.2}, ...]
        """
        if not self.disponible:
            return []
        resultado = []
        for tok in tokens:
            perp = self.puntuar(tok)
            tipo = "zapoteca" if perp < self._umbral_perp else "otro"
            resultado.append({
                "palabra": tok,
                "tipo":    tipo,
                "puntaje": round(perp, 2),
            })
        return resultado

    def info(self) -> str:
        if not self.disponible:
            return "NanoGPT: no entrenado"
        n = self._modelo.n_params()
        return f"NanoGPT zapoteco ({n/1e3:.0f}K params, umbral={self._umbral_perp:.1f})"
