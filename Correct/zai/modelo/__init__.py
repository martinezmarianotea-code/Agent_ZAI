"""
zai/modelo
==========
Modelo NanoGPT local entrenado con texto zapoteco.
Proporciona clasificación y scoring sin depender de APIs externas.
"""
from zai.modelo.inferencia import ModeloZapoteco

__all__ = ["ModeloZapoteco"]
