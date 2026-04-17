"""Punto de entrada: python -m zai"""
from zai.config import configurar_logging
configurar_logging()

from zai.gui.app import lanzar

if __name__ == "__main__":
    lanzar()
