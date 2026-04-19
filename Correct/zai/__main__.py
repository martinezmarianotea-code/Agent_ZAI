"""Punto de entrada: python -m zai  /  comando: zai"""
from zai.config import configurar_logging
configurar_logging()

from zai.gui.app import lanzar


def main() -> None:
    lanzar()


if __name__ == "__main__":
    main()
