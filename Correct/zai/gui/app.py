"""
gui/app.py
==========
Ventana principal de ZAI.
Sidebar con 4 modos + barra de estado inferior.
"""

from __future__ import annotations
import logging
import threading
import tkinter as tk
from tkinter import messagebox, filedialog
from pathlib import Path

import customtkinter as ctk

logger = logging.getLogger(__name__)

from zai.config import cargar_config, guardar_config, DATASET_PATH, DICCIONARIO_PATH, OLLAMA_MODEL_MAIN, OLLAMA_MODEL
from zai.config import PDF_FONÉTICA, PDF_VOCAB
from zai.excel import GestorDataset
from zai.context import ContextoLinguistico
from zai.agent import ZAIAgent
from zai.modes.corrector import ModoCorrector
from zai.modes.chat import ModoChat
from zai.modes.extractor import ModoExtractor
from zai.modes.organizer import ModoOrganizador


from zai.gui.theme import (
    BG_MAIN, BG_CARD, SIDEBAR_BG, SIDEBAR_SEL,
    ACCENT, ACCENT_L as ACCENT_LIGHT, TEXT as TEXT_MAIN, TEXT_DIM,
    VERDE, AMARILLO, ROJO,
)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


class AppZAI(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("ZAI — Agente Zapoteco del Istmo")
        self.geometry("1280x800")
        self.minsize(1000, 650)
        self.configure(fg_color=BG_MAIN)

        self.cfg     = cargar_config()
        self._vista_activa: str = ""
        self._cerrando = False

        self.protocol("WM_DELETE_WINDOW", self._on_cerrar)

        # Módulos (se inicializan después de cargar recursos)
        self.dataset:    GestorDataset    | None = None
        self.contexto:   ContextoLinguistico | None = None
        self.agente:     ZAIAgent          | None = None
        self.corrector:  ModoCorrector     | None = None
        self.chat:       ModoChat          | None = None
        self.extractor:  ModoExtractor     | None = None
        self.organizer:  ModoOrganizador   | None = None

        self._construir_layout()
        self._inicializar_recursos()

    # ── Layout ───────────────────────────────────────────────────────

    def _construir_layout(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self._sidebar = _Sidebar(self, self._cambiar_vista)
        self._sidebar.grid(row=0, column=0, sticky="nsw", padx=0, pady=0)

        # Contenido principal
        self._frame_contenido = ctk.CTkFrame(self, fg_color=BG_MAIN, corner_radius=0)
        self._frame_contenido.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self._frame_contenido.grid_rowconfigure(0, weight=1)
        self._frame_contenido.grid_columnconfigure(0, weight=1)

        self._vista_actual_widget = None

        # Barra de estado
        self._statusbar = _StatusBar(self)
        self._statusbar.grid(row=1, column=0, columnspan=2, sticky="ew")

    # ── Inicialización de recursos ───────────────────────────────────

    def _inicializar_recursos(self) -> None:
        self._statusbar.set("Cargando recursos…", color=AMARILLO)
        self._sidebar.deshabilitar()

        def _cargar():
            try:
                dataset_path = Path(self.cfg.get("dataset", str(DATASET_PATH)))
                dic_path     = Path(self.cfg.get("diccionario", str(DICCIONARIO_PATH)))

                self.dataset  = GestorDataset(dataset_path)
                self.contexto = ContextoLinguistico(PDF_FONÉTICA, PDF_VOCAB, dic_path).cargar()
                self.agente   = ZAIAgent(self.contexto)

                # Modos
                self.corrector = ModoCorrector(self.dataset, self.agente)
                self.chat      = ModoChat(self.agente)
                self.extractor = ModoExtractor(self.dataset, self.agente, self.contexto)
                self.organizer = ModoOrganizador(self.dataset, self.contexto)

                self.after(0, self._recursos_listos)
            except Exception as e:
                self.after(0, lambda: self._error_recursos(str(e)))

        threading.Thread(target=_cargar, daemon=True).start()

    def _recursos_listos(self) -> None:
        if self.agente and self.agente.ollama_disponible:
            modelos = f"Ollama {OLLAMA_MODEL_MAIN} + {OLLAMA_MODEL}"
        else:
            modelos = "Ollama no disponible — ejecuta: ollama serve"
        self._statusbar.set(
            f"✓ {self.dataset.total:,} transcripciones · "
            f"{len(self.contexto.diccionario):,} palabras en diccionario · {modelos}",
            color=VERDE if (self.agente and self.agente.ollama_disponible) else AMARILLO,
        )
        self._sidebar.habilitar()
        self._cambiar_vista("corrector")

    def _error_recursos(self, msg: str) -> None:
        self._statusbar.set(f"Error al cargar: {msg}", color=ROJO)
        messagebox.showerror("Error de inicio", msg)

    # ── Navegación ───────────────────────────────────────────────────

    def _cambiar_vista(self, nombre: str) -> None:
        if nombre == self._vista_activa:
            return
        self._vista_activa = nombre
        self._sidebar.marcar_activo(nombre)

        if self._vista_actual_widget:
            self._vista_actual_widget.destroy()
            self._vista_actual_widget = None

        if not self.dataset:
            return

        from zai.gui.views.corrector_view import CorrectorView
        from zai.gui.views.chat_view      import ChatView
        from zai.gui.views.extractor_view import ExtractorView
        from zai.gui.views.organizer_view import OrganizerView

        mapa = {
            "corrector": lambda: CorrectorView(self._frame_contenido, self.corrector, self._statusbar),
            "chat":      lambda: ChatView(self._frame_contenido, self.chat),
            "extractor": lambda: ExtractorView(self._frame_contenido, self.extractor, self._statusbar),
            "organizer": lambda: OrganizerView(self._frame_contenido, self.organizer, self.dataset, self.contexto),
        }
        if nombre in mapa:
            widget = mapa[nombre]()
            widget.grid(row=0, column=0, sticky="nsew")
            self._vista_actual_widget = widget

    # ── Cierre seguro ────────────────────────────────────────────────

    def _on_cerrar(self) -> None:
        if self._cerrando:
            return
        self._cerrando = True

        # Guardar cambios pendientes antes de salir
        if self.corrector and self.corrector.cambios_pendientes:
            try:
                guardadas = self.corrector.guardar()
                logger.info("Auto-guardado al cerrar: %d correcciones", guardadas)
            except Exception:
                logger.exception("Error al guardar correcciones al cerrar")

        self.destroy()

    # ── Helpers públicos ─────────────────────────────────────────────

    def actualizar_status(self, msg: str, color: str = TEXT_MAIN) -> None:
        self._statusbar.set(msg, color=color)


# ── Sidebar ──────────────────────────────────────────────────────────

class _Sidebar(ctk.CTkFrame):

    _ITEMS = [
        ("corrector", "✏️", "Corrector"),
        ("chat",      "💬", "Chat"),
        ("extractor", "🔍", "Extractor"),
        ("organizer", "📚", "Diccionario"),
    ]

    def __init__(self, master, on_click):
        super().__init__(master, fg_color=SIDEBAR_BG, corner_radius=0, width=180)
        self._on_click = on_click
        self._botones: dict[str, ctk.CTkButton] = {}
        self._construir()

    def _construir(self) -> None:
        # Logo / título
        ctk.CTkLabel(
            self, text="ZAI", font=ctk.CTkFont("Arial", 28, "bold"),
            text_color=ACCENT_LIGHT,
        ).pack(pady=(24, 4))
        ctk.CTkLabel(
            self, text="Diidxazá Agent", font=ctk.CTkFont("Arial", 11),
            text_color=TEXT_DIM,
        ).pack(pady=(0, 24))

        ctk.CTkFrame(self, fg_color=ACCENT, height=1).pack(fill="x", padx=16, pady=(0, 16))

        for key, emoji, label in self._ITEMS:
            btn = ctk.CTkButton(
                self,
                text=f"  {emoji}  {label}",
                anchor="w",
                fg_color="transparent",
                hover_color=SIDEBAR_SEL,
                text_color=TEXT_DIM,
                font=ctk.CTkFont("Arial", 13),
                height=44,
                corner_radius=8,
                command=lambda k=key: self._on_click(k),
            )
            btn.pack(fill="x", padx=12, pady=2)
            self._botones[key] = btn

        # Spacer
        ctk.CTkFrame(self, fg_color="transparent").pack(fill="both", expand=True)

        # Footer
        ctk.CTkLabel(
            self, text="v2.1 — Ollama + NanoGPT",
            font=ctk.CTkFont("Arial", 9), text_color=TEXT_DIM,
        ).pack(pady=12)

    def marcar_activo(self, key: str) -> None:
        for k, btn in self._botones.items():
            if k == key:
                btn.configure(fg_color=ACCENT, text_color=TEXT_MAIN)
            else:
                btn.configure(fg_color="transparent", text_color=TEXT_DIM)

    def habilitar(self) -> None:
        for btn in self._botones.values():
            btn.configure(state="normal")

    def deshabilitar(self) -> None:
        for btn in self._botones.values():
            btn.configure(state="disabled")


# ── Barra de estado ──────────────────────────────────────────────────

class _StatusBar(ctk.CTkFrame):

    def __init__(self, master):
        super().__init__(master, fg_color=SIDEBAR_BG, height=28, corner_radius=0)
        self._lbl = ctk.CTkLabel(
            self, text="Iniciando…",
            font=ctk.CTkFont("Arial", 11), text_color=TEXT_DIM,
            anchor="w",
        )
        self._lbl.pack(side="left", padx=12)

    def set(self, texto: str, color: str = TEXT_DIM) -> None:
        self._lbl.configure(text=texto, text_color=color)


def lanzar() -> None:
    app = AppZAI()
    app.mainloop()
