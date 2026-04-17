"""
views/chat_view.py
==================
Modo Chat: conversación libre sobre Zapoteco del Istmo.
Burbujas de chat, streaming en tiempo real.
"""

from __future__ import annotations
import tkinter as tk
from tkinter import scrolledtext
import customtkinter as ctk

from zai.modes.chat import ModoChat, MensajeChat

BG_MAIN   = "#0d1117"
BG_CARD   = "#161b22"
BG_INPUT  = "#21262d"
ACCENT    = "#0f3460"
ACCENT_L  = "#e94560"
TEXT      = "#eaeaea"
TEXT_DIM  = "#8892a0"
VERDE     = "#2ea043"
AZUL      = "#1f6feb"
BURBUJA_U = "#1f6feb"   # burbuja usuario
BURBUJA_A = "#21262d"   # burbuja asistente


class ChatView(ctk.CTkFrame):

    def __init__(self, master, chat: ModoChat):
        super().__init__(master, fg_color=BG_MAIN, corner_radius=0)
        self.chat = chat
        self._en_progreso = False
        self._construir()
        self._mostrar_bienvenida()

    # ── Layout ───────────────────────────────────────────────────────

    def _construir(self) -> None:
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Header
        header = ctk.CTkFrame(self, fg_color=BG_CARD, height=52, corner_radius=0)
        header.grid(row=0, column=0, sticky="ew")

        ctk.CTkLabel(
            header, text="💬  Chat con ZAI",
            font=ctk.CTkFont("Arial", 16, "bold"), text_color=TEXT,
        ).pack(side="left", padx=16, pady=14)

        ctk.CTkLabel(
            header, text="Pregunta cualquier cosa sobre Zapoteco del Istmo",
            font=ctk.CTkFont("Arial", 11), text_color=TEXT_DIM,
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            header, text="🗑 Limpiar", width=90, height=30,
            fg_color="transparent", hover_color=BG_INPUT, text_color=TEXT_DIM,
            command=self._limpiar,
        ).pack(side="right", padx=12)

        # Área de mensajes
        self._canvas = tk.Canvas(self, bg=BG_MAIN, highlightthickness=0)
        self._canvas.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        self.grid_rowconfigure(1, weight=1)

        scroll = ctk.CTkScrollbar(self, command=self._canvas.yview)
        scroll.grid(row=1, column=1, sticky="ns")
        self._canvas.configure(yscrollcommand=scroll.set)

        self._msg_frame = tk.Frame(self._canvas, bg=BG_MAIN)
        self._canvas_window = self._canvas.create_window(
            (0, 0), window=self._msg_frame, anchor="nw")

        self._msg_frame.bind("<Configure>", self._on_frame_resize)
        self._canvas.bind("<Configure>", self._on_canvas_resize)
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # Input
        input_frame = ctk.CTkFrame(self, fg_color=BG_CARD, height=68, corner_radius=0)
        input_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        input_frame.grid_columnconfigure(0, weight=1)

        self._entrada = ctk.CTkEntry(
            input_frame,
            placeholder_text="¿Cómo se dice 'lluvia' en zapoteco? ¿Qué significa 'guendaranaxhii'?…",
            fg_color=BG_INPUT, text_color=TEXT, border_color=ACCENT,
            font=ctk.CTkFont("Arial", 13), height=40,
        )
        self._entrada.grid(row=0, column=0, padx=(16, 8), pady=14, sticky="ew")
        self._entrada.bind("<Return>", lambda e: self._enviar())

        self._btn_enviar = ctk.CTkButton(
            input_frame, text="Enviar ↩",
            fg_color=AZUL, hover_color="#2979ff", text_color="white",
            font=ctk.CTkFont("Arial", 13, "bold"), width=100, height=40,
            command=self._enviar,
        )
        self._btn_enviar.grid(row=0, column=1, padx=(0, 16))

        # Sugerencias rápidas
        sugs = ctk.CTkFrame(self, fg_color=BG_MAIN, height=36)
        sugs.grid(row=3, column=0, columnspan=2, sticky="ew", padx=16, pady=(0, 4))

        ejemplos = [
            "¿Cómo se dice 'pescado'?",
            "¿Qué significa 'biñi'?",
            "Explica las vocales cortadas",
            "¿Qué es una vocal quebrada?",
            "Diferencia entre x y xh",
        ]
        for ej in ejemplos:
            ctk.CTkButton(
                sugs, text=ej, height=28,
                fg_color=BG_CARD, hover_color=ACCENT, text_color=TEXT_DIM,
                font=ctk.CTkFont("Arial", 10), corner_radius=14,
                command=lambda t=ej: self._enviar_texto(t),
            ).pack(side="left", padx=4)

    # ── Mensajes ─────────────────────────────────────────────────────

    def _mostrar_bienvenida(self) -> None:
        bienvenida = (
            "¡Hola! Soy ZAI, tu asistente de Zapoteco del Istmo (Diidxazá) 🌿\n\n"
            "Tengo acceso al diccionario completo con más de 9,000 palabras, "
            "las reglas fonéticas del idioma y vocabulario histórico.\n\n"
            "Puedo ayudarte a:\n"
            "  • Traducir palabras o frases (zapoteco ↔ español)\n"
            "  • Explicar reglas de pronunciación y escritura\n"
            "  • Verificar si una palabra existe en el diccionario\n"
            "  • Responder dudas lingüísticas o culturales\n\n"
            "¿En qué te puedo ayudar?"
        )
        self._agregar_burbuja("assistant", bienvenida)

    def _agregar_burbuja(self, rol: str, texto: str) -> tk.Widget:
        """Crea y retorna el widget de burbuja."""
        es_usuario = (rol == "user")

        outer = tk.Frame(self._msg_frame, bg=BG_MAIN)
        outer.pack(fill="x", padx=16, pady=4)

        if es_usuario:
            burbuja_frame = tk.Frame(outer, bg=BG_MAIN)
            burbuja_frame.pack(side="right")
        else:
            # Emoji avatar
            tk.Label(outer, text="🤖", bg=BG_MAIN, font=("Arial", 16)).pack(
                side="left", anchor="n", padx=(0, 8))
            burbuja_frame = tk.Frame(outer, bg=BG_MAIN)
            burbuja_frame.pack(side="left")

        bg_color = BURBUJA_U if es_usuario else BURBUJA_A
        fg_color = "white" if es_usuario else TEXT

        lbl = tk.Text(
            burbuja_frame,
            wrap=tk.WORD,
            bg=bg_color,
            fg=fg_color,
            font=("Arial", 12),
            relief="flat", bd=0,
            padx=14, pady=10,
            width=60,
            cursor="arrow",
            state="normal",
        )
        lbl.insert("1.0", texto)
        lbl.configure(state="disabled" if texto else "normal",
                      height=max(2, texto.count("\n") + 2))
        lbl.pack()

        self._scroll_abajo()
        return lbl

    def _scroll_abajo(self) -> None:
        self.after(50, lambda: self._canvas.yview_moveto(1.0))

    def _limpiar(self) -> None:
        self.chat.limpiar()
        for widget in self._msg_frame.winfo_children():
            widget.destroy()
        self._mostrar_bienvenida()

    # ── Envío ────────────────────────────────────────────────────────

    def _enviar(self) -> None:
        texto = self._entrada.get().strip()
        if not texto or self._en_progreso:
            return
        self._entrada.delete(0, tk.END)
        self._enviar_texto(texto)

    def _enviar_texto(self, texto: str) -> None:
        if self._en_progreso:
            return
        # Burbuja del usuario
        self._agregar_burbuja("user", texto)
        self._en_progreso = True
        self._btn_enviar.configure(state="disabled")

        # Burbuja del asistente (vacía, se irá llenando)
        burbuja_ia = self._agregar_burbuja("assistant", "")

        def _chunk(texto_chunk: str) -> None:
            def _upd():
                burbuja_ia.configure(state="normal")
                burbuja_ia.insert(tk.END, texto_chunk)
                # Ajustar alto dinámicamente
                lineas = int(burbuja_ia.index(tk.END).split(".")[0])
                burbuja_ia.configure(height=max(2, lineas))
                burbuja_ia.configure(state="disabled")
                self._scroll_abajo()
            self.after(0, _upd)

        def _done() -> None:
            self._en_progreso = False
            self.after(0, lambda: self._btn_enviar.configure(state="normal"))

        def _nuevo_msg(msg: MensajeChat) -> None:
            pass  # ya creamos la burbuja manualmente

        self.chat.enviar(
            texto            = texto,
            on_chunk         = _chunk,
            on_done          = _done,
            on_nuevo_mensaje = _nuevo_msg,
        )

    # ── Resize helpers ───────────────────────────────────────────────

    def _on_frame_resize(self, event) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_resize(self, event) -> None:
        self._canvas.itemconfig(self._canvas_window, width=event.width - 16)

    def _on_mousewheel(self, event) -> None:
        self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
