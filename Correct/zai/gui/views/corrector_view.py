"""
views/corrector_view.py
=======================
Vista del Corrector IA:
  ┌─────────────────────────────────────────────────────────────┐
  │ Progreso                                  [ < ]  [ > ]  [✓]│
  ├──────────────────────┬──────────────────────────────────────┤
  │  ZAPOTECO            │  ESPAÑOL (referencia)                │
  │  (editable)          │  (read-only)                         │
  ├──────────────────────┴──────────────────────────────────────┤
  │  Sugerencia IA  (streaming, con pre-análisis Ollama)        │
  ├─────────────────────────────────────────────────────────────┤
  │  [Aplicar corrección]  [Rechazar]  [Mensaje a IA…]         │
  └─────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox
import customtkinter as ctk

from zai.modes.corrector import ModoCorrector

# Colores
BG_MAIN  = "#0d1117"
BG_CARD  = "#161b22"
BG_INPUT = "#21262d"
ACCENT   = "#0f3460"
ACCENT_L = "#e94560"
TEXT     = "#eaeaea"
TEXT_DIM = "#8892a0"
VERDE    = "#2ea043"
AMARILLO = "#d29922"
ROJO     = "#da3633"
AZUL     = "#1f6feb"


class CorrectorView(ctk.CTkFrame):

    def __init__(self, master, corrector: ModoCorrector, statusbar):
        super().__init__(master, fg_color=BG_MAIN, corner_radius=0)
        self.corrector = corrector
        self.statusbar = statusbar
        self._ia_activa = False
        self._construir()
        self._cargar_fila_actual()

    # ── Layout ───────────────────────────────────────────────────────

    def _construir(self) -> None:
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=2)
        self.grid_columnconfigure(0, weight=1)

        # — Barra superior —
        top = ctk.CTkFrame(self, fg_color=BG_CARD, height=52, corner_radius=0)
        top.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        top.grid_columnconfigure(1, weight=1)

        self._lbl_audio = ctk.CTkLabel(
            top, text="", font=ctk.CTkFont("Arial", 13, "bold"),
            text_color=ACCENT_L,
        )
        self._lbl_audio.grid(row=0, column=0, padx=16, pady=12)

        self._lbl_progreso = ctk.CTkLabel(
            top, text="", font=ctk.CTkFont("Arial", 12), text_color=TEXT_DIM,
        )
        self._lbl_progreso.grid(row=0, column=1, padx=8)

        btn_frame = ctk.CTkFrame(top, fg_color="transparent")
        btn_frame.grid(row=0, column=2, padx=12)

        self._btn_prev = ctk.CTkButton(btn_frame, text="◀", width=36, height=32,
            fg_color=BG_INPUT, hover_color=ACCENT, text_color=TEXT,
            command=self._anterior)
        self._btn_prev.pack(side="left", padx=2)

        self._btn_next = ctk.CTkButton(btn_frame, text="▶", width=36, height=32,
            fg_color=BG_INPUT, hover_color=ACCENT, text_color=TEXT,
            command=self._siguiente)
        self._btn_next.pack(side="left", padx=2)

        self._btn_analizar = ctk.CTkButton(btn_frame, text="🤖 Analizar", width=100, height=32,
            fg_color=AZUL, hover_color="#2979ff", text_color=TEXT,
            font=ctk.CTkFont("Arial", 12, "bold"),
            command=self._lanzar_analisis)
        self._btn_analizar.pack(side="left", padx=(8, 2))

        # — Panel central: Zapoteco | Español —
        mid = ctk.CTkFrame(self, fg_color=BG_MAIN, corner_radius=0)
        mid.grid(row=1, column=0, sticky="nsew", padx=12, pady=(8, 0))
        mid.grid_columnconfigure(0, weight=1)
        mid.grid_columnconfigure(1, weight=1)
        mid.grid_rowconfigure(1, weight=1)

        # Zapoteco
        ctk.CTkLabel(mid, text="ZAPOTECO", font=ctk.CTkFont("Arial", 11, "bold"),
            text_color=ACCENT_L).grid(row=0, column=0, sticky="w", padx=8)

        self._txt_zapoteco = _TextBox(mid, bg=BG_INPUT, fg=TEXT, editable=True)
        self._txt_zapoteco.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(4, 0))

        # Español
        ctk.CTkLabel(mid, text="ESPAÑOL (referencia)", font=ctk.CTkFont("Arial", 11, "bold"),
            text_color=AMARILLO).grid(row=0, column=1, sticky="w", padx=8)

        self._txt_espanol = _TextBox(mid, bg="#1a1f2a", fg="#c9d1d9", editable=False)
        self._txt_espanol.grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=(4, 0))

        # — Panel IA —
        ia_outer = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=12)
        ia_outer.grid(row=2, column=0, sticky="nsew", padx=12, pady=8)
        ia_outer.grid_rowconfigure(1, weight=1)
        ia_outer.grid_columnconfigure(0, weight=1)

        ia_header = ctk.CTkFrame(ia_outer, fg_color="transparent")
        ia_header.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 0))

        ctk.CTkLabel(ia_header, text="🤖 Análisis ZAI",
            font=ctk.CTkFont("Arial", 13, "bold"), text_color=AZUL).pack(side="left")

        self._lbl_modelo = ctk.CTkLabel(ia_header, text="",
            font=ctk.CTkFont("Arial", 10), text_color=TEXT_DIM)
        self._lbl_modelo.pack(side="right")

        self._txt_ia = _TextBox(ia_outer, bg=BG_CARD, fg=TEXT, editable=False, wrap=tk.WORD)
        self._txt_ia.grid(row=1, column=0, sticky="nsew", padx=8, pady=6)

        # — Barra de acciones —
        acciones = ctk.CTkFrame(self, fg_color=BG_CARD, height=56, corner_radius=0)
        acciones.grid(row=3, column=0, sticky="ew", padx=0, pady=0)

        self._btn_aplicar = ctk.CTkButton(acciones, text="✅ Aplicar corrección",
            fg_color=VERDE, hover_color="#3fb950", text_color="white",
            font=ctk.CTkFont("Arial", 12, "bold"), width=160, height=36,
            command=self._aplicar)
        self._btn_aplicar.pack(side="left", padx=12, pady=10)

        self._btn_rechazar = ctk.CTkButton(acciones, text="❌ Sin cambios",
            fg_color=BG_INPUT, hover_color=ROJO, text_color=TEXT,
            font=ctk.CTkFont("Arial", 12), width=120, height=36,
            command=self._rechazar)
        self._btn_rechazar.pack(side="left", padx=4)

        ctk.CTkFrame(acciones, fg_color="transparent").pack(side="left", fill="x", expand=True)

        # Entrada para preguntas al IA
        self._entrada_msg = ctk.CTkEntry(acciones,
            placeholder_text="Pregunta o aclaración para ZAI…",
            fg_color=BG_INPUT, text_color=TEXT, border_color=ACCENT,
            font=ctk.CTkFont("Arial", 12), height=36, width=320)
        self._entrada_msg.pack(side="left", padx=4)
        self._entrada_msg.bind("<Return>", lambda e: self._enviar_mensaje())

        self._btn_msg = ctk.CTkButton(acciones, text="Enviar ↩",
            fg_color=ACCENT, hover_color="#1a4a80", text_color=TEXT,
            font=ctk.CTkFont("Arial", 12), width=80, height=36,
            command=self._enviar_mensaje)
        self._btn_msg.pack(side="left", padx=(0, 12))

        ctk.CTkButton(acciones, text="💾 Guardar",
            fg_color=BG_INPUT, hover_color=ACCENT, text_color=TEXT,
            font=ctk.CTkFont("Arial", 12), width=90, height=36,
            command=self._guardar).pack(side="right", padx=12)

    # ── Carga de fila ────────────────────────────────────────────────

    def _cargar_fila_actual(self) -> None:
        fila = self.corrector.fila_actual
        if not fila:
            self._txt_zapoteco.set("— Fin del dataset —")
            self._txt_espanol.set("")
            return

        n, total = self.corrector.progreso
        self._lbl_audio.configure(text=f"🎙 {fila.audio}")
        self._lbl_progreso.configure(text=f"Fila {n + 1} / {total}")

        self._txt_zapoteco.set(fila.texto_vigente)
        self._txt_espanol.set(self.corrector.espanol_actual or "(sin traducción)")
        self._txt_ia.set("")
        self._lbl_modelo.configure(text="")

        # Pre-análisis con Ollama en background (no bloquea)
        if self.corrector.agente.ollama_disponible:
            import threading
            def _pre():
                resultado = self.corrector.preanalizar_actual()
                if resultado and resultado.get("observacion"):
                    self.after(0, lambda: self._txt_ia.append(
                        f"[Ollama pre-análisis]: {resultado['observacion']}\n\n", color=AMARILLO))
            threading.Thread(target=_pre, daemon=True).start()

    # ── Acciones ─────────────────────────────────────────────────────

    def _lanzar_analisis(self) -> None:
        if self._ia_activa:
            return
        self._ia_activa = True
        self._btn_analizar.configure(state="disabled", text="Analizando…")
        self._txt_ia.set("")
        self._lbl_modelo.configure(text="qwen3:4b · streaming")

        def _chunk(texto: str) -> None:
            self.after(0, lambda: self._txt_ia.append(texto))

        def _done(texto: str) -> None:
            self._ia_activa = False
            self.after(0, lambda: self._btn_analizar.configure(
                state="normal", text="🤖 Analizar"))

        self.corrector.analizar_con_claude(on_chunk=_chunk, on_done=_done)

    def _enviar_mensaje(self) -> None:
        msg = self._entrada_msg.get().strip()
        if not msg or self._ia_activa:
            return
        self._entrada_msg.delete(0, tk.END)
        self._ia_activa = True
        self._txt_ia.append(f"\n\n[Tú]: {msg}\n\n[ZAI]: ", color=TEXT_DIM)

        def _chunk(texto: str) -> None:
            self.after(0, lambda: self._txt_ia.append(texto))

        def _done(_: str) -> None:
            self._ia_activa = False

        self.corrector.responder_usuario(msg, on_chunk=_chunk, on_done=_done)

    def _aplicar(self) -> None:
        texto = self._txt_zapoteco.get_all()
        self.corrector.aplicar_correccion(texto)
        self._siguiente()

    def _rechazar(self) -> None:
        self.corrector.rechazar_correccion()
        self._siguiente()

    def _siguiente(self) -> None:
        self.corrector.avanzar()
        self._cargar_fila_actual()

    def _anterior(self) -> None:
        n, _ = self.corrector.progreso
        if n > 0:
            self.corrector.ir_a(n - 1)
            self._cargar_fila_actual()

    def _guardar(self) -> None:
        guardadas = self.corrector.guardar()
        self.statusbar.set(f"✓ {guardadas} filas guardadas", color="#2ea043")


# ── Widget TextBox reutilizable ──────────────────────────────────────

class _TextBox(tk.Text):
    """Text widget con helpers de color y edición."""

    def __init__(self, master, bg="#161b22", fg="#eaeaea",
                 editable=True, wrap=tk.WORD, **kw):
        super().__init__(
            master,
            bg=bg, fg=fg,
            insertbackground=fg,
            selectbackground="#264f78",
            relief="flat",
            borderwidth=0,
            padx=12, pady=8,
            wrap=wrap,
            font=("Consolas", 12),
            **kw,
        )
        if not editable:
            self.configure(state="disabled", cursor="arrow")
        # Tags de color
        self.tag_configure("amarillo", foreground="#d29922")
        self.tag_configure("azul",     foreground="#58a6ff")
        self.tag_configure("dimmed",   foreground="#8892a0")

    def set(self, texto: str) -> None:
        self.configure(state="normal")
        self.delete("1.0", tk.END)
        self.insert("1.0", texto)

    def append(self, texto: str, color: str | None = None) -> None:
        self.configure(state="normal")
        if color == "#d29922":
            self.insert(tk.END, texto, "amarillo")
        elif color == "#8892a0":
            self.insert(tk.END, texto, "dimmed")
        else:
            self.insert(tk.END, texto)
        self.see(tk.END)

    def get_all(self) -> str:
        return self.get("1.0", tk.END).strip()
