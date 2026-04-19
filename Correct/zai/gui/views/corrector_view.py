"""
views/corrector_view.py
=======================
Vista del Corrector IA.

Flujo:
  1. Pantalla de entrada — pegar texto, cargar .txt, o cargar Excel
  2. Pantalla de corrección — fila por fila con análisis bajo demanda

El análisis solo corre cuando el usuario presiona "Analizar".
"""

from __future__ import annotations
import threading
import tkinter as tk
from tkinter import messagebox, filedialog
import customtkinter as ctk

from zai.modes.corrector import ModoCorrector
from zai.gui.theme import (
    BG_MAIN, BG_CARD, BG_INPUT, ACCENT, ACCENT_L,
    TEXT, TEXT_DIM, VERDE, AMARILLO, ROJO, AZUL,
)


class CorrectorView(ctk.CTkFrame):

    def __init__(self, master, corrector: ModoCorrector, statusbar):
        super().__init__(master, fg_color=BG_MAIN, corner_radius=0)
        self.corrector = corrector
        self.statusbar = statusbar
        self._ia_activa = False

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Arranca en pantalla de entrada, a menos que ya haya datos del Excel
        if self.corrector.tiene_datos and not self.corrector._modo_libre:
            self._mostrar_entrada(precargar_excel=True)
        else:
            self._mostrar_entrada()

    # ════════════════════════════════════════════════════════════════
    #  PANTALLA 1 — Entrada de datos
    # ════════════════════════════════════════════════════════════════

    def _mostrar_entrada(self, precargar_excel: bool = False) -> None:
        """Muestra la pantalla de carga/pegado de transcripciones."""
        if hasattr(self, "_frame_actual") and self._frame_actual:
            self._frame_actual.destroy()

        frame = ctk.CTkFrame(self, fg_color=BG_MAIN, corner_radius=0)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.grid_rowconfigure(2, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        self._frame_actual = frame

        # — Header —
        header = ctk.CTkFrame(frame, fg_color=BG_CARD, height=56, corner_radius=0)
        header.grid(row=0, column=0, sticky="ew")

        ctk.CTkLabel(
            header, text="✏️  Corrector de Transcripciones",
            font=ctk.CTkFont("Arial", 16, "bold"), text_color=TEXT,
        ).pack(side="left", padx=20, pady=16)

        ctk.CTkLabel(
            header, text="Zapoteco del Istmo — Diidxazá",
            font=ctk.CTkFont("Arial", 11), text_color=TEXT_DIM,
        ).pack(side="left", padx=4)

        # — Botones de carga —
        btn_row = ctk.CTkFrame(frame, fg_color=BG_MAIN)
        btn_row.grid(row=1, column=0, sticky="ew", padx=24, pady=(20, 0))

        ctk.CTkButton(
            btn_row, text="📄 Cargar .txt",
            fg_color=ACCENT, hover_color="#1a4a80", text_color=TEXT,
            font=ctk.CTkFont("Arial", 12), height=38, width=140,
            command=self._cargar_txt,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            btn_row, text="📊 Cargar Excel",
            fg_color=BG_CARD, hover_color=ACCENT, text_color=TEXT,
            font=ctk.CTkFont("Arial", 12), height=38, width=140,
            command=self._cargar_excel,
        ).pack(side="left", padx=8)

        if precargar_excel:
            n, total = self.corrector.progreso
            ctk.CTkLabel(
                btn_row,
                text=f"  ← Excel cargado: {total} transcripciones",
                font=ctk.CTkFont("Arial", 11), text_color=VERDE,
            ).pack(side="left", padx=12)

        ctk.CTkLabel(
            btn_row,
            text="— o pega el texto abajo —",
            font=ctk.CTkFont("Arial", 11), text_color=TEXT_DIM,
        ).pack(side="right", padx=8)

        # — Área de texto —
        ctk.CTkLabel(
            frame,
            text="Una transcripción por línea:",
            font=ctk.CTkFont("Arial", 11), text_color=TEXT_DIM, anchor="w",
        ).grid(row=1, column=0, sticky="w", padx=28, pady=(56, 0))

        self._txt_entrada = tk.Text(
            frame,
            bg=BG_INPUT, fg=TEXT, insertbackground=TEXT,
            relief="flat", borderwidth=0,
            font=("Consolas", 12),
            padx=16, pady=12,
            wrap=tk.WORD,
        )
        self._txt_entrada.grid(row=2, column=0, sticky="nsew", padx=24, pady=(4, 0))

        if precargar_excel:
            # Muestra preview de las primeras filas del Excel
            preview = "\n".join(
                f.zapoteco for f in self.corrector._filas[:30]
            )
            if self.corrector._total > 30:
                preview += f"\n… ({self.corrector._total - 30} filas más)"
            self._txt_entrada.insert("1.0", preview)
            self._txt_entrada.configure(state="disabled", fg=TEXT_DIM)

        # — Footer con botón Iniciar —
        footer = ctk.CTkFrame(frame, fg_color=BG_CARD, height=60, corner_radius=0)
        footer.grid(row=3, column=0, sticky="ew")

        self._btn_iniciar = ctk.CTkButton(
            footer,
            text="▶  Iniciar corrección",
            fg_color=VERDE, hover_color="#3fb950", text_color="white",
            font=ctk.CTkFont("Arial", 13, "bold"), height=40, width=200,
            command=self._iniciar_correccion,
        )
        self._btn_iniciar.pack(side="left", padx=20, pady=10)

        if precargar_excel:
            ctk.CTkLabel(
                footer,
                text=f"Se corregirán {self.corrector._total} filas del Excel",
                font=ctk.CTkFont("Arial", 11), text_color=TEXT_DIM,
            ).pack(side="left", padx=8)
        else:
            ctk.CTkLabel(
                footer,
                text="Pega el texto o carga un archivo, luego presiona Iniciar",
                font=ctk.CTkFont("Arial", 11), text_color=TEXT_DIM,
            ).pack(side="left", padx=8)

    # ── Carga de archivos ────────────────────────────────────────────

    def _cargar_txt(self) -> None:
        ruta = filedialog.askopenfilename(
            title="Seleccionar archivo de transcripciones",
            filetypes=[("Texto", "*.txt"), ("Todos", "*.*")],
        )
        if not ruta:
            return
        try:
            texto = open(ruta, encoding="utf-8").read()
            self._txt_entrada.configure(state="normal", fg=TEXT)
            self._txt_entrada.delete("1.0", tk.END)
            self._txt_entrada.insert("1.0", texto)
        except Exception as e:
            messagebox.showerror("Error al leer archivo", str(e))

    def _cargar_excel(self) -> None:
        ruta = filedialog.askopenfilename(
            title="Seleccionar DataSet Excel",
            filetypes=[("Excel", "*.xlsx *.xls"), ("Todos", "*.*")],
        )
        if not ruta:
            return
        try:
            from zai.excel import GestorDataset
            nuevo_ds = GestorDataset(ruta)
            self.corrector.ds = nuevo_ds
            self.corrector.limpiar_sesion()
            self.corrector._modo_libre = False
            self._mostrar_entrada(precargar_excel=True)
            self.statusbar.set(f"✓ Excel cargado: {nuevo_ds.total} filas", color=VERDE)
        except Exception as e:
            messagebox.showerror("Error al cargar Excel", str(e))

    def _iniciar_correccion(self) -> None:
        # Si hay datos del Excel y el texto está deshabilitado, usar Excel directo
        if self.corrector.tiene_datos and not self.corrector._modo_libre:
            self._mostrar_corrector()
            return

        # Cargar desde el texto pegado
        try:
            texto = self._txt_entrada.get("1.0", tk.END)
        except Exception:
            texto = ""
        n = self.corrector.cargar_texto(texto)
        if n == 0:
            messagebox.showwarning(
                "Sin texto",
                "Pega al menos una línea de texto o carga un archivo.",
            )
            return
        self.statusbar.set(f"✓ {n} líneas cargadas", color=VERDE)
        self._mostrar_corrector()

    # ════════════════════════════════════════════════════════════════
    #  PANTALLA 2 — Corrección fila por fila
    # ════════════════════════════════════════════════════════════════

    def _mostrar_corrector(self) -> None:
        if hasattr(self, "_frame_actual") and self._frame_actual:
            self._frame_actual.destroy()

        frame = ctk.CTkFrame(self, fg_color=BG_MAIN, corner_radius=0)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_rowconfigure(2, weight=2)
        frame.grid_columnconfigure(0, weight=1)
        self._frame_actual = frame

        # — Barra superior —
        top = ctk.CTkFrame(frame, fg_color=BG_CARD, height=52, corner_radius=0)
        top.grid(row=0, column=0, sticky="ew")
        top.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(
            top, text="← Volver", width=80, height=32,
            fg_color="transparent", hover_color=BG_INPUT, text_color=TEXT_DIM,
            command=lambda: self._mostrar_entrada(
                precargar_excel=not self.corrector._modo_libre
            ),
        ).grid(row=0, column=0, padx=(8, 0), pady=10)

        self._lbl_audio = ctk.CTkLabel(
            top, text="", font=ctk.CTkFont("Arial", 13, "bold"), text_color=ACCENT_L,
        )
        self._lbl_audio.grid(row=0, column=1, padx=8)

        self._lbl_progreso = ctk.CTkLabel(
            top, text="", font=ctk.CTkFont("Arial", 12), text_color=TEXT_DIM,
        )
        self._lbl_progreso.grid(row=0, column=2, padx=8)

        btn_nav = ctk.CTkFrame(top, fg_color="transparent")
        btn_nav.grid(row=0, column=3, padx=12)

        self._btn_prev = ctk.CTkButton(
            btn_nav, text="◀", width=36, height=32,
            fg_color=BG_INPUT, hover_color=ACCENT, text_color=TEXT,
            command=self._anterior,
        )
        self._btn_prev.pack(side="left", padx=2)

        self._btn_next = ctk.CTkButton(
            btn_nav, text="▶", width=36, height=32,
            fg_color=BG_INPUT, hover_color=ACCENT, text_color=TEXT,
            command=self._siguiente,
        )
        self._btn_next.pack(side="left", padx=2)

        self._btn_analizar = ctk.CTkButton(
            btn_nav, text="🤖 Analizar", width=100, height=32,
            fg_color=AZUL, hover_color="#2979ff", text_color=TEXT,
            font=ctk.CTkFont("Arial", 12, "bold"),
            command=self._lanzar_analisis,
        )
        self._btn_analizar.pack(side="left", padx=(8, 2))

        # — Panel central: Zapoteco | Español —
        mid = ctk.CTkFrame(frame, fg_color=BG_MAIN, corner_radius=0)
        mid.grid(row=1, column=0, sticky="nsew", padx=12, pady=(8, 0))
        mid.grid_columnconfigure(0, weight=1)
        mid.grid_columnconfigure(1, weight=1)
        mid.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            mid, text="ZAPOTECO", font=ctk.CTkFont("Arial", 11, "bold"), text_color=ACCENT_L,
        ).grid(row=0, column=0, sticky="w", padx=8)

        self._txt_zapoteco = _TextBox(mid, bg=BG_INPUT, fg=TEXT, editable=True)
        self._txt_zapoteco.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(4, 0))

        ctk.CTkLabel(
            mid, text="ESPAÑOL (referencia)", font=ctk.CTkFont("Arial", 11, "bold"), text_color=AMARILLO,
        ).grid(row=0, column=1, sticky="w", padx=8)

        self._txt_espanol = _TextBox(mid, bg="#1a1f2a", fg="#c9d1d9", editable=False)
        self._txt_espanol.grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=(4, 0))

        # — Panel IA —
        ia_outer = ctk.CTkFrame(frame, fg_color=BG_CARD, corner_radius=12)
        ia_outer.grid(row=2, column=0, sticky="nsew", padx=12, pady=8)
        ia_outer.grid_rowconfigure(1, weight=1)
        ia_outer.grid_columnconfigure(0, weight=1)

        ia_header = ctk.CTkFrame(ia_outer, fg_color="transparent")
        ia_header.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 0))

        ctk.CTkLabel(
            ia_header, text="🤖 Análisis ZAI",
            font=ctk.CTkFont("Arial", 13, "bold"), text_color=AZUL,
        ).pack(side="left")

        self._lbl_modelo = ctk.CTkLabel(
            ia_header, text="Presiona 'Analizar' para empezar",
            font=ctk.CTkFont("Arial", 10), text_color=TEXT_DIM,
        )
        self._lbl_modelo.pack(side="right")

        self._txt_ia = _TextBox(ia_outer, bg=BG_CARD, fg=TEXT, editable=False, wrap=tk.WORD)
        self._txt_ia.grid(row=1, column=0, sticky="nsew", padx=8, pady=6)

        # — Barra de acciones —
        acciones = ctk.CTkFrame(frame, fg_color=BG_CARD, height=56, corner_radius=0)
        acciones.grid(row=3, column=0, sticky="ew")

        self._btn_aplicar = ctk.CTkButton(
            acciones, text="✅ Aplicar corrección",
            fg_color=VERDE, hover_color="#3fb950", text_color="white",
            font=ctk.CTkFont("Arial", 12, "bold"), width=160, height=36,
            command=self._aplicar,
        )
        self._btn_aplicar.pack(side="left", padx=12, pady=10)

        self._btn_rechazar = ctk.CTkButton(
            acciones, text="❌ Sin cambios",
            fg_color=BG_INPUT, hover_color=ROJO, text_color=TEXT,
            font=ctk.CTkFont("Arial", 12), width=120, height=36,
            command=self._rechazar,
        )
        self._btn_rechazar.pack(side="left", padx=4)

        ctk.CTkFrame(acciones, fg_color="transparent").pack(side="left", fill="x", expand=True)

        self._entrada_msg = ctk.CTkEntry(
            acciones,
            placeholder_text="Pregunta o aclaración para ZAI…",
            fg_color=BG_INPUT, text_color=TEXT, border_color=ACCENT,
            font=ctk.CTkFont("Arial", 12), height=36, width=320,
        )
        self._entrada_msg.pack(side="left", padx=4)
        self._entrada_msg.bind("<Return>", lambda e: self._enviar_mensaje())

        ctk.CTkButton(
            acciones, text="Enviar ↩",
            fg_color=ACCENT, hover_color="#1a4a80", text_color=TEXT,
            font=ctk.CTkFont("Arial", 12), width=80, height=36,
            command=self._enviar_mensaje,
        ).pack(side="left", padx=(0, 4))

        ctk.CTkButton(
            acciones, text="💾 Guardar",
            fg_color=BG_INPUT, hover_color=ACCENT, text_color=TEXT,
            font=ctk.CTkFont("Arial", 12), width=90, height=36,
            command=self._guardar,
        ).pack(side="right", padx=12)

        self._cargar_fila_actual()

    # ── Carga de fila (sin auto-análisis) ────────────────────────────

    def _cargar_fila_actual(self) -> None:
        fila = self.corrector.fila_actual
        if not fila:
            self._txt_zapoteco.set("— Fin —")
            self._txt_espanol.set("")
            self._lbl_progreso.configure(text="Completado ✓")
            return

        n, total = self.corrector.progreso
        self._lbl_audio.configure(text=f"🎙 {fila.audio}")
        self._lbl_progreso.configure(text=f"{n + 1} / {total}")
        self._txt_zapoteco.set(fila.texto_vigente)
        espanol = self.corrector.espanol_actual
        self._txt_espanol.set(espanol or "(sin traducción)")
        self._txt_ia.set("")
        self._lbl_modelo.configure(text="Presiona 'Analizar' para empezar")

    # ── Análisis (solo cuando el usuario lo pide) ────────────────────

    def _lanzar_analisis(self) -> None:
        if self._ia_activa:
            return
        self._ia_activa = True
        self._btn_analizar.configure(state="disabled", text="Analizando…")
        self._txt_ia.set("")
        self._lbl_modelo.configure(text="qwen3:4b · streaming")

        def _chunk(texto: str) -> None:
            self.after(0, lambda: self._txt_ia.append(texto))

        def _done(_: str) -> None:
            self._ia_activa = False
            self.after(0, lambda: self._btn_analizar.configure(
                state="normal", text="🤖 Analizar"))

        self.corrector.analizar(on_chunk=_chunk, on_done=_done)

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

    # ── Navegación ───────────────────────────────────────────────────

    def _aplicar(self) -> None:
        self.corrector.aplicar_correccion(self._txt_zapoteco.get_all())
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
        self.statusbar.set(f"✓ {guardadas} filas guardadas", color=VERDE)


# ── Widget TextBox reutilizable ──────────────────────────────────────

class _TextBox(tk.Text):

    def __init__(self, master, bg=BG_INPUT, fg=TEXT, editable=True, wrap=tk.WORD, **kw):
        super().__init__(
            master,
            bg=bg, fg=fg,
            insertbackground=fg,
            selectbackground="#264f78",
            relief="flat", borderwidth=0,
            padx=12, pady=8,
            wrap=wrap,
            font=("Consolas", 12),
            **kw,
        )
        if not editable:
            self.configure(state="disabled", cursor="arrow")
        self.tag_configure("amarillo", foreground=AMARILLO)
        self.tag_configure("dimmed",   foreground=TEXT_DIM)

    def set(self, texto: str) -> None:
        self.configure(state="normal")
        self.delete("1.0", tk.END)
        self.insert("1.0", texto)

    def append(self, texto: str, color: str | None = None) -> None:
        self.configure(state="normal")
        tag = "amarillo" if color == AMARILLO else ("dimmed" if color == TEXT_DIM else None)
        if tag:
            self.insert(tk.END, texto, tag)
        else:
            self.insert(tk.END, texto)
        self.see(tk.END)

    def get_all(self) -> str:
        return self.get("1.0", tk.END).strip()
