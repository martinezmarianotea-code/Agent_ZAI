"""
views/extractor_view.py
=======================
Vista del Extractor de Palabras Nuevas.
Pipeline: Tokenización → Ollama pre-clasificación → Claude traducciones
"""

from __future__ import annotations
import tkinter as tk
import customtkinter as ctk
from tkinter import messagebox
import threading

from zai.modes.extractor import ModoExtractor
from zai.excel import PalabraNueva

BG_MAIN  = "#0d1117"
BG_CARD  = "#161b22"
BG_INPUT = "#21262d"
ACCENT   = "#0f3460"
ACCENT_L = "#e94560"
TEXT     = "#eaeaea"
TEXT_DIM = "#8892a0"
VERDE    = "#2ea043"
AMARILLO = "#d29922"
AZUL     = "#1f6feb"
ROJO     = "#da3633"


class ExtractorView(ctk.CTkFrame):

    def __init__(self, master, extractor: ModoExtractor, statusbar):
        super().__init__(master, fg_color=BG_MAIN, corner_radius=0)
        self.extractor = extractor
        self.statusbar = statusbar
        self._construir()

    # ── Layout ───────────────────────────────────────────────────────

    def _construir(self) -> None:
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Header
        header = ctk.CTkFrame(self, fg_color=BG_CARD, height=52, corner_radius=0)
        header.grid(row=0, column=0, sticky="ew")

        ctk.CTkLabel(header, text="🔍  Extractor de Palabras Nuevas",
            font=ctk.CTkFont("Arial", 16, "bold"), text_color=TEXT,
        ).pack(side="left", padx=16, pady=14)

        self._btn_analizar = ctk.CTkButton(
            header, text="▶ Analizar Dataset",
            fg_color=AZUL, hover_color="#2979ff", text_color="white",
            font=ctk.CTkFont("Arial", 13, "bold"), width=150, height=34,
            command=self._lanzar_analisis,
        )
        self._btn_analizar.pack(side="right", padx=12)

        # Panel principal
        main = ctk.CTkFrame(self, fg_color=BG_MAIN, corner_radius=0)
        main.grid(row=1, column=0, sticky="nsew", padx=12, pady=8)
        main.grid_rowconfigure(1, weight=1)
        main.grid_columnconfigure(0, weight=1)
        main.grid_columnconfigure(1, weight=2)

        # — Columna izquierda: estadísticas + log —
        izq = ctk.CTkFrame(main, fg_color=BG_CARD, corner_radius=12)
        izq.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 6))
        izq.grid_rowconfigure(1, weight=1)
        izq.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(izq, text="Progreso", font=ctk.CTkFont("Arial", 13, "bold"),
            text_color=ACCENT_L).grid(row=0, column=0, padx=12, pady=10, sticky="w")

        self._txt_log = tk.Text(
            izq, bg=BG_CARD, fg=TEXT_DIM, font=("Consolas", 11),
            relief="flat", bd=0, padx=10, pady=8, wrap=tk.WORD, state="disabled",
        )
        self._txt_log.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 8))
        self._txt_log.tag_configure("verde",   foreground=VERDE)
        self._txt_log.tag_configure("amarillo",foreground=AMARILLO)
        self._txt_log.tag_configure("azul",    foreground=AZUL)

        # — Columna derecha: tabla de palabras —
        der = ctk.CTkFrame(main, fg_color=BG_CARD, corner_radius=12)
        der.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(6, 0))
        der.grid_rowconfigure(1, weight=1)
        der.grid_columnconfigure(0, weight=1)

        top_der = ctk.CTkFrame(der, fg_color="transparent")
        top_der.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 0))

        ctk.CTkLabel(top_der, text="Palabras candidatas",
            font=ctk.CTkFont("Arial", 13, "bold"), text_color=TEXT,
        ).pack(side="left")

        self._lbl_count = ctk.CTkLabel(top_der, text="",
            font=ctk.CTkFont("Arial", 11), text_color=TEXT_DIM)
        self._lbl_count.pack(side="left", padx=8)

        # Botones de acción
        acc = ctk.CTkFrame(top_der, fg_color="transparent")
        acc.pack(side="right")

        self._btn_traducciones = ctk.CTkButton(acc, text="🤖 Pedir traducciones",
            fg_color=AZUL, hover_color="#2979ff", text_color="white",
            font=ctk.CTkFont("Arial", 11), width=160, height=30,
            command=self._pedir_traducciones, state="disabled",
        )
        self._btn_traducciones.pack(side="left", padx=4)

        self._btn_guardar = ctk.CTkButton(acc, text="💾 Guardar selección",
            fg_color=VERDE, hover_color="#3fb950", text_color="white",
            font=ctk.CTkFont("Arial", 11), width=140, height=30,
            command=self._guardar_seleccion, state="disabled",
        )
        self._btn_guardar.pack(side="left", padx=4)

        # Tabla (listbox simple con scrollbar)
        tabla_frame = ctk.CTkFrame(der, fg_color=BG_INPUT, corner_radius=8)
        tabla_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        tabla_frame.grid_rowconfigure(0, weight=1)
        tabla_frame.grid_columnconfigure(0, weight=1)

        self._lista = tk.Listbox(
            tabla_frame,
            bg=BG_INPUT, fg=TEXT, selectbackground=ACCENT,
            font=("Consolas", 12), relief="flat", bd=0,
            selectmode=tk.EXTENDED,
        )
        self._lista.grid(row=0, column=0, sticky="nsew")

        scroll = ctk.CTkScrollbar(tabla_frame, command=self._lista.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self._lista.configure(yscrollcommand=scroll.set)

        # Panel de sugerencias Claude
        sug_frame = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=12)
        sug_frame.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 8))
        sug_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(sug_frame, text="🤖 Sugerencias de traducción (Claude)",
            font=ctk.CTkFont("Arial", 12, "bold"), text_color=AZUL,
        ).grid(row=0, column=0, padx=12, pady=(8, 0), sticky="w")

        self._txt_sugerencias = tk.Text(
            sug_frame, bg=BG_CARD, fg=TEXT, font=("Consolas", 11),
            relief="flat", bd=0, padx=10, pady=6, wrap=tk.WORD,
            height=6, state="disabled",
        )
        self._txt_sugerencias.grid(row=1, column=0, sticky="ew", padx=8, pady=(4, 8))

    # ── Análisis ─────────────────────────────────────────────────────

    def _lanzar_analisis(self) -> None:
        self._btn_analizar.configure(state="disabled", text="Analizando…")
        self._lista.delete(0, tk.END)
        self._limpiar_log()
        self._limpiar_sugerencias()

        def _run():
            resultado = self.extractor.analizar_dataset(
                on_progreso=lambda msg: self.after(0, lambda m=msg: self._log(m))
            )
            self.after(0, lambda: self._mostrar_resultado(resultado))

        threading.Thread(target=_run, daemon=True).start()

    def _mostrar_resultado(self, resultado) -> None:
        self._btn_analizar.configure(state="normal", text="▶ Analizar Dataset")
        self._lbl_count.configure(
            text=f"{len(resultado.palabras_nuevas):,} candidatas")

        for p in resultado.palabras_nuevas:
            self._lista.insert(tk.END, f"  {p.zapoteco}")

        self._log(
            f"\n✓ {len(resultado.palabras_nuevas):,} palabras nuevas identificadas", "verde")
        self._btn_traducciones.configure(state="normal")
        self._btn_guardar.configure(state="normal")
        self.statusbar.set(
            f"Extracción completa: {len(resultado.palabras_nuevas):,} candidatas", "#2ea043")

    def _pedir_traducciones(self) -> None:
        if not self.extractor.resultado:
            return
        self._btn_traducciones.configure(state="disabled")
        palabras = [p.zapoteco for p in self.extractor.resultado.palabras_nuevas[:40]]

        self._txt_sugerencias.configure(state="normal")
        self._txt_sugerencias.delete("1.0", tk.END)

        def _chunk(texto: str) -> None:
            def _u():
                self._txt_sugerencias.configure(state="normal")
                self._txt_sugerencias.insert(tk.END, texto)
                self._txt_sugerencias.see(tk.END)
                self._txt_sugerencias.configure(state="disabled")
            self.after(0, _u)

        def _done(_: str) -> None:
            self.after(0, lambda: self._btn_traducciones.configure(state="normal"))

        self.extractor.pedir_traducciones_claude(palabras, _chunk, _done)

    def _guardar_seleccion(self) -> None:
        if not self.extractor.resultado:
            return
        seleccion = self._lista.curselection()
        if not seleccion:
            # Guardar todas
            palabras = self.extractor.resultado.palabras_nuevas
        else:
            palabras = [self.extractor.resultado.palabras_nuevas[i] for i in seleccion]

        guardadas = self.extractor.guardar_seleccionadas(palabras)
        messagebox.showinfo("Guardadas", f"{guardadas} palabras nuevas guardadas en el dataset.")
        self.statusbar.set(f"✓ {guardadas} palabras guardadas", VERDE)

    # ── Helpers ──────────────────────────────────────────────────────

    def _log(self, msg: str, tag: str = "") -> None:
        self._txt_log.configure(state="normal")
        if tag:
            self._txt_log.insert(tk.END, msg + "\n", tag)
        else:
            self._txt_log.insert(tk.END, msg + "\n")
        self._txt_log.see(tk.END)
        self._txt_log.configure(state="disabled")

    def _limpiar_log(self) -> None:
        self._txt_log.configure(state="normal")
        self._txt_log.delete("1.0", tk.END)
        self._txt_log.configure(state="disabled")

    def _limpiar_sugerencias(self) -> None:
        self._txt_sugerencias.configure(state="normal")
        self._txt_sugerencias.delete("1.0", tk.END)
        self._txt_sugerencias.configure(state="disabled")
