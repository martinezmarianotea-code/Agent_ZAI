"""
views/organizer_view.py
=======================
Vista del Organizador / Diccionario:
  - Tabla de palabras nuevas del dataset (ordenables)
  - Vista del diccionario completo con búsqueda
  - Exportar / integrar al diccionario
  - Entrenamiento del NanoGPT zapoteco local
"""

from __future__ import annotations
import threading
import tkinter as tk
from tkinter import messagebox, filedialog
import customtkinter as ctk
from pathlib import Path

from zai.modes.organizer import ModoOrganizador
from zai.excel import GestorDataset, PalabraNueva
from zai.context import ContextoLinguistico
from zai.config import DICCIONARIO_PATH

from zai.gui.theme import BG_MAIN, BG_CARD, BG_INPUT, ACCENT, ACCENT_L, TEXT, TEXT_DIM, VERDE, AMARILLO, ROJO, AZUL


class OrganizerView(ctk.CTkFrame):

    def __init__(self, master, organizer: ModoOrganizador,
                 dataset: GestorDataset, contexto: ContextoLinguistico):
        super().__init__(master, fg_color=BG_MAIN, corner_radius=0)
        self.organizer = organizer
        self.dataset   = dataset
        self.contexto  = contexto
        self._palabras_nuevas: list[PalabraNueva] = []
        self._dic_filtrado: list[tuple[str, str]] = []
        self._construir()
        self._cargar_datos()

    # ── Layout ───────────────────────────────────────────────────────

    def _construir(self) -> None:
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Header
        header = ctk.CTkFrame(self, fg_color=BG_CARD, height=52, corner_radius=0)
        header.grid(row=0, column=0, sticky="ew")

        ctk.CTkLabel(header, text="📚  Organizador de Diccionario",
            font=ctk.CTkFont("Arial", 16, "bold"), text_color=TEXT,
        ).pack(side="left", padx=16, pady=14)

        self._btn_integrar = ctk.CTkButton(
            header, text="📥 Integrar al diccionario",
            fg_color=VERDE, hover_color="#3fb950", text_color="white",
            font=ctk.CTkFont("Arial", 12, "bold"), width=180, height=34,
            command=self._integrar_diccionario,
        )
        self._btn_integrar.pack(side="right", padx=4)

        self._btn_exportar = ctk.CTkButton(
            header, text="💾 Exportar Excel",
            fg_color=AZUL, hover_color="#2979ff", text_color="white",
            font=ctk.CTkFont("Arial", 12), width=130, height=34,
            command=self._exportar_excel,
        )
        self._btn_exportar.pack(side="right", padx=4)

        ctk.CTkButton(
            header, text="🔄 Recargar",
            fg_color="transparent", hover_color=BG_INPUT, text_color=TEXT_DIM,
            font=ctk.CTkFont("Arial", 11), width=90, height=34,
            command=self._cargar_datos,
        ).pack(side="right", padx=(12, 4))

        # Tabs
        self._tabview = ctk.CTkTabview(self, fg_color=BG_MAIN,
            segmented_button_fg_color=BG_CARD,
            segmented_button_selected_color=ACCENT,
            segmented_button_selected_hover_color=ACCENT,
            segmented_button_unselected_color=BG_CARD,
            segmented_button_unselected_hover_color=BG_INPUT,
            text_color=TEXT,
        )
        self._tabview.grid(row=1, column=0, sticky="nsew", padx=12, pady=8)

        self._tab_nuevas = self._tabview.add("✨ Palabras nuevas")
        self._tab_dic    = self._tabview.add("📖 Diccionario completo")
        self._tab_nano   = self._tabview.add("🧠 NanoGPT Zapoteco")

        self._construir_tab_nuevas()
        self._construir_tab_diccionario()
        self._construir_tab_nanogpt()

    def _construir_tab_nuevas(self) -> None:
        tab = self._tab_nuevas
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        # Barra superior
        top = ctk.CTkFrame(tab, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", pady=(4, 8))

        self._lbl_nuevas = ctk.CTkLabel(top, text="",
            font=ctk.CTkFont("Arial", 12), text_color=TEXT_DIM)
        self._lbl_nuevas.pack(side="left", padx=4)

        ctk.CTkButton(top, text="↕ Ordenar",
            fg_color=BG_CARD, hover_color=ACCENT, text_color=TEXT,
            font=ctk.CTkFont("Arial", 11), width=90, height=28,
            command=self._ordenar_nuevas,
        ).pack(side="right", padx=4)

        # Tabla
        tabla_frame = ctk.CTkFrame(tab, fg_color=BG_CARD, corner_radius=10)
        tabla_frame.grid(row=1, column=0, sticky="nsew")
        tabla_frame.grid_rowconfigure(0, weight=1)
        tabla_frame.grid_columnconfigure(0, weight=1)

        # Cabecera de columnas
        cabecera = ctk.CTkFrame(tabla_frame, fg_color=BG_INPUT, height=32)
        cabecera.pack(fill="x")
        ctk.CTkLabel(cabecera, text="ZAPOTECO", width=200,
            font=ctk.CTkFont("Arial", 11, "bold"), text_color=ACCENT_L,
            anchor="w").pack(side="left", padx=12)
        ctk.CTkLabel(cabecera, text="ESPAÑOL", width=200,
            font=ctk.CTkFont("Arial", 11, "bold"), text_color=AMARILLO,
            anchor="w").pack(side="left", padx=12)
        ctk.CTkLabel(cabecera, text="FUENTE",
            font=ctk.CTkFont("Arial", 11, "bold"), text_color=TEXT_DIM,
            anchor="w").pack(side="left", padx=12)

        list_frame = tk.Frame(tabla_frame, bg=BG_CARD)
        list_frame.pack(fill="both", expand=True)

        self._list_nuevas = tk.Listbox(
            list_frame,
            bg=BG_CARD, fg=TEXT, selectbackground=ACCENT,
            font=("Consolas", 12), relief="flat", bd=0,
            selectmode=tk.EXTENDED,
            activestyle="none",
        )
        self._list_nuevas.pack(side="left", fill="both", expand=True)

        scroll_n = ctk.CTkScrollbar(list_frame, command=self._list_nuevas.yview)
        scroll_n.pack(side="right", fill="y")
        self._list_nuevas.configure(yscrollcommand=scroll_n.set)

    def _construir_tab_diccionario(self) -> None:
        tab = self._tab_dic
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        # Búsqueda
        top = ctk.CTkFrame(tab, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", pady=(4, 8))

        ctk.CTkLabel(top, text="Buscar:",
            font=ctk.CTkFont("Arial", 12), text_color=TEXT_DIM,
        ).pack(side="left", padx=(4, 8))

        self._entrada_buscar = ctk.CTkEntry(top,
            placeholder_text="palabra en zapoteco o español…",
            fg_color=BG_INPUT, text_color=TEXT, border_color=ACCENT,
            font=ctk.CTkFont("Arial", 12), height=32, width=300,
        )
        self._entrada_buscar.pack(side="left")
        self._entrada_buscar.bind("<KeyRelease>", lambda e: self._filtrar_diccionario())

        self._lbl_dic = ctk.CTkLabel(top, text="",
            font=ctk.CTkFont("Arial", 11), text_color=TEXT_DIM)
        self._lbl_dic.pack(side="left", padx=12)

        # Tabla
        tabla_frame = ctk.CTkFrame(tab, fg_color=BG_CARD, corner_radius=10)
        tabla_frame.grid(row=1, column=0, sticky="nsew")

        # Cabecera
        cabecera = ctk.CTkFrame(tabla_frame, fg_color=BG_INPUT, height=32)
        cabecera.pack(fill="x")
        ctk.CTkLabel(cabecera, text="ZAPOTECO", width=200,
            font=ctk.CTkFont("Arial", 11, "bold"), text_color=ACCENT_L,
            anchor="w").pack(side="left", padx=12)
        ctk.CTkLabel(cabecera, text="ESPAÑOL",
            font=ctk.CTkFont("Arial", 11, "bold"), text_color=AMARILLO,
            anchor="w").pack(side="left", padx=12)

        list_frame = tk.Frame(tabla_frame, bg=BG_CARD)
        list_frame.pack(fill="both", expand=True)

        self._list_dic = tk.Listbox(
            list_frame,
            bg=BG_CARD, fg=TEXT, selectbackground=ACCENT,
            font=("Consolas", 12), relief="flat", bd=0,
            activestyle="none",
        )
        self._list_dic.pack(side="left", fill="both", expand=True)

        scroll_d = ctk.CTkScrollbar(list_frame, command=self._list_dic.yview)
        scroll_d.pack(side="right", fill="y")
        self._list_dic.configure(yscrollcommand=scroll_d.set)

    # ── Carga de datos ───────────────────────────────────────────────

    def _cargar_datos(self) -> None:
        # Palabras nuevas del dataset
        self._palabras_nuevas = list(self.dataset.palabras)
        self._refrescar_lista_nuevas(self._palabras_nuevas)

        # Diccionario completo ordenado
        self._dic_filtrado = self.organizer.ordenar_diccionario()
        self._refrescar_lista_diccionario(self._dic_filtrado)

    def _refrescar_lista_nuevas(self, palabras: list[PalabraNueva]) -> None:
        self._list_nuevas.delete(0, tk.END)
        for p in palabras:
            zap = p.zapoteco.ljust(22)
            esp = p.espanol.ljust(22) if p.espanol else "(sin traducción)".ljust(22)
            fte = p.fuente or ""
            self._list_nuevas.insert(tk.END, f"  {zap}  {esp}  {fte}")
        self._lbl_nuevas.configure(
            text=f"{len(palabras):,} palabras nuevas en el dataset")

    def _refrescar_lista_diccionario(self, entradas: list[tuple[str, str]]) -> None:
        self._list_dic.delete(0, tk.END)
        for zap, esp in entradas:
            z = zap.ljust(22)
            self._list_dic.insert(tk.END, f"  {z}  {esp}")
        self._lbl_dic.configure(
            text=f"{len(entradas):,} / {len(self.contexto.diccionario):,} entradas")

    # ── Acciones ─────────────────────────────────────────────────────

    def _ordenar_nuevas(self) -> None:
        ordenadas = self.organizer.ordenar(self._palabras_nuevas)
        self._palabras_nuevas = ordenadas
        self._refrescar_lista_nuevas(ordenadas)

    def _filtrar_diccionario(self) -> None:
        query = self._entrada_buscar.get().strip().lower()
        if not query:
            self._dic_filtrado = self.organizer.ordenar_diccionario()
        else:
            self._dic_filtrado = [
                (z, e) for z, e in self.organizer.ordenar_diccionario()
                if query in z.lower() or query in (e or "").lower()
            ]
        self._refrescar_lista_diccionario(self._dic_filtrado)

    def _exportar_excel(self) -> None:
        if not self._palabras_nuevas:
            messagebox.showwarning("Sin datos", "No hay palabras nuevas para exportar.")
            return

        ruta = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx")],
            initialfile="palabras_nuevas_ordenadas.xlsx",
            title="Guardar como…",
        )
        if not ruta:
            return

        n = self.organizer.exportar_excel_independiente(
            self._palabras_nuevas, Path(ruta))
        messagebox.showinfo("Exportado",
            f"{n} palabras exportadas a:\n{ruta}")

    def _integrar_diccionario(self) -> None:
        if not self._palabras_nuevas:
            messagebox.showwarning("Sin datos",
                "No hay palabras nuevas para integrar.")
            return

        seleccion = self._list_nuevas.curselection()
        if seleccion:
            palabras = [self._palabras_nuevas[i] for i in seleccion]
            detalle = f"{len(palabras)} palabras seleccionadas"
        else:
            palabras = self._palabras_nuevas
            detalle = f"todas las {len(palabras)} palabras nuevas"

        confirmar = messagebox.askyesno(
            "Integrar al diccionario",
            f"¿Integrar {detalle} al diccionario?\n\n"
            f"Ruta: {DICCIONARIO_PATH}\n\n"
            "Las palabras ya existentes serán omitidas. "
            "El diccionario quedará reordenado en el alfabeto zapoteco.",
        )
        if not confirmar:
            return

        try:
            n = self.organizer.integrar_al_diccionario(palabras, Path(DICCIONARIO_PATH))
            messagebox.showinfo("Integración completa",
                f"{n} palabras nuevas agregadas al diccionario.\n"
                "El diccionario ha sido reordenado.")
            # Recargar el diccionario en memoria
            self.contexto.diccionario = self._leer_diccionario_actualizado()
            self._cargar_datos()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo integrar:\n{e}")

    def _construir_tab_nanogpt(self) -> None:
        tab = self._tab_nano
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        # Info superior
        info = ctk.CTkFrame(tab, fg_color=BG_CARD, corner_radius=10)
        info.grid(row=0, column=0, sticky="ew", pady=(4, 8))
        info.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(info, text="🧠 NanoGPT Zapoteco",
            font=ctk.CTkFont("Arial", 14, "bold"), text_color=AZUL,
        ).grid(row=0, column=0, columnspan=2, padx=16, pady=(12, 4), sticky="w")

        ctk.CTkLabel(info,
            text=(
                "Entrena un modelo de lenguaje pequeño (NanoGPT) con todas las transcripciones\n"
                "y el diccionario zapoteco. Una vez entrenado, reemplaza las heurísticas en el\n"
                "Extractor: clasifica tokens como 'zapoteco' o 'no-zapoteco' localmente,\n"
                "sin necesidad de Ollama ni Claude."
            ),
            font=ctk.CTkFont("Arial", 11), text_color=TEXT_DIM,
            justify="left",
        ).grid(row=1, column=0, columnspan=2, padx=16, pady=(0, 8), sticky="w")

        # Estado actual
        self._lbl_nano_estado = ctk.CTkLabel(info, text="",
            font=ctk.CTkFont("Arial", 12, "bold"), text_color=AMARILLO)
        self._lbl_nano_estado.grid(row=2, column=0, padx=16, pady=(0, 12), sticky="w")

        self._btn_entrenar = ctk.CTkButton(info,
            text="▶ Entrenar modelo",
            fg_color=AZUL, hover_color="#2979ff", text_color="white",
            font=ctk.CTkFont("Arial", 13, "bold"), width=160, height=36,
            command=self._lanzar_entrenamiento,
        )
        self._btn_entrenar.grid(row=2, column=1, padx=16, pady=(0, 12), sticky="e")

        # Log de entrenamiento
        log_frame = ctk.CTkFrame(tab, fg_color=BG_CARD, corner_radius=10)
        log_frame.grid(row=1, column=0, sticky="nsew")
        log_frame.grid_rowconfigure(1, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(log_frame, text="Progreso del entrenamiento",
            font=ctk.CTkFont("Arial", 11, "bold"), text_color=TEXT_DIM,
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(8, 0))

        self._txt_nano_log = tk.Text(
            log_frame,
            bg=BG_CARD, fg=TEXT, font=("Consolas", 11),
            relief="flat", bd=0, padx=10, pady=6,
            wrap=tk.WORD, state="disabled",
        )
        self._txt_nano_log.grid(row=1, column=0, sticky="nsew", padx=6, pady=(4, 8))
        self._txt_nano_log.tag_configure("verde",   foreground=VERDE)
        self._txt_nano_log.tag_configure("amarillo",foreground=AMARILLO)
        self._txt_nano_log.tag_configure("azul",    foreground=AZUL)
        self._txt_nano_log.tag_configure("rojo",    foreground=ROJO)

        scroll_nl = ctk.CTkScrollbar(log_frame, command=self._txt_nano_log.yview)
        scroll_nl.grid(row=1, column=1, sticky="ns")
        self._txt_nano_log.configure(yscrollcommand=scroll_nl.set)

        # Mostrar estado inicial
        self._actualizar_estado_nano()

    def _actualizar_estado_nano(self) -> None:
        from zai.modelo.entrenador import RUTA_PESOS
        if RUTA_PESOS.exists():
            import os
            mb = os.path.getsize(RUTA_PESOS) / 1024 / 1024
            self._lbl_nano_estado.configure(
                text=f"✓ Modelo entrenado ({mb:.1f} MB) — listo para usar",
                text_color=VERDE)
            self._btn_entrenar.configure(text="🔄 Re-entrenar")
        else:
            self._lbl_nano_estado.configure(
                text="⚠ Modelo no entrenado — el Extractor usa heurísticas",
                text_color=AMARILLO)
            self._btn_entrenar.configure(text="▶ Entrenar modelo")

    def _log_nano(self, msg: str, tag: str = "") -> None:
        def _u():
            self._txt_nano_log.configure(state="normal")
            if tag:
                self._txt_nano_log.insert(tk.END, msg + "\n", tag)
            else:
                self._txt_nano_log.insert(tk.END, msg + "\n")
            self._txt_nano_log.see(tk.END)
            self._txt_nano_log.configure(state="disabled")
        self.after(0, _u)

    def _lanzar_entrenamiento(self) -> None:
        self._btn_entrenar.configure(state="disabled", text="Entrenando…")
        self._txt_nano_log.configure(state="normal")
        self._txt_nano_log.delete("1.0", tk.END)
        self._txt_nano_log.configure(state="disabled")
        self._log_nano("Iniciando entrenamiento NanoGPT zapoteco…", "azul")
        self._log_nano("(CPU — puede tardar 10–30 min dependiendo del hardware)\n", "amarillo")

        def _run():
            try:
                from zai.modelo.entrenador import entrenar

                def _progreso(msg: str) -> None:
                    tag = ""
                    if msg.startswith("✓"):
                        tag = "verde"
                    elif "ERROR" in msg or "error" in msg.lower():
                        tag = "rojo"
                    elif "iter" in msg:
                        tag = "amarillo"
                    self._log_nano(msg, tag)

                entrenar(on_progreso=_progreso)
                self.after(0, self._entrenamiento_completo)
            except Exception as e:
                self._log_nano(f"\nERROR: {e}", "rojo")
                self.after(0, lambda: self._btn_entrenar.configure(
                    state="normal", text="▶ Entrenar modelo"))

        threading.Thread(target=_run, daemon=True).start()

    def _entrenamiento_completo(self) -> None:
        self._actualizar_estado_nano()
        self._btn_entrenar.configure(state="normal")
        self._log_nano("\n✓ ¡Entrenamiento completado!", "verde")
        self._log_nano("Reinicia la aplicación para que el Extractor use el nuevo modelo.", "azul")

    def _leer_diccionario_actualizado(self) -> list[tuple[str, str]]:
        """Recarga el diccionario desde disco tras una integración."""
        try:
            import openpyxl
            wb = openpyxl.load_workbook(str(DICCIONARIO_PATH), read_only=True, data_only=True)
            ws = wb.active
            entradas = []
            for row in ws.iter_rows(min_row=2, values_only=True):
                if row[0]:
                    zap = str(row[0]).strip()
                    esp = str(row[1]).strip() if row[1] else ""
                    entradas.append((zap, esp))
            wb.close()
            return entradas
        except Exception:
            return self.contexto.diccionario
