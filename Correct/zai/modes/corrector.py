"""
modes/corrector.py
==================
Lógica del Modo Corrector.

Mantiene el estado de una sesión de corrección:
  - fila actual, historial de cambios
  - análisis con Ollama bajo demanda (no automático)
  - auto-guardado configurable

Fuentes de datos aceptadas:
  - Excel (GestorDataset) — comportamiento original
  - Texto libre (lista de líneas) — pegar o cargar .txt
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional

from zai.excel import GestorDataset, FilaTranscripcion
from zai.agent import ZAIAgent
from zai.config import AUTOGUARDADO_INTERVALO


@dataclass
class EstadoSesion:
    """Estado mutable de la sesión de corrección."""
    indice:       int  = 0
    guardadas:    int  = 0
    analizadas:   int  = 0
    conversacion: list[dict] = field(default_factory=list)  # historial de conversación

    def registrar_respuesta(self, zapoteco: str, respuesta_ia: str) -> None:
        """Añade el turno al historial para mantener contexto multi-turno."""
        # Mantener historial corto (últimos 6 mensajes = 3 turnos)
        self.conversacion.append({"role": "assistant", "content": respuesta_ia})
        if len(self.conversacion) > 12:
            self.conversacion = self.conversacion[-12:]

    def registrar_pregunta(self, pregunta: str) -> None:
        self.conversacion.append({"role": "user", "content": pregunta})
        if len(self.conversacion) > 12:
            self.conversacion = self.conversacion[-12:]

    def limpiar_conversacion(self) -> None:
        """Nueva fila → conversación fresca."""
        self.conversacion = []


class ModoCorrector:
    """
    Orquestador del flujo de corrección fila por fila.
    La GUI llama a los métodos públicos; este objeto gestiona el estado.
    """

    def __init__(self, dataset: GestorDataset, agente: ZAIAgent):
        self.ds     = dataset
        self.agente = agente
        self.sesion = EstadoSesion()
        self._pendientes: list[FilaTranscripcion] = []
        # Filas cargadas desde texto libre (no Excel)
        self._filas_libres: list[FilaTranscripcion] = []
        self._modo_libre = False   # True cuando se cargó texto plano

    # ── Carga de texto libre ─────────────────────────────────────────

    def cargar_texto(self, texto: str) -> int:
        """
        Carga líneas de texto plano como filas a corregir.
        Cada línea no-vacía se convierte en una fila independiente.
        Retorna el número de filas cargadas.
        """
        lineas = [l.strip() for l in texto.splitlines() if l.strip()]
        self._filas_libres = [
            FilaTranscripcion(indice=i, audio=f"línea {i + 1}", zapoteco=linea)
            for i, linea in enumerate(lineas)
        ]
        self.sesion = EstadoSesion()
        self._pendientes.clear()
        self._modo_libre = True
        return len(self._filas_libres)

    def limpiar_sesion(self) -> None:
        """Vuelve al estado inicial vacío."""
        self._filas_libres.clear()
        self._modo_libre = False
        self.sesion = EstadoSesion()
        self._pendientes.clear()

    # ── Navegación ───────────────────────────────────────────────────

    @property
    def _filas(self) -> list[FilaTranscripcion]:
        return self._filas_libres if self._modo_libre else self.ds.filas

    @property
    def _total(self) -> int:
        return len(self._filas)

    @property
    def fila_actual(self) -> Optional[FilaTranscripcion]:
        if self.sesion.indice < self._total:
            return self._filas[self.sesion.indice]
        return None

    @property
    def espanol_actual(self) -> str:
        if self._modo_libre:
            return ""
        fila = self.fila_actual
        if fila:
            return self.ds.traduccion_de(fila.audio)
        return ""

    @property
    def progreso(self) -> tuple[int, int]:
        return self.sesion.indice, self._total

    @property
    def hay_mas(self) -> bool:
        return self.sesion.indice < self._total

    @property
    def tiene_datos(self) -> bool:
        """True si hay filas cargadas (Excel o texto libre)."""
        return self._total > 0

    def avanzar(self) -> None:
        self.sesion.indice += 1
        self.sesion.limpiar_conversacion()
        self.sesion.analizadas += 1
        # Auto-guardado
        if self.sesion.analizadas % AUTOGUARDADO_INTERVALO == 0:
            self.guardar()

    def ir_a(self, n: int) -> None:
        if 0 <= n < self.ds.total:
            self.sesion.indice = n
            self.sesion.limpiar_conversacion()

    # ── Análisis ─────────────────────────────────────────────────────

    def preanalizar_actual(self) -> dict:
        """Pre-análisis rápido con Ollama (no bloquea)."""
        fila = self.fila_actual
        if not fila:
            return {}
        return self.agente.preanalizar(fila.zapoteco, self.espanol_actual)

    def analizar(
        self,
        on_chunk: Callable[[str], None],
        on_done:  Callable[[str], None],
    ) -> None:
        """Lanza análisis de la fila actual con Ollama (streaming, bajo demanda)."""
        fila = self.fila_actual
        if not fila:
            on_done("")
            return

        def _done(texto: str) -> None:
            self.sesion.registrar_respuesta(fila.zapoteco, texto)
            on_done(texto)

        self.agente.corregir_streaming(
            zapoteco  = fila.texto_vigente,
            espanol   = self.espanol_actual,
            n         = self.sesion.indice,
            total     = self._total,
            historial = self.sesion.conversacion.copy(),
            on_chunk  = on_chunk,
            on_done   = _done,
        )

    # Alias para compatibilidad con código existente
    analizar_con_claude = analizar

    def responder_usuario(
        self,
        mensaje:  str,
        on_chunk: Callable[[str], None],
        on_done:  Callable[[str], None],
    ) -> None:
        """El usuario hace una pregunta/aclaración sobre la fila actual."""
        fila = self.fila_actual
        if not fila:
            return
        self.sesion.registrar_pregunta(mensaje)

        def _done(texto: str) -> None:
            self.sesion.registrar_respuesta(fila.zapoteco, texto)
            on_done(texto)

        self.agente.corregir_streaming(
            zapoteco  = fila.texto_vigente,
            espanol   = self.espanol_actual,
            n         = self.sesion.indice,
            total     = self._total,
            historial = self.sesion.conversacion.copy(),
            on_chunk  = on_chunk,
            on_done   = _done,
        )

    # ── Aplicar cambios ──────────────────────────────────────────────

    def aplicar_correccion(self, texto_nuevo: str) -> None:
        fila = self.fila_actual
        if fila and texto_nuevo.strip():
            fila.zapoteco_corregido = texto_nuevo.strip()
            if fila not in self._pendientes:
                self._pendientes.append(fila)

    def rechazar_correccion(self) -> None:
        """Mantiene el texto original."""
        pass

    # ── Persistencia ─────────────────────────────────────────────────

    def guardar(self) -> int:
        guardadas = self.ds.guardar_correcciones(self._pendientes)
        if guardadas:
            self.sesion.guardadas += guardadas
            self._pendientes.clear()
        return guardadas

    @property
    def cambios_pendientes(self) -> int:
        return len([f for f in self._pendientes if f.fue_modificada])
