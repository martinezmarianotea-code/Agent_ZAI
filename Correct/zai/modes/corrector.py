"""
modes/corrector.py
==================
Lógica del Modo Corrector.

Mantiene el estado de una sesión de corrección:
  - fila actual, historial de cambios
  - integra el pre-análisis de Ollama + la corrección de Claude
  - auto-guardado configurable
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

    # ── Navegación ───────────────────────────────────────────────────

    @property
    def fila_actual(self) -> Optional[FilaTranscripcion]:
        if self.sesion.indice < self.ds.total:
            return self.ds.obtener_fila(self.sesion.indice)
        return None

    @property
    def espanol_actual(self) -> str:
        fila = self.fila_actual
        if fila:
            return self.ds.traduccion_de(fila.audio)
        return ""

    @property
    def progreso(self) -> tuple[int, int]:
        return self.sesion.indice, self.ds.total

    @property
    def hay_mas(self) -> bool:
        return self.sesion.indice < self.ds.total

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

    def analizar_con_claude(
        self,
        on_chunk: Callable[[str], None],
        on_done:  Callable[[str], None],
    ) -> None:
        """Lanza análisis profundo con Claude (streaming en background thread)."""
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
            total     = self.ds.total,
            historial = self.sesion.conversacion.copy(),
            on_chunk  = on_chunk,
            on_done   = _done,
        )

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
            total     = self.ds.total,
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
