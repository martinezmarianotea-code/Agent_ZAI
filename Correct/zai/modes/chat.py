"""
modes/chat.py
=============
Modo Chat: conversación libre sobre zapoteco del Istmo.
Mantiene historial de conversación para contexto multi-turno.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable


@dataclass
class MensajeChat:
    rol:    str   # "user" | "assistant"
    texto:  str
    en_progreso: bool = False   # True mientras Claude escribe


class ModoChat:
    """
    Gestiona el historial y delega las respuestas al agente.
    """

    MAX_HISTORIAL = 20   # turnos conservados en memoria

    def __init__(self, agente):
        self.agente   = agente
        self.mensajes: list[MensajeChat] = []
        self._historial_api: list[dict]  = []   # formato Anthropic
        self._lock = Lock()

    # ── Historial ────────────────────────────────────────────────────

    def limpiar(self) -> None:
        with self._lock:
            self.mensajes.clear()
            self._historial_api.clear()

    def _agregar_usuario(self, texto: str) -> MensajeChat:
        msg = MensajeChat(rol="user", texto=texto)
        with self._lock:
            self.mensajes.append(msg)
            self._historial_api.append({"role": "user", "content": texto})
        return msg

    def _agregar_asistente_stream(self) -> MensajeChat:
        msg = MensajeChat(rol="assistant", texto="", en_progreso=True)
        with self._lock:
            self.mensajes.append(msg)
        return msg

    def _finalizar_asistente(self, msg: MensajeChat, texto: str) -> None:
        with self._lock:
            msg.texto       = texto
            msg.en_progreso = False
            self._historial_api.append({"role": "assistant", "content": texto})
            if len(self._historial_api) > self.MAX_HISTORIAL * 2:
                self._historial_api = self._historial_api[-(self.MAX_HISTORIAL * 2):]

    # ── Enviar mensaje ───────────────────────────────────────────────

    def enviar(
        self,
        texto:    str,
        on_chunk: Callable[[str], None],
        on_done:  Callable[[], None],
        on_nuevo_mensaje: Callable[[MensajeChat], None],
    ) -> None:
        """
        Envía un mensaje del usuario y obtiene respuesta de Claude.
        on_nuevo_mensaje se llama con cada MensajeChat nuevo (usuario y asistente).
        on_chunk se llama con cada token de la respuesta.
        on_done se llama al terminar.
        """
        self._agregar_usuario(texto)

        msg_asistente = self._agregar_asistente_stream()
        on_nuevo_mensaje(msg_asistente)

        with self._lock:
            historial_copia = self._historial_api[:-1].copy()   # sin el turno actual del usuario

        def _chunk(texto_chunk: str) -> None:
            msg_asistente.texto += texto_chunk
            on_chunk(texto_chunk)

        def _done(texto_completo: str) -> None:
            self._finalizar_asistente(msg_asistente, texto_completo)
            on_done()

        self.agente.chat_streaming(
            mensaje   = texto,
            historial = historial_copia,
            on_chunk  = _chunk,
            on_done   = _done,
        )
