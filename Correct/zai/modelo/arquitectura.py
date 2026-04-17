"""
modelo/arquitectura.py
======================
NanoGPT para Zapoteco del Istmo.
Arquitectura idéntica a la Práctica 2 (transformersModel.ipynb),
limpiada en módulo reutilizable.

Uso:
    cfg = ConfigGPT(vocab_size=512)
    modelo = NanoGPT(cfg)
    logits, loss = modelo(x, y)          # entrenamiento
    logits, _    = modelo(x)             # inferencia
"""

from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class ConfigGPT:
    vocab_size:  int   = 512
    block_size:  int   = 32     # ventana de contexto (tokens)
    n_embd:      int   = 64     # dimensión de embeddings
    n_head:      int   = 4      # cabezas de atención
    n_layer:     int   = 3      # capas del transformer
    dropout:     float = 0.1


# ── Bloques del Transformer ──────────────────────────────────────────

class _Head(nn.Module):
    """Una cabeza de self-attention causal."""

    def __init__(self, cfg: ConfigGPT, head_size: int) -> None:
        super().__init__()
        self.key   = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)),
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v   = self.value(x)
        return wei @ v


class _MultiHead(nn.Module):
    def __init__(self, cfg: ConfigGPT) -> None:
        super().__init__()
        head_size  = cfg.n_embd // cfg.n_head
        self.heads = nn.ModuleList([_Head(cfg, head_size) for _ in range(cfg.n_head)])
        self.proj  = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop  = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.drop(self.proj(out))


class _FFN(nn.Module):
    def __init__(self, cfg: ConfigGPT) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.ReLU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Block(nn.Module):
    def __init__(self, cfg: ConfigGPT) -> None:
        super().__init__()
        self.sa   = _MultiHead(cfg)
        self.ffn  = _FFN(cfg)
        self.ln1  = nn.LayerNorm(cfg.n_embd)
        self.ln2  = nn.LayerNorm(cfg.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ── Modelo principal ─────────────────────────────────────────────────

class NanoGPT(nn.Module):
    """
    Modelo de lenguaje NanoGPT para Zapoteco.
    Compatible con la arquitectura de la Práctica 2.
    """

    def __init__(self, cfg: ConfigGPT) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks  = nn.Sequential(*[_Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f    = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)

    def forward(
        self,
        idx:     torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x   = tok + pos
        x   = self.blocks(x)
        x   = self.ln_f(x)
        logits = self.lm_head(x)          # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, C),
                targets.view(B * T),
            )
        return logits, loss

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def generar(
        self,
        idx:            torch.Tensor,
        max_new_tokens: int,
        temperature:    float = 1.0,
    ) -> torch.Tensor:
        """Genera tokens auto-regresivamente (para debug/exploración)."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs  = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

    # ── Perplejidad ──────────────────────────────────────────────────

    @torch.no_grad()
    def perplejidad(self, ids: list[int], device: str = "cpu") -> float:
        """
        Calcula la perplejidad de una secuencia de tokens.
        Perplejidad baja → el modelo "reconoce" el texto como zapoteco.
        Perplejidad alta → el texto es extraño para el modelo.
        """
        if len(ids) < 2:
            return float("inf")

        self.eval()
        block = self.cfg.block_size
        total_loss = 0.0
        n_windows  = 0

        for start in range(0, max(1, len(ids) - 1), block):
            chunk = ids[start: start + block + 1]
            if len(chunk) < 2:
                break
            x = torch.tensor(chunk[:-1], dtype=torch.long, device=device).unsqueeze(0)
            y = torch.tensor(chunk[1:],  dtype=torch.long, device=device).unsqueeze(0)
            _, loss = self(x, y)
            if loss is not None:
                total_loss += loss.item()
                n_windows  += 1

        if n_windows == 0:
            return float("inf")

        import math
        return math.exp(total_loss / n_windows)
