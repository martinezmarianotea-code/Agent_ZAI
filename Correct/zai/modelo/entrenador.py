"""
modelo/entrenador.py
====================
Construye el corpus zapoteco y entrena el NanoGPT.

Corpus de entrenamiento:
  - Todas las transcripciones del DataSet
  - Todas las palabras del diccionario (zapoteco)
  - Palabras nuevas acumuladas

Uso desde terminal:
    cd /home/mariano/ZAI/Correct
    python -m zai.modelo.entrenador

Uso programático:
    from zai.modelo.entrenador import entrenar
    entrenar(on_progreso=print)
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable

import torch

from zai.config   import DATASET_PATH, DICCIONARIO_PATH, DATA_DIR
from zai.excel    import GestorDataset
from zai.modelo.tokenizador  import CharTokenizer
from zai.modelo.arquitectura import NanoGPT, ConfigGPT

# ── Rutas de salida ──────────────────────────────────────────────────
MODELO_DIR        = DATA_DIR / "modelo_zapoteco"
RUTA_TOKENIZADOR  = MODELO_DIR / "tokenizador.json"
RUTA_PESOS        = MODELO_DIR / "nanogpt.pt"
RUTA_CONFIG       = MODELO_DIR / "config.pt"
RUTA_CORPUS       = MODELO_DIR / "corpus.txt"     # guardado para referencia

# ── Hiperparámetros ──────────────────────────────────────────────────
HP = dict(
    # vocab_size se calcula automáticamente del corpus
    block_size  = 32,    # ventana de contexto en caracteres
    n_embd      = 64,
    n_head      = 4,
    n_layer     = 3,
    dropout     = 0.1,
    batch_size  = 64,
    max_iters   = 3000,
    eval_iters  = 100,
    eval_interval = 300,
    lr          = 5e-4,
)


# ── Construcción del corpus ──────────────────────────────────────────

def construir_corpus(log: Callable[[str], None] = print) -> str:
    """
    Extrae todo el texto zapoteco disponible y lo devuelve como string.
    """
    partes: list[str] = []

    # 1. Transcripciones del DataSet
    log("  Cargando transcripciones del DataSet...")
    ds = GestorDataset(DATASET_PATH)
    for fila in ds.filas:
        partes.append(fila.texto_vigente.strip())
    log(f"    → {len(ds.filas):,} transcripciones")

    # 2. Palabras del diccionario (solo columna zapoteco)
    log("  Cargando diccionario...")
    import openpyxl
    wb = openpyxl.load_workbook(str(DICCIONARIO_PATH), read_only=True, data_only=True)
    ws = wb.active
    dic_words: list[str] = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0]:
            dic_words.append(str(row[0]).strip())
    wb.close()
    partes.extend(dic_words)
    log(f"    → {len(dic_words):,} palabras del diccionario")

    # 3. Palabras nuevas del DataSet
    for p in ds.palabras:
        if p.zapoteco:
            partes.append(p.zapoteco.strip())

    # Unir con saltos de línea
    corpus = "\n".join(partes)
    log(f"  Corpus total: {len(corpus):,} caracteres")
    return corpus


# ── Utilidades de entrenamiento ──────────────────────────────────────

def _get_batch(
    data:       torch.Tensor,
    block_size: int,
    batch_size: int,
    device:     str,
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x  = torch.stack([data[i:i + block_size]     for i in ix])
    y  = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def _estimate_loss(
    model:      NanoGPT,
    train_data: torch.Tensor,
    val_data:   torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    device:     str,
) -> dict[str, float]:
    model.eval()
    result: dict[str, float] = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = _get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()  # type: ignore
        result[split] = losses.mean().item()
    model.train()
    return result


# ── Entrenamiento principal ──────────────────────────────────────────

def entrenar(
    on_progreso: Callable[[str], None] = print,
    hp: dict | None = None,
) -> None:
    """
    Entrena el NanoGPT zapoteco desde cero y guarda los pesos.
    on_progreso(msg) se llama con mensajes de progreso.
    """
    params = {**HP, **(hp or {})}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    on_progreso(f"Dispositivo: {device}")

    MODELO_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Corpus
    on_progreso("Construyendo corpus zapoteco...")
    corpus = construir_corpus(on_progreso)
    RUTA_CORPUS.write_text(corpus, encoding="utf-8")

    if len(corpus) < 500:
        on_progreso("ERROR: corpus muy pequeño, revisa el DataSet.")
        return

    # 2. Tokenizador carácter-a-carácter (instantáneo)
    on_progreso("Construyendo vocabulario de caracteres...")
    tok = CharTokenizer()
    tok.train(corpus)
    tok.guardar(RUTA_TOKENIZADOR)
    on_progreso(f"  Vocabulario: {tok.vocab_size} caracteres únicos")

    # 3. Codificar corpus (O(n), inmediato)
    on_progreso("Codificando corpus...")
    ids  = tok.encode(corpus)
    data = torch.tensor(ids, dtype=torch.long)

    split     = int(0.9 * len(data))
    train_data = data[:split]
    val_data   = data[split:]
    on_progreso(f"  Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")

    # 4. Modelo
    cfg = ConfigGPT(
        vocab_size = tok.vocab_size,   # se determina del corpus
        block_size = params["block_size"],
        n_embd     = params["n_embd"],
        n_head     = params["n_head"],
        n_layer    = params["n_layer"],
        dropout    = params["dropout"],
    )
    model = NanoGPT(cfg).to(device)
    on_progreso(f"  Parámetros del modelo: {model.n_params()/1e6:.2f}M")

    # 5. Entrenamiento
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"])

    on_progreso(f"\nEntrenando {params['max_iters']:,} iteraciones...")
    for step in range(params["max_iters"]):
        if (step % params["eval_interval"] == 0
                or step == params["max_iters"] - 1):
            losses = _estimate_loss(
                model, train_data, val_data,
                params["block_size"], params["batch_size"],
                params["eval_iters"], device,
            )
            on_progreso(
                f"  iter {step:4d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f}"
            )

        x, y = _get_batch(
            train_data, params["block_size"], params["batch_size"], device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    on_progreso("\nGuardando pesos...")
    torch.save(model.state_dict(), RUTA_PESOS)
    torch.save(cfg, RUTA_CONFIG)
    on_progreso(f"✓ Modelo guardado en {MODELO_DIR}")

    # 6. Demo rápida
    on_progreso("\nGenerando muestra zapoteca:")
    ctx_tensor = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample_ids = model.generar(ctx_tensor, max_new_tokens=100)[0].tolist()
    muestra = tok.decode(sample_ids)
    on_progreso(f"  {muestra[:200]}")


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("ZAI — Entrenador NanoGPT Zapoteco")
    print("=" * 60)

    def _log(msg: str) -> None:
        print(msg, flush=True)

    try:
        entrenar(on_progreso=_log)
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido.")
        sys.exit(0)
