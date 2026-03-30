# Pool Architecture Ideas

## 1. Post-Training Pool Pruning

### Concept
After training completes, the pool coverage tracker knows exactly which vectors were
ever accessed (via `access_frequency` in the checkpoint JSON). Untouched vectors hold
random-initialized weights and contribute nothing to model quality — they are pure
memory waste.

Pruning removes them, producing a smaller model with identical output quality.

### How It Works
1. Load final coverage report (`pool_coverage_step_500000.json`)
2. Identify all (row, col) coordinates that were accessed at least once
3. Remap the pool grid to only include those coordinates
4. Update the indexer's output mapping to match the new grid
5. Save compressed checkpoint

### Expected Gains
| Training coverage | Pool size before | Pool size after | Saving |
|---|---|---|---|
| 65% | 2.15 GB | 1.40 GB | 35% |
| 80% | 2.15 GB | 1.72 GB | 20% |
| 50% | 2.15 GB | 1.07 GB | 50% |

### Implementation Complexity
Low — coverage data already exists. Requires a post-training script (~100 lines).

---

## 2. Domain-Specific Fine-Tuning into Unused Pool Regions

### Concept
Because the pool is a spatially organized 2D grid and the indexer learns to route
different content types to different regions, new domain knowledge can be injected
into UNUSED pool regions without disturbing existing knowledge.

This is fundamentally different from dense model fine-tuning, where new knowledge
overwrites old weights (catastrophic forgetting).

### How It Works
**Base training** (web text, books, code):
- Fills pool regions A, B, C with general knowledge
- Controller + indexer learn general language understanding

**Domain fine-tuning** (e.g. private GitHub repos):
1. Freeze controller weights entirely
2. Identify unused pool coordinate ranges (from coverage report)
3. Mask gradients for already-trained pool vectors
4. Fine-tune only: indexer + empty pool vectors
5. Indexer learns to route code patterns → fresh empty region D

**Result:**
```
Pool grid after fine-tuning:
┌─────────────────────────────────┐
│  Web text knowledge  (region A) │  ← frozen, untouched
│  Book knowledge      (region B) │  ← frozen, untouched
│  General language    (region C) │  ← frozen, untouched
│  Private GitHub code (region D) │  ← newly trained
│  [empty]             (region E) │  ← reserved for next domain
└─────────────────────────────────┘
```

### Why This Is Possible
Dense models (GPT-4, Llama, Mistral) cannot do this. Fine-tuning them on new data
overwrites existing weights — the model forgets old knowledge proportionally.

DPSN separates:
- **Routing** (indexer, 63M params) — learns what to look up
- **Storage** (pool, 1.07B params) — stores actual knowledge

Fine-tuning only requires updating the indexer's routing + filling empty storage.
The base model is never touched.

### Advantages Over Standard Fine-Tuning
| Property | Dense model fine-tuning | DPSN domain fine-tuning |
|---|---|---|
| Catastrophic forgetting | Yes — unavoidable | No — base regions frozen |
| Compute cost | Full model backward pass | Indexer + empty pool only |
| Reversibility | Cannot undo | Zero out pool region |
| Multi-domain | Overwrites previous domain | Each domain gets own region |
| Data privacy | Weights entangle all data | Domains physically separated |

---

## 3. Per-Customer Pool Slices (Product Application)

### Concept
Partition the pool grid into reserved ranges per customer. Each customer's private
data is fine-tuned into their own pool slice. The controller and shared knowledge
regions are never modified.

```
Pool grid (1024 × 1024):
rows 0–600:    shared base knowledge (all customers)
rows 601–700:  Customer A (private codebase)
rows 701–800:  Customer B (private documents)
rows 801–900:  Customer C (private books)
rows 901–1023: reserved for new customers
```

### Business Model
- Sell base model as SaaS
- Each customer fine-tunes their data into their slice (~hours, not days)
- Customer data never touches shared weights — provably private
- Revoke access: zero out their pool rows
- Export: ship only their pool slice + base model

### Implementation
1. Add `--pool_row_start` / `--pool_row_end` args to training command
2. During fine-tuning, mask gradients outside customer's assigned range
3. Indexer learns a "customer prefix" routing signal
4. At inference, merge customer pool slice with base pool

### Implementation Complexity
Medium — requires gradient masking by coordinate range (~200 lines).

---

## Priority Order

1. **Pool pruning** — implement now, low effort, immediate benefit (smaller inference model)
2. **Domain fine-tuning** — implement after base training validates, high value
3. **Per-customer slices** — implement when productizing, requires inference serving infra
