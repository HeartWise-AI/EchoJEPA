# RADAR training: severity-weighted positives and per-epoch validation retrieval

Two opt-in options of `evals/radar_frozen/train.py`, taken from DeepCORO_CLIP. Both are off when their key is absent, so
existing configs train exactly as before (same losses, same fingerprint).

## Severity weights (`loss.severity_weights`)

DeepCORO_CLIP weights positives by severity (normal < mild < moderate < severe). Here each row of the structure loss
(both directions) is weighted by its caption's grade, and the per-structure loss is the weighted mean
`sum_i w_i CE_i / sum_i w_i` (DeepCORO's importance-weighted multi-positive InfoNCE). The targets Q are unchanged, so
the option composes with every arm (A0-A3).

| Grade | Default weight | From |
|---|---|---|
| normal | 1.0 | `normality` = normal |
| mild | 1.5 | abnormal, a `sources` value with `mild`/`mildly` |
| moderate | 2.5 | abnormal, a `sources` value with `moderate`/`moderately` (`mild_to_moderate` too) |
| severe | 3.0 | abnormal, a `sources` value with `severe`/`severely` |
| abnormal | 1.5 | abnormal without a graded value (v0 tables have no `sources`) |
| unknown | 1.0 | `normality` = unknown |

The grade is read from structured columns (`normality` and the merged builder's `sources` values), never from the
caption text (`src/models/radar/severity.py`). Config: `severity_weights: true` for the defaults, or a mapping that
overrides some of them (`{severe: 4.0}`). The run writes `severity_weights.json` (rows per grade and structure); each
weighted structure's step stats carry `mean_weight`. The key is part of the fingerprint when set. The merged
builder's `normality` is provisional: with this option on, A0/A1 runs lean on it although they otherwise do not.

## Validation monitoring (`validation`)

After each epoch's checkpoint, rank 0 embeds the validation split from its cached tokens
(`evals/radar_frozen/val_monitor.py`) and reports, per trained structure:

- video -> caption Recall@1/5/10, MRR and median rank, with de-duplicated captions as gallery (DeepCORO's validation).
  Many studies share a caption, so chance Recall@1 is about 1 / `unique_captions`, not 1 / studies;
- with `validation.labels`, the zero-shot AUROC of each binary prompt task (`zero_shot_eval.py` scoring) and their mean.

Results go to `validation.jsonl` and to W&B under `val/` (no caption, no study id). The section is run control:
outside the fingerprint, no checkpoint selection, training random streams restored, so the run trains bit for bit as
without it. Keys: `clips`, `captions`, `split` (val), `cache_dir` (data.cache_dir), `every` (1; the last epoch is always
validated), `num_bags` (1), `eval_seed` (0), `batch_size` (32), `recall_k` ([1, 5, 10]), `labels`, `prompts`
(`configs/radar/prompts_v0.yaml`). Clips only (a stills run is validated on its clips). Under DDP the other ranks
wait at the epoch barrier, so a large validation split should use `every` > 1 or a subset to stay within the process
group timeout.
