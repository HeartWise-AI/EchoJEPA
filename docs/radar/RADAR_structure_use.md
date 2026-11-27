# RADAR structure training use: tricuspid, pulmonic and right atrium

A structure contributes to a step only when the global batch has at least `min_pairs` valid rows for it and its
targets are not uniform (`src/models/radar/loss.py`). With exact-caption targets (A1 and its descendants), the
targets are uniform whenever every valid row has the same caption. In the merged English captions, about 95% of
tricuspid captions and 99% of pulmonic captions were "... valve has a normal appearance.", and the pulmonic valve was
routed to PSAX_AV only. As a result, these two structures were trained in few steps. `epochs.jsonl` logs each
structure's `use_fraction` and its skip reasons.

## Options (all off by default; a config without them keeps its fingerprint)

| Where | Key / flag | Effect |
|---|---|---|
| routing | `data.routing: configs/radar/radar_routing_v2.yaml` | Routes generic PSAX to the pulmonic and tricuspid valves (weak, `weak_views`). Trains the right atrium (A4C primary; also A4C_ZOOM, A5C, A5C_ZOOM, PSAX_AV, RVINF and SUBCOSTAL, the last one weak). Routes SUBCOSTAL (weak) to the RV and the pericardium. v0 and v1 are unchanged. |
| captions | `build_radar_captions_merged.py --regurgitation-grade-valves tricuspid pulmonic [mitral aortic]` | Writes a `color_flow` row per valve: its B-mode sentences plus the structured regurgitation grade, written only when the report text agrees with the grade. |
| training | `data.color_flow_captions: true` | When a bag holds a colour clip routed to the structure, reads that structure's `color_flow` row; otherwise reads the `bmode` row. Without this key, `color_flow` rows are refused, as before. |
| loss | `loss.uniform_fallback: identity` (or `{tricuspid: identity, ...}`) | When a step's targets are uniform, trains on identity targets instead of skipping the step. `epochs.jsonl` counts these steps as `fallback_identity` (distinct captions) and `fallback_identity_identical` (one caption). |
| loss | `loss.uniform_fallback_min_distinct` (default 1) | Sets the number of distinct captions a fallback needs. Use 2 to fall back only when the captions differ. |
| loss | `loss.min_pairs: {ra: 3}` | Sets `min_pairs` per structure; a structure the mapping does not name keeps 2. |

When every row has the same caption, identity targets carry no alignment signal. The image-to-text gradient is zero,
and the text-to-image loss only makes the images equally similar to that one caption. The loss detaches the
temperature in that case, so these steps cannot push it up. Captions that differ (`color_flow` rows) are what make
these valves trainable. The fallback only stops whole steps from being skipped.

## Right atrium captions

The RADAR parser has no right-atrium grammar, so the merged builder writes the RA captions itself (`ra_sentences`):

- **Size from the report text.** The RA cell is read first, using the column review's `ra_size` reading and a
  grade word when one is present ("Right atrium is mildly dilated.").
- **Size from the conclusion.** Read only when the cell says nothing, and only from segments that name the right
  atrium and no other structure.
- **Size from a measurement.** Read only when neither text source states a size. The RA area (4C) is used first,
  then the major axis, against the ASE 2015 limits (18 cm², 5.3 cm), with a grey zone that decides nothing.
- **The area value.** Written as its own sentence when it is recorded ("Right atrial area is 24 cm2.").
- **Conflicts.** When a measurement contradicts the text beyond the grey zone, nothing is written.

The report export fills the RA in few studies. Across all 249,084 reports, 9,224 get an RA caption (3.7%), with 148
distinct captions; the most common one covers 11% of RA rows. The RA cell is filled from 2020 only; about 600 of the RA sizes come from
the conclusion.

## Simulated training use (share of 48-study batches, A1, view-balanced bags of 12)

The simulation uses 9,600 random linked studies, the repository's own builders, bags and skip rules, and the
current main code. Each clip is labelled colour flow with the colour rate of its view class. These rates were
estimated from 200 random clips per view, by counting coloured pixels in the centre of the sector. They are an
estimate: the clip table has no acquisition column for the whole archive.

| structure | before (v0, B-mode rows) | v2 routing only | v2 + colour rows | v2 + colour rows + fallback |
|---|---|---|---|---|
| tricuspid | 24% | 24% | 100% | 100% |
| pulmonic | 2% | 28% | 100% | 100% |
| right atrium | not trained | 46% | 46% | 46% |
| pericardium | 58% | 60% | 60% | 100% (40% of steps with identical captions) |

In the "before" column, tricuspid fails mostly on uniform targets (51%), pulmonic mostly on too few rows (76%). In
the "v2 + colour rows" column, the most common caption's share falls from 90% to 35% for tricuspid and from 95% to 27%
for pulmonic. The right atrium is skipped for too few rows in 54% of steps: 3.3% of the simulated studies have an RA
caption.
