# DeepECHO: confirm prefilled columns in Labelbox

**This is the proposed Labelbox review interface for the current PR.** It replaces the generic five-slot
proposal with named clinical columns, dropdown values, report-based draft captions, and typed omissions and
unsupported statements. The website uses the same field IDs and choices. The older RADAR v1 taxonomy and
approval record remain versioned separately; this change does not record medical approval.

The current files are selected by [current_review.json](../../configs/radar/labelbox/current_review.json):

- [Column/value/view specification](../../configs/radar/column_review_v2.json)
- [Native Labelbox ontology](../../configs/radar/labelbox/column_review_v2.json)
- [Prefill builder and scorer](../../tools/labelbox/column_review.py)
- [Ontology/context/MAL importer](../../tools/labelbox/import_column_review.py)
- [Whole-study demonstration](https://deepecho-three-view-demo.papirobbi.chatgpt.site)

## The usual annotation takes one confirmation

Each clip opens with its proposed view and acquisition mode, relevant report values, and a draft caption.
Explicitly reported normal findings are prefilled too:

| Report field | Explicit report text | Prefilled column/value | Proposed caption |
|---|---|---|---|
| Tricuspid valve findings | Valve tricuspide normale | tricuspid_appearance = normal | The visible tricuspid valve has a normal appearance. |
| Right ventricule findings | Fonction systolique normale du ventricule droit | rv_function = normal | RV systolic function appears normal in this input. |
| Left ventricule findings | Ventricule gauche non-dilaté | lv_size = normal | The visible LV cavity is not dilated. |

The native form selects applicable clinical columns, fills their values, and suggests **Supported**.
It also suggests that the view and mode are correct and that no additional information is missing or unsupported.
The reviewer checks the displayed input and target coverage, changes exceptions, and selects the required
**Review completed** answer before submitting. That answer is never part of imported predictions.
Imported suggestions cannot be scored as submitted annotations.

An empty report section stays empty. Mild TR, no TR, or a normal RV does not establish normal tricuspid
leaflet appearance. Normal appearance does not establish normal motion or absence of a jet.
The narrow extractor handles explicit phrases, uncertainty, conflicting assertions, and verified RV function codes.
It does not convert numerical EF into arbitrary qualitative thresholds. Unrecognized phrases produce no proposal
and remain available as report context.

## Per-column exceptions

| Reviewer action | Required structured detail | Scoring and corrected caption |
|---|---|---|
| Supported | Confirm the original value against this input | Original claim supported and retained |
| Should not be here | Error reason plus removal or corrected value | Original claim rejected; correction recorded separately |
| Cannot assess | Assessment limitation | Unresolved; neither an error nor a clinical negative |
| Important finding missing | Missing column/value and visible-evidence confirmation | Omission recorded separately and added after review |
| Additional unsupported statement | Clinical column or numerical/other statement type plus error reason | Counted separately and omitted from rebuilt caption |

Error reasons are **wrong value/diagnosis, wrong severity, wrong structure/segment, contradicted by visible
evidence, not visible, wrong view/target, wrong modality, study-level claim in a local caption, and no supporting
source/invented assertion**. Limitations are poor image quality, incomplete cycle, target outside the displayed
input, unresolved view/slice level, and insufficient input requiring other views or measurements.

Original values and source text remain in an immutable context attachment and the local candidate.
A value correction uses its explicit correction control. An existing column cannot also be counted as an
omission. Removing a rejected claim never inverts it to normal or disease absent.

## Routing and values

The specification contains **46 named clinical columns** and all **20 current view labels**. It includes chamber
function/size, four valves' appearance/motion/devices and regurgitant jets, en-face aortic cusp count, the routine
16 LV wall-motion segments, local pericardial fluid, IVC findings, and separately located aortic findings.
The full tables below are generated from the same specification.

Confirm focused/zoom coverage, generic PSAX slice level, and subcostal subtype. A4C does not route aortic-valve
findings; A2C does not route RV or tricuspid findings. Mitral prolapse assessment uses PLAX/A3C, and en-face
aortic cusp count uses PSAX_AV. Regurgitant jets require color flow and matching target coverage.

Integrated valve severity, exact EF, pressures, velocities and gradients stay in study context or separately
validated targets. Still images cannot establish contraction, regional motion, leaflet excursion, or respiratory collapse.
WMS proposals retain ASE ambiguity: score 1 means normal/hyperkinetic; score 3 means akinetic/severely hypokinetic.
The dataset's WMS codebook still needs independent certification. A3C apical segments have no dedicated report
column in this export and receive no inferred normal proposal. SAX columns use the mid-level template and are
not broadcast to basal/apical slices. The apical cap is excluded.

## Native ontology and prelabels

Build candidates with the builder's report, view, acquisition mode and opaque row-key arguments. Retain exact
input provenance and use confirmed targets for focused acquisitions. Whitelisted clinical report context includes
relevant numerical fields without turning those numbers into local claims.
Raw report pages, accession numbers and linkage manifests are not annotation payloads.

Generate the schema, native annotations and text context offline:

~~~bash
python -m tools.labelbox.column_review --check
python -m tools.labelbox.column_review --candidates /absolute/path/candidates.json --output /absolute/path/review-bundle
~~~

Apply to an explicitly selected video project whose data rows already exist in Labelbox:

~~~bash
python -m tools.labelbox.import_column_review --project-id YOUR_PROJECT_ID --candidates /absolute/path/candidates.json
# With LABELBOX_API_KEY in the environment, apply the reviewed bundle:
python -m tools.labelbox.import_column_review --project-id YOUR_PROJECT_ID --candidates /absolute/path/candidates.json --apply
~~~

The importer checks every existing video row, the complete schema and original context before writing.
It creates or reuses the v2 ontology, connects a compatible project, creates a stable batch of existing rows,
adds original-proposal context, and imports **MAL predictions**. A retry reuses its matching job.
It refuses conflicting schemas or context and never uploads videos, imports ground truth, invites users, or
replaces an incompatible live ontology.

The native form uses global video classifications and searchable choices. JSON annotations preserve the
full nested hierarchy. Labelbox documents arbitrary nesting for JSON and one nested level for its Python
annotation types; the implementation therefore uses native NDJSON dictionaries.

## Scoring

Decode native annotation dictionaries and validate required answers at their exact nested paths.
Supply the scorer's submitted flag only for a verified human submission; its default is false.
Mutable caption text is not accepted as clinical evidence.

- Original support rate = supported original columns / populated original columns.
- Original error rate = rejected original columns / (supported + rejected original columns).
- Cannot-assess columns, corrections, omissions and extra unsupported statements have separate counts.
- Empty denominators return null.

The rebuilt caption includes supported findings and explicit visible corrections/omissions. Wrong or unresolved
view, acquisition or target coverage, missing details or absent final confirmation prevents a clinical caption.
Exact spatial/temporal provenance additionally gates readiness for downstream validation. That flag is not
medical approval or a release to training.

## Complete study

Study B accounts for **all 79 DICOM objects** in available metadata: 44 cine loops, 33 cropped image exports,
one identifying report screenshot represented as deidentified clinical text, and one structured-report record.
OTHER/unclassified acquisitions remain available for correction. Spectral and analysis acquisitions remain
visible as context without automatically routed qualitative claims.

The report distinguishes autoEF 48% and narrative visual EF 50%; its structured EF field is 48%. This stays
separate from the mildly reduced qualitative LV-function proposal. The earlier A4C/A2C_LV/A3C_LV demonstrations
belong to study A; PLAX/PSAX_AV belong to study B and retain that report.

The site shows default captions, saves reviews locally, and exports original and corrected candidates.
It provides the native ontology and all 44 video prelabels. Display crops have unverified production-input
provenance and are clinical review illustrations.

## Sources and verification

- Actual DeepECHO report fields and complete study-linked acquisition metadata.
- [ASE chamber quantification and segment maps](https://www.asecho.org/wp-content/uploads/2016/02/2015_ChamberQuantificationREV.pdf).
- [ASE reporting, Table 7](https://www.asecho.org/wp-content/uploads/2025/09/PIIS0894731725002925.pdf).
- [ASE comprehensive TTE acquisition](https://www.asecho.org/wp-content/uploads/2019/01/2019_Comprehensive-TTE.pdf).
- [ASE native valve morphology](https://www.asecho.org/wp-content/uploads/2025/04/2017VavularRegurgitationGuideline.pdf).
- [Labelbox native video annotation formats](https://docs.labelbox.com/horizon/developer-guides/import-video-annotations).
- [Labelbox model-assisted prelabels](https://docs.labelbox.com/horizon/guides/model-assisted-labeling).

Offline checks cover explicit normals, conservative extraction, routing, typed omissions/errors, submission-only
scoring, SDK roundtrip, conflicting schemas, context attachments, fake-client import retries, and site/Python parity.
Live Labelbox editor inspection and installation require the target project and authenticated API access.

<!-- BEGIN GENERATED: column_review_tables -->
## Clinical columns and choices

| Column | Values | Report source |
|---|---|---|
| `lv_function` | hyperdynamic, normal, mildly_reduced, moderately_reduced, severely_reduced | `Left ventricule findings` |
| `lv_size` | normal, dilated | `Left ventricule findings` |
| `lv_wall_thickness` | not_increased, increased | `Left ventricule findings`; `Ventricular septum findings` |
| `rv_function` | normal, mildly_reduced, moderately_reduced, severely_reduced | `Right ventricule findings`; `MHI VD fonction systolique` |
| `rv_size` | normal, dilated | `Right ventricule findings` |
| `la_size` | normal, dilated | `Left atrial findings` |
| `ra_size` | normal, dilated | `Right Atrial findings` |
| `pericardial_effusion` | present, not_demonstrated | `Pericardium findings` |
| `ivc_size` | normal, dilated | `Inferior vena cava findings` |
| `ivc_collapse` | preserved, reduced | `Inferior vena cava findings` |
| `mitral_appearance` | normal, thickened, calcified, sclerotic | `Mitral valve findings` |
| `mitral_motion` | normal, restricted, tethered, flail, prolapse | `Mitral valve findings` |
| `mitral_device` | edge_to_edge_clip, annuloplasty_ring, bioprosthesis, mechanical_prosthesis | `Mitral valve findings` |
| `mitral_regurgitant_jet` | demonstrated, not_demonstrated | `Mitral valve findings` |
| `aortic_appearance` | normal, thickened, calcified, sclerotic | `Aortic valve findings` |
| `aortic_motion` | normal, restricted | `Aortic valve findings` |
| `aortic_device` | edge_to_edge_clip, annuloplasty_ring, bioprosthesis, mechanical_prosthesis | `Aortic valve findings` |
| `aortic_regurgitant_jet` | demonstrated, not_demonstrated | `Aortic valve findings` |
| `tricuspid_appearance` | normal, thickened, calcified, sclerotic | `Tricuspid valve findings` |
| `tricuspid_motion` | normal, restricted, tethered, flail, prolapse | `Tricuspid valve findings` |
| `tricuspid_device` | edge_to_edge_clip, annuloplasty_ring, bioprosthesis, mechanical_prosthesis | `Tricuspid valve findings` |
| `tricuspid_regurgitant_jet` | demonstrated, not_demonstrated | `Tricuspid valve findings` |
| `pulmonic_appearance` | normal, thickened, calcified, sclerotic | `Pulmonary valve findings` |
| `pulmonic_motion` | normal, restricted | `Pulmonary valve findings` |
| `pulmonic_device` | edge_to_edge_clip, annuloplasty_ring, bioprosthesis, mechanical_prosthesis | `Pulmonary valve findings` |
| `pulmonic_regurgitant_jet` | demonstrated, not_demonstrated | `Pulmonary valve findings` |
| `aortic_cusp_count` | two, three | `Aortic valve findings` |
| `aorta_root_size` | normal, dilated | `Aorta findings` |
| `aorta_ascending_size` | normal, dilated | `Aorta findings` |
| `aorta_arch_size` | normal, dilated | `Aorta findings` |
| `wall_basal_inferoseptal` | normal, hyperkinetic, normal_or_hyperkinetic, hypokinetic, severely_hypokinetic, akinetic, akinetic_or_severely_hypokinetic, dyskinetic, aneurysmal | `WMS Basal Inferoseptal` |
| `wall_mid_inferoseptal` | normal, hyperkinetic, normal_or_hyperkinetic, hypokinetic, severely_hypokinetic, akinetic, akinetic_or_severely_hypokinetic, dyskinetic, aneurysmal | `WMS 4C Mid Inferoseptal`; `WMS SAX Inferoseptal` |
| `wall_apical_septal` | normal, hyperkinetic, normal_or_hyperkinetic, hypokinetic, severely_hypokinetic, akinetic, akinetic_or_severely_hypokinetic, dyskinetic, aneurysmal | `WMS 4C Apical septum` |
| `wall_basal_anterolateral` | normal, hyperkinetic, normal_or_hyperkinetic, hypokinetic, severely_hypokinetic, akinetic, akinetic_or_severely_hypokinetic, dyskinetic, aneurysmal | `WMS 4C Basal Anterolateral` |
| `wall_mid_anterolateral` | normal, hyperkinetic, normal_or_hyperkinetic, hypokinetic, severely_hypokinetic, akinetic, akinetic_or_severely_hypokinetic, dyskinetic, aneurysmal | `WMS 4C Mid Anterolateral`; `WMS SAX Anterolateral` |
| `wall_apical_lateral` | normal, hyperkinetic, normal_or_hyperkinetic, hypokinetic, severely_hypokinetic, akinetic, akinetic_or_severely_hypokinetic, dyskinetic, aneurysmal | `WMS 4C Apical Lateral` |
| `wall_basal_inferior` | normal, hyperkinetic, normal_or_hyperkinetic, hypokinetic, severely_hypokinetic, akinetic, akinetic_or_severely_hypokinetic, dyskinetic, aneurysmal | `WMS 2C Basal Inferior` |
| `wall_mid_inferior` | normal, hyperkinetic, normal_or_hyperkinetic, hypokinetic, severely_hypokinetic, akinetic, akinetic_or_severely_hypokinetic, dyskinetic, aneurysmal | `WMS 2C Mid Inferior`; `WMS SAX Inferior` |
| `wall_apical_inferior` | normal, hyperkinetic, normal_or_hyperkinetic, hypokinetic, severely_hypokinetic, akinetic, akinetic_or_severely_hypokinetic, dyskinetic, aneurysmal | `WMS 2C Apical Inferior` |
| `wall_basal_anterior` | normal, hyperkinetic, normal_or_hyperkinetic, hypokinetic, severely_hypokinetic, akinetic, akinetic_or_severely_hypokinetic, dyskinetic, aneurysmal | `WMS 2C Basal Anterior` |
| `wall_mid_anterior` | normal, hyperkinetic, normal_or_hyperkinetic, hypokinetic, severely_hypokinetic, akinetic, akinetic_or_severely_hypokinetic, dyskinetic, aneurysmal | `WMS 2C Mid Anterior`; `WMS SAX Anterior` |
| `wall_apical_anterior` | normal, hyperkinetic, normal_or_hyperkinetic, hypokinetic, severely_hypokinetic, akinetic, akinetic_or_severely_hypokinetic, dyskinetic, aneurysmal | `WMS 2C Apical Anterior` |
| `wall_basal_inferolateral` | normal, hyperkinetic, normal_or_hyperkinetic, hypokinetic, severely_hypokinetic, akinetic, akinetic_or_severely_hypokinetic, dyskinetic, aneurysmal | `WMS LAX Basal Inferolateral` |
| `wall_mid_inferolateral` | normal, hyperkinetic, normal_or_hyperkinetic, hypokinetic, severely_hypokinetic, akinetic, akinetic_or_severely_hypokinetic, dyskinetic, aneurysmal | `WMS LAX Mid Inferolateral`; `WMS SAX Inferolateral` |
| `wall_basal_anteroseptal` | normal, hyperkinetic, normal_or_hyperkinetic, hypokinetic, severely_hypokinetic, akinetic, akinetic_or_severely_hypokinetic, dyskinetic, aneurysmal | `WMS LAX Basal Anteroseptal` |
| `wall_mid_anteroseptal` | normal, hyperkinetic, normal_or_hyperkinetic, hypokinetic, severely_hypokinetic, akinetic, akinetic_or_severely_hypokinetic, dyskinetic, aneurysmal | `WMS LAX Mid Anteroseptal`; `WMS SAX Anteroseptal` |

## Routes for all 20 labels

| View | Possible columns (confirm coverage) | Conditional targets |
|---|---|---|
| A4C | `lv_function`, `lv_size`, `lv_wall_thickness`, `rv_function`, `rv_size`, `la_size`, `ra_size`, `pericardial_effusion`, `mitral_appearance`, `mitral_motion`, `mitral_device`, `mitral_regurgitant_jet`, `tricuspid_appearance`, `tricuspid_motion`, `tricuspid_device`, `tricuspid_regurgitant_jet`, `wall_basal_inferoseptal`, `wall_mid_inferoseptal`, `wall_apical_septal`, `wall_basal_anterolateral`, `wall_mid_anterolateral`, `wall_apical_lateral` |  |
| A4C_LV | `lv_function`, `lv_size`, `lv_wall_thickness`, `rv_function`, `rv_size`, `la_size`, `ra_size`, `pericardial_effusion`, `mitral_appearance`, `mitral_motion`, `mitral_device`, `mitral_regurgitant_jet`, `tricuspid_appearance`, `tricuspid_motion`, `tricuspid_device`, `tricuspid_regurgitant_jet`, `wall_basal_inferoseptal`, `wall_mid_inferoseptal`, `wall_apical_septal`, `wall_basal_anterolateral`, `wall_mid_anterolateral`, `wall_apical_lateral` | rv, la, ra, mitral, tricuspid, pericardium |
| A2C | `lv_function`, `lv_size`, `lv_wall_thickness`, `la_size`, `pericardial_effusion`, `mitral_appearance`, `mitral_motion`, `mitral_device`, `mitral_regurgitant_jet`, `wall_basal_inferior`, `wall_mid_inferior`, `wall_apical_inferior`, `wall_basal_anterior`, `wall_mid_anterior`, `wall_apical_anterior` |  |
| A2C_LV | `lv_function`, `lv_size`, `lv_wall_thickness`, `la_size`, `pericardial_effusion`, `mitral_appearance`, `mitral_motion`, `mitral_device`, `mitral_regurgitant_jet`, `wall_basal_inferior`, `wall_mid_inferior`, `wall_apical_inferior`, `wall_basal_anterior`, `wall_mid_anterior`, `wall_apical_anterior` | la, mitral, pericardium |
| A3C | `lv_function`, `lv_size`, `lv_wall_thickness`, `la_size`, `pericardial_effusion`, `mitral_appearance`, `mitral_motion`, `mitral_device`, `mitral_regurgitant_jet`, `aortic_appearance`, `aortic_motion`, `aortic_device`, `aortic_regurgitant_jet`, `wall_apical_septal`, `wall_apical_lateral`, `wall_basal_inferolateral`, `wall_mid_inferolateral`, `wall_basal_anteroseptal`, `wall_mid_anteroseptal` |  |
| A3C_LV | `lv_function`, `lv_size`, `lv_wall_thickness`, `la_size`, `pericardial_effusion`, `mitral_appearance`, `mitral_motion`, `mitral_device`, `mitral_regurgitant_jet`, `aortic_appearance`, `aortic_motion`, `aortic_device`, `aortic_regurgitant_jet`, `wall_apical_septal`, `wall_apical_lateral`, `wall_basal_inferolateral`, `wall_mid_inferolateral`, `wall_basal_anteroseptal`, `wall_mid_anteroseptal` | la, mitral, aortic, pericardium |
| A5C | `lv_function`, `lv_size`, `lv_wall_thickness`, `la_size`, `pericardial_effusion`, `mitral_appearance`, `mitral_motion`, `mitral_device`, `mitral_regurgitant_jet`, `aortic_appearance`, `aortic_motion`, `aortic_device`, `aortic_regurgitant_jet` |  |
| PLAX | `lv_function`, `lv_size`, `lv_wall_thickness`, `la_size`, `pericardial_effusion`, `mitral_appearance`, `mitral_motion`, `mitral_device`, `mitral_regurgitant_jet`, `aortic_appearance`, `aortic_motion`, `aortic_device`, `aortic_regurgitant_jet`, `aorta_root_size`, `aorta_ascending_size`, `wall_basal_inferolateral`, `wall_mid_inferolateral`, `wall_basal_anteroseptal`, `wall_mid_anteroseptal` |  |
| PLAX_DEEP | `pericardial_effusion` |  |
| PSAX | `lv_function`, `lv_size`, `lv_wall_thickness`, `pericardial_effusion`, `mitral_appearance`, `mitral_motion`, `mitral_device`, `mitral_regurgitant_jet`, `wall_mid_inferoseptal`, `wall_mid_anterolateral`, `wall_mid_inferior`, `wall_mid_anterior`, `wall_mid_inferolateral`, `wall_mid_anteroseptal` | lv, mitral, pericardium |
| PSAX_AV | `aortic_appearance`, `aortic_motion`, `aortic_device`, `aortic_regurgitant_jet`, `tricuspid_appearance`, `tricuspid_motion`, `tricuspid_device`, `tricuspid_regurgitant_jet`, `pulmonic_appearance`, `pulmonic_motion`, `pulmonic_device`, `pulmonic_regurgitant_jet`, `aortic_cusp_count` | tricuspid, pulmonic |
| RVINF | `rv_function`, `rv_size`, `ra_size`, `tricuspid_appearance`, `tricuspid_motion`, `tricuspid_device`, `tricuspid_regurgitant_jet` |  |
| SUBCOSTAL | `lv_function`, `lv_size`, `lv_wall_thickness`, `rv_function`, `rv_size`, `la_size`, `ra_size`, `pericardial_effusion`, `ivc_size`, `ivc_collapse`, `mitral_appearance`, `mitral_motion`, `mitral_device`, `mitral_regurgitant_jet`, `tricuspid_appearance`, `tricuspid_motion`, `tricuspid_device`, `tricuspid_regurgitant_jet` | lv, rv, la, ra, mitral, tricuspid, pericardium, ivc |
| SUPRASTERNAL | `aorta_arch_size` |  |
| A2C_ZOOM | `lv_function`, `lv_size`, `lv_wall_thickness`, `la_size`, `pericardial_effusion`, `mitral_appearance`, `mitral_motion`, `mitral_device`, `mitral_regurgitant_jet`, `wall_basal_inferior`, `wall_mid_inferior`, `wall_apical_inferior`, `wall_basal_anterior`, `wall_mid_anterior`, `wall_apical_anterior` | lv, la, mitral, pericardium |
| A3C_ZOOM | `lv_function`, `lv_size`, `lv_wall_thickness`, `la_size`, `pericardial_effusion`, `mitral_appearance`, `mitral_motion`, `mitral_device`, `mitral_regurgitant_jet`, `aortic_appearance`, `aortic_motion`, `aortic_device`, `aortic_regurgitant_jet`, `wall_apical_septal`, `wall_apical_lateral`, `wall_basal_inferolateral`, `wall_mid_inferolateral`, `wall_basal_anteroseptal`, `wall_mid_anteroseptal` | lv, la, mitral, aortic, pericardium |
| A4C_ZOOM | `lv_function`, `lv_size`, `lv_wall_thickness`, `rv_function`, `rv_size`, `la_size`, `ra_size`, `pericardial_effusion`, `mitral_appearance`, `mitral_motion`, `mitral_device`, `mitral_regurgitant_jet`, `tricuspid_appearance`, `tricuspid_motion`, `tricuspid_device`, `tricuspid_regurgitant_jet`, `wall_basal_inferoseptal`, `wall_mid_inferoseptal`, `wall_apical_septal`, `wall_basal_anterolateral`, `wall_mid_anterolateral`, `wall_apical_lateral` | lv, rv, la, ra, mitral, tricuspid, pericardium |
| A5C_ZOOM | `lv_function`, `lv_size`, `lv_wall_thickness`, `la_size`, `pericardial_effusion`, `mitral_appearance`, `mitral_motion`, `mitral_device`, `mitral_regurgitant_jet`, `aortic_appearance`, `aortic_motion`, `aortic_device`, `aortic_regurgitant_jet` | lv, la, mitral, aortic, pericardium |
| PLAX_ZOOM | `lv_function`, `lv_size`, `lv_wall_thickness`, `la_size`, `pericardial_effusion`, `mitral_appearance`, `mitral_motion`, `mitral_device`, `mitral_regurgitant_jet`, `aortic_appearance`, `aortic_motion`, `aortic_device`, `aortic_regurgitant_jet`, `aorta_root_size`, `aorta_ascending_size`, `wall_basal_inferolateral`, `wall_mid_inferolateral`, `wall_basal_anteroseptal`, `wall_mid_anteroseptal` | lv, la, mitral, aortic, aorta_root, aorta_ascending, pericardium |
| OTHER |  |  |
<!-- END GENERATED: column_review_tables -->
