# RADAR caption ontology v1 (`radar-caption-ontology-v1`)

**Approval.** The medical approval of this version (issue #21) is recorded in [approval_v1.md](approval_v1.md) and
nowhere else. No clinical data may be uploaded for review under this ontology until that record is `approved`.

The ontology fixes what a RADAR caption may say about one echo clip: its view, its acquisition mode, which cardiac
structures it shows, and which findings it supports. The Labelbox review form, the reviewer instructions and the
[pilot metrics](pilot_metrics_v1.md) are all built on it.

The sources of truth are [`configs/radar/radar-caption-ontology-v1.yaml`](../../configs/radar/radar-caption-ontology-v1.yaml)
and [`configs/radar/label_mappings_v1.yaml`](../../configs/radar/label_mappings_v1.yaml). The tables below are
generated from them:

```bash
python -m app.radar.ontology_doc          # rewrite the generated tables
python -m app.radar.ontology_doc --check  # fails if a table no longer matches the YAML (also a unit test)
```

## How the view and the acquisition limit a caption

A caption is view-conditioned: it may only mention what a clip of that view and acquisition mode can show. For each
finding, the ontology lists a prior: the views and acquisition modes in which the finding may be visible. Combined
with the clip's view and acquisition, the prior gives one of five outcomes:

| Outcome | Meaning | In a caption? |
|---|---|---|
| `relevant` | the view and the modality can show it, and it is judged visually | yes, as a claim a reviewer must confirm |
| `modality_absent` | the view fits but the clip's modality cannot show it (for example, RV systolic pressure in a color loop) | no, listed as a gap |
| `measurement_required` | it needs calipers, tracing or spectral measurement (TAPSE, TR ERO, RVSP …) | no, listed as a gap |
| `acquisition_unknown` | the clip's acquisition mode is `unknown` or `other` | no, listed as a gap |
| `not_in_view` | the view is outside the prior | no |

The prior only says what a clip **may** show. It is never evidence that a finding, or even a structure, is visible
in a particular clip: a zoomed, deep, off-axis or poor-quality clip can miss a structure its view usually shows. That
is why every claim is reviewed.

Three further rules come with the prior:

- **Modality.** Each concept lists the acquisition modes that can carry its evidence (any one suffices). `other` and
  `unknown` never satisfy that requirement.
- **Measurement.** Concepts marked `measurement` (for example TAPSE or TR peak velocity) stay out of captions even
  when the view fits, because watching a clip cannot confirm a measured number.
- **Several views.** Some findings are judged across views (the "Needs views from" column, for example regional wall
  motion across the apical 4-, 2- and 3-chamber views). A clip from one of those views may carry the claim, but it
  supports the finding only for the part it shows; the finding is supported for the study only when every listed
  group is covered.

Visibility and disease are separate concepts: `mitral_valve_visible` says nothing about `mitral_regurgitation`, and a
reported disease says nothing about what one clip shows.

## Views

The view taxonomy is the list of classes of the DeepECHO view classifier (the archive's `predicted_class`). `OTHER`
is the classifier's catch-all: it belongs to no group and carries no prior, so a caption for an `OTHER` clip states
no structure or finding.

<!-- BEGIN GENERATED: views -->
| View | Group | Window | Variant | English | French | The standard view shows |
|---|---|---|---|---|---|---|
| `A2C` | `apical_2c` | apical | standard | Apical 2-chamber view | Coupe apicale 2 cavités | Left ventricle, left atrium and mitral valve seen from the apex; the right heart is excluded. |
| `A2C_LV` | `apical_2c` | apical | lv_focused | Apical 2-chamber view, LV-focused | Coupe apicale 2 cavités centrée sur le VG | Apical 2-chamber view with depth and sector set on the left ventricle. |
| `A2C_ZOOM` | `apical_2c` | apical | zoomed | Apical 2-chamber view, zoomed | Coupe apicale 2 cavités zoomée | Magnified apical 2-chamber view, often on the mitral valve; part of the view may be outside the sector. |
| `A3C` | `apical_3c` | apical | standard | Apical 3-chamber view | Coupe apicale 3 cavités | Apical long-axis view; left ventricle, left atrium, mitral valve, LV outflow tract and aortic valve. |
| `A3C_LV` | `apical_3c` | apical | lv_focused | Apical 3-chamber view, LV-focused | Coupe apicale 3 cavités centrée sur le VG | Apical 3-chamber view with depth and sector set on the left ventricle. |
| `A3C_ZOOM` | `apical_3c` | apical | zoomed | Apical 3-chamber view, zoomed | Coupe apicale 3 cavités zoomée | Magnified apical 3-chamber view, often on the mitral or aortic valve; part of the view may be outside the sector. |
| `A4C` | `apical_4c` | apical | standard | Apical 4-chamber view | Coupe apicale 4 cavités | All four chambers seen from the apex, with the mitral and tricuspid valves and the interatrial septum. |
| `A4C_LV` | `apical_4c` | apical | lv_focused | Apical 4-chamber view, LV-focused | Coupe apicale 4 cavités centrée sur le VG | Apical 4-chamber view set on the left ventricle; the right heart may be partly outside the sector. |
| `A4C_ZOOM` | `apical_4c` | apical | zoomed | Apical 4-chamber view, zoomed | Coupe apicale 4 cavités zoomée | Magnified apical 4-chamber view, often on one valve or chamber; part of the view may be outside the sector. |
| `A5C` | `apical_5c` | apical | standard | Apical 5-chamber view | Coupe apicale 5 cavités | Apical 4-chamber view tilted anteriorly to show the LV outflow tract and the aortic valve. |
| `A5C_ZOOM` | `apical_5c` | apical | zoomed | Apical 5-chamber view, zoomed | Coupe apicale 5 cavités zoomée | Magnified apical 5-chamber view, often on the LV outflow tract or the aortic valve. |
| `PLAX` | `parasternal_long` | parasternal | standard | Parasternal long-axis view | Coupe parasternale grand axe | RV outflow tract, left ventricle, left atrium, mitral and aortic valves, aortic root. |
| `PLAX_DEEP` | `parasternal_long` | parasternal | deep | Parasternal long-axis view, deep | Coupe parasternale grand axe profonde | Parasternal long-axis view with increased depth, also showing structures behind the heart (descending aorta, pericardial and pleural spaces). |
| `PLAX_ZOOM` | `parasternal_long` | parasternal | zoomed | Parasternal long-axis view, zoomed | Coupe parasternale grand axe zoomée | Magnified parasternal long-axis view, often on the aortic valve, aortic root or mitral valve. |
| `PSAX` | `parasternal_short` | parasternal | standard | Parasternal short-axis view | Coupe parasternale petit axe | Left ventricle in short axis, at the mitral, papillary-muscle or apical level. |
| `PSAX_AV` | `parasternal_short_av` | parasternal | standard | Parasternal short-axis view at the aortic valve | Coupe parasternale petit axe aortique | Short axis at the aortic valve level; atria, interatrial septum, tricuspid valve, RV outflow tract and pulmonic valve. |
| `RVINF` | `rv_inflow` | parasternal | standard | Right ventricular inflow view | Coupe de la voie d'entrée du VD | Right atrium, tricuspid valve and right ventricle, from the parasternal window. |
| `SUBCOSTAL` | `subcostal` | subcostal | standard | Subcostal view | Coupe sous-costale | Subcostal views, including the four-chamber view and the inferior vena cava, grouped in one class. |
| `SUPRASTERNAL` | `suprasternal` | suprasternal | standard | Suprasternal view | Coupe sus-sternale | Aortic arch from the suprasternal window. |
| `OTHER` | none | unclassified | unclassified | Unclassified view | Coupe non classée | A clip the classifier did not assign to a listed view (off-axis, non-standard or not cardiac). No structure or finding prior. |
<!-- END GENERATED: views -->

The descriptions are those of the standard echo views. Where the classifier's training labels put the boundary
between neighbouring classes (for example `A4C_LV` and `A4C_ZOOM`) is listed under
[Clinical choices in v1](#clinical-choices-in-v1).

## Acquisition modes

Each clip's acquisition mode comes from the acquisition index (DICOM header flags), not from the view classifier: the
view classes do not separate grayscale from color clips.

<!-- BEGIN GENERATED: acquisitions -->
| Category | English | French |
|---|---|---|
| `bmode` | B-mode | mode B |
| `color_flow` | color Doppler | Doppler couleur |
| `spectral_doppler` | spectral Doppler | Doppler spectral |
| `mmode` | M-mode | mode M |
| `other` | other acquisition | acquisition autre |
| `unknown` | unknown acquisition | acquisition indéterminée |
<!-- END GENERATED: acquisitions -->

## Cardiac structures

Every concept names the structures it is about. A structure has at most one visibility concept; the views where the
structure may be seen are that concept's prior.

<!-- BEGIN GENERATED: structures -->
| Structure | Category | English | French | Visibility concept | May be visible in (views, acquisitions) | Findings |
|---|---|---|---|---|---|---|
| `left_ventricle` | chamber | Left ventricle | Ventricule gauche | `lv_visible` | `apical_4c`, `apical_5c`, `apical_2c`, `apical_3c`, `parasternal_long`, `parasternal_short`, `subcostal` (bmode, color_flow) | `lv_systolic_function`, `lv_ejection_fraction`, `lv_size`, `lv_wall_thickness`, `lv_regional_wall_motion`, `lv_diastolic_function`, `lv_global_longitudinal_strain`, `mechanical_circulatory_support` |
| `right_ventricle` | chamber | Right ventricle | Ventricule droit | `rv_visible` | `apical_4c`, `apical_5c`, `parasternal_long`, `parasternal_short`, `parasternal_short_av`, `rv_inflow`, `subcostal` (bmode, color_flow) | `rv_systolic_function`, `rv_size`, `tapse`, `rv_systolic_pressure`, `pacing_lead` |
| `left_atrium` | chamber | Left atrium | Oreillette gauche | `la_visible` | `apical_4c`, `apical_5c`, `apical_2c`, `apical_3c`, `parasternal_long`, `parasternal_short_av`, `subcostal` (bmode, color_flow) | `la_size`, `elevated_la_pressure` |
| `right_atrium` | chamber | Right atrium | Oreillette droite | `ra_visible` | `apical_4c`, `apical_5c`, `parasternal_short_av`, `rv_inflow`, `subcostal` (bmode, color_flow) | `ra_pressure`, `ra_size`, `pacing_lead` |
| `interatrial_septum` | septum | Interatrial septum | Septum interauriculaire | `interatrial_septum_visible` | `apical_4c`, `parasternal_short_av`, `subcostal` (bmode, color_flow) | `atrial_septal_hypertrophy` |
| `mitral_valve` | valve | Mitral valve | Valve mitrale | `mitral_valve_visible` | `apical_4c`, `apical_5c`, `apical_2c`, `apical_3c`, `parasternal_long`, `parasternal_short` (bmode, color_flow) | `mitral_regurgitation`, `mitral_stenosis`, `mitral_annular_calcification`, `mitral_valve_device` |
| `aortic_valve` | valve | Aortic valve | Valve aortique | `aortic_valve_visible` | `apical_5c`, `apical_3c`, `parasternal_long`, `parasternal_short_av` (bmode, color_flow) | `aortic_regurgitation`, `aortic_stenosis`, `aortic_valve_peak_velocity`, `bicuspid_aortic_valve`, `aortic_valve_prosthesis` |
| `tricuspid_valve` | valve | Tricuspid valve | Valve tricuspide | `tricuspid_valve_visible` | `apical_4c`, `apical_5c`, `parasternal_short_av`, `rv_inflow`, `subcostal` (bmode, color_flow) | `tr_peak_velocity`, `tricuspid_regurgitation`, `tricuspid_regurgitation_ero`, `tricuspid_stenosis` |
| `pulmonic_valve` | valve | Pulmonic valve | Valve pulmonaire | `pulmonic_valve_visible` | `parasternal_short_av`, `subcostal` (bmode, color_flow) | `pulmonic_regurgitation` |
| `left_ventricular_outflow_tract` | outflow_tract | Left ventricular outflow tract | Chambre de chasse du ventricule gauche | none | no prior in v1 | `lvot_obstruction` |
| `aortic_root` | great_vessel | Aortic root | Racine aortique | `aortic_root_visible` | `apical_5c`, `apical_3c`, `parasternal_long`, `parasternal_short_av`, `suprasternal` (bmode, color_flow) | `aortic_root_dilation` |
| `pericardium` | pericardium | Pericardium | Péricarde | `pericardium_visible` | `apical_4c`, `apical_2c`, `apical_3c`, `parasternal_long`, `parasternal_short`, `subcostal` (bmode) | `pericardial_effusion` |
| `inferior_vena_cava` | systemic_vein | Inferior vena cava | Veine cave inférieure | `ivc_visible` | `subcostal` (bmode, color_flow, mmode) | `ivc_dilation`, `ivc_collapse` |
<!-- END GENERATED: structures -->

## Findings

<!-- BEGIN GENERATED: findings -->
| Finding | English | French | Structures | Value | Evidence | Modalities | Prior (views, acquisitions) | Needs views from | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `lv_systolic_function` | Left ventricular systolic function | Fonction systolique ventriculaire gauche | `left_ventricle` | scale `systolic_function` | visual | bmode | `apical_4c`, `apical_2c`, `apical_3c`, `parasternal_long`, `parasternal_short`, `subcostal` (bmode) |  |  |
| `lv_ejection_fraction` | LV ejection fraction (visual estimate) | Fraction d'éjection VG (estimation visuelle) | `left_ventricle` | %, 5 to 90 | visual | bmode | `apical_4c`, `apical_2c` (bmode) | `apical_4c`, `apical_2c` | A visual estimate integrates apical 4- and 2-chamber views; one clip supports it only partly. |
| `lv_size` | Left ventricular size | Taille du ventricule gauche | `left_ventricle` | scale `dilation` | visual | bmode | `apical_4c`, `apical_2c`, `parasternal_long`, `parasternal_short` (bmode) |  |  |
| `lv_wall_thickness` | Left ventricular hypertrophy | Hypertrophie ventriculaire gauche | `left_ventricle` | scale `hypertrophy` | visual | bmode | `apical_4c`, `parasternal_long`, `parasternal_short` (bmode) |  |  |
| `lv_regional_wall_motion` | LV regional wall motion (worst segment) | Cinétique segmentaire du VG (segment le plus atteint) | `left_ventricle` | scale `wall_motion` | visual | bmode | `apical_4c`, `apical_2c`, `apical_3c`, `parasternal_short` (bmode) | `apical_4c`, `apical_2c`, `apical_3c` | The 17 segments are seen across apical 4-, 2- and 3-chamber views; a single clip covers some segments. |
| `lv_diastolic_function` | LV diastolic function | Fonction diastolique du VG | `left_ventricle` | scale `diastolic_function` | measurement | spectral_doppler | `apical_4c` (spectral_doppler) |  | Graded from mitral inflow and tissue Doppler; spectral Doppler stills are not in the video corpus. |
| `lvot_obstruction` | LV outflow tract obstruction | Obstruction de la chambre de chasse du VG | `left_ventricular_outflow_tract` | yes / no | measurement | spectral_doppler | `apical_5c`, `apical_3c` (spectral_doppler) |  |  |
| `lv_global_longitudinal_strain` | LV global longitudinal strain | Strain longitudinal global du VG | `left_ventricle` | %, -40 to 0 | measurement | bmode | `apical_4c`, `apical_2c`, `apical_3c` (bmode) | `apical_4c`, `apical_2c`, `apical_3c` |  |
| `rv_systolic_function` | Right ventricular systolic function | Fonction systolique ventriculaire droite | `right_ventricle` | scale `systolic_function` | visual | bmode | `apical_4c`, `parasternal_short`, `rv_inflow`, `subcostal` (bmode) |  |  |
| `rv_size` | Right ventricular size | Taille du ventricule droit | `right_ventricle` | scale `dilation` | visual | bmode | `apical_4c`, `parasternal_long`, `parasternal_short`, `subcostal` (bmode) |  |  |
| `tapse` | TAPSE | TAPSE | `right_ventricle` | mm, 3 to 50 | measurement | mmode | `apical_4c` (mmode) |  |  |
| `rv_systolic_pressure` | Right ventricular systolic pressure | Pression systolique ventriculaire droite | `right_ventricle` | mmHg, 10 to 150 | measurement | spectral_doppler | `apical_4c`, `parasternal_short_av`, `rv_inflow`, `subcostal` (spectral_doppler) |  | CW Doppler of the TR jet plus estimated RA pressure; never inferred from a color loop. |
| `tr_peak_velocity` | TR peak velocity | Vitesse maximale de l'insuffisance tricuspide | `tricuspid_valve` | m/s, 0.5 to 7 | measurement | spectral_doppler | `apical_4c`, `parasternal_short_av`, `rv_inflow`, `subcostal` (spectral_doppler) |  |  |
| `ra_pressure` | Estimated right atrial pressure | Pression auriculaire droite estimée | `right_atrium` | mmHg, 1 to 30 | measurement | bmode, mmode | `subcostal` (bmode, mmode) |  | Estimated from IVC diameter and collapse; ivc_dilation and ivc_collapse are the visual counterparts. |
| `la_size` | Left atrial size | Taille de l'oreillette gauche | `left_atrium` | scale `dilation` | visual | bmode | `apical_4c`, `apical_2c`, `parasternal_long`, `parasternal_short_av` (bmode) |  |  |
| `ra_size` | Right atrial size | Taille de l'oreillette droite | `right_atrium` | scale `dilation` | visual | bmode | `apical_4c`, `rv_inflow`, `subcostal` (bmode) |  |  |
| `atrial_septal_hypertrophy` | Lipomatous atrial septal hypertrophy | Hypertrophie lipomateuse du septum interauriculaire | `interatrial_septum` | yes / no | visual | bmode | `apical_4c`, `subcostal` (bmode) |  |  |
| `mitral_regurgitation` | Mitral regurgitation | Insuffisance mitrale | `mitral_valve` | scale `regurgitation` | visual | color_flow | `apical_4c`, `apical_5c`, `apical_2c`, `apical_3c`, `parasternal_long` (color_flow) |  | A color loop supports a visual impression of the jet; the reported grade may integrate PISA and spectral data. |
| `mitral_stenosis` | Mitral stenosis | Sténose mitrale | `mitral_valve` | scale `stenosis` | measurement | spectral_doppler | `apical_4c` (spectral_doppler) |  | Graded by mean gradient or planimetry; leaflet morphology on B-mode is not a severity grade. |
| `mitral_annular_calcification` | Mitral annular calcification | Calcification de l'anneau mitral | `mitral_valve` | yes / no | visual | bmode | `apical_4c`, `apical_2c`, `apical_3c`, `parasternal_long`, `parasternal_short` (bmode) |  |  |
| `mitral_valve_device` | Mitral valve device or prosthesis | Dispositif ou prothèse mitrale | `mitral_valve` | `none`, `edge_to_edge_clip`, `annuloplasty_ring`, `bioprosthesis`, `mechanical_prosthesis` | visual | bmode, color_flow | `apical_4c`, `apical_2c`, `apical_3c`, `parasternal_long`, `parasternal_short` (bmode, color_flow) |  |  |
| `aortic_regurgitation` | Aortic regurgitation | Insuffisance aortique | `aortic_valve` | scale `regurgitation` | visual | color_flow | `apical_5c`, `apical_3c`, `parasternal_long`, `parasternal_short_av` (color_flow) |  |  |
| `aortic_stenosis` | Aortic stenosis | Sténose aortique | `aortic_valve` | scale `stenosis` | measurement | spectral_doppler | `apical_5c`, `apical_3c`, `suprasternal` (spectral_doppler) |  | Graded by peak velocity, mean gradient and valve area. Calcified, restricted cusps on B-mode are not a grade. |
| `aortic_valve_peak_velocity` | Aortic valve peak velocity | Vitesse maximale transvalvulaire aortique | `aortic_valve` | m/s, 0.5 to 7 | measurement | spectral_doppler | `apical_5c`, `apical_3c`, `suprasternal` (spectral_doppler) |  |  |
| `bicuspid_aortic_valve` | Bicuspid aortic valve | Bicuspidie aortique | `aortic_valve` | yes / no | visual | bmode | `parasternal_long`, `parasternal_short_av` (bmode) |  |  |
| `aortic_valve_prosthesis` | Aortic valve prosthesis | Prothèse valvulaire aortique | `aortic_valve` | `none`, `transcatheter`, `surgical_bioprosthesis`, `mechanical_prosthesis` | visual | bmode, color_flow | `apical_5c`, `apical_3c`, `parasternal_long`, `parasternal_short_av` (bmode, color_flow) |  |  |
| `aortic_root_dilation` | Aortic root dilation | Dilatation de la racine aortique | `aortic_root` | scale `dilation` | visual | bmode | `apical_5c`, `apical_3c`, `parasternal_long`, `suprasternal` (bmode) |  |  |
| `tricuspid_regurgitation` | Tricuspid regurgitation | Insuffisance tricuspide | `tricuspid_valve` | scale `regurgitation` | visual | color_flow | `apical_4c`, `apical_5c`, `parasternal_short_av`, `rv_inflow`, `subcostal` (color_flow) |  |  |
| `tricuspid_regurgitation_ero` | Tricuspid regurgitant orifice area (PISA) | Surface de l'orifice régurgitant tricuspide (PISA) | `tricuspid_valve` | cm², 0.01 to 3 | measurement | color_flow | `apical_4c`, `rv_inflow` (color_flow) |  | PISA needs a zoomed color frame with a measured radius plus CW Doppler; a loop cannot confirm the number. |
| `tricuspid_stenosis` | Tricuspid stenosis | Sténose tricuspide | `tricuspid_valve` | scale `stenosis` | measurement | spectral_doppler | `apical_4c`, `rv_inflow` (spectral_doppler) |  |  |
| `pulmonic_regurgitation` | Pulmonic regurgitation | Insuffisance pulmonaire | `pulmonic_valve` | scale `regurgitation` | visual | color_flow | `parasternal_short_av`, `subcostal` (color_flow) |  |  |
| `ivc_dilation` | Inferior vena cava dilation | Dilatation de la veine cave inférieure | `inferior_vena_cava` | yes / no | visual | bmode, mmode | `subcostal` (bmode, mmode) |  |  |
| `ivc_collapse` | IVC inspiratory collapse | Collapsus inspiratoire de la VCI | `inferior_vena_cava` | `normal`, `reduced` | visual | bmode, mmode | `subcostal` (bmode, mmode) |  |  |
| `pericardial_effusion` | Pericardial effusion | Épanchement péricardique | `pericardium` | scale `effusion` | visual | bmode | `apical_4c`, `apical_2c`, `parasternal_long`, `parasternal_short`, `subcostal` (bmode) |  |  |
| `pacing_lead` | Intracardiac pacing lead | Sonde de stimulation intracardiaque | `right_atrium`, `right_ventricle` | yes / no | visual | bmode | `apical_4c`, `parasternal_short`, `rv_inflow`, `subcostal` (bmode) |  |  |
| `mechanical_circulatory_support` | Mechanical circulatory support | Assistance circulatoire mécanique | `left_ventricle` | `none`, `microaxial_pump`, `lvad` | visual | bmode, color_flow | `apical_4c`, `apical_5c`, `apical_3c`, `parasternal_long` (bmode, color_flow) |  |  |
| `elevated_la_pressure` | Elevated LV filling pressure | Pressions de remplissage du VG élevées | `left_atrium` | yes / no | measurement | spectral_doppler | `apical_4c` (spectral_doppler) |  |  |
<!-- END GENERATED: findings -->

Ordinal findings use these scales. A value outside the order, such as `indeterminate` diastolic function, is a valid
value but no grade: it ranks neither above nor below the others.

<!-- BEGIN GENERATED: scales -->
| Scale | Values, in order | Outside the order | Negated mention |
|---|---|---|---|
| `regurgitation` | `none` < `trace` < `mild` < `mild_to_moderate` < `moderate` < `moderate_to_severe` < `severe` |  | `none` |
| `stenosis` | `none` < `mild` < `moderate` < `severe` |  | `none` |
| `dilation` | `normal` < `mild` < `moderate` < `severe` |  | `normal` |
| `hypertrophy` | `none` < `mild` < `moderate` < `severe` |  | `none` |
| `systolic_function` | `hyperdynamic` < `normal` < `mildly_reduced` < `moderately_reduced` < `severely_reduced` |  | unresolved |
| `diastolic_function` | `normal` < `grade_1` < `grade_2` < `grade_3` | `indeterminate` | unresolved |
| `wall_motion` | `normal` < `hypokinetic` < `akinetic` < `dyskinetic` < `aneurysmal` |  | `normal` |
| `effusion` | `none` < `trivial` < `small` < `moderate` < `large` |  | `none` |
<!-- END GENERATED: scales -->

## Assessment limitations

The reasons a reviewer can give when a clip, or one concept in it, cannot be judged:

<!-- BEGIN GENERATED: limitations -->
| Reason | English | French |
|---|---|---|
| `quality` | Image quality | Qualité d'image |
| `crop` | Crop, zoom or outside the square | Cadrage, zoom ou hors du carré |
| `modality` | Modality | Modalité |
| `other` | Other | Autre |
<!-- END GENERATED: limitations -->

## What a caption claims

A candidate caption is its text plus a structured list of the statements it makes. Reviewers judge those
statements, and the [pilot metrics](pilot_metrics_v1.md) count them, so the two must match exactly:

- **View.** Every caption states the clip's view class. The view is judged by its own review question and is not
  counted as a statement.
- **Acquisition mode.** Every caption states the clip's acquisition mode; this is a statement.
- **Findings.** A finding statement names a concept and a value. Only findings `relevant` to the clip may be stated;
  gaps and findings outside the view never are.
- **Visibility.** A visibility statement names a visibility concept and says whether the structure is visible. It may
  only name a structure whose prior includes the clip's view and acquisition. The visibility prior shown to reviewers
  as expected structures is not a statement.
- **Absence.** "No mitral regurgitation" is a finding statement with the value `none` (or the scale's normal value),
  judged like any other statement.
- **Text and list match one to one.** Every statement appears in the text, and everything the text asserts is a
  statement. A caption whose text says "mitral valve visible" without the matching visibility statement is invalid
  and is not sent for review.
- **Nothing else.** No identifiers, dates, report text, or values the ontology does not define.
- **Corrected captions** keep the same form. A view the reviewer cannot tell is written "Coupe indéterminée"
  (`review.undetermined_view`), which is not a view class: `OTHER` stays for off-axis or non-standard clips. An
  acquisition mode the reviewer cannot tell is written with the label of `unknown`, "acquisition indéterminée".

The code that generates captions, uploads clips and imports reviews is outside issue #21. Besides enforcing this
contract, it must:

- upload clips only under an approved ontology (`require_approval` in `app/radar/approval.py`);
- draw on every uploaded clip the square the model uses (its evaluation crop: shorter side resized to 256, center
  crop of 224, mapped back to the clip's pixels), and record that square with the clip;
- record with each clip the ontology tag and the candidate caption's view, acquisition mode and statements, and
  preload the candidate caption as the answer to "corrected caption" (the form itself cannot hold a per-clip default);
- count a review as valid only when `review_problems` in `app/radar/review_rules.py` finds nothing: every required
  answer given with an allowed value, every consistency rule met, and no flag on a statement the candidate caption
  does not make.

## Model-input review

The square outline must come from the actual preprocessing transform, and the review item must identify the
selected source frames received by the encoder, including any padding. The default evaluation crop described
above is configurable and differs from random training crops. A fixed outline alone cannot establish temporal
support. Verify the displayed geometry and selected frames using synthetic inputs before clinical review.

The full loop provides context for view identification. Statements are supported only in the selected model-input
frames inside the crop. Findings visible only in other frames are unconfirmed with reason `other` and the comment
"outside model-input frames". Missing or unverifiable crop/frame provenance requires display correction before a
medical label is submitted. An optional synchronized crop-only panel must show the same selected frames. Overlays
are review aids and must never be burned into encoder inputs.

The [ontology request and manuscript review](ontology_request.md) describes proposed schema extensions and
contrastive dataset construction. Its proposals do not replace the approved v1 rules.

## Review form

The Labelbox review form is defined in the `review` section of the YAML and generated as
[`configs/radar/labelbox/radar-caption-ontology-v1.json`](../../configs/radar/labelbox/radar-caption-ontology-v1.json)
(`python -m tools.labelbox.review_ontology`; a unit test fails when the JSON drifts, and another checks it against
the Labelbox SDK). It asks the six questions of issue #21 (uninterpretable clip, view, caption supported, missing
information, unsupported information, corrected caption), an optional list of unconfirmed statements and optional
comments. The reviewer instructions, in [English](labeling_instructions_v1.md) and
[French](labeling_instructions_v1.fr.md), explain every answer with worked synthetic examples.

## How DeepECHO, EchoPrime and PanEcho labels relate to v1

[`configs/radar/label_mappings_v1.yaml`](../../configs/radar/label_mappings_v1.yaml) lists every label of each
source, and every class of each classification task; a unit test checks the lists against the files vendored in this
repository. Each label has one relation:

| Relation | Meaning |
|---|---|
| `exact` | Same definition: a view maps to one view, a class to one value, a regression has the concept's unit. |
| `grouped` | Same concept, coarser labels: a view class covers several views, or a task class covers several values (PanEcho's `Moderate\|Severe`). A prediction never becomes one precise value; values no class covers are listed as "no class". |
| `transformed` | Same quantity after the stated conversion (for example PanEcho's TR peak gradient in mmHg to a velocity in m/s). |
| `related` | Informative about the listed concepts, but not the same quantity or definition, or too uncertain to imply a value: every EchoPrime binary task is a phrase match that may be negated ("No evidence of severe mitral regurgitation" is positive). No value is ever derived from it. |
| `unmapped` | No v1 concept; the structures it is about and the reason are given. |

The mappings describe what each label means for a caption. They are not used to write captions: candidate captions
are built from the ICM report fields listed in the ontology.

<!-- BEGIN GENERATED: mapping_summary -->
| Source | Kind | Labels | exact | grouped | transformed | related | unmapped |
|---|---|---|---|---|---|---|---|
| `deepecho_views` | view | 20 | 20 | 0 | 0 | 0 | 0 |
| `echoprime_views` | view | 11 | 2 | 9 | 0 | 0 | 0 |
| `echoprime_findings` | task | 25 | 1 | 0 | 0 | 24 | 0 |
| `panecho_tasks` | task | 40 | 4 | 17 | 3 | 14 | 2 |
<!-- END GENERATED: mapping_summary -->

### DeepECHO views

<!-- BEGIN GENERATED: mapping_deepecho_views -->
DeepECHO view classifier classes (the archive's `predicted_class`). They are the ontology's views. Source: `configs/radar/radar-caption-ontology-v1.yaml (views)`.

Every class maps exactly to the view of the same name: `A2C`, `A2C_LV`, `A2C_ZOOM`, `A3C`, `A3C_LV`, `A3C_ZOOM`, `A4C`, `A4C_LV`, `A4C_ZOOM`, `A5C`, `A5C_ZOOM`, `PLAX`, `PLAX_DEEP`, `PLAX_ZOOM`, `PSAX`, `PSAX_AV`, `RVINF`, `SUBCOSTAL`, `SUPRASTERNAL`, `OTHER`.
<!-- END GENERATED: mapping_deepecho_views -->

### EchoPrime views

<!-- BEGIN GENERATED: mapping_echoprime_views -->
EchoPrime's coarse view classes. Doppler clips have classes of their own; whether the other classes exclude color clips is not documented, so their acquisition is `unspecified`. EchoPrime has no RV inflow class, and which class its classifier gives RVINF clips is unknown. Source: `evals/video_classification_frozen/modelcustom/EchoPrime/utils/utils.py (COARSE_VIEWS)`.

| Label | Relation | Views | Acquisitions | Notes |
|---|---|---|---|---|
| `A2C` | grouped | `apical_2c` | unspecified |  |
| `A3C` | grouped | `apical_3c` | unspecified |  |
| `A4C` | grouped | `apical_4c` | unspecified |  |
| `A5C` | grouped | `apical_5c` | unspecified |  |
| `Apical_Doppler` | grouped | `apical_4c`, `apical_5c`, `apical_2c`, `apical_3c` | color_flow, spectral_doppler | Any apical Doppler clip. The view cannot be narrowed down to one apical view. |
| `Doppler_Parasternal_Long` | grouped | `parasternal_long` | color_flow, spectral_doppler |  |
| `Doppler_Parasternal_Short` | grouped | `parasternal_short`, `parasternal_short_av` | color_flow, spectral_doppler |  |
| `Parasternal_Long` | grouped | `parasternal_long` | unspecified |  |
| `Parasternal_Short` | grouped | `parasternal_short`, `parasternal_short_av` | unspecified |  |
| `SSN` | exact | `SUPRASTERNAL` | unspecified |  |
| `Subcostal` | exact | `SUBCOSTAL` | unspecified |  |
<!-- END GENERATED: mapping_echoprime_views -->

### EchoPrime findings

<!-- BEGIN GENERATED: mapping_echoprime_findings -->
EchoPrime's study-level report findings. A binary task is `positive` when one of its phrases appears in its report section, as a case-insensitive substring without negation handling: "No evidence of severe mitral regurgitation" is positive for mitral regurgitation. A phrase match is not a verified statement about the study, so no binary task implies a value: each is related to its concept, and its notes say what its phrases state. `negative` only means that no phrase matched. A regression task is missing (not zero) when no phrase matches. Source: `evals/video_classification_frozen/modelcustom/EchoPrime/assets/per_section.json`.

| Label | Relation | Unit | Maps to | Notes |
|---|---|---|---|---|
| `pacemaker` | related |  | related to `pacing_lead` (classes `positive`, `negative` imply no value) | The phrases are "pacer" and "pacemaker", which also match other words ("spacer"). |
| `impella` | related |  | related to `mechanical_circulatory_support` (classes `positive`, `negative` imply no value) | The phrase states that an Impella catheter is seen. |
| `tavr` | related |  | related to `aortic_valve_prosthesis` (classes `positive`, `negative` imply no value) | The phrase states a bioprosthetic stent-valve in the aortic position. |
| `mitraclip` | related |  | related to `mitral_valve_device` (classes `positive`, `negative` imply no value) | The phrases state one or two MitraClips on the mitral leaflets. |
| `aortic_root_dilation` | related |  | related to `aortic_root_dilation` (classes `positive`, `negative` imply no value) | The phrases state moderate or severe aortic root dilation. |
| `bicuspid_aov_morphology` | related |  | related to `bicuspid_aortic_valve` (classes `positive`, `negative` imply no value) | The phrases state a bicuspid aortic valve, including a "possible" one. |
| `aortic_stenosis` | related |  | related to `aortic_stenosis` (classes `positive`, `negative` imply no value) | The phrases state moderate or severe aortic stenosis. |
| `tricuspid_stenosis` | related |  | related to `tricuspid_stenosis` (classes `positive`, `negative` imply no value) | The phrases state moderate or severe tricuspid stenosis. |
| `aortic_regurgitation` | related |  | related to `aortic_regurgitation` (classes `positive`, `negative` imply no value) | The phrases state moderate or severe aortic regurgitation (moderate to severe included). |
| `dilated_ivc` | related |  | related to `ivc_dilation` (classes `positive`, `negative` imply no value) | The phrases state a dilated IVC or any IVC diameter starting with 2 or 3 cm, which includes normal diameters. |
| `left_atrium_dilation` | related |  | related to `la_size` (classes `positive`, `negative` imply no value) | The phrases state a moderately or severely dilated left atrium. |
| `ejection_fraction` | exact | % | `lv_ejection_fraction` | The reported EF, whatever the method; the v1 concept is the visual estimate of the same quantity. |
| `mitral_annular_calcification` | related |  | related to `mitral_annular_calcification` (classes `positive`, `negative` imply no value) | The phrases state moderate or severe mitral annular calcification. |
| `mitral_stenosis` | related |  | related to `mitral_stenosis` (classes `positive`, `negative` imply no value) | The phrases state moderate or severe mitral stenosis. |
| `mitral_regurgitation` | related |  | related to `mitral_regurgitation` (classes `positive`, `negative` imply no value) | The phrases state moderate or severe mitral regurgitation (moderate to severe included). |
| `pericardial_effusion` | related |  | related to `pericardial_effusion` (classes `positive`, `negative` imply no value) | The phrases state a moderate or larger pericardial effusion, or tamponade. |
| `pulmonary_artery_pressure_continuous` | related | mmHg | related to `rv_systolic_pressure` | PA systolic pressure equals RV systolic pressure only without pulmonic stenosis or RV outflow obstruction. |
| `right_atrium_dilation` | related |  | related to `ra_size` (classes `positive`, `negative` imply no value) | The phrases state a moderately or severely dilated right atrium. |
| `rv_systolic_function_depressed` | related |  | related to `rv_systolic_function` (classes `positive`, `negative` imply no value) | The phrases state moderately or severely depressed RV systolic function. |
| `right_ventricle_dilation` | related |  | related to `rv_size` (classes `positive`, `negative` imply no value) | The phrases state a moderately or severely dilated right ventricle. |
| `tricuspid_valve_regurgitation` | related |  | related to `tricuspid_regurgitation` (classes `positive`, `negative` imply no value) | The phrases state moderate or severe tricuspid regurgitation (moderate to severe included). |
| `pulmonic_valve_regurgitation` | related |  | related to `pulmonic_regurgitation` (classes `positive`, `negative` imply no value) | The phrases state moderate or severe pulmonic regurgitation (moderate to severe included). |
| `elevated_left_atrial_pressure` | related |  | related to `elevated_la_pressure` (classes `positive`, `negative` imply no value) | The phrase is "elevated left atrial pressure". |
| `wall_motion_hypokinesis` | related |  | related to `lv_regional_wall_motion` (classes `positive`, `negative` imply no value) | The phrase is "hypokinesis" anywhere in the wall-motion section; the worst segment may also be worse. |
| `atrial_septum_hypertrophy` | related |  | related to `atrial_septal_hypertrophy` (classes `positive`, `negative` imply no value) | The phrases state moderate or severe lipomatous hypertrophy of the atrial septum. |
<!-- END GENERATED: mapping_echoprime_findings -->

### PanEcho tasks

<!-- BEGIN GENERATED: mapping_panecho_tasks -->
PanEcho's study-level tasks, with their class names. A `|` inside a class name joins grades PanEcho merged. PanEcho has no view labels. Source: `evals/video_classification_frozen/modelcustom/PanEcho/content/tasks.pkl`.

| Label | Relation | Unit | Maps to | Notes |
|---|---|---|---|---|
| `pericardial-effusion` | grouped |  | `pericardial_effusion`: `mild_mod_severe` → `small`, `moderate`, `large`; `none_trace` → `none`, `trivial` | PanEcho's "mild" effusion is taken to be the v1 "small" grade. |
| `EF` | exact | % | `lv_ejection_fraction` |  |
| `GLS` | transformed | % | `lv_global_longitudinal_strain`: GLS = -1 x output (PanEcho outputs the magnitude). |  |
| `LVEDV` | related | mL | related to `lv_size` |  |
| `LVESV` | related | mL | related to `lv_ejection_fraction` |  |
| `LVSV` | unmapped | mL | structures `left_ventricle`: Stroke volume (volumetric or Doppler); v1 has no caption concept for it. |  |
| `LVSize` | grouped |  | `lv_size`: `Mildly Increased` → `mild`; `Moderately\|Severely Increased` → `moderate`, `severe`; `Normal` → `normal` |  |
| `LVWallThickness-increased-any` | grouped |  | `lv_wall_thickness`: `Increased` → `mild`, `moderate`, `severe`; `Normal` → `none` |  |
| `LVWallThickness-increased-modsev` | grouped |  | `lv_wall_thickness`: `Moderately\|severely increased` → `moderate`, `severe`; `Normal or mildly increased` → `none`, `mild` |  |
| `LVSystolicFunction` | grouped |  | `lv_systolic_function`: `Mildly Decreased` → `mildly_reduced`; `Moderately\|Severely Decreased` → `moderately_reduced`, `severely_reduced`; `Normal\|Hyperdynamic` → `normal`, `hyperdynamic` |  |
| `LVWallMotionAbnormalities` | grouped |  | `lv_regional_wall_motion`: `None` → `normal`; `Present` → `hypokinetic`, `akinetic`, `dyskinetic`, `aneurysmal` |  |
| `IVSd` | related | cm | related to `lv_wall_thickness` |  |
| `LVPWd` | related | cm | related to `lv_wall_thickness` |  |
| `LVIDs` | related | cm | related to `lv_size`, `lv_systolic_function` |  |
| `LVIDd` | related | cm | related to `lv_size` |  |
| `LVOTDiam` | unmapped | cm | structures `left_ventricular_outflow_tract`: A caliper measurement for the continuity equation; v1 has no caption concept for it. |  |
| `LVDiastolicFunction` | grouped |  | `lv_diastolic_function`: `Mild\|Indeterminate` → `grade_1`, `indeterminate`; `Moderate\|Severe` → `grade_2`, `grade_3`; `Normal` → `normal` |  |
| `E\|EAvg` | related | ratio | related to `lv_diastolic_function`, `elevated_la_pressure` |  |
| `RVSP` | exact | mmHg | `rv_systolic_pressure` |  |
| `RVSize` | grouped |  | `rv_size`: `Mildly Increased` → `mild`; `Moderately\|Severely Increased` → `moderate`, `severe`; `Normal` → `normal` |  |
| `RVSystolicFunction` | grouped |  | `rv_systolic_function`: `Decreased` → `mildly_reduced`, `moderately_reduced`, `severely_reduced`; `Normal` → `normal`, `hyperdynamic` |  |
| `RVIDd` | related | cm | related to `rv_size` |  |
| `TAPSE` | transformed | cm | `tapse`: TAPSE (mm) = 10 x output (cm). |  |
| `RVSVel` | related | cm/s | related to `rv_systolic_function` |  |
| `LASize` | grouped |  | `la_size`: `Mildly Dilated` → `mild`; `Moderately\|Severely Dilated` → `moderate`, `severe`; `Normal` → `normal` |  |
| `LAIDs2D` | related | cm | related to `la_size` |  |
| `LAVol` | related | mL | related to `la_size` |  |
| `RASize` | grouped |  | `ra_size`: `Dilated` → `mild`, `moderate`, `severe`; `Normal` → `normal` |  |
| `RADimensionM-L(cm)` | related | cm | related to `ra_size` |  |
| `AVStructure` | exact |  | `bicuspid_aortic_valve`: `Bicuspid` → `true`; `Normal` → `false` |  |
| `AVStenosis` | grouped |  | `aortic_stenosis`: `Mild\|Moderate` → `mild`, `moderate`; `None` → `none`; `Severe` → `severe` |  |
| `AVPkVel(m\|s)` | exact | m/s | `aortic_valve_peak_velocity` |  |
| `AVRegurg` | grouped |  | `aortic_regurgitation`: `Mild` → `mild`; `Moderate\|Severe` → `moderate`, `moderate_to_severe`, `severe`; `None\|Trace` → `none`, `trace`; no class: `mild_to_moderate` |  |
| `LVOT20mmHg` | related |  | related to `lvot_obstruction` (classes `0.0`, `1.0` imply no value) | PanEcho's label is an LV outflow peak gradient of at least 20 mmHg; the v1 concept is the report's statement of obstruction, which may use another threshold. |
| `MVStenosis` | grouped |  | `mitral_stenosis`: `Mild\|Moderate\|Severe` → `mild`, `moderate`, `severe`; `None` → `none` |  |
| `MVRegurgitation` | grouped |  | `mitral_regurgitation`: `Mild` → `mild`; `Moderate\|Severe` → `moderate`, `moderate_to_severe`, `severe`; `None\|Trace` → `none`, `trace`; no class: `mild_to_moderate` |  |
| `TVRegurgitation` | grouped |  | `tricuspid_regurgitation`: `Mild` → `mild`; `Moderate\|Severe` → `moderate`, `moderate_to_severe`, `severe`; `None\|Trace` → `none`, `trace`; no class: `mild_to_moderate` |  |
| `TVPkGrad` | transformed | mmHg | `tr_peak_velocity`: velocity (m/s) = sqrt(gradient / 4), the simplified Bernoulli equation. |  |
| `RAP-8-or-higher` | grouped |  | `ra_pressure`: `1.0` → at least 8; `0.0` → below 8; also related to `ivc_dilation`, `ivc_collapse` |  |
| `AORoot` | related | cm | related to `aortic_root_dilation` |  |
<!-- END GENERATED: mapping_panecho_tasks -->

## Clinical choices in v1

The medical review covers the whole ontology. These choices rest most on clinical judgment; a change the review asks
for is made in the approved files before their content commit (see [approval_v1.md](approval_v1.md)).

1. **View class boundaries.** The descriptions are written to match how the DeepECHO classes were labelled, in
   particular the `_LV` and `_ZOOM` variants and `SUBCOSTAL`, which groups the subcostal four-chamber and IVC views.
2. **LV outflow tract.** v1 has no visibility concept for it (no `lvot_visible`); it appears only in
   `lvot_obstruction`, a measurement.
3. **Priors.** Each structure's and finding's prior (views and acquisition modes), listed in full in the tables above.
4. **Measurement concepts.** Which concepts are marked `measurement`, and so stay out of captions, and which are
   judged visually.
5. **Mappings.** The relations and class-to-value assignments, in particular: PanEcho's merged grades, where
   `mild_to_moderate` regurgitation is left unassigned; PanEcho "mild" effusion as "small"; PA systolic pressure as
   only related to RVSP; EchoPrime's reported EF as the same quantity as the visual estimate; and every EchoPrime
   binary task kept as only related, because its phrases also match negated, possible or borderline mentions.
6. **Field of view.** Reviewers judge statements inside the square the model uses, with the rest of the clip as
   context. A statement that only the area outside the square shows is unconfirmed ("crop, zoom or outside the
   square"), not unsupported, so the pilot's error rates count caption errors and not crop effects.
