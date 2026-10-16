# Revue des légendes RADAR : instructions aux relecteurs (v1)

Version : `radar-caption-ontology-v1`, avec la [revue par colonnes v2](column_review_v2.md) et la
[revue de coupe et de type](view_type_review_v1.md). Leur approbation médicale est consignée dans
[approval_v1.md](approval_v1.md) ; aucune donnée clinique ne peut être téléversée pour revue avant que ce registre soit
`approved`. [English version](labeling_instructions_v1.md). Les deux versions décrivent les mêmes formulaires, dont
les libellés sont en anglais, comme dans Labelbox.

## Ce que vous relisez

Chaque vidéo valide d'une étude pilote va dans l'un de deux projets Labelbox :

- **Revue par colonnes**, pour un clip d'une coupe classée, en mode B ou en Doppler couleur. Vous voyez le clip, sa
  coupe et son mode d'acquisition proposés, ses **constatations préremplies** et une **légende proposée**, une phrase
  par constatation. Les constatations sont préremplies à partir de ce que le compte rendu de l'étude affirme
  explicitement : une section du compte rendu qui ne dit rien ne préremplit rien, et aucune constatation n'est
  supposée normale. Ce sont des suggestions ; votre revue décide de ce que ce clip étaye.
- **Revue de coupe et de type**, pour un clip que le classifieur de coupes classe `OTHER`, ou ne classe pas. Vous
  voyez le clip seul, sans légende, et dites ce qu'il est et quelle coupe il montre.

Les clips figés, les clips avec des mesures visibles et les coupes classées dans d'autres modes d'acquisition sont
retirés avant le téléversement. Si l'un d'eux vous parvient quand même, signalez-le dans **Comments**.

À côté de chaque clip de la revue par colonnes, une pièce jointe texte liste les propositions d'origine, chacune avec
le champ du compte rendu dont elle vient, et les nombres du compte rendu (FE, dimensions, scores de cinétique) comme
contexte de l'étude. Ces nombres décrivent l'étude, pas ce clip, et ne sont jamais des constatations à confirmer.

Le clip est montré en entier, à sa résolution d'origine, après dé-identification : le texte identifiant est masqué en
noir. Le modèle reçoit le même clip, chaque image redimensionnée en entier : rien n'est rogné. Jugez chaque
constatation sur l'image entière et la boucle entière (voir
[Model input and review display](ontology_v1.md#model-input-and-review-display)).

Les termes employés ci-dessous :

- **Constatation :** une colonne avec sa valeur, par exemple « LV function: Mildly reduced ». Chacune donne une phrase
  de la légende.
- **Étayée :** ce clip la montre, à la valeur indiquée.
- **Fausse :** ce clip montre une autre valeur, ou la contredit.
- **Impossible à évaluer :** ce clip ne permet pas de trancher (voir [Incertitude](#incertitude)).

## Revue par colonnes

### Comment relire un clip

1. Regardez la boucle entière, plusieurs fois si besoin.
2. Décidez si le clip est interprétable (question 1). S'il ne l'est pas, donnez la raison, sélectionnez **Review
   completed** et soumettez : rien d'autre n'est lu.
3. Vérifiez la coupe et le mode d'acquisition préremplis, et corrigez directement toute valeur erronée (questions 2 et 3).
4. Vérifiez les cibles que le clip couvre réellement et, pour un clip PSAX ou sous-costal, son niveau (questions 4 et
   5).
5. Passez les constatations préremplies une à une : étayée, fausse ou impossible à évaluer (question 6).
6. Cherchez une constatation importante et visible que la proposition omet (question 7), et un énoncé de la légende
   hors de ses constatations (question 8).
7. Ajoutez un commentaire seulement si nécessaire, sélectionnez **Review completed** (question 11) et soumettez.

Toutes les réponses sauf **Review completed** sont préremplies : clip interprétable, coupe et mode corrects, chaque
constatation étayée, rien de manquant ni de superflu. Modifiez ce que ce clip n'étaye pas ; une réponse laissée telle
quelle vaut confirmation.

### Le formulaire

<!-- BEGIN GENERATED: column_form -->
| # | Question | Id | Obligatoire | Réponses |
|---|---|---|---|---|
| 1 | Is this clip uninterpretable? | `uninterpretable` | oui | `no` No, it can be interpreted; `yes` Yes, nothing in it can be judged → Why? (une raison ci-dessous) |
| 2 | View (prefilled - correct if needed) | `view` | oui | une des 20 classes de coupe |
| 3 | Acquisition mode (prefilled - correct if needed) | `modality` | oui | `bmode` B-mode; `color_flow` Color Doppler; `spectral` Spectral Doppler; `m_mode` M-mode; `mixed` Mixed / analysis acquisition; `unresolved` Unresolved |
| 4 | Confirm targets actually covered and assessable | `visible_targets` | non | `lv` LV; `rv` RV; `la` LA; `ra` RA; `mitral` Mitral; `tricuspid` Tricuspid; `pericardium` Pericardium; `aortic` Aortic; `aorta_root` Aorta root; `aorta_ascending` Aorta ascending; `pulmonic` Pulmonic; `ivc` IVC; `aorta_arch` Aorta arch |
| 5 | PSAX slice / SUBCOSTAL subtype (when relevant) | `psax_level` | oui | `not_applicable` Not applicable; `unresolved` Unresolved; `mid` PSAX mid / papillary level; `basal` PSAX basal; `apical` PSAX apical; `mitral` PSAX mitral leaflet level; `ivc` IVC focused; `four_chamber` Subcostal four chamber |
| 6 | Prefilled findings — confirm or correct selected columns | `review_columns` | non | les colonnes préremplies, chacune : Proposed value (original retained in context) (une des valeurs de la colonne) + Confirm this finding (`yes` Supported in this input; `no` Incorrect / should not be here → Why should it not be here? (une raison d'erreur) + Correction (`remove` Remove this claim, ou une des valeurs de la colonne); `cannot_assess` Cannot assess → Why can it not be assessed? (une limite d'évaluation)) |
| 7 | Important visible finding missing from the candidate? | `missing_information` | oui | `no` No; `yes` Yes → Select missing columns and values (toute colonne que la coupe porte, chacune : Missing visible value (une des valeurs de la colonne) + Visible in this input? (`yes` Yes; `cannot_assess` Cannot assess)) |
| 8 | Additional statement that should not be here? | `unsupported_information` | oui | `no` No; `yes` Yes → Select offending columns / statements (toute colonne ou tout type d'énoncé, chacune : Why should this statement not be here? (une raison d'erreur); un type d'énoncé demande aussi Copy the offending statement (texte libre)) |
| 9 | Original caption (for reference only; edits are ignored) | `candidate_caption` | non | texte libre |
| 10 | Comments (no patient details) | `comments` | non | texte libre |
| 11 | I reviewed this input, checked the proposed values and targets, and confirmed or corrected the suggestions. | `review_completed` | oui | `yes` Review completed |
<!-- END GENERATED: column_form -->

Les colonnes de constatations, leurs valeurs et les coupes qui les portent sont listées dans
[column_review_v2.md](column_review_v2.md#clinical-columns-and-choices). Les raisons :

<!-- BEGIN GENERATED: column_reasons -->
**Raisons d'erreur (constatation fausse, énoncé supplémentaire)**

| Id | Libellé |
|---|---|
| `wrong_value` | Wrong value / diagnosis |
| `wrong_severity` | Wrong severity |
| `wrong_location` | Wrong structure / segment |
| `contradicted` | Contradicted by visible evidence |
| `not_visible` | Claimed structure/finding not visible |
| `wrong_view` | Wrong view / target |
| `wrong_modality` | Needs another acquisition mode |
| `study_only` | Study-level claim in a local caption |
| `unsupported_report` | No supporting source / invented assertion |

**Limites d'évaluation (constatation impossible à évaluer)**

| Id | Libellé |
|---|---|
| `poor_quality` | Poor image quality |
| `incomplete_cycle` | Incomplete cardiac/respiratory cycle |
| `target_outside` | Target outside the displayed input |
| `unresolved_view` | View / slice level unresolved |
| `insufficient_input` | Needs additional views or measurements |

**Clips ininterprétables**

| Id | Libellé |
|---|---|
| `poor_quality` | Poor image quality or artifacts throughout |
| `no_cardiac_structure` | No identifiable cardiac structure (crop, zoom or depth) |
| `not_cardiac` | Not a cardiac ultrasound acquisition |
| `other` | Other: say why in Comments |

**Types d'énoncé supplémentaire (énoncé de la légende hors de ses colonnes)**

| Id | Libellé |
|---|---|
| `exact_ef` | Exact numerical EF |
| `valve_severity` | Integrated valve severity |
| `numeric_measurement` | Numeric dimension / velocity / gradient / pressure |
| `other_statement` | Other unsupported statement |
<!-- END GENERATED: column_reasons -->

### Règles

#### Juger ce clip seulement

Jugez ce que montre ce clip, pas ce que vous savez du patient : le compte rendu, ses nombres, les autres clips de
l'étude et vos propres mesures ne comptent pas. Une constatation réelle chez le patient mais invisible sur ce clip
n'est pas étayée ici.

#### Étayée, fausse, impossible à évaluer

- **Étayée :** vous la voyez à la valeur indiquée, comme vous la graderiez à l'œil. Laissez « Supported in this
  input ».
- **Fausse :** choisissez pourquoi (les raisons d'erreur ci-dessus), puis la valeur correcte ou « Remove this
  claim ». Ne donnez une valeur que si ce clip la montre ; sinon retirez l'énoncé. Retirer un énoncé n'affirme jamais
  son contraire.
- **Impossible à évaluer :** choisissez la limite. Ce n'est ni une erreur ni une constatation négative, et la
  constatation reste hors de la légende.

**Proposed value** garde la valeur d'origine : une correction va dans **Correction**, jamais dans **Proposed value**.

#### Incertitude

Quand vous ne pouvez pas décider si une constatation est étayée, ne devinez pas : choisissez **Cannot assess** avec la
limite la plus proche. « View / slice level unresolved » sert pour une constatation qui dépend d'une coupe ou d'un
niveau que vous ne pouvez pas dire ; « Needs additional views or measurements » pour une constatation que ce clip seul
ne peut pas trancher ; « Target outside the displayed input » pour une structure coupée au bord du secteur, hors de la
boîte couleur, ou hors d'un clip zoomé.

#### Constatations absentes

Certaines constatations affirment une absence : « No mitral regurgitant jet is demonstrated in this acquisition »,
« No pericardial effusion is demonstrated in the imaged region ». Jugez-les comme toute autre constatation : étayées
seulement si ce clip pouvait montrer la constatation (bonne coupe, bon mode, la structure dans l'image et, pour un jet,
le Doppler couleur sur la valve pendant la systole) et qu'elle n'y est pas. Si le clip ne pouvait pas la montrer, choisissez
**Cannot assess**.

Un compte rendu qui ne mentionne pas une constatation ne prouve pas son absence, et une constatation absente qui n'est
pas préremplie n'est pas une information manquante.

#### Constatations normales préremplies

Une constatation normale n'est préremplie que si le compte rendu l'affirme (« Valve tricuspide normale »). Ne la
confirmez que si ce clip la montre. Un aspect normal des feuillets n'établit pas une mobilité normale, et aucun des
deux n'établit l'absence de jet.

#### Sévérité clinique et portée

Les valeurs de fonction et de taille sont des impressions visuelles à l'échelle du clip, pas le diagnostic d'un
examen complet. Un jet de régurgitation est seulement démontré ou non démontré dans cette acquisition : aucune
constatation ne le grade, et une légende qui affirme un grade intégré, une FE exacte ou un nombre fait un énoncé
supplémentaire (question 8). La cinétique segmentaire ne décrit que les segments que montre ce clip. Les scores de
cinétique du compte rendu gardent leur ambiguïté : le score 1 donne « Normal / hyperkinetic », le score 3
« Akinetic / severely hypokinetic ». N'inférez pas une cinétique normale dans des segments non vus, une FE de l'étude,
une sévérité de sténose, des pressions de remplissage ni une hypertension pulmonaire.

#### Cibles et structures non visibles

**Confirm targets actually covered and assessable** est prérempli avec les cibles que la coupe montre habituellement.
Décochez une cible que ce clip ne couvre pas assez pour l'évaluer ; cochez une cible qu'un clip ciblé ou zoomé couvre.

Une constatation sur une structure absente du clip est **fausse** : « Claimed structure/finding not visible ». Une
structure seulement en partie dans l'image, trop coupée pour être jugée, est **impossible à évaluer** : « Target
outside the displayed input ».

#### Niveau PSAX et sous-type sous-costal

Pour un clip PSAX, choisissez le niveau de coupe ; pour un clip sous-costal, VCI ou quatre cavités. Les segments
pariétaux ne sont jugés qu'au niveau moyen (papillaire) et les feuillets mitraux qu'au niveau mitral, et les
constatations sous-costales sur la VCI et sur les cavités ne se mélangent jamais. Pour les autres coupes, laissez
« Not applicable ».

#### Information manquante

Répondez « Yes » à la question 7 seulement pour une constatation **importante**, visible sur ce clip, que la coupe
porte et qu'aucune constatation préremplie n'énonce : choisissez la colonne et sa valeur, et confirmez qu'elle est
visible. N'ajoutez pas de valeurs normales que le compte rendu omet, de constatations absentes, de mesures, ni ce que
seuls d'autres clips montrent. Une constatation préremplie à la valeur fausse se corrige à la question 6 ; elle ne
s'ajoute pas comme manquante.

#### Énoncés supplémentaires

La légende proposée est construite à partir de ses constatations, donc la question 8 est rarement « Yes ».
Utilisez-la quand la légende affirme quelque chose hors de ses constatations, comme un grade, une FE exacte ou un
nombre : choisissez la colonne ou le type d'énoncé et la raison, et recopiez l'énoncé quand le formulaire le demande.

#### Clips ininterprétables

Répondez « Yes » à la question 1 seulement quand rien dans le clip ne peut être jugé : artéfacts ou mauvaise qualité
d'image tout du long, aucune structure cardiaque reconnaissable, ou pas une acquisition d'échographie cardiaque.
Donnez la raison ; avec « Other », dites pourquoi dans **Comments**. Les autres réponses ne sont alors pas lues, et
vous n'avez pas à les modifier.

Si seules certaines constatations ne peuvent pas être jugées, le clip est interprétable : utilisez **Cannot assess**
pour celles-là.

#### La coupe et le mode d'acquisition

Vérifiez la question 2 d'après les classes de coupe décrites dans [ontology_v1.md](ontology_v1.md#views). Si le clip
appartient à une autre classe, remplacez la valeur préremplie par cette classe ; utilisez `OTHER` pour les clips hors
axe ou non standard. Si rien dans le clip ne permet une classification fiable, marquez le clip ininterprétable au
lieu de choisir au hasard. Vérifiez et corrigez directement le mode d'acquisition à la question 3.

Jugez tout de même les constatations préremplies sur le clip : elles sont comptées dans tous les cas. Une coupe ou un
mode corrigé ne donne pas de légende au clip.

#### La légende

Il n'y a pas de légende à modifier. Après la revue, la légende est reconstruite à partir de vos réponses : les
constatations étayées, les valeurs corrigées et les constatations manquantes confirmées, une phrase chacune.
**Original caption** montre la proposition et n'est pas lu.

#### Quand une revue est renvoyée

Une revue est renvoyée pour correction quand :

- une réponse obligatoire est vide, ou **Review completed** n'est pas sélectionné ;
- une constatation fausse n'a pas de raison ou pas de correction, ou une constatation impossible à évaluer n'a pas de
  limite ;
- la question 7 est « Yes » sans colonne, sans sa valeur ou sans « Visible in this input? Yes », ou liste une
  constatation déjà préremplie ;
- la question 8 est « Yes » sans colonne ou type d'énoncé et sa raison, ou nomme une constatation laissée étayée ;
- la coupe ou le mode d'acquisition est répondu « No » mais laissé tel que proposé, ou modifié sans répondre « No » ;
- une constatation préremplie est décochée, ou sa valeur proposée modifiée ;
- le clip est ininterprétable sans raison, ou avec « Other » et sans commentaire.

Les règles complètes sont dans [pilot_metrics_v1.md](pilot_metrics_v1.md#which-reviews-count).

## Exemples commentés : revue par colonnes

Les clips sont synthétiques, décrits en mots.

<!-- BEGIN GENERATED: column_examples -->
### 1. Constatations étayées

- **Clip (synthétique) :** Coupe apicale 4 cavités en mode B : ventricule gauche de taille normale, ventricule droit qui se contracte normalement, feuillets tricuspides fins.
- **Proposé :** `A4C`, B-mode
- **Constatations préremplies :** `lv_size` Normal; `rv_function` Normal; `tricuspid_appearance` Normal
- **Légende proposée :** « The visible LV cavity is not dilated. RV systolic function appears normal in this input. The visible tricuspid valve has a normal appearance. »
- **Ce que le relecteur modifie :** rien
- **Légende reconstruite :** « The visible LV cavity is not dilated. RV systolic function appears normal in this input. The visible tricuspid valve has a normal appearance. »
- **Pourquoi :** Chaque constatation préremplie est visible sur ce clip, à la valeur indiquée, et rien d'important ne manque : il suffit de confirmer et de soumettre.

### 2. Sévérité fausse

- **Clip (synthétique) :** Coupe apicale 4 cavités en mode B : le ventricule gauche se contracte nettement moins que la normale, une atteinte modérée à l'œil.
- **Proposé :** `A4C`, B-mode
- **Constatations préremplies :** `lv_function` Mildly reduced
- **Légende proposée :** « LV systolic function appears mildly reduced in this input. »
- **Ce que le relecteur modifie :**
  - LV function, Confirm this finding : **Incorrect / should not be here**
  - LV function, Why should it not be here? **Wrong severity**
  - LV function, Correction : **Moderately reduced**
- **Légende reconstruite :** « LV systolic function appears moderately reduced in this input. »
- **Pourquoi :** La constatation est juste mais pas sa sévérité : « Wrong severity », avec la valeur que montre le clip. La légende reconstruite reprend la correction ; la valeur d'origine reste enregistrée.

### 3. Constatation contredite

- **Clip (synthétique) :** Coupe parasternale grand axe en mode B : le péricarde postérieur est bien visible, sans liquide autour du cœur.
- **Proposé :** `PLAX`, B-mode
- **Constatations préremplies :** `pericardial_effusion` Present
- **Légende proposée :** « Pericardial effusion is demonstrated in the imaged region. »
- **Ce que le relecteur modifie :**
  - Pericardial effusion, Confirm this finding : **Incorrect / should not be here**
  - Pericardial effusion, Why should it not be here? **Contradicted by visible evidence**
  - Pericardial effusion, Correction : **Not demonstrated**
- **Légende reconstruite :** « No pericardial effusion is demonstrated in the imaged region. »
- **Pourquoi :** Le clip montre bien le péricarde et aucun épanchement : la constatation préremplie est contredite. Comme ce clip peut montrer un épanchement, la correction peut dire qu'aucun n'est démontré ; s'il ne le pouvait pas, il faudrait retirer l'énoncé (« Remove this claim »).

### 4. Constatation manquante

- **Clip (synthétique) :** Coupe apicale 4 cavités en Doppler couleur : un jet d'insuffisance mitrale et un jet d'insuffisance tricuspide bien visible.
- **Proposé :** `A4C`, Color Doppler
- **Constatations préremplies :** `mitral_regurgitant_jet` Jet demonstrated
- **Légende proposée :** « A mitral regurgitant jet is demonstrated in this color acquisition. »
- **Ce que le relecteur modifie :**
  - Important visible finding missing from the candidate? **Yes**
  - Select missing columns and values : **Tricuspid regurgitant jet**
  - Tricuspid regurgitant jet, Missing visible value : **Jet demonstrated**
  - Tricuspid regurgitant jet, Visible in this input? **Yes**
- **Légende reconstruite :** « A mitral regurgitant jet is demonstrated in this color acquisition. A tricuspid regurgitant jet is demonstrated in this color acquisition. »
- **Pourquoi :** Ce qui est prérempli est étayé ; le jet tricuspide est visible, pertinent et absent des constatations : il est ajouté comme manquant. Aucun des deux jets ne reçoit de grade : une boucle couleur montre si un jet est démontré, pas une sévérité intégrée.

### 5. Absence impossible à juger

- **Clip (synthétique) :** Coupe apicale 4 cavités en Doppler couleur, la boîte couleur placée sur la valve tricuspide ; la valve mitrale est hors de la boîte couleur.
- **Proposé :** `A4C`, Color Doppler
- **Constatations préremplies :** `mitral_regurgitant_jet` No jet demonstrated in this acquisition
- **Légende proposée :** « No mitral regurgitant jet is demonstrated in this acquisition. »
- **Ce que le relecteur modifie :**
  - Mitral regurgitant jet, Confirm this finding : **Cannot assess**
  - Mitral regurgitant jet, Why can it not be assessed? **Target outside the displayed input**
- **Légende reconstruite :** aucune : aucune constatation n'a été confirmée
- **Pourquoi :** Un jet ne peut être dit absent que là où le Doppler couleur couvre la valve. Ici ce n'est pas le cas : l'absence ne peut pas être jugée, ce qui n'est ni une erreur ni une constatation négative.

### 6. Segment non évaluable

- **Clip (synthétique) :** Coupe apicale 2 cavités en mode B : ventricule gauche de taille normale ; l'endocarde de la paroi inférieure moyenne n'est pas visible (perte d'écho).
- **Proposé :** `A2C`, B-mode
- **Constatations préremplies :** `lv_size` Normal; `wall_mid_inferior` Normal / hyperkinetic (report score 1)
- **Légende proposée :** « The visible LV cavity is not dilated. The visible mid inferior segment shows normal or hyperkinetic motion. »
- **Ce que le relecteur modifie :**
  - Mid inferior wall motion, Confirm this finding : **Cannot assess**
  - Mid inferior wall motion, Why can it not be assessed? **Poor image quality**
- **Légende reconstruite :** « The visible LV cavity is not dilated. »
- **Pourquoi :** Un segment dont l'endocarde n'est pas vu ne peut pas être gradé. « Cannot assess » avec sa limite retire l'énoncé de la légende sans le compter comme une erreur ; l'autre constatation reste.

### 7. Coupe fausse

- **Clip (synthétique) :** Boucle en mode B classée apicale 4 cavités, mais la chambre de chasse du ventricule gauche et la valve aortique sont ouvertes dans l'image : c'est une coupe apicale 5 cavités. Les deux ventricules sont de taille normale.
- **Proposé :** `A4C`, B-mode
- **Constatations préremplies :** `lv_size` Normal; `rv_size` Normal
- **Légende proposée :** « The visible LV cavity is not dilated. The visible RV cavity is not dilated. »
- **Ce que le relecteur modifie :**
  - View (prefilled - correct if needed) : **A5C**
- **Légende reconstruite :** aucune : la coupe ou le mode d'acquisition a été corrigé
- **Pourquoi :** Remplacer la coupe préremplie par la coupe que montre le clip. Les constatations préremplies sont tout de même jugées sur le clip, car elles sont comptées. Le clip n'a pas de légende sous la coupe proposée.

### 8. Clip ininterprétable

- **Clip (synthétique) :** Boucle apicale en mode B envahie d'artéfacts du début à la fin : aucune paroi ni valve n'est reconnaissable.
- **Proposé :** `A4C`, B-mode
- **Constatations préremplies :** `lv_function` Normal
- **Légende proposée :** « LV systolic function appears normal in this input. »
- **Ce que le relecteur modifie :**
  - Is this clip uninterpretable? **Yes, nothing in it can be judged**
  - Why? **Poor image quality or artifacts throughout**
- **Légende reconstruite :** aucune : le clip est ininterprétable
- **Pourquoi :** Rien dans ce clip ne peut être jugé : répondre « Yes », donner la raison et soumettre ; les autres réponses ne sont pas lues. Quand seules certaines constatations ne peuvent pas être jugées, le clip est interprétable : utiliser « Cannot assess » pour celles-là.
<!-- END GENERATED: column_examples -->

## Revue de coupe et de type (clips `OTHER`)

Chaque élément est un clip que le classifieur de coupes classe `OTHER` ou ne classe pas. Dites ce qu'il est et, pour
une boucle échographique, quelle coupe il montre et son mode d'acquisition. Vos réponses mesurent ce que contient
`OTHER`, et une boucle échographique d'une coupe que porte la revue par colonnes, en mode B ou en Doppler couleur,
retourne à la revue par colonnes sous cette coupe.

<!-- BEGIN GENERATED: view_type_form -->
| Question | Id | Obligatoire | Réponses |
|---|---|---|---|
| What is this item? | `input_type` | oui | `cine_loop` Echo cine loop (moving echo images); `analysis_screen` Analysis or quantification screen (strain, 3D, EF tool); `no_echo_image` No echo image |
| Which view does it show? | `view` | oui, demandée si `input_type` vaut `cine_loop` | `A2C` A2C: Apical 2-chamber view; `A2C_LV` A2C_LV: Apical 2-chamber view, LV-focused; `A2C_ZOOM` A2C_ZOOM: Apical 2-chamber view, zoomed; `A3C` A3C: Apical 3-chamber view; `A3C_LV` A3C_LV: Apical 3-chamber view, LV-focused; `A3C_ZOOM` A3C_ZOOM: Apical 3-chamber view, zoomed; `A4C` A4C: Apical 4-chamber view; `A4C_LV` A4C_LV: Apical 4-chamber view, LV-focused; `A4C_ZOOM` A4C_ZOOM: Apical 4-chamber view, zoomed; `A5C` A5C: Apical 5-chamber view; `A5C_ZOOM` A5C_ZOOM: Apical 5-chamber view, zoomed; `PLAX` PLAX: Parasternal long-axis view; `PLAX_DEEP` PLAX_DEEP: Parasternal long-axis view, deep; `PLAX_ZOOM` PLAX_ZOOM: Parasternal long-axis view, zoomed; `PSAX` PSAX: Parasternal short-axis view; `PSAX_AV` PSAX_AV: Parasternal short-axis view at the aortic valve; `RVINF` RVINF: Right ventricular inflow view; `SUBCOSTAL` SUBCOSTAL: Subcostal view; `SUPRASTERNAL` SUPRASTERNAL: Suprasternal view; `OTHER` OTHER: off-axis or non-standard echo view; `cannot_tell` Cannot tell the view |
| Acquisition mode | `modality` | oui, demandée si `input_type` vaut `cine_loop` | `bmode` B-mode; `color_flow` Color Doppler; `spectral` Spectral Doppler; `m_mode` M-mode; `mixed` Mixed / analysis acquisition; `unresolved` Unresolved |
| Is identifying text visible (a name, a record number, a date of birth)? | `identifier_visible` | oui | `no` No; `yes` Yes: the item must be deleted |
| Comments (no patient details) | `comments` | non | texte libre |
<!-- END GENERATED: view_type_form -->

- **Qu'est-ce que cet élément ?** « Echo cine loop » pour une image échographique en mouvement, dans tout mode
  d'acquisition : alors seulement la coupe et le mode d'acquisition sont demandés. Pour toute autre réponse, ils ne
  sont pas demandés et sont enregistrés comme sans objet.
- **Coupe.** `OTHER` sert à une coupe échographique hors axe ou non standard ; « Cannot tell the view » quand vous ne
  pouvez pas trancher.
- **Texte identifiant.** Répondez « Yes » si un nom, un numéro de dossier ou une date de naissance est lisible où que
  ce soit dans l'élément. Il est supprimé.

## Exemples commentés : revue de coupe et de type

<!-- BEGIN GENERATED: view_type_examples -->
### 1. Veine cave inférieure classée OTHER

- **Élément (synthétique) :** Boucle sous-costale en mode B de la veine cave inférieure qui se jette dans l'oreillette droite.
- **Réponses :** What is this item? **Echo cine loop (moving echo images)**; Which view does it show? **SUBCOSTAL: Subcostal view**; Acquisition mode : **B-mode**; Is identifying text visible (a name, a record number, a date of birth)? **No**
- **Ensuite :** renvoyé à la revue par colonnes en `SUBCOSTAL`, B-mode
- **Pourquoi :** Une boucle échographique d'une coupe que la revue par colonnes sait router : elle y retourne sous cette coupe.

### 2. Coupe hors axe

- **Élément (synthétique) :** Boucle en mode B entre les plans apicaux 4 et 5 cavités, raccourcie, sans coupe standard.
- **Réponses :** What is this item? **Echo cine loop (moving echo images)**; Which view does it show? **OTHER: off-axis or non-standard echo view**; Acquisition mode : **B-mode**; Is identifying text visible (a name, a record number, a date of birth)? **No**
- **Ensuite :** reste hors de la revue par colonnes
- **Pourquoi :** OTHER sert aux coupes échographiques hors axe ou non standard ; « Cannot tell the view » sert quand on ne peut pas trancher.

### 3. Écran d'analyse

- **Élément (synthétique) :** Écran d'analyse de déformation : une coupe apicale 4 cavités aux segments colorés et une cible en œil de bœuf.
- **Réponses :** What is this item? **Analysis or quantification screen (strain, 3D, EF tool)**; Is identifying text visible (a name, a record number, a date of birth)? **No**; coupe et mode d'acquisition non demandés, enregistrés comme sans objet
- **Ensuite :** reste hors de la revue par colonnes
- **Pourquoi :** Un écran d'analyse n'est pas un clip qu'une légende doit décrire : sa coupe et son mode ne sont pas demandés, et il ne retourne pas à la revue par colonnes.

### 4. Texte identifiant visible

- **Élément (synthétique) :** Boucle apicale 4 cavités en mode B où un nom reste lisible dans le bandeau du haut.
- **Réponses :** What is this item? **Echo cine loop (moving echo images)**; Which view does it show? **A4C: Apical 4-chamber view**; Acquisition mode : **B-mode**; Is identifying text visible (a name, a record number, a date of birth)? **Yes: the item must be deleted**
- **Ensuite :** supprimé de Labelbox : texte identifiant
- **Pourquoi :** Répondre « Yes » sans recopier le texte nulle part. L'élément est supprimé de Labelbox et l'étape de dé-identification est corrigée.
<!-- END GENERATED: view_type_examples -->

## Confidentialité

Les commentaires ne doivent contenir ni nom de patient, ni identifiant, ni date, ni texte de compte rendu.

Si vous voyez du texte identifiant dans l'image (un nom, un numéro de dossier, une date de naissance) : dans la revue
par colonnes, ne relisez pas le clip, passez-le dans Labelbox et donnez à l'équipe de l'étude son identifiant
Labelbox, sans recopier le texte ; dans la revue de coupe et de type, répondez « Yes » à la question sur le texte
identifiant. Dans les deux cas, le clip est retiré de Labelbox et du jeu de légendes.
