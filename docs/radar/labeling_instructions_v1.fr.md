# Revue des légendes RADAR : consignes aux relecteurs (v1)

Version : `radar-caption-ontology-v1`. Son approbation médicale est consignée dans [approval_v1.md](approval_v1.md).
Aucune donnée clinique ne peut être téléversée pour relecture selon ces consignes tant que le statut qui y figure
n'est pas `approved`.
[English version](labeling_instructions_v1.md). Les deux versions utilisent les mêmes identifiants de questions et de
réponses.

## Ce que vous relisez

Chaque élément dans Labelbox est un clip d'échocardiographie accompagné d'une **légende proposée** : une ou deux
phrases courtes, en français, qui indiquent la coupe, le mode d'acquisition et quelques constatations. Les légendes
proposées sont rédigées automatiquement à partir du compte rendu de l'examen et de ce que la coupe et l'acquisition
du clip peuvent montrer. Elles ne sont pas validées : votre relecture décide de ce que ce clip étaye réellement.

À côté du clip figurent aussi les structures que sa coupe montre habituellement (les « structures attendues »). C'est
un a priori pour vous aider, pas une affirmation de la légende.

Termes utilisés ci-dessous :

- **Énoncé :** une chose que dit la légende, en dehors de la coupe : le mode d'acquisition, la visibilité d'une
  structure, ou une constatation avec sa valeur (« Insuffisance mitrale : modérée », « Insuffisance mitrale :
  absente »).
- **Étayé :** ce clip le montre, avec la valeur ou la sévérité indiquée.
- **Impossible à vérifier :** ce clip ne permet pas de trancher (voir [Incertitude](#incertitude)).

L'ontologie sur laquelle repose le formulaire, avec chaque coupe, structure et constatation, est décrite dans
[ontology_v1.md](ontology_v1.md) (en anglais, libellés en français).

## Déroulement pour un clip

1. Regarder toute la boucle, plusieurs fois si nécessaire.
2. Décider si le clip est interprétable (question 1).
3. Vérifier la coupe (question 2).
4. Parcourir la légende énoncé par énoncé : étayé, non étayé (question 5) ou impossible à vérifier (question 6).
   Répondre ensuite à la question 3, qui les résume.
5. Chercher une information visible et pertinente que la légende omet (question 4).
6. Corriger la légende (question 7), et n'ajouter un commentaire que si nécessaire.

## Le formulaire

<!-- BEGIN GENERATED: review_form -->
| # | Question | Id | Obligatoire | Réponses |
|---|---|---|---|---|
| 1 | Ce clip est-il ininterprétable ? | `uninterpretable_clip` | oui | `no` Non, le clip est interprétable; `yes` Oui, il est ininterprétable → Raison : Qualité d'image, Cadrage ou zoom, Modalité, Autre |
| 2 | La coupe indiquée est-elle correcte ? | `view_correct` | oui | `yes` Oui; `no` Non → Coupe correcte (une des classes de coupe); `unsure` Incertain; `not_applicable` Sans objet : clip ininterprétable |
| 3 | En dehors de la coupe, tout ce que la légende affirme est-il étayé par ce clip ? | `caption_supported` | oui | `yes` Oui, chaque énoncé est étayé; `no` Non, au moins un énoncé n'est pas étayé ou est faux; `unsure` Incertain : rien n'est faux, mais au moins un énoncé est impossible à vérifier; `not_applicable` Sans objet : clip ininterprétable |
| 4 | La légende omet-elle une information visible et cliniquement pertinente ? | `missing_information` | oui | `no` Non; `yes` Oui → Concepts manquants (les énoncés ci-dessous) + Autre information manquante (hors ontologie) (texte libre); `not_applicable` Sans objet : clip ininterprétable |
| 5 | En dehors de la coupe, la légende affirme-t-elle quelque chose que ce clip n'étaye pas (information non étayée ou hallucinée) ? | `unsupported_information` | oui | `no` Non; `yes` Oui → Énoncés non étayés ou faux (les énoncés ci-dessous), chacun : Non montré ou contredit par le clip (halluciné) / Présent, mais valeur ou sévérité fausse; `not_applicable` Sans objet : clip ininterprétable |
| 6 | Énoncés impossibles à vérifier sur ce clip (laisser vide s'il n'y en a aucun) | `unconfirmed_statements` | non | les énoncés ci-dessous, chacun : Raison (Visible, mais je ne suis pas certain / Qualité d'image / Cadrage ou zoom / Modalité / Autre) |
| 7 | La légende telle qu'elle devrait se lire (préremplie avec la légende proposée) | `corrected_caption` | oui | texte libre |
| 8 | Commentaires (aucune information sur le patient) | `comments` | non | texte libre |
<!-- END GENERATED: review_form -->

Les énoncés que vous pouvez signaler ou indiquer comme manquants :

<!-- BEGIN GENERATED: statements -->
| Énoncé | Libellé | Type |
|---|---|---|
| `acquisition` | Mode d'acquisition | mode d'acquisition |
| `lv_visible` | Ventricule gauche visible | visibilité |
| `rv_visible` | Ventricule droit visible | visibilité |
| `la_visible` | Oreillette gauche visible | visibilité |
| `ra_visible` | Oreillette droite visible | visibilité |
| `mitral_valve_visible` | Valve mitrale visible | visibilité |
| `aortic_valve_visible` | Valve aortique visible | visibilité |
| `tricuspid_valve_visible` | Valve tricuspide visible | visibilité |
| `pulmonic_valve_visible` | Valve pulmonaire visible | visibilité |
| `aortic_root_visible` | Racine aortique visible | visibilité |
| `interatrial_septum_visible` | Septum interauriculaire visible | visibilité |
| `pericardium_visible` | Péricarde visible | visibilité |
| `ivc_visible` | Veine cave inférieure visible | visibilité |
| `lv_systolic_function` | Fonction systolique ventriculaire gauche | constatation |
| `lv_ejection_fraction` | Fraction d'éjection VG (estimation visuelle) | constatation |
| `lv_size` | Taille du ventricule gauche | constatation |
| `lv_wall_thickness` | Hypertrophie ventriculaire gauche | constatation |
| `lv_regional_wall_motion` | Cinétique segmentaire du VG (segment le plus atteint) | constatation |
| `rv_systolic_function` | Fonction systolique ventriculaire droite | constatation |
| `rv_size` | Taille du ventricule droit | constatation |
| `la_size` | Taille de l'oreillette gauche | constatation |
| `ra_size` | Taille de l'oreillette droite | constatation |
| `atrial_septal_hypertrophy` | Hypertrophie lipomateuse du septum interauriculaire | constatation |
| `mitral_regurgitation` | Insuffisance mitrale | constatation |
| `mitral_annular_calcification` | Calcification de l'anneau mitral | constatation |
| `mitral_valve_device` | Dispositif ou prothèse mitrale | constatation |
| `aortic_regurgitation` | Insuffisance aortique | constatation |
| `bicuspid_aortic_valve` | Bicuspidie aortique | constatation |
| `aortic_valve_prosthesis` | Prothèse valvulaire aortique | constatation |
| `aortic_root_dilation` | Dilatation de la racine aortique | constatation |
| `tricuspid_regurgitation` | Insuffisance tricuspide | constatation |
| `pulmonic_regurgitation` | Insuffisance pulmonaire | constatation |
| `ivc_dilation` | Dilatation de la veine cave inférieure | constatation |
| `ivc_collapse` | Collapsus inspiratoire de la VCI | constatation |
| `pericardial_effusion` | Épanchement péricardique | constatation |
| `pacing_lead` | Sonde de stimulation intracardiaque | constatation |
| `mechanical_circulatory_support` | Assistance circulatoire mécanique | constatation |
<!-- END GENERATED: statements -->

## Règles

### Juger ce clip seulement

Jugez ce que ce clip montre, pas ce que vous savez du patient : le compte rendu, les autres clips de l'examen et vos
propres mesures ne comptent pas. Une constatation réelle chez le patient mais invisible sur ce clip n'est pas étayée
ici.

### Étayé, non étayé, valeur fausse

- **Étayé :** vous le voyez sur ce clip, avec la valeur ou la sévérité indiquée, telle que vous l'évalueriez
  visuellement.
- **Non étayé (halluciné) :** le clip ne le montre pas, ou le contredit. Signalez-le à la question 5 avec « non
  montré ou contredit ».
- **Valeur fausse :** la constatation est présente mais sa valeur ou sa sévérité est fausse (par exemple sévère au
  lieu de modérée). Signalez-la à la question 5 avec « valeur ou sévérité fausse » et donnez la bonne valeur dans la
  légende corrigée.
- Le **mode d'acquisition** est aussi un énoncé : si la légende indique Doppler couleur et que le clip est en mode B,
  signalez « mode d'acquisition » comme non étayé.

### Incertitude

Quand vous ne pouvez pas décider si un énoncé est étayé, ne devinez pas. Indiquez-le à la question 6 avec une raison :

- **incertain :** la constatation est visible mais vous ne pouvez pas la juger avec assurance (par exemple un jet
  excentré dont le grade est indéterminable) ;
- **qualité d'image, cadrage ou zoom, modalité, autre :** quelque chose dans le clip empêche de juger.

Si rien n'est faux dans la légende mais qu'au moins un énoncé est impossible à vérifier, répondez « incertain » à la
question 3. Un énoncé est soit non étayé, soit impossible à vérifier, jamais les deux. Pour la coupe, répondez
« incertain » si vous ne pouvez pas déterminer la classe.

### Constatations absentes

Une légende peut affirmer qu'une constatation est absente (« Insuffisance mitrale : absente »). Jugez-la comme tout
autre énoncé : elle n'est étayée que si ce clip pouvait montrer la constatation (bonne coupe, bonne modalité,
structure dans le secteur) et qu'elle est absente. Si le clip ne pouvait pas la montrer, l'énoncé est impossible à
vérifier, pas étayé.

Une constatation absente et non mentionnée dans la légende n'est pas une information manquante.

### Structures non visibles

Les structures attendues sont un a priori, pas une affirmation. Une structure attendue mais non visible sur ce clip
n'est ni une information manquante ni une erreur, sauf si la légende affirme qu'elle est visible : cet énoncé est
alors non étayé.

Une structure partiellement visible compte comme visible si vous pouvez l'identifier avec assurance ; sinon,
indiquez l'énoncé comme impossible à vérifier (cadrage ou zoom).

### Information manquante

Répondez « oui » à la question 4 seulement pour une information visible sur ce clip, cliniquement pertinente, que la
légende omet. Cochez les énoncés correspondants ; utilisez le champ de texte libre pour ce que la liste ne contient
pas. N'indiquez pas les mesures (TAPSE, PSVD, PISA…), ce que seuls d'autres clips montrent, ni les constatations
absentes.

### Clips ininterprétables

Répondez « oui » à la question 1 seulement quand rien dans la légende ne peut être jugé : artefacts majeurs, modalité
inadaptée à toute la légende, cadrage ne laissant rien d'identifiable. Donnez la raison, répondez « sans objet » aux
questions 2 à 5 et laissez la légende corrigée inchangée (elle est ignorée).

Si seule une partie de la légende ne peut pas être jugée, le clip est interprétable : utilisez les énoncés impossibles
à vérifier. Un clip interprétable ne reçoit jamais la réponse « sans objet » : si vous ne pouvez pas trancher,
répondez « incertain » quand la question le propose, et indiquez à la question 6 les énoncés impossibles à vérifier.

### La coupe

Répondez à la question 2 d'après les classes de coupe décrites dans [ontology_v1.md](ontology_v1.md#views). Si le
clip relève d'une autre classe, répondez « non » et choisissez cette classe, jamais celle qui est indiquée ; utilisez
`OTHER` pour un clip hors axe ou non standard. Si vous ne pouvez pas déterminer la classe, répondez « incertain » : ne
choisissez pas une classe au hasard et n'utilisez pas `OTHER` pour cela.

La coupe se juge seulement ici. Les questions 3 et 5 portent sur les autres énoncés : une coupe fausse ou incertaine,
à elle seule, donne « légende étayée : oui » si chaque énoncé est étayé, et n'est jamais signalée comme énoncé non
étayé. Dans tous les cas, jugez les énoncés tels qu'ils sont écrits.

### La légende corrigée

Le champ est prérempli avec la légende proposée. Réécrivez-la pour qu'elle n'affirme que ce que ce clip étaye :

- retirer les énoncés non étayés et ceux impossibles à vérifier, sauf le mode d'acquisition, que toute légende garde
  (voir ci-dessous) ;
- corriger les valeurs fausses ;
- ajouter l'information manquante quand la liste des énoncés la contient ;
- donner la bonne coupe si la réponse à la question 2 était « non » ;
- garder la même forme : la coupe, le mode d'acquisition, puis les constatations, en français ;
- ne jamais ajouter d'identifiants, de dates, de mesures ni de texte du compte rendu.

Quand vous ne pouvez pas déterminer la coupe ou le mode d'acquisition, la légende corrigée garde sa forme et
l'indique :

- question 2 « incertain » : écrivez « Coupe indéterminée » à la place de la coupe (« Coupe indéterminée, Doppler
  couleur. … ») ; la classe du classifieur reste enregistrée avec le clip ;
- mode d'acquisition impossible à vérifier : écrivez « acquisition indéterminée » à la place du mode. Si le mode
  indiqué est faux, écrivez celui que montre le clip, ou « acquisition indéterminée » si vous ne pouvez pas le
  déterminer.

Les énoncés que le clip étaye restent. Si la légende est juste telle quelle, laissez-la inchangée.

### Quand une relecture est renvoyée

Une relecture est renvoyée pour correction quand une réponse obligatoire est vide, comme une légende corrigée laissée
vide, ou quand ses réponses se contredisent : par exemple « légende étayée : oui » avec un énoncé non étayé ou
impossible à vérifier, « non » sans énoncé non étayé, « information manquante : oui » sans rien d'indiqué,
« sans objet » pour un clip interprétable, une coupe fausse corrigée en la coupe déjà indiquée, ou un énoncé signalé
que la légende ne contient pas. La liste complète est dans
[pilot_metrics_v1.md](pilot_metrics_v1.md#which-reviews-count) (en anglais).

### Confidentialité

Les commentaires et les légendes corrigées ne doivent contenir ni nom de patient, ni identifiant, ni date, ni texte du
compte rendu.

## Exemples commentés

Les clips sont synthétiques, décrits en mots.

<!-- BEGIN GENERATED: examples -->
### 1. Légende étayée

- **Clip (synthétique) :** Coupe apicale 4 cavités en Doppler couleur ; jet d'insuffisance mitrale modérée, bien aligné.
- **Coupe et acquisition du classifieur :** `A4C`, Doppler couleur
- **Légende proposée :** « Coupe apicale 4 cavités (A4C), Doppler couleur. Insuffisance mitrale : modérée. »
- **Réponses :**

  - Ce clip est-il ininterprétable ? **Non, le clip est interprétable**
  - La coupe indiquée est-elle correcte ? **Oui**
  - En dehors de la coupe, tout ce que la légende affirme est-il étayé par ce clip ? **Oui, chaque énoncé est étayé**
  - La légende omet-elle une information visible et cliniquement pertinente ? **Non**
  - En dehors de la coupe, la légende affirme-t-elle quelque chose que ce clip n'étaye pas (information non étayée ou hallucinée) ? **Non**
  - Énoncés impossibles à vérifier sur ce clip (laisser vide s'il n'y en a aucun) : **aucun**

- **Légende corrigée :** inchangée
- **Pourquoi :** Chaque énoncé est visible sur ce clip, et rien de pertinent ne manque.

### 2. Sévérité fausse

- **Clip (synthétique) :** Coupe apicale 4 cavités en Doppler couleur ; jet d'insuffisance mitrale d'allure modérée.
- **Coupe et acquisition du classifieur :** `A4C`, Doppler couleur
- **Légende proposée :** « Coupe apicale 4 cavités (A4C), Doppler couleur. Insuffisance mitrale : sévère. »
- **Réponses :**

  - Ce clip est-il ininterprétable ? **Non, le clip est interprétable**
  - La coupe indiquée est-elle correcte ? **Oui**
  - En dehors de la coupe, tout ce que la légende affirme est-il étayé par ce clip ? **Non, au moins un énoncé n'est pas étayé ou est faux**
  - La légende omet-elle une information visible et cliniquement pertinente ? **Non**
  - En dehors de la coupe, la légende affirme-t-elle quelque chose que ce clip n'étaye pas (information non étayée ou hallucinée) ? **Oui** (Insuffisance mitrale : Présent, mais valeur ou sévérité fausse)
  - Énoncés impossibles à vérifier sur ce clip (laisser vide s'il n'y en a aucun) : **aucun**

- **Légende corrigée :** « Coupe apicale 4 cavités (A4C), Doppler couleur. Insuffisance mitrale : modérée. »
- **Pourquoi :** La fuite est présente mais la sévérité est fausse : « valeur ou sévérité fausse », et la légende corrigée donne la sévérité que montre le clip.

### 3. Constatation hallucinée

- **Clip (synthétique) :** Coupe parasternale grand axe en mode B ; péricarde bien visible, sans épanchement.
- **Coupe et acquisition du classifieur :** `PLAX`, mode B
- **Légende proposée :** « Coupe parasternale grand axe (PLAX), mode B. Épanchement péricardique : de moyenne abondance. »
- **Réponses :**

  - Ce clip est-il ininterprétable ? **Non, le clip est interprétable**
  - La coupe indiquée est-elle correcte ? **Oui**
  - En dehors de la coupe, tout ce que la légende affirme est-il étayé par ce clip ? **Non, au moins un énoncé n'est pas étayé ou est faux**
  - La légende omet-elle une information visible et cliniquement pertinente ? **Non**
  - En dehors de la coupe, la légende affirme-t-elle quelque chose que ce clip n'étaye pas (information non étayée ou hallucinée) ? **Oui** (Épanchement péricardique : Non montré ou contredit par le clip (halluciné))
  - Énoncés impossibles à vérifier sur ce clip (laisser vide s'il n'y en a aucun) : **aucun**

- **Légende corrigée :** « Coupe parasternale grand axe (PLAX), mode B. Épanchement péricardique : absent. »
- **Pourquoi :** Le clip montre bien le péricarde et aucun épanchement : l'énoncé n'est pas étayé (halluciné). Comme le clip permet de juger l'absence, la légende corrigée peut l'indiquer.

### 4. Information manquante

- **Clip (synthétique) :** Coupe apicale 4 cavités en Doppler couleur ; insuffisance mitrale légère et jet d'insuffisance tricuspide modérée bien visible.
- **Coupe et acquisition du classifieur :** `A4C`, Doppler couleur
- **Légende proposée :** « Coupe apicale 4 cavités (A4C), Doppler couleur. Insuffisance mitrale : légère. »
- **Réponses :**

  - Ce clip est-il ininterprétable ? **Non, le clip est interprétable**
  - La coupe indiquée est-elle correcte ? **Oui**
  - En dehors de la coupe, tout ce que la légende affirme est-il étayé par ce clip ? **Oui, chaque énoncé est étayé**
  - La légende omet-elle une information visible et cliniquement pertinente ? **Oui** (Insuffisance tricuspide)
  - En dehors de la coupe, la légende affirme-t-elle quelque chose que ce clip n'étaye pas (information non étayée ou hallucinée) ? **Non**
  - Énoncés impossibles à vérifier sur ce clip (laisser vide s'il n'y en a aucun) : **aucun**

- **Légende corrigée :** « Coupe apicale 4 cavités (A4C), Doppler couleur. Insuffisance mitrale : légère. Insuffisance tricuspide : modérée. »
- **Pourquoi :** Ce que la légende affirme est étayé ; la fuite tricuspide, visible et pertinente, manque. Les deux questions sont indépendantes.

### 5. Absence impossible à vérifier

- **Clip (synthétique) :** Coupe apicale 4 cavités en Doppler couleur ; la boîte couleur est placée sur la valve tricuspide et ne couvre pas la valve mitrale.
- **Coupe et acquisition du classifieur :** `A4C`, Doppler couleur
- **Légende proposée :** « Coupe apicale 4 cavités (A4C), Doppler couleur. Insuffisance mitrale : absente. »
- **Réponses :**

  - Ce clip est-il ininterprétable ? **Non, le clip est interprétable**
  - La coupe indiquée est-elle correcte ? **Oui**
  - En dehors de la coupe, tout ce que la légende affirme est-il étayé par ce clip ? **Incertain : rien n'est faux, mais au moins un énoncé est impossible à vérifier**
  - La légende omet-elle une information visible et cliniquement pertinente ? **Non**
  - En dehors de la coupe, la légende affirme-t-elle quelque chose que ce clip n'étaye pas (information non étayée ou hallucinée) ? **Non**
  - Énoncés impossibles à vérifier sur ce clip (laisser vide s'il n'y en a aucun) : **Insuffisance mitrale : Cadrage ou zoom**

- **Légende corrigée :** « Coupe apicale 4 cavités (A4C), Doppler couleur. »
- **Pourquoi :** Une absence s'évalue comme toute constatation : elle n'est étayée que si le clip pouvait montrer la fuite. Ici le Doppler ne couvre pas la valve mitrale, donc l'énoncé est impossible à vérifier (cadrage) et sort de la légende corrigée.

### 6. Structure attendue mais non visible

- **Clip (synthétique) :** Coupe apicale 4 cavités zoomée sur la valve mitrale en mode B ; la valve tricuspide est hors du secteur.
- **Coupe et acquisition du classifieur :** `A4C_ZOOM`, mode B
- **Légende proposée :** « Coupe apicale 4 cavités zoomée (A4C_ZOOM), mode B. Valve tricuspide visible. »
- **Réponses :**

  - Ce clip est-il ininterprétable ? **Non, le clip est interprétable**
  - La coupe indiquée est-elle correcte ? **Oui**
  - En dehors de la coupe, tout ce que la légende affirme est-il étayé par ce clip ? **Non, au moins un énoncé n'est pas étayé ou est faux**
  - La légende omet-elle une information visible et cliniquement pertinente ? **Oui** (Valve mitrale visible)
  - En dehors de la coupe, la légende affirme-t-elle quelque chose que ce clip n'étaye pas (information non étayée ou hallucinée) ? **Oui** (Valve tricuspide visible : Non montré ou contredit par le clip (halluciné))
  - Énoncés impossibles à vérifier sur ce clip (laisser vide s'il n'y en a aucun) : **aucun**

- **Légende corrigée :** « Coupe apicale 4 cavités zoomée (A4C_ZOOM), mode B. Valve mitrale visible. »
- **Pourquoi :** Les structures attendues pour une coupe ne sont qu'un a priori : une structure attendue mais absente n'est ni manquante ni fausse, sauf si la légende affirme qu'elle est visible. Ici elle l'affirme, donc l'énoncé n'est pas étayé.

### 7. Grade incertain

- **Clip (synthétique) :** Coupe apicale 4 cavités en Doppler couleur ; jet tricuspide excentré, en partie hors du plan de coupe.
- **Coupe et acquisition du classifieur :** `A4C`, Doppler couleur
- **Légende proposée :** « Coupe apicale 4 cavités (A4C), Doppler couleur. Insuffisance tricuspide : légère. »
- **Réponses :**

  - Ce clip est-il ininterprétable ? **Non, le clip est interprétable**
  - La coupe indiquée est-elle correcte ? **Oui**
  - En dehors de la coupe, tout ce que la légende affirme est-il étayé par ce clip ? **Incertain : rien n'est faux, mais au moins un énoncé est impossible à vérifier**
  - La légende omet-elle une information visible et cliniquement pertinente ? **Non**
  - En dehors de la coupe, la légende affirme-t-elle quelque chose que ce clip n'étaye pas (information non étayée ou hallucinée) ? **Non**
  - Énoncés impossibles à vérifier sur ce clip (laisser vide s'il n'y en a aucun) : **Insuffisance tricuspide : Visible, mais je ne suis pas certain**

- **Légende corrigée :** « Coupe apicale 4 cavités (A4C), Doppler couleur. »
- **Pourquoi :** La fuite est visible mais son grade ne peut pas être jugé sur ce clip : « incertain », jamais oui ou non au hasard. Un énoncé impossible à vérifier sort de la légende corrigée.

### 8. Mode d'acquisition faux

- **Clip (synthétique) :** Coupe apicale 4 cavités en mode B, sans Doppler couleur.
- **Coupe et acquisition du classifieur :** `A4C`, Doppler couleur
- **Légende proposée :** « Coupe apicale 4 cavités (A4C), Doppler couleur. Insuffisance mitrale : légère. »
- **Réponses :**

  - Ce clip est-il ininterprétable ? **Non, le clip est interprétable**
  - La coupe indiquée est-elle correcte ? **Oui**
  - En dehors de la coupe, tout ce que la légende affirme est-il étayé par ce clip ? **Non, au moins un énoncé n'est pas étayé ou est faux**
  - La légende omet-elle une information visible et cliniquement pertinente ? **Non**
  - En dehors de la coupe, la légende affirme-t-elle quelque chose que ce clip n'étaye pas (information non étayée ou hallucinée) ? **Oui** (Mode d'acquisition : Non montré ou contredit par le clip (halluciné))
  - Énoncés impossibles à vérifier sur ce clip (laisser vide s'il n'y en a aucun) : **Insuffisance mitrale : Modalité**

- **Légende corrigée :** « Coupe apicale 4 cavités (A4C), mode B. »
- **Pourquoi :** Le mode d'acquisition est lui aussi un énoncé : ici il est faux. La fuite mitrale ne peut pas être jugée sans Doppler couleur, elle est donc impossible à vérifier (modalité). Un clip peut avoir à la fois des énoncés non étayés et des énoncés impossibles à vérifier.

### 9. Mode d'acquisition impossible à vérifier

- **Clip (synthétique) :** Coupe apicale 4 cavités aux bords coupés : l'échelle couleur et le contour d'une éventuelle boîte couleur sont hors de l'image, et les rares pixels colorés dans le ventricule gauche peuvent être du Doppler couleur ou du bruit ; le ventricule gauche est bien visible.
- **Coupe et acquisition du classifieur :** `A4C`, Doppler couleur
- **Légende proposée :** « Coupe apicale 4 cavités (A4C), Doppler couleur. Ventricule gauche visible. »
- **Réponses :**

  - Ce clip est-il ininterprétable ? **Non, le clip est interprétable**
  - La coupe indiquée est-elle correcte ? **Oui**
  - En dehors de la coupe, tout ce que la légende affirme est-il étayé par ce clip ? **Incertain : rien n'est faux, mais au moins un énoncé est impossible à vérifier**
  - La légende omet-elle une information visible et cliniquement pertinente ? **Non**
  - En dehors de la coupe, la légende affirme-t-elle quelque chose que ce clip n'étaye pas (information non étayée ou hallucinée) ? **Non**
  - Énoncés impossibles à vérifier sur ce clip (laisser vide s'il n'y en a aucun) : **Mode d'acquisition : Cadrage ou zoom**

- **Légende corrigée :** « Coupe apicale 4 cavités (A4C), acquisition indéterminée. Ventricule gauche visible. »
- **Pourquoi :** Le mode d'acquisition est un énoncé : ici il est impossible à vérifier (cadrage), il est donc listé à la question 6 et la question 3 reçoit « incertain ». La légende corrigée garde sa forme : au lieu de supprimer le mode, elle écrit « acquisition indéterminée ». Le ventricule gauche est visible quel que soit le mode, son énoncé reste.

### 10. Coupe fausse

- **Clip (synthétique) :** Coupe apicale 5 cavités en Doppler couleur (chambre de chasse et valve aortique visibles) ; insuffisance mitrale légère.
- **Coupe et acquisition du classifieur :** `A4C`, Doppler couleur
- **Légende proposée :** « Coupe apicale 4 cavités (A4C), Doppler couleur. Insuffisance mitrale : légère. »
- **Réponses :**

  - Ce clip est-il ininterprétable ? **Non, le clip est interprétable**
  - La coupe indiquée est-elle correcte ? **Non** (A5C: Coupe apicale 5 cavités)
  - En dehors de la coupe, tout ce que la légende affirme est-il étayé par ce clip ? **Oui, chaque énoncé est étayé**
  - La légende omet-elle une information visible et cliniquement pertinente ? **Non**
  - En dehors de la coupe, la légende affirme-t-elle quelque chose que ce clip n'étaye pas (information non étayée ou hallucinée) ? **Non**
  - Énoncés impossibles à vérifier sur ce clip (laisser vide s'il n'y en a aucun) : **aucun**

- **Légende corrigée :** « Coupe apicale 5 cavités (A5C), Doppler couleur. Insuffisance mitrale : légère. »
- **Pourquoi :** La coupe se juge à part : « non » et la bonne classe. Les énoncés se jugent tels qu'ils sont écrits ; la légende corrigée donne la bonne coupe.

### 11. Coupe incertaine

- **Clip (synthétique) :** Coupe apicale en Doppler couleur, entre 4 et 5 cavités : la chambre de chasse du ventricule gauche n'apparaît que sur certains battements ; insuffisance mitrale légère.
- **Coupe et acquisition du classifieur :** `A4C`, Doppler couleur
- **Légende proposée :** « Coupe apicale 4 cavités (A4C), Doppler couleur. Insuffisance mitrale : légère. »
- **Réponses :**

  - Ce clip est-il ininterprétable ? **Non, le clip est interprétable**
  - La coupe indiquée est-elle correcte ? **Incertain**
  - En dehors de la coupe, tout ce que la légende affirme est-il étayé par ce clip ? **Oui, chaque énoncé est étayé**
  - La légende omet-elle une information visible et cliniquement pertinente ? **Non**
  - En dehors de la coupe, la légende affirme-t-elle quelque chose que ce clip n'étaye pas (information non étayée ou hallucinée) ? **Non**
  - Énoncés impossibles à vérifier sur ce clip (laisser vide s'il n'y en a aucun) : **aucun**

- **Légende corrigée :** « Coupe indéterminée, Doppler couleur. Insuffisance mitrale : légère. »
- **Pourquoi :** Quand la classe de coupe ne peut pas être tranchée, répondre « incertain » : ne pas choisir une classe au hasard, ni OTHER, réservée aux coupes hors axe ou non standard. Les énoncés se jugent tels qu'ils sont écrits. La légende corrigée remplace la coupe par « Coupe indéterminée » ; la classe du classifieur reste enregistrée avec le clip.

### 12. Clip ininterprétable

- **Clip (synthétique) :** Coupe sous-costale en mode B ; artefacts majeurs, aucune structure identifiable.
- **Coupe et acquisition du classifieur :** `SUBCOSTAL`, mode B
- **Légende proposée :** « Coupe sous-costale (SUBCOSTAL), mode B. Veine cave inférieure visible. »
- **Réponses :**

  - Ce clip est-il ininterprétable ? **Oui, il est ininterprétable** (Qualité d'image)
  - La coupe indiquée est-elle correcte ? **Sans objet : clip ininterprétable**
  - En dehors de la coupe, tout ce que la légende affirme est-il étayé par ce clip ? **Sans objet : clip ininterprétable**
  - La légende omet-elle une information visible et cliniquement pertinente ? **Sans objet : clip ininterprétable**
  - En dehors de la coupe, la légende affirme-t-elle quelque chose que ce clip n'étaye pas (information non étayée ou hallucinée) ? **Sans objet : clip ininterprétable**
  - Énoncés impossibles à vérifier sur ce clip (laisser vide s'il n'y en a aucun) : **aucun**

- **Légende corrigée :** inchangée
- **Pourquoi :** Rien ne peut être jugé : « oui », la raison, puis « sans objet » aux quatre autres questions. La légende corrigée reste inchangée ; elle est ignorée. Si seule une partie de la légende ne peut pas être jugée, le clip est interprétable : utiliser les énoncés impossibles à vérifier.
<!-- END GENERATED: examples -->
