# data/radar_normality.py

"""Caption normality for RADAR structure alignment: attribute facts first, then one flag per caption.

The flag feeds the A2 and A3 arms. It answers "does this caption, as written, describe the structure as normal?", and
it comes from facts about separate attributes (`lv_size`, `lv_function`, `pericardial_effusion`...), never from the
caption as a whole. Each `Fact` keeps its source (`text`, `appended` or `structured`), normalized value, numbers,
units, the rule or report column behind it and the rule version. The clinical choices (cutoffs, scope, what a normal
caption must state) are in `configs/radar/radar_normality_policy_v0.yaml`; the wording rules are here.

`assess` derives the flag:

- `abnormal`: a confirmed abnormal finding. Other attributes may be unknown.
- `normal`: every part of the caption was understood, every claim is normal, and the policy's required attributes are
  stated.
- `unknown`: anything else, including missing or unrecognized evidence and conflicts.

Conflicts about one attribute are recorded, never resolved in favour of a side: normal against abnormal wording in the
caption (`text_text`, as `VG non dilaté mais dilatation apicale` or `FEVG diminuée à 55 %`) or a claim against a
measurement (`text_structured`). A conflicted attribute is neither confirmed abnormal nor normal.

The caption is what gets aligned, so the facts that decide the flag must agree with it. A structured value is compared
with what the caption says about the same attribute. It never sets the flag alone: when it shows an abnormality the
caption does not mention, the caption cannot be normal, and stays unknown. `retain_caption` removes what V1's B-mode
clips cannot ground before anything is judged, and a caption that still mentions it is not understood.

A measure can have a `mild` band: a value over the limit but within the band is no contradiction once the report shows
the guideline's normal geometry, meaning the `settled_by` attribute normal, in the caption or by measurement, and a
relative wall thickness in range. Wall thickness works this way, because LV mass index and relative wall thickness, not
one wall, classify the LV. A caption calling the walls or the whole LV normal does not settle it: only mass does. Above
the band a value counts in full, whatever the caption says.

Reading is conservative, by construction rather than by a list of bad phrases. A clause is cut into segments at
commas, parentheses and connectives (`mais`, `sans`, `avec`, `et`, `ni`, `ou`...), and every word of a segment must be
accounted for: a structure label, an attribute (`fonction systolique`), a finding (`dilatation`), a state (`normale`),
a negation, a hedge, a modifier (`modérée`, `apicale`) or filler. One unexplained word makes the whole segment
unrecognized, its claims are dropped, and the caption cannot be normal. So new wording falls to unknown rather than to
a wrong flag. The coverage report lists the most frequent unrecognized wording, which is where the rules grow.

Scope of a negation: it denies the first claim after it in its segment, or for `absent` the last one before it. `sans
tamponnade` negates only the tamponade, and `non normale` turns a normal state abnormal. Any other claim in its reach
is uncertain: nothing says whether `VG non dilaté à fonction altérée` denies the function too. `ni` and `ou` carry a
negation on; `et`, a partitive or a bare list item after a negated segment is ambiguous (`pas de dilatation et
d'hypertrophie`), so its claims are uncertain, and a hedge reaches the next member the same way. A plural `absentes`
reaches back over `et`. A device is only denied by a negation aimed at it: `pas de végétation sur la prothèse` still
says there is a prosthesis.

A hedged claim (`probable`, `limite`, `?`) is uncertain; a claim that could not be made (`non évaluée`, `mal vue`) is
not assessed; a negated claim with a modifier (`pas d'épanchement important`) is uncertain, because it does not exclude
a smaller or more local finding; a normal claim with a modifier is uncertain for the same reason.

EF values are read by `radar_captions`, whose regular expression spans the words between the label and the number.
Those words are not taken as read: only the bound wording is removed, and the rest is judged like any other text. So
`FEVG avec hypokinésie globale 55 %` gives both a normal EF and an abnormal wall motion finding, and an EF with
unexplained wording is not understood. A comma or `et` can cut a value from its label (`FEVG 55 % au Simpson, 50 % par
WMS`), so a percentage also counts as an EF when it is tied to an EF method, or follows `évaluée à` in a clause about
LV function. No other percentage does.

Synthetic tests establish how the parser behaves, not how accurate it is on real reports: that takes the manually
checked sample of `radar_normality_audit.py`.
"""

import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache

import radar_captions as rc
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_POLICY = os.path.join(REPO, "configs", "radar", "radar_normality_policy_v0.yaml")
POLICY_SCHEMA = "radar-normality-policy"
SCOPES = ("bmode",)
COMBINE = ("every", "first_available")
DEVICE_VALUES = ("abnormal", "unknown")
RULE_VERSION = 15  # the number readers see; the policy digest changes with any edit to the rules or the policy
FLAGS = ("normal", "abnormal", "unknown")
DECIDED = ("normal", "abnormal")
OVERALL = "overall"
EF_EVIDENCE = "ef_value"

_FLIP = {"normal": "abnormal", "abnormal": "normal"}
_SIGN = re.compile(r"(?:^|[\s:(])[-+]+[\s).]*$")
_JOINING_WORD = re.compile(r"\b(?:mais|et|ou|est|sont|cependant|toutefois|donc)\b")


# ------------------------------------------------------------------------------------------------------ the policy

@dataclass(frozen=True, eq=False)
class Policy:
    version: int
    scope: str
    ef: dict
    measures: dict
    required: dict
    device_present: str
    digest: str
    status: str               # provisional, or approved
    approval: dict | None

    @property
    def rule_version(self):
        return f"{RULE_VERSION}+{self.digest[:12]}"


def policy_digest(raw):
    """sha256 of the policy without its approval block, and of the rules that read it."""
    body = {k: v for k, v in raw.items() if k != "approval"}
    h = hashlib.sha256(json.dumps(body, sort_keys=True).encode() + b"\0")
    for path in (__file__, rc.__file__):
        with open(path, "rb") as f:
            h.update(f.read() + b"\0")
    return h.hexdigest()


def _source_problems(attribute, source):
    if not isinstance(source, dict) or not {"column", "unit", "valid", "normal_max", "abnormal_above"} <= set(source):
        return [f"{attribute}: a source needs column, unit, valid, normal_max and abnormal_above"]
    if source["normal_max"] > source["abnormal_above"]:
        return [f"{attribute}: normal_max is above abnormal_above"]
    low, high = source["valid"]
    if not low <= source["normal_max"] <= high:
        return [f"{attribute}: normal_max is outside the valid range"]
    return []


def _attributes(structure):
    """What the wording rules of a structure can say something about."""
    grammar = GRAMMARS[structure]
    return ({a.split(":")[0] for a, _ in grammar.subjects} | {r.attribute for r in grammar.findings}) - {OVERALL}


def _mild_problems(attribute, measure, measures):
    mild = measure.get("mild")
    if mild is None:
        return []
    if not isinstance(mild, dict) or not {"up_to", "settled_by", "geometry"} <= set(mild):
        return [f"{attribute}: mild needs up_to, settled_by and geometry"]
    other = measures.get(mild["settled_by"])
    if other is None or other.get("structure") != measure.get("structure"):
        return [f"{attribute}: mild is settled by {mild['settled_by']!r}, not a measure of the same structure"]
    if any(isinstance(s, dict) and mild["up_to"] < s.get("abnormal_above", 0) for s in measure.get("sources") or ()):
        return [f"{attribute}: mild.up_to is below abnormal_above"]
    geometry = mild["geometry"]
    if not isinstance(geometry, dict) or not {"wall", "cavity", "max"} <= set(geometry):
        return [f"{attribute}: mild.geometry needs wall, cavity and max"]
    columns = {s.get("column") for m in measures.values() if m.get("structure") == measure.get("structure")
               for s in m.get("sources") or () if isinstance(s, dict)}
    if not {geometry["wall"], geometry["cavity"]} <= columns:
        return [f"{attribute}: mild.geometry names a column no measure of this structure reads"]
    return []


def _problems(raw):
    if not isinstance(raw, dict):
        return ["the policy is not a mapping"]
    problems = []
    if raw.get("schema") != POLICY_SCHEMA:
        problems.append(f"schema must be {POLICY_SCHEMA!r}")
    if raw.get("scope") not in SCOPES:
        problems.append(f"scope must be one of {SCOPES}")
    ef = raw.get("ef")
    if not isinstance(ef, dict) or not {"column", "unit", "valid", "normal_min", "tolerance",
                                        "boundary_margin"} <= set(ef):
        problems.append("ef needs column, unit, valid, normal_min, tolerance and boundary_margin")
    for attribute, measure in (raw.get("measures") or {}).items():
        if measure.get("structure") not in GRAMMARS:
            problems.append(f"{attribute}: unknown structure {measure.get('structure')!r}")
        elif attribute not in _attributes(measure["structure"]):
            problems.append(f"{attribute}: no wording rule states this attribute, so no caption could agree with it")
        if measure.get("combine") not in COMBINE:
            problems.append(f"{attribute}: combine must be one of {COMBINE}")
        if not measure.get("sources"):
            problems.append(f"{attribute}: no sources")
        for source in measure.get("sources") or ():
            problems.extend(_source_problems(attribute, source))
        problems.extend(_mild_problems(attribute, measure, raw.get("measures")))
    required = raw.get("required_for_normal")
    if not isinstance(required, dict) or set(required) != set(GRAMMARS):
        problems.append(f"required_for_normal must name exactly {sorted(GRAMMARS)}")
    else:
        for structure, attributes in required.items():
            unknown = sorted(set(attributes) - _attributes(structure))
            if unknown:
                problems.append(f"required_for_normal: no wording rule states {unknown} for {structure}")
    if raw.get("device_present") not in DEVICE_VALUES:
        problems.append(f"device_present must be one of {DEVICE_VALUES}")
    approval = raw.get("approval")
    if approval is not None and (not isinstance(approval, dict)
                                 or not all(approval.get(k) for k in ("by", "on", "audit_report", "digest"))):
        problems.append("approval needs by, on, audit_report and digest")
    return problems


def load_policy(path=DEFAULT_POLICY):
    """The validated policy. An approval whose digest no longer matches is refused: it approved something else."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    problems = _problems(raw)
    if problems:
        raise ValueError(f"Invalid normality policy {path}: " + "; ".join(problems))
    digest = policy_digest(raw)
    approval = raw.get("approval")
    if approval is not None and approval["digest"] != digest:
        raise ValueError(f"The approval in {path} is for digest {approval['digest']}, but the rules and policy are "
                         f"now {digest}: audit and approve again.")
    return Policy(version=raw["version"], scope=raw["scope"], ef=raw["ef"], measures=raw.get("measures") or {},
                  required=raw["required_for_normal"], device_present=raw["device_present"], digest=digest,
                  status="approved" if approval else "provisional", approval=approval)


# -------------------------------------------------------------------------------------------- the wording rules

@dataclass(frozen=True)
class Rule:
    name: str
    attribute: str
    value: str       # normal, abnormal, uncertain, device (the policy's device_present), or note (read, and neutral)
    pattern: re.Pattern


@dataclass(frozen=True)
class Grammar:
    labels: re.Pattern
    subjects: tuple    # (attribute, pattern): what a state word such as `normale` is about
    findings: tuple    # Rules carrying their own attribute and meaning


def _rule(name, attribute, pattern, value="abnormal"):
    return Rule(name, attribute, value, re.compile(pattern))


def _subject(attribute, pattern):
    return attribute, re.compile(pattern)


_NORMAL_STATE = re.compile(r"\bnormale?s?\b|\bnormaux\b|\bconservee?s?\b|\bpreservee?s?\b|\bbonnes?\b"
                           r"|\bsatisfaisante?s?\b")
_ABNORMAL_STATE = re.compile(r"\banormale?s?\b|\banormaux\b|\bdiminuee?s?\b|\babaissee?s?\b|\balteree?s?\b"
                             r"|\breduite?s?\b|\beffondree?s?\b|\baugmentee?s?\b|\baugmentations?\b"
                             r"|\bdiminutions?\b|\breductions?\b|\bpathologiques?\b")
_DIASTOLIC = r"(?: (?:tele[- ]*)?diastoliques?)?"
_SIZE = (rf"\btailles?\b|\bdimensions?{_DIASTOLIC}\b|\bdiametres?{_DIASTOLIC}\b"
         rf"|\bvolumes?(?: indexee?s?|{_DIASTOLIC})\b|\bcalibres?\b|\bcavites?\b")
_FUNCTION = r"\bfonctions?(?: systoliques?)?(?: globales?)?\b|\bperformances? systoliques?\b"
_WALL_MOTION = r"\bcontractilite(?: segmentaire| regionale)?\b|\bcinetique(?: segmentaire| regionale)?\b"
_MOTION_DISORDER = (r"\b(?:troubles?|anomalies?) de(?:s| la)? (?:cinetique|contractilite)"
                    r"(?: segmentaire| regionale)?\b|\banomalies? (?:contractiles? )?(?:segmentaires?|regionales?)\b"
                    r"|\banomalies? contractiles?\b|\barcs?\b|\bdiffuses? de la (?:cinetique|contractilite)\b"
                    r"|\bhypokine\w*|\bakine\w*|\bdyskine\w*")
_DEVICE = (r"\bbioprothe\w*|\bprothe\w*|\bmitraclip\b|\btriclip\b|\bclips?\b|\bannuloplast\w*|\btavi\b|\btavr\b"
           r"|\bhomogreffes?\b|\bvalvuloplast\w*|\bstat(?:ut|us) post remplacements? valvulaires?\b")

_COMMON = (
    _rule("nothing_to_report", OVERALL, r"\brien a signaler\b", "normal"),
    _rule("anomaly", OVERALL, r"\banomalies?\b|\bparticularites?\b|\batteintes?\b|\blesions?\b"),
)


def _valve(prefix, label, extra=()):
    appearance, motion = f"{prefix}_appearance", f"{prefix}_motion"
    return Grammar(
        labels=re.compile(label + r"|\b(?:valve|valvule)s?\b"),
        subjects=(_subject(appearance, r"\bstructures?\b|\bmorphologies?\b|\baspects?\b|\bfeuillets?\b|\bcuspides?\b"
                                       r"|\bsigmoides?\b"),
                  _subject(motion, r"\bmobilites?\b|\bouvertures?\b|\bexcursions?\b|\bcinetiques?\b"),
                  _subject(f"{prefix}_function", r"\bfonctions?\b")),
        findings=(
            *extra,
            _rule("thin_leaflets", appearance, r"\b(?:feuillets?|sigmoides?|cuspides?) (?:fine?s?|souples?)\b",
                  "normal"),
            _rule("calcified", appearance, r"\bcalcifi\w*|\bcalcaires?\b"),
            _rule("sclerotic", appearance, r"\bscler\w*"),
            _rule("thickened", appearance, r"\bepaissi\w*"),
            _rule("degenerative", appearance, r"\bdegenerati\w*|\bmyxoid\w*|\brhumatism\w*|\bremani\w*|\bfibros\w*"),
            _rule("vegetation", appearance, r"\bvegetations?\b"),
            _rule("prolapse", motion, r"\bprolaps\w*"),
            _rule("flail", motion, r"\bflail\b|\brupture de cordages?\b"),
            _rule("restricted", motion, r"\btether\w*|\brestrictions?\b|\brestrictive?s?\b"),
            _rule("device", f"{prefix}_device", _DEVICE, "device"),
            *_COMMON,
        ))


GRAMMARS = {
    "lv": Grammar(
        labels=re.compile(r"\bventricule gauche\b|\bventriculaires? gauches?\b|\bvg\b"),
        subjects=(_subject("lv_size", _SIZE),
                  _subject("lv_function", _FUNCTION + r"|\bfevg\b|\bfraction d' ejection\b|\bcontractilite globale\b"),
                  _subject("lv_wall_motion", _WALL_MOTION),
                  _subject("lv_wall_thickness", r"\bepaisseurs?(?: parietales?)?\b|\bparois?\b"),
                  _subject("lv_mass", r"\bmasses?(?: ventriculaire gauche| du ventricule gauche| du vg| vg)?\b"),
                  # No cutoff yet: a value alone says nothing, so only worded claims count.
                  _subject("lv_strain", r"\bstrains?(?: longitudinale?s?(?: globale?s?)?)?\b|\bdeformations?\b"
                                        r"|\bmyocardiques? longitudinale?s? globale?s?\b|\bafi\b")),
        findings=(
            # The outflow tract is not the LV cavity, and a small LV is not a stated normal size: neither group.
            _rule("lvot", "lvot_size", r"\b(?:chambre|voie)s? de chasse(?: du ventricule gauche| ventriculaire gauche"
                                       r"| du vg| vg)?(?: de (?:grande|petite) dimension)?\b", "uncertain"),
            _rule("small_cavity", "lv_size", r"\bpetite?s? (?:dimensions?|tailles?|volumes?)\b", "uncertain"),
            _rule("vsd_repair", "lv_vsd", r"\bstatut post-fermeture de communications? inter-?ventriculaires?"
                                          r"(?: par voie chirurgicale)?", "uncertain"),
            _rule("vsd", "lv_vsd", r"\bcommunications? inter-?ventriculaires?(?: residuelles?)?\b|\bciv\b"
                                   r"|\bshunts?(?: residuels?| a l' etage ventriculaire)?\b"),
            _rule("sam", "lv_sam", r"\bmouvements? systoliques? anterieurs?(?: mitral)?\b|\bsam\b"),
            _rule("septal_motion", "lv_septal_motion",
                  r"\bmouvements? septale?s? anormale?s?(?: post-operatoires?| secondaires? a "
                  r"(?:l' electro-entrainement|un bloc de branche gauche|un trouble de conduction))?"),
            _rule("dyssynchrony", "lv_dyssynchrony",
                  r"\bcriteres? d' asynchronisme (?:inter|intra)-?ventriculaires?\b"),
            _rule("d_shaped", "lv_geometry", r"\b(?:ventricule gauche|vg) en d\b"),
            _rule("cardiomyopathy", "lv_cardiomyopathy", r"\bcardiomyopathies?(?: hypertrophiques?)?"
                                                         r"(?: obstructives?| apicales?)?\b"),
            # A diagnosis that may come from the history, not something these clips show: neither group.
            _rule("ischemic_disease", "lv_cardiomyopathy", r"\bcardiopathies? ischemiques?\b", "uncertain"),
            _rule("motion_disorder", "lv_wall_motion", _MOTION_DISORDER),
            _rule("dysfunction", "lv_function", r"\bdysfonctions?(?: systoliques?)?\b"),
            _rule("hyperdynamic", "lv_function", r"\bhyperdynami\w*", "uncertain"),
            _rule("dilated", "lv_size", r"\bdilat\w*"),
            # Concentric hypertrophy has thick walls; other hypertrophy (eccentric, unqualified, denied) is about the
            # mass, and the walls vary. Same attribute and value: only the finding tells them apart.
            _rule("concentric_hypertrophy", "lv_wall_thickness",
                  r"\b(?:hypertroph\w*|hvg)(?: ventriculaires? gauches?| du ventricule gauche| du vg)?"
                  r" concentriques?\b"),
            _rule("hypertrophy", "lv_wall_thickness", r"\bhypertroph\w*|\bhvg\b"),
            _rule("thickened", "lv_wall_thickness", r"\bepaissi\w*"),
            _rule("thrombus", "lv_thrombus", r"\bthrombus(?: intra-?cavitaires?| visualisee?s?)?\b"),
            _rule("aneurysm", "lv_morphology", r"\banevr[iy]sm\w*"),
            _rule("remodelling", "lv_geometry", r"\bremodelage\b"),
            _rule("septal_bulge", "lv_morphology",
                  r"\bbourrelets?(?: septale?s?)?(?: basale?s?)?(?: non[- ]obstructifs?| obstructifs?)?"
                  r"(?: (?:mesurant|a|de) \d+(?:[.,]\d+)? ?mm(?: d' epaisseur)?)?(?: non[- ]obstructifs?)?",
                  "uncertain"),
            *_COMMON,
        )),
    "rv": Grammar(
        labels=re.compile(r"\bventricule droit\b|\bventriculaire droite?\b|\bvd\b"),
        subjects=(_subject("rv_size", _SIZE), _subject("rv_function", _FUNCTION)),
        findings=(
            _rule("dysfunction", "rv_function", r"\bdysfonctions?(?: systoliques?)?\b"),
            _rule("hypokinesis", "rv_function", r"\bhypokine\w*|\bakine\w*"),
            # A lead is neither a normal nor an abnormal RV, so a caption with one stays out of both groups.
            _rule("lead", "rv_device", r"\b(?:electrodes?|sondes?)(?: de (?:cardiostimulateur|stimulateur"
                                       r"|defibrillateur|stimulation|pacemaker))?\b", "uncertain"),
            _rule("dilated", "rv_size", r"\bdilat\w*"),
            _rule("hypertrophy", "rv_wall_thickness", r"\bhypertroph\w*"),
            *_COMMON,
        )),
    "la": Grammar(
        labels=re.compile(r"\boreillette gauche\b|\bauriculaire gauche\b|\bog\b"),
        subjects=(_subject("la_size", _SIZE + r"|\bsurfaces?\b"),),
        findings=(_rule("dilated", "la_size", r"\bdilat\w*"), *_COMMON)),
    "mitral": _valve("mitral", r"\b(?:valve|valvule) mitrale\b|\bmitrale?\b|\bvm\b"),
    "aortic": _valve("aortic", r"\b(?:valve|valvule) aortique\b|\baortique\b|\bva\b", extra=(
        _rule("tricuspid_cusps", "aortic_cusps", r"\btricuspide\b", "normal"),
        _rule("bicuspid", "aortic_cusps", r"\bbicuspid\w*"),
    )),
    "tricuspid": _valve("tricuspid", r"\b(?:valve|valvule) tricuspide\b|\btricuspide\b|\bvt\b"),
    "pulmonic": _valve("pulmonic", r"\b(?:valve|valvule) pulmonaire\b|\bpulmonaire\b|\bvp\b"),
    "aorta": Grammar(
        labels=re.compile(r"\baortique\b|\baorte\b|\bao\b"),
        subjects=(_subject("aorta_size", _SIZE),
                  _subject("aorta_size:root", r"\b(?:au niveau (?:des|du) )?sinus de valsalva\b|\bracine\b|\bsinus\b"),
                  _subject("aorta_size:ascending", r"\bascendante(?: proximale)?\b"),
                  _subject("aorta_size:arch", r"\bcrosse\b"),
                  _subject("aorta_size:descending", r"\bdescendante\b")),
        findings=(_rule("dilated", "aorta_size", r"\bdilat\w*|\bectasi\w*|\banevr[iy]sm\w*"),
                  _rule("coarctation", "aorta_coarctation", r"\bcoarctation\b"), *_COMMON)),
    "ivc": Grammar(
        labels=re.compile(r"\bveine cave inferieure\b|\bveine cave\b|\bvci\b"),
        subjects=(_subject("ivc_size", _SIZE),
                  _subject("ivc_collapse", r"\bcollapsus\b|\bcompliance\b|\bvariations? respiratoires?\b"),
                  _subject("ivc_pressure", r"\btension veineuse centrale\b")),
        findings=(
            _rule("reduced_collapse", "ivc_collapse", r"\b(?:collapsus|compliance|variations? respiratoires?) "
                                                      r"(?:absente?s?|diminuee?s?|reduite?s?|abolie?s?)\b"),
            _rule("compliant", "ivc_collapse", r"\bcompliante\b", "normal"),
            _rule("stiff", "ivc_collapse", r"\bpeu souples?\b|\bfixes?\b"),
            _rule("supple", "ivc_collapse", r"\bsouples?\b", "normal"),
            _rule("dilated", "ivc_size", r"\bdilat\w*"),
            *_COMMON,
        )),
    "pericardium": Grammar(
        labels=re.compile(r"\bsac pericardique\b|\bpericardiques?\b|\bpericarde\b"),
        subjects=(),
        findings=(
            _rule("dry", "pericardial_effusion", r"\bsec\b", "normal"),
            # The pleura is not the pericardium: neutral here, like fat.
            _rule("pleural_effusion", "pleural_effusion",
                  r"\bepanchements? pleura(?:l|ux)(?: droits?| gauches?| bilateraux| bilateral)?\b", "note"),
            _rule("effusion", "pericardial_effusion", r"\bepanchements?\b"),
            _rule("tamponade", "pericardial_tamponade", r"\btamponnade\b"),
            _rule("thickened", "pericardial_appearance", r"\bepaissi\w*|\bcalcifi\w*"),
            # Fat is not pericardial disease: it neither confirms nor denies an effusion, which the caption must state.
            _rule("epicardial_fat", "epicardial_fat", r"\bgraisses? (?:epi|peri)cardiques?\b", "note"),
            *_COMMON,
        )),
}

# Only Doppler can ground these: not in a V1 caption. Spectral and tissue Doppler, colour jets, pressures, diastology.
_DOPPLER = re.compile(
    r"\binsuffisance\w*|\b(?:valves?|valvules?|aortiques?|mitrales?|tricuspides?|pulmonaires?) insuffisante?s?\b"
    r"|\bfuites?\b|\bfuyante?s?\b|\bregurgit\w*|\bjets?\b|\b(?:im|it|ia|iao|ip|rao|rm)\b|\bstenos\w*"
    r"|\bretrecissement\w*|\bgradients?\b|\bvitesses?\b|\bvelocites?\b|\baccelerations?\b|\bvmax\b|\bitv\b|\bvti\b"
    r"|\bpht\b|\b(?:surface|aire)s? (?:valvulaire|fonctionnelle|aortique|mitrale)s?\b|\bava\b|\bore\b|\bpisa\b"
    r"|\bvena contracta\b|\bdoppler\b|\bflux\b|(?<![a-z])mm ?hg\b|(?<![a-z])c?m/s(?:ec)?\b|\bpap[sm]?\b"
    r"|\bpressions?\b|\brvsp\b|\bhta?p\b|\bhypertension (?:arterielle )?pulmonaire\b|\bdiastoli\w*|\brelaxation\b"
    r"|\bremplissage\b|\be/a\b|\be/ea?\b|\btapse\b|\btriv\b|\bonde s\b|\b[se]' (?:[=:] ?)?\d"
    r"|\bprofils? (?:pseudo-?normal|restrictif)\b|\bfusion des ondes\b"
    r"|\bdebits? cardiaques?\b|\bindex cardiaques?\b|\bobstructions?\b|\bmanoeuvres? de valsalva\b"
    r"|\b(?:avec(?: le)?|per) valsalva\b|\bero\b|\bflots?\b|\breflux\b|\brenversements?\b")
# A chamber's size at end-diastole is B-mode, whatever the word `diastolique` says elsewhere, and central venous
# pressure stays in the caption, as the policy says, whatever its unit: the IVC grounds it.
_NOT_DOPPLER = re.compile(r"\b(?:diametre|dimension|volume|aire|surface)s? (?:tele[- ]*)?diastoli\w*"
                          r"|\bd[- ]?shape [\w-]*diastolique\b|\bap+latissement (?:\w+ )?diastolique\b"
                          r"|\b(?:tension veineuse centrale|tvc)\b[^,;()]*")
# Doppler statements as a whole, for the words in them that say something else on their own: `normale`, `ventricule`,
# `dysfonction`, `anomalie` are accounted for here, and only here. `Augmentation du ventricule droit` is not.
_DOPPLER_STATEMENT = re.compile(
    r"\b(?:(?:dys)?fonctions? )?diastoliques?(?: normales?)?(?: de type (?:restrictif|pseudo-?normal))?"
    r"|\banomalies? d(?:e|u) (?:la )?(?:relaxation|remplissage)\b"
    r"|\b(?:centro|intra|para|peri)[- ]?prothetiques?\b"
    r"|\+ ?tvc\b|\b(?:(?:augmentation|elevation)(?: (?:legere|moderee|severe))? de (?:la )?)?pressions?"
    r"(?: de remplissage| systoliques?| pulmonaires?| arterielles?)*"
    r"(?: du ventricule (?:droit|gauche)| ventriculaires? (?:droite|gauche)| du v[dg])?"
    r"(?: (?:legerement |moderement |severement )?(?:normale?s?|elevee?s?|augmentee?s?|limite))?"
    r"(?: du ventricule (?:droit|gauche))?")
# Where a statement ends and what it leads to begins: `Hypertrophie` | `secondaire à une sténose`.
_LEADS_TO = re.compile(r":|\s/\s|\b(?:qui )?(?:ne |n['’])?(?:entra[iî]n|cr[ée]|caus|provoqu|sugg[èée]r)(?:ant|e)\b"
                       r"|\b(?:non |probablement |possiblement |en partie )?(?:secondaires?|dues?|li[ée]e?s?) [àa]\b"
                       r"|\b(?:non )?responsables? d|\bnon (?=st[ée]nos|obstructi)|\bpouvant\b"
                       r"|\bpour (?=st[ée]nos|insuffisance|fuite)"
                       r"|(?<!non )\b(?:qui appara[iî]t (?:visuellement )?)?"
                       r"(?:(?:l[ée]g[èe]re|mod[ée]r[ée]|s[ée]v[èe]re)ment )?st[ée]nosante\b", re.IGNORECASE)
# What a Doppler statement can rest on, seen in 2D, and the words that lead to it: `Insuffisance mitrale` | `sur
# dilatation de l'anneau`. Chambers are not here: the mitral cell's `sur dilatation de l'oreillette` is not the valve.
_STRUCTURAL = re.compile(r"\bcalcifi\w*|\bremani\w*|\bprolapsus\b|\brestrictions?\b|\btenting\b|\banneau\b"
                         r"|\bcoaptation\b|\bbicuspide\b|\brhumatism\w*|\bcordages?\b|\b(?:bio)?protheses?\b"
                         r"|\b(?:auto|homo)greffe\b"
                         r"|\bsondes?\b|\belectrodes?\b|\bclips?\b|\bmouvement systolique anterieur\b|\bsam\b")
_RESTS_ON = re.compile(r"\b(?:(?:probablement|possiblement|en partie) )?(?:sur|par|via|secondaires? [àa]"
                       r"|d[uû]e?s? [àa]|li[ée]e?s? [àa]|associ[ée]e?s? (?:[àa]|au)|au niveau d[eu])(?=\W)",
                       re.IGNORECASE)
# A finding that is an adjective of the Doppler noun, and nothing else in the segment: no negation, no doubt. Only here
# does a word change: the valve takes the noun's place, and the grade, which is the Doppler's, goes.
_SEVERITY = r"(?:l[ée]g[èe]re|mod[ée]r[ée]e|s[ée]v[èe]re)"
_GRADE = rf"(?:d['’]allure )?{_SEVERITY}(?: [àa] {_SEVERITY})?"
_GLUED = re.compile(rf"(?P<lead>\s*(?:avec )?)(?:st[ée]nose|r[ée]tr[ée]cissement)(?: valvulaire)? "
                    rf"(?P<valve>aortique|mitrale)(?: valvulaire)?(?: {_GRADE})? "
                    rf"(?P<finding>calcifi[ée]e|rhumatismale|d[ée]g[ée]n[ée]rative)"
                    rf"(?: {_GRADE})?(?: au doppler)?(?P<more> valve aortique (?:tri|bi)cuspide)?(?P<end>[\s,.;)]*)",
                    re.IGNORECASE)
# A past procedure is still in the clip; its date, and what follows it, a place or a name as often as not, are not. An
# EF or a value of that time is not in it either.
_PROCEDURE = re.compile(r"\b(?:valvulo)?plastie\b|\b(?:bio)?protheses?\b|\bremplacement\b|\brv[am]\b|\btavi\b"
                        r"|\bmyomectomie\b|\b(?:auto|homo)?greffe\b")
# A procedure or a device: a claim that it happened, which what was lost beside it may deny.
_DONE_TO_THE_HEART = re.compile(_PROCEDURE.pattern + r"|\bsondes?\b|\belectrodes?\b|\b(?:mitra)?clips?\b"
                                r"|\bpacemaker\b|\b(?:cardio)?stimulateur\b|\bdefibrillateur\b")
_THEN = re.compile(r"%|\bfe(?:vg)?\b")
# What may follow a date for the date alone to go: no word at all. A word may say the procedure never happened in any
# spelling (`absente`, `Absente`, `ABSENTE`), and nothing in a word shows it is a place or a name.
_SAYS_MORE = re.compile(r"[^\W\d_]|\?")
_DATE = re.compile(r"\s*\b(?:(?:du|de|en|le|depuis)\s+)?(?:\d{1,2}\s*(?P<s>[/.-])\s*\d{1,2}\s*(?P=s)\s*\d{2,4}"
                   r"|(?:\d{1,2}\s+)?(?:janvier|f[ée]vrier|mars|avril|mai|juin|juillet|ao[uû]t|septembre|octobre"
                   r"|novembre|d[ée]cembre)\s+(?:19|20)\d{2}|(?:19|20)\d{2})\b", re.IGNORECASE)

_NOT_ASSESSED = re.compile(
    r"\b(?:non|pas|mal) (?:evalu\w*|visualis\w*|vue?s?\b|mesur\w*|calcul\w*|interpret\w*|analys\w*|explor\w*"
    r"|quantifi\w*|appreci\w*|etudi\w*|definie?s?\b)|\bdifficile\w*|\bimpossible\w*|\bechogenicite\b|\bfenetres?\b")
_HEDGE = re.compile(
    r"\bne peut etre (?:exclu|ecarte|elimine)\w*|\b(?:non|pas) (?:exclu|ecarte|elimine)\w*|\bnormale? basse?\b"
    r"|\bprobabl\w*|\bpossibl\w*|\bsuspicion\b|\bsuspect\w*|\bpeut[- ]etre\b|\bsembl\w*|\bparai\w*|\bdoute\w*"
    r"|\bevoqu\w*|\bevocat\w*|\bcompatibles?\b|\?"
    r"|\ba (?:verifier|controler|recontroler|confirmer|surveiller|eliminer|exclure|ecarter|preciser|discuter)\b"
    r"|\blimites?\b|\bborderline\b|\bsub[- ]?normal\w*|\bpresque\b|\bquasi\w*|\benviron\b|\bapprox\w*|\btraces?\b"
    r"|\bdiscutable\w*|\bincertain\w*|\bindetermine\w*")
_NEGATION_AFTER = re.compile(r"\babsente?s?\b|\bexclue?s?\b|\becartee?s?\b|\bnon retrouvee?s?\b")
_NEGATION = re.compile(
    r"\bn' (?:est|sont|a|ont|y a) pas(?: d[e'] ?)?|\bne (?:presente|montre|retrouve|note|voit)\w* pas(?: d[e'] ?)?"
    r"|\bpas d[e'] ?|\bpas\b|\bnon\b(?![- ]obstructi)|\baucune?s?\b|\babsence d[e'] ?|\bsans\b"
    r"|\bni(?: d[e'] ?)?\b")
_MEASURE = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)?\s*(?:mm|cm2|cm|ml/m2|ml|%)?(?![\w])")
_EF_TOKEN = re.compile(r"\befvalue(\d+)\b")
_EF_PERCENT = r"(?<![\w.,-])(?P<first>\d{1,3}(?:[.,]\d+)?)(?:\s*-\s*(?P<second>\d{1,3}(?:[.,]\d+)?))?\s*%"
_EF_METHOD = (r"(?:visuellement|wmsi?|(?:par|au|en)(?: la| le)?(?: methodes?)?(?: de)? "
              r"(?:simpsons?(?: bi-?plan)?|wmsi?|wall motion score|auto-?ef|[23] ?d))\b")
_EF_WITH_METHOD = (re.compile(rf"{_EF_PERCENT}\s*{_EF_METHOD}"),
                   re.compile(rf"\b{_EF_METHOD}\s*(?:a\s+|:\s*)?{_EF_PERCENT}"))
_EF_VERB = re.compile(rf"\b(?:estimee?|evaluee?|calculee?)\s+a\s+{_EF_PERCENT}")
_EF_SUBJECT = re.compile(r"\bfevg\b|\bfraction d' ejection\b|\bfonctions?\b|\bdysfonctions?\b")
# A comparison with an earlier study: what the clause mentions may no longer hold, so it is not read.
_PRIOR = re.compile(r"\bamelioration\b|\bcompare?e?s? (?:a|au|avec)\b|\bvs\b|\bpar rapport (?:a|au)\b|\balors\b"
                    r"|\bprecedente?s?\b|\banterieurement\b")
_STRAIN_VALUE = re.compile(r"(?<![\w.,])-\s*\d+(?:[.,]\d+)?\s*%")
_ACQUISITION = re.compile(r"\b(?:sans )?contraste\b|\bdefinity\b")     # how the images were taken, not a finding
_TERRITORY = re.compile(r"\b(?:dans le )?territoires? de l' (?:artere )?(?:coronaire droite"
                        r"|interventriculaire anterieure|circonflexe(?: gauche)?)\b")
_EF_BOUND = re.compile(r"<=|>=|<|>|≤|≥|\binf[ée]rieure?s?\b|\bsup[ée]rieure?s?\b|\binf\b|\bsup\b|\bplus de\b"
                       r"|\bmoins de\b|\bau moins\b|\bau plus\b|\bau[- ]del[àa] de\b|\bentre\b")
_MODIFIER = re.compile(
    r"legere?s?|legerement|minimes?|discrete?s?|discretement|moderee?s?|moderement|severes?|severement|importante?s?"
    r"|marquee?s?|franche?s?|franchement|massive?s?|volumineu(?:x|se|ses)|abondante?s?|abondance|faible|moyenne|grande"
    r"|petite?s?|globale?s?|diffuse?s?|segmentaires?|regionale?s?|localisee?s?|circonferentielle?s?|concentriques?"
    r"|excentriques?|asymetriques?|apicale?s?|apex|basale?s?|septale?s?|laterale?s?|inferieure?s?|anterieure?s?"
    r"|posterieure?s?|anneau|annulaires?|lame|focale?s?"
    r"|inspiris|evolut|corevalve|magna|sapien|perimount|trifecta|position|bien|fonctionnelle?s?"
    r"|moyens?|moyennes|infero-?laterale?s?|antero-?laterale?s?|infero-?septale?s?|antero-?septale?s?|septum"
    r"|territoire|valvulaires?|mecaniques?|biologiques?|tricuspidien(?:ne)?s?")
_FILLER = frozenset(
    "le la les l' un une des du de d' a au aux en est sont il elle on ce cette cet ces son sa ses leur leurs y par "
    "pour sur dans qui que qu' note notee notees evidence presence present presente presents presentes presentent "
    "estime estimee evalue "
    "evaluee mesure mesuree calcule calculee visuellement visuelle biplan bi-plan simpson simpsons methode methodes "
    "plusieurs selon wms wmsi wall motion score auto-ef autoef diastole voir cf schema diagramme statut status post "
    "2d 3d"
    .split())
_METHOD_WORDS = frozenset("wms wmsi wall motion score simpson simpsons biplan bi-plan methode methodes auto-ef autoef "
                          "diastole 2d 3d".split())      # a state on one of these says nothing of the whole structure
_REFERENCE_WORDS = frozenset("voir cf schema diagramme".split())    # the finding is elsewhere, not a state carried on
# All a Doppler segment may hold besides the above. A word earns its place by saying nothing of what a clip shows: how
# much, where the jet goes, how it was measured, why it could not be. No state word and no chamber: anything not listed
# is unexplained, whatever it means, and that is what keeps an unknown from becoming a normal.
_NEUTRAL = _FILLER | frozenset(
    "et ou ni ne n' ' pas non sans avec mais absence aucun aucune sous s' se ceci cela "
    "valve valves valvule valvules aortique mitrale mitral tricuspidienne tricuspidien tricuspide pulmonaire "
    "pulmonaires valvulaire "
    "veine veines veineux sus-hepatique sus-hepatiques niveau sous-aortique "
    "intraventriculaire intra-ventriculaire mid-ventriculaire midventriculaire intravg "
    "trace traces legere leger legeres legerement minime moderee modere moderement severe severes severement massive "
    "significative significatives significatif significatifs grade i ii iii iv tout plus moins tres "
    "moyen moyenne maximal maximale max mesurant systolique systoliques repos valsalva vasalva calculee "
    "centrale central excentrique dirige dirigee posterieurement anterieurement posterieur posterieure anterieur "
    "anterieure couleur fonctionnelle fonctionelle dynamique critere criteres "
    "inversion rapport e partie responsable responsables secondaire secondaires due dues lie liee lies liees "
    "entrainant entraine creant cree causant provoquant provoque suggerant suggere "
    "examen permettant evaluer adequatement impossibilite impossible evaluable reserve profil imparfait incomplet "
    "raison cause rapportee possiblement probablement sous-estimee pouvant etre etant donne "
    "fibrillation auriculaire fa bigeminisme fc age mm cm cm2 ms msec ml cc bpm hg"
    .split())
_VALUE = re.compile(r"\d[\d.,%+-]*(?:mmhg|mm|cm2|cm|ms|msec|ml|cc|bpm|hg)?")
_SEGMENT = re.compile(r"[()]|,(?!\d)|\s-\s|(?=\b(?:mais|cependant|toutefois|n[ée]anmoins|par contre|en revanche"
                      r"|ainsi que|sans|avec|et|ni|ou)\b)", re.IGNORECASE)
_LEAD = re.compile(r"\s*(mais|cependant|toutefois|neanmoins|par contre|en revanche|ainsi que|avec|et|ou)\b")
_PARTITIVE = re.compile(r"\s*(?:d' |(?:de|du|des)\b)")
_CONTINUES = re.compile(r"\s*(?:d' |(?:et|ni|ou|de|du|des)\b)")
_PLURAL_MARK = re.compile(r"(?<=\w)\((e?s|e|x)\)")
_ESCAPED_CHARACTER = re.compile(r"_x[0-9a-f]{4}_", re.IGNORECASE)
_POSSESSIVE = re.compile(r"\bsimpson'\s*s\b")
# A removed last subject after `et`, one state word and the chamber at most: the one form whose state surely covers the
# list. A comma, a colon or a parenthesis opens a statement of its own: `Dimension et fonction: pression normale`.
_SHARED_TAIL = re.compile(r"et (?:(?:de|la|sa) )*(?:fonction )?"
                          r"(?:pressions?(?: systoliques?| pulmonaires?| arterielles?)*|diastoliques?)"
                          r" (?P<state>\S+)(?P<chamber> du ventricule (?:droit|gauche))?")
_PARENTHESES = re.compile(r"\([^()]*\)")
_LIST_WORDS = frozenset("ventricule oreillette droit droite gauche vd vg de d' du des la le sa et systolique "
                        "systoliques globale globales".split())
_BEFORE_CLOSING = re.compile(r"[\s,;]+\)")
_EMPTY_PARENTHESES = re.compile(r"\s*\(\W*\)")
_BEFORE_COMMA = re.compile(r"\s+(?=,)")
_REPEATED_COMMA = re.compile(r",(?:\s*,)+")
_DANGLING_END = re.compile(r"[\s,;]*\b(?:mais|et|ou|avec|ainsi que)\W*$", re.IGNORECASE)
_COORDINATED = re.compile(r"\s*(?:et|ou|ni)\b")
_TRAILING_HEDGE = re.compile(
    r"\ba (?:confirmer|verifier|controler|recontroler|surveiller|eliminer|exclure|ecarter|preciser|discuter)\b|\?"
    r"|\b(?:probables?|possibles?|suspecte?s?|douteu(?:x|ses?)|discutables?|incertaine?s?)\W*$")
_BUT = re.compile(r"\s*(?:mais|cependant|toutefois|neanmoins|par contre|en revanche)\b")
# A denial that opens a list: `Pas de X, Y` denies Y too. `VG non dilaté, HVG légère` opens none.
_LIST_DENIAL = re.compile(r"\b(?:pas d[e'] ?|sans\b|absence d[e'] ?|aucune?s?\b)")
_LONE_VALUE = re.compile(r"\W*\d[\d.,+-]*\s*(?:mm ?hg|c?m/s(?:ec)?)\W*")
_CF = re.compile(r"\bc\.\s*f\b\.?", re.IGNORECASE)
_TITLE = re.compile(r"(?i:\b(?:dr|dre|docteure?|pr|prof|professeur|mme|mlle)\b)\.?\s*[^\W\d_][\w'-]*"
                    r"|\bM\.\s?[A-ZÀ-Ý][\w'-]*")
# What no clip of this study can show and what may identify someone: dates, years, a named colleague. The segment goes.
_NOT_IN_THE_CLIP = re.compile(
    r"\bdocteure?\b|\b(?:19|20)\d{2}\b|\b\d{1,2}\s*(?P<s>[/.-])\s*\d{1,2}\s*(?P=s)\s*\d{2,4}\b"
    r"|\b(?:du|le) \d{1,2}/\d{1,2}\b"
    r"|\b\d{1,2}(?:er)? (?:janvier|fevrier|mars|avril|mai|juin|juillet|aout|septembre|octobre|novembre|decembre)\b")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_STRAIN_CLEARLY_LOW = 16    # percent, in absolute value; see the policy's wording choices
_RATIO_TOLERANCE = 1e-9    # float noise only: 2 x 1.05 / 5.0 comes out at 0.42000000000000004
_SUBJECT_END = re.compile(r"\b(?:dimensions?|fonctions?(?: systoliques?)?|tailles?|volumes?|masses?|structures?"
                          r"|morphologies?|aspects?|epaisseurs?|calibres?|cinetiques?|contractilites?)\W*$")


def _plain(text):
    """Lower case, no accents, an apostrophe always ending its token. A plural mark (`anomalie(s)`) joins its word, or
    the parentheses would cut it into segments; a spreadsheet's escape for a control character (`_x000b_`) is a
    space."""
    text = _ESCAPED_CHARACTER.sub(" ", rc.canonical_caption(text).replace("œ", "oe"))
    text = _POSSESSIVE.sub("simpson", _PLURAL_MARK.sub(r"\1", text))
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).replace("'", "' ")
    return " ".join(text.split())


def _slices(clause):
    """The segments of a clause as consecutive slices of its text (separators stay at the end of a slice)."""
    cuts = sorted({m.end() for m in _SEGMENT.finditer(clause)} - {0, len(clause)})
    return [clause[a:b] for a, b in zip([0, *cuts], [*cuts, len(clause)])]


def _stands_alone(part):
    """Whether a segment says something without the text before it: it names a finding, or a subject with its state.
    What follows a removed segment and does not (`1-2/4`, `centrale`, `42 mmHg`) was only ever about that segment. A
    normal descriptor is no finding here: `60 % en tricuspide` is not a tricuspid aortic valve, and what must not be
    lost is a finding."""
    text = _plain(part)
    if any(rule.value != "normal" and rule.pattern.search(text)
           for grammar in GRAMMARS.values() for rule in grammar.findings):
        return True
    has_state = _NORMAL_STATE.search(text) or _ABNORMAL_STATE.search(text)
    return bool(has_state) and any(pattern.search(text) for grammar in GRAMMARS.values()
                                   for _, pattern in grammar.subjects)


def _names_doppler(part):
    return bool(_DOPPLER.search(_NOT_DOPPLER.sub(" ", _plain(part))))


def _leaves(part):
    """Whether a segment leaves the caption: only Doppler can ground it, or it holds a date or a name."""
    return _names_doppler(part) or bool(_NOT_IN_THE_CLIP.search(_plain(part)))


def _unexplained(part):
    """The words of a segment that are not established as saying nothing: all but its Doppler wording, its values and
    its `_NEUTRAL` words, and a date in any spelling. There is no rule to write for a word never seen: `calcifiée` and
    `masse` are both in here."""
    rest = _DOPPLER.sub(" ", _DOPPLER_STATEMENT.sub(" ", _plain(_DATE.sub(" ", part))))
    words = [word.strip(":=;-().,%<>~+\"«»") for word in re.split(r"[\s/]+", rest)]
    return [word for word in words if word and word not in _NEUTRAL and not _VALUE.fullmatch(word)]


def _separated(part):
    """`part`, in two where what comes before a connector says something of its own and only what follows is Doppler
    wording, so that only that wording goes: `Calcification responsable de sténose`, `Prothèse Inspiris 23: gradient
    moyen de 11 mmHg`. The rules need not read it: `Masse intracavitaire` stays, and the caption is read as unknown. A
    bare label before the connector goes with it."""
    if _names_doppler(part):
        cuts = [m.start() for m in _LEADS_TO.finditer(part) if not _names_doppler(part[:m.start()])]
        if cuts and _unexplained(part[:cuts[-1]]) and not _restricted_denial(part, cuts[-1]):
            return [part[:cuts[-1]].rstrip() + ", ", part[cuts[-1]:]]
        for match in _RESTS_ON.finditer(part):
            head, tail = part[:match.start()], part[match.start():]
            if _names_doppler(head) and _only_measures(head) and _is_structural(_separated(tail)[0]):
                return [head, *_separated(tail)]
    return [part]


def _only_measures(part):
    """Whether a Doppler statement is plain: what it rests on can stay as a finding only then. A denial, a doubt, a
    condition or a word the rules do not know may reach that finding too: `Probable insuffisance sur prolapsus`, `Si
    fuite sur prothèse`, `Pour éliminer une fuite sur prothèse`."""
    text = _plain(part)
    return not (_unexplained(part) or _NEGATION.search(text) or _HEDGE.search(text))


def _is_structural(part):
    """Whether what a Doppler statement rests on is a finding to keep, as written and with any doubt in it. Not a
    denial or a normal state: the statement that held it is gone, and a normal claim must not come out of it."""
    text = _plain(part)
    return (bool(_STRUCTURAL.search(text)) and not _leaves(part) and not _NEGATION.search(text)
            and not _NORMAL_STATE.search(text))


def _unglued(part):
    """`Sténose aortique calcifiée modérée` as `Valve aortique calcifiée`; see `_GLUED`."""
    match = _GLUED.fullmatch(part)
    if not match:
        return part
    valve = "Valve" if not match["lead"].strip() and part.lstrip()[:1].isupper() else "valve"
    more = f",{match['more']}" if match["more"] else ""
    return f"{match['lead']}{valve} {match['valve']} {match['finding']}{more}{match['end']}"


def _undated(part, following=""):
    """`part`, in two where it is a past procedure and a date and nothing else: `Plastie mitrale` stays and `du 12 mars
    2019` leaves like any segment with a date. A word after the date, or a next segment that cannot stand without it
    (`following`), may say the procedure never happened (`non implantée`, `, annulée`), so the procedure must not stay
    without it: such a segment leaves whole and counts as lost, a name or a place with it, and so does anything else
    with a date."""
    date = _DATE.search(part)
    if not date or _THEN.search(_plain(part)) or not _PROCEDURE.search(_plain(part[:date.start()])):
        return [part]
    if _SAYS_MORE.search(part[date.end():]) or not _says_its_own(following):
        return [part]
    comma = "," * part.rstrip().endswith(",")       # it still ends the procedure
    return [part[:date.start()].rstrip() + comma + " ", part[date.start():]]


def _says_its_own(part):
    """Whether a segment owes nothing to the one before: there is none, it stands alone, or it is Doppler wording."""
    return not _plain(part).strip(" .,;:()-") or _stands_alone(part) or (_leaves(part) and not _unexplained(part))


def _restricted_denial(part, cut):
    """Whether the Doppler wording limits a denial: `Pas de calcification responsable de sténose` does not deny all
    calcification. A colon or a slash only ends the statement."""
    return part[cut] not in ":/ " and bool(_NEGATION.search(_plain(part[:cut])))


def _has_state(text):
    return bool(_NORMAL_STATE.search(text) or _ABNORMAL_STATE.search(text))


def _reaches_back(part):
    """Whether a coordinated part also says something of the member before it: a plural state or denial, or a doubt at
    its end: `Calcification et sténose absentes`, `Végétation et fuite à confirmer`, `Thrombus et fuite à éliminer`."""
    text = _plain(part)
    plural = any(match.group().endswith(("s", "x")) for pattern in (_NORMAL_STATE, _ABNORMAL_STATE, _NEGATION_AFTER)
                 for match in pattern.finditer(text))
    return plural or bool(_TRAILING_HEDGE.search(text))


def _surviving(parts, doppler):
    """(the parts of a clause that stay, how many were stranded). A stranded part follows a removed one, and only
    continues it (`ni de calcification`), rests on it when it was not plain (`_only_measures`) or cannot stand alone:
    it leaves the text, and it stays lost, because nothing says what it meant. After a removed denial, a bare list item
    was denied with it: `Pas de sténose, calcification` does not say calcified. A removed part with anything
    `_unexplained` counts as stranded too, and so does a value with no subject: `15 mmHg` of what? What comes before a
    removed part stays, unless it is a coordinated member that needed the removed one: a subject waiting for a state,
    or a finding it qualified (`_reaches_back`). A closing parenthesis stays too, whatever it ended: it may close what
    a kept part opened."""
    kept, stranded, after_removed, denied, qualified = [], 0, False, False, False
    for index, (part, gone) in enumerate(zip(parts, doppler)):
        text = _plain(part)
        blank = not text.strip(" .,;:()-")
        closing = ")" * part.rstrip().endswith(")")
        leaning = bool(_RESTS_ON.match(part.lstrip()))
        rests_on = leaning and _is_structural(part)   # `du à un SAM` continues nothing
        if gone:
            stranded += bool(_unexplained(part)) or (not after_removed and bool(_LONE_VALUE.fullmatch(text)))
            reach = index          # back over the list, as far as its members needed this one
            while reach and kept[reach - 1] == parts[reach - 1] and _needed(parts[reach - 1], part,
                                                                            index == len(parts) - 1):
                kept[reach - 1] = ")" * parts[reach - 1].rstrip().endswith(")") + " "
                stranded += 1
                reach -= 1
            after_removed, denied = True, denied or bool(_LIST_DENIAL.search(text))
            qualified = not _only_measures(part)
            kept.append(closing + " ")
        elif after_removed and not blank and (
                (leaning and qualified)         # `Insuffisance possible, sur prolapsus`: the doubt reaches it
                or (not rests_on and (_CONTINUES.match(text) or not _stands_alone(part)))
                or (denied and not (_NEGATION.search(text) or _has_state(text) or _BUT.match(text)))):
            stranded += 1
            kept.append(closing + " ")
        else:
            kept.append(part)
            after_removed, denied = after_removed and blank, denied and blank
    return kept, stranded


def _plural_denial(text):
    """Whether a plain text denies in the plural: `sténose absentes` denies more than its own noun."""
    return any(match.group().endswith(("s", "x")) for match in _NEGATION_AFTER.finditer(text))


def _needed(previous, removed, last):
    """Whether the kept part `previous` meant nothing without the part `removed` after it: a coordinated part, or a
    plural denial after a comma."""
    before, text = _plain(previous), _plain(removed)
    if not before.strip(" .,;:()-") or (last and _shared_state(removed)):
        return False
    if _COORDINATED.match(text):
        return bool(_SUBJECT_END.search(before)) or (_reaches_back(removed) and not _has_state(before))
    return _plural_denial(text) and not _has_state(before)


def _shared_state(part):
    """The state word a removed last subject shares with the subjects before it, as written, with its chamber if it
    names one, or None. It takes the removed segment being exactly that subject and one state. A negation or a hedge
    (`non normales`, `probablement normales`) would be lost if the word moved alone. A singular agrees with the last
    subject only, except `normale` where it is the report's usual spelling: the subjects before it have no other state,
    and they are only read at all when they ask for one (`_SUBJECT_END`). An abnormal singular is not shared. What the
    subjects themselves must be is `_bare_subjects`."""
    match = _SHARED_TAIL.fullmatch(_plain(part).strip(" .,;"))     # with its parentheses: they do not match
    state = match and match["state"]
    if not match or not (state == "normale" or (state[-1] in "sx" and (
            _NORMAL_STATE.fullmatch(state) or _ABNORMAL_STATE.fullmatch(state)))):
        return None
    return " ".join(part.rstrip(" .,;)").split()[-(1 + len((match["chamber"] or "").split())):])


def _bare_subjects(text):
    """Whether `text`, from its last comma on and leaving out what is in parentheses, is a list of subjects and nothing
    else. A state moved onto anything more would say something new: `Mauvaise fonction` and `Pas d'évaluation de la
    fonction` do not become normal when `et pression systolique normale` goes, and nothing after a colon is a list."""
    last = _plain(re.split(r"[,;:]", _PARENTHESES.sub(" ", text))[-1]).split()
    return bool(last) and all(word in _LIST_WORDS or _SUBJECT_END.fullmatch(word) for word in last)


def retain_caption(text):
    """(the caption without what only Doppler can ground, how many parts were removed, how many were stranded). Removal
    is by segment, so `Valve mitrale calcifiée sans sténose` keeps its calcification, and a segment that goes from
    anything else to Doppler wording (`Hypertrophie secondaire à une sténose`) keeps what it said (`_separated`). What
    such a statement rests on stays too, as written (`Insuffisance mitrale` goes, `sur dilatation de l'anneau` stays)
    if the statement is plain (`_only_measures`), and a finding glued to its Doppler noun takes the valve for noun
    (`_unglued`). A segment on a plain past procedure stays without its date (`_undated`). A
    segment leaves quietly only if nothing in it is `_unexplained`. Anything else, a finding that cannot be told from
    its Doppler wording or a word the rules have never seen, leaves with it and counts as stranded: the caption would
    not say what the report said. What follows a removed segment goes too unless it stands alone (`_stands_alone`) and
    is not its continuation: in `Insuffisance légère (1-2/4), valve calcifiée` the grade goes and the calcification
    stays, and all of `Pas de sténose ni de calcification` goes. Such a stranded part may have been a finding the rules
    cannot read or that lost its meaning with its segment, so `assess` refuses to call a caption with a stranded part
    normal. Coordinated subjects share the state word at the end of the list: in `dimension, fonction et pression
    systolique normales` it stays when the pressure goes, or `dimension, fonction` would say nothing. When what remains
    ends on such a subject with no state that surely covers it (`_shared_state`), the clause goes and counts as
    stranded: changing what it says would be worse than losing it, but what it said is still unknown. A segment with a
    date or a name leaves like a Doppler one (`_NOT_IN_THE_CLIP`), and a clause with no word at all goes too: `1-2/4`
    says nothing of what it grades. Every caption comes back as clauses joined by full stops, whatever separated them
    in the report, so that the same statements are the same caption."""
    text = _TITLE.sub(" docteur ", _CF.sub("cf", rc.caption_text(text)))
    kept, dropped, stranded = [], 0, 0
    for clause in rc.text_clauses(text):
        clause = " ".join(_ESCAPED_CHARACTER.sub(" ", clause).split())
        if rc.caption_text(clause) == "" or not clause.strip(" .,;:()-"):
            continue
        if not re.search(r"[^\W\d_]", clause):
            dropped, stranded = dropped + 1, stranded + 1
            continue
        parts = [piece for part in _slices(clause) for piece in _separated(part)]
        restated = [_unglued(part) for part in parts]
        dropped += sum(new != old for new, old in zip(restated, parts))
        parts = [piece for part, after in zip(restated, [*restated[1:], ""]) for piece in _undated(part, after)]
        clause = "".join(parts)
        doppler = [_leaves(part) for part in parts]
        dropped += sum(doppler)
        if not any(doppler):
            kept.append(clause)
            continue
        surviving, lost = _surviving(parts, doppler)
        stranded += lost
        rest = _tidied("".join(surviving))
        if lost and _DONE_TO_THE_HEART.search(_plain(rest)):
            rest = ""       # what was lost may deny it: `Bioprothèse mitrale, non implantée en 2019`
        if _SUBJECT_END.search(_plain(rest)):
            shared = _shared_state(parts[-1]) if doppler[-1] and _bare_subjects("".join(surviving)) else None
            stranded += not shared
            rest = f"{rest} {shared}" if shared else ""
        if _plain(rest).strip(" .,;:()-"):
            kept.append(rest)
    return ". ".join(kept) + ("." if kept else ""), dropped, stranded


def _tidied(text):
    """What is left of a clause, without the punctuation of what left it. An emptied parenthesis leaves a comma: its
    two sides were not one statement."""
    text = " ".join(_EMPTY_PARENTHESES.sub(", ", _BEFORE_CLOSING.sub(")", text)).split())
    text = _REPEATED_COMMA.sub(",", _BEFORE_COMMA.sub("", text)).strip(" ,(-")
    if text.count("(") != text.count(")"):
        text = text.replace("(", "").replace(")", "").strip(" ,-")
    return _DANGLING_END.sub("", text)


# ------------------------------------------------------------------------------------------------------ the facts

@dataclass(frozen=True)
class Fact:
    """One statement about one attribute. value: normal, abnormal, uncertain, not_assessed, or, for a measurement,
    indeterminate or invalid. `finding` is the normalized finding (`dilated`), `negated` says the caption denies it,
    `low` and `high` are the numeric value in `unit` (a point has low == high)."""
    attribute: str
    value: str
    source: str               # text, appended or structured
    evidence: str             # the rule's name, or the report column
    rule_version: str
    finding: str = ""
    negated: bool = False
    low: float | None = None
    high: float | None = None
    unit: str = ""
    narrow: bool = False      # denies one finding of a valve's appearance, with the valve not named: states too little


def _take(text, pattern):
    """(text with every match blanked out, the matches' spans). Blanking keeps the positions of what remains."""
    spans = [m.span() for m in pattern.finditer(text)]
    for a, b in spans:
        text = text[:a] + " " * (b - a) + text[b:]
    return text, spans


def _mask_ef(text):
    """Each EF value replaced by a token, keeping the words between label and number, minus the bound wording that the
    EF statement has already taken in. Returns (text, EF statements)."""
    parts, statements, last = [], [], 0
    for match, statement in rc.ef_expressions(text):
        gap = _EF_BOUND.sub(" ", match.group("gap"))
        parts += [text[last:match.start()], f" fevg {gap} efvalue{len(statements)} "]
        statements.append(statement)
        last = match.end()
    parts.append(text[last:])
    return _mask_unlabelled_ef(" ".join("".join(parts).split()), statements), statements


def _mask_unlabelled_ef(text, statements):
    """The same for an EF cut from its label: a percentage tied to an EF method (`50 % par WMS`), or `évaluée à 60 %`
    after an LV function subject in this clause. Any other percentage stays a number the rules do not judge."""
    def token(match):
        statement = rc.ef_value(match.group("first"), match.group("second"))
        if statement is None:
            return match.group()
        statements.append(statement)
        return f" fevg efvalue{len(statements) - 1} "

    for pattern in _EF_WITH_METHOD:
        text = pattern.sub(token, text)
    return _EF_VERB.sub(lambda m: token(m) if _EF_SUBJECT.search(text, 0, m.start()) else m.group(), text)


class _ClauseState:
    """What one segment hands to the next within a clause."""

    def __init__(self):
        self.pending = []        # attributes named without a state yet (`de taille et de fonction normales`)
        self.negated = False     # the previous segment negated something
        self.last_state = None   # (value, negations) of the previous segment's state word
        self.hedged = False      # the previous segment was hedged
        self.labelled = False    # the clause has named the structure
        self.reaches_back = False    # this segment also denies the claims of the one before it
        self.denies_list = False     # this segment denies in the plural: what came before may be denied too
        self.abandoned = False       # a subject waited for a state and the list ended first
        self.listing = False     # the previous segment opened a list of denied items


def _ef_fact(statement, policy):
    value = rc.ef_rule_flag([statement], policy.ef["normal_min"])
    return Fact("lv_function", value if value in DECIDED else "uncertain", "text", EF_EVIDENCE, policy.rule_version,
                low=statement.low, high=statement.high, unit=policy.ef["unit"])


def _value(text):
    """The first number of `text`, without its sign."""
    return float(_NUMBER.search(text).group().replace(",", "."))


def _read_segment(segment, grammar, statements, state, policy):
    """(facts, understood) for one segment."""
    lead = _LEAD.match(segment)
    text = segment[lead.end():] if lead else segment
    lead = lead.group(1) if lead else ""
    text, _ = _take(text, _ACQUISITION)
    text = _TERRITORY.sub(" territoire ", text)      # a modifier: a denial limited to one territory is not a plain no
    partitive = bool(_PARTITIVE.match(text))
    text, not_assessed = _take(text, _NOT_ASSESSED)
    text, hedged = _take(text, _HEDGE)
    state.denies_list = _plural_denial(text)
    state.reaches_back = lead in ("et", "ou") and state.denies_list
    listing = bool(_LIST_DENIAL.search(text))
    text, after = _take(text, _NEGATION_AFTER)
    text, before = _take(text, _NEGATION)

    found = []
    for rule in grammar.findings:
        text, spans = _take(text, rule.pattern)
        found += [(rule, span) for span in spans]
    text, normal_states = _take(text, _NORMAL_STATE)
    text, abnormal_states = _take(text, _ABNORMAL_STATE)
    subjects = []
    for attribute, pattern in grammar.subjects:
        text, spans = _take(text, pattern)
        subjects += [attribute] * bool(spans)
    text, labels = _take(text, grammar.labels)
    state.labelled = state.labelled or bool(labels)
    text, _ = _take(text, _EF_TOKEN)
    ef_ids = [int(m.group(1)) for m in _EF_TOKEN.finditer(segment)]
    strain_values = [m.group() for m in _STRAIN_VALUE.finditer(text)]
    text, _ = _take(text, _STRAIN_VALUE)
    zero = any(_value(m.group()) == 0 for m in _MEASURE.finditer(text))
    text, numbers = _take(text, _MEASURE)

    named = [w.strip(":=;-#") for w in text.split()]
    words = [w for w in named if w and w not in _FILLER]
    modifiers = [w for w in words if _MODIFIER.fullmatch(w)]
    leftover = len(words) > len(modifiers)

    own_negation = bool(before or after)
    carried = state.negated and not own_negation and lead == "ou"
    states_normal = bool(found) and all(rule.value == "normal" for rule, _ in found)    # `non-dilatée et souple`
    stated = bool(normal_states or abnormal_states)
    list_item = state.listing and not lead and not stated and not ef_ids and bool(found)
    ambiguous = not (own_negation or carried or states_normal) and (
        (state.negated and (lead == "et" or partitive)) or list_item
        or (state.hedged and not stated and (lead in ("et", "ou") or partitive)))
    locator = next((a for a in subjects if ":" in a), None)

    claims = sorted([span for _, span in found] + normal_states + abnormal_states)
    denied, doubted = Counter(), set()
    for _, end in before:
        following = [span for span in claims if span[0] >= end]
        if following:
            denied[following[0]] += 1
        elif claims:
            doubted.add(claims[-1])          # `Épanchement : aucun`: what it denies is before it, against the rule
    for start, _ in after:
        leading = [span for span in claims if span[1] <= start]
        if leading:
            denied[leading[-1]] += 1
        elif claims:
            doubted.add(claims[0])           # `Exclue dilatation`: what it denies is after it, against the rule

    def located(attribute):
        return locator if locator and locator.split(":")[0] == attribute else attribute

    def judged(value, start, end):
        """(value in context, whether a negation applies). A negation denies one claim, and leaves the others it could
        reach uncertain."""
        negations = denied[(start, end)] + carried
        reached = any(b <= start for _, b in before) or any(a >= end for a, _ in after)
        if value in _FLIP and ((start, end) in doubted or (reached and not denied[(start, end)])
                               or (carried and len(claims) > 1)):
            return "uncertain", True
        if value in _FLIP and negations == 1:
            value = _FLIP[value]
        elif value in _FLIP and negations > 1:
            value = "uncertain"
        return value, negations > 0

    def qualified(value):
        """The cues of the whole segment, applied to a claim."""
        if not_assessed:
            return "not_assessed"
        if hedged or ambiguous:
            return "uncertain"
        if value == "normal" and modifiers:     # narrows or grades: it does not state a plain normal
            return "uncertain"
        return value

    facts = []
    for rule, (start, end) in found:
        base = policy.device_present if rule.value == "device" else rule.value
        base = "uncertain" if base == "unknown" else base
        value, negated = judged(base, start, end)
        if rule.value == "device" and not denied[(start, end)]:
            value, negated = base, False
        narrow = negated and value == "normal" and rule.attribute.endswith("_appearance") and not state.labelled
        # `Pas d'anomalie de la fonction` is about the function, not the whole structure.
        targets = dict.fromkeys(map(located, subjects)) if rule.attribute == OVERALL and subjects else [
            located(rule.attribute)]
        facts += [Fact(target, qualified(value), "text", rule.name, policy.rule_version, finding=rule.name,
                       negated=negated, narrow=narrow) for target in targets]
        state.last_state = None

    if lead not in ("", "et", "ou", "ainsi que"):       # `Dimensions mais fonction normale`: no list goes on past it
        state.abandoned, state.pending = state.abandoned or bool(state.pending), []
    attributes = list(dict.fromkeys(map(located, [*state.pending, *subjects])))
    states = [("normal", span) for span in normal_states] + [("abnormal", span) for span in abnormal_states]
    if states and not attributes and _METHOD_WORDS & set(named):
        leftover = True                      # `WMSI normal` is no claim about the whole structure
    for value, (start, end) in states:
        value, negated = judged(value, start, end)
        state.last_state = (value, negated)
        for attribute in attributes or [OVERALL]:
            facts.append(Fact(attribute, qualified(value), "text", "state", policy.rule_version, finding="state",
                              negated=negated))
    if states:
        state.pending = []
    elif found:            # `Dimensions, pas de thrombus, fonction normale`: a finding ends the list
        state.abandoned, state.pending = state.abandoned or bool(state.pending), []
    elif subjects and not found and not ef_ids and lead == "et" and state.last_state and not (
            _REFERENCE_WORDS & set(named)):
        value, negated = state.last_state
        facts += [Fact(a, qualified(value), "text", "state", policy.rule_version, finding="state", negated=negated)
                  for a in attributes]
    elif subjects and not found and not ef_ids:
        if not_assessed or hedged or _REFERENCE_WORDS & set(named):      # `Dimensions : voir schéma`: said elsewhere
            facts += [Fact(a, qualified("uncertain"), "text", "cue", policy.rule_version) for a in attributes]
            state.pending = []
        else:
            state.pending = attributes
    elif not_assessed and not found:
        facts.append(Fact(OVERALL, "not_assessed", "text", "cue", policy.rule_version))

    if ef_ids:
        state.last_state = None         # `Fonction normale, FEVG 60 % et taille`: no state goes on past an EF
    for index in ef_ids:
        fact = _ef_fact(statements[index], policy)
        if hedged or own_negation or carried or ambiguous or (fact.value == "normal" and modifiers):
            fact = replace(fact, value="uncertain")
        facts.append(fact)

    # A number the rules do not judge may contradict a normal claim; beside an abnormal one it only grades it, unless
    # it is zero. A strain value sets no flag, but a clearly low one cannot stand beside a normal claim.
    low_strain = any(_value(value) < _STRAIN_CLEARLY_LOW for value in strain_values)
    if (numbers and any(f.value != "abnormal" for f in facts)) or zero or (
            low_strain and any(f.value == "normal" for f in facts)):
        leftover = True
    stray = bool(numbers or modifiers or hedged or before or after) and not facts
    state.negated = own_negation or carried or ambiguous
    state.hedged = bool(hedged) or (state.hedged and ambiguous)
    state.listing = listing or list_item
    return facts, not (leftover or stray)


def _only_doubt(segment):
    """Whether a segment is a doubt and nothing else: `Calcifiée, à confirmer` does not say what it doubts, so the
    whole clause is in doubt."""
    text, doubts = _take(segment, _HEDGE)
    return bool(doubts) and not _JOINING_WORD.sub(" ", text).strip(" .,;:()-?")


def _mixed_state(facts):
    """Whether a clause holds both a normal and an abnormal state, one of them with no subject of its own: `Fonction
    normale ou légèrement diminuée` does not say the structure is abnormal."""
    stated = [fact for fact in facts if fact.finding == "state" and fact.value in _FLIP]
    return len({fact.value for fact in stated}) > 1 and any(fact.attribute == OVERALL for fact in stated)


@lru_cache(maxsize=1 << 18)
def read_caption(structure, caption, policy):
    """(text facts, unrecognized wording) of a caption. A segment that is not fully understood gives no fact."""
    grammar = GRAMMARS[structure]
    facts, unrecognized = [], []
    for clause in rc.split_clauses(caption):
        text = _plain(clause)
        if _PRIOR.search(text):
            unrecognized.append(rc.clause_template(text))
            continue
        statements = []
        text = _EF_TOKEN.sub(" illisible ", text)      # the token is ours: in a report it is a word nobody can read
        if structure == "lv":
            text, statements = _mask_ef(text)
        state = _ClauseState()
        last = start = len(facts)
        doubted = bool(_SIGN.search(text))     # `Épanchement -`: a sign may deny what came before it
        if doubted:
            unrecognized.append(rc.clause_template(text))
        for part in _slices(text):
            segment = part.strip(" ,()-+")
            doubted = doubted or _only_doubt(segment)
            if not segment:
                if part.strip(" ,()") in ("-", "+"):       # `Épanchement (-)`: it may deny what came before it
                    unrecognized.append(part.strip(" ,()"))
                    facts[last:] = [replace(f, value="uncertain") if f.value in DECIDED else f for f in facts[last:]]
                continue
            found, understood = _read_segment(segment, grammar, statements, state, policy)
            if state.reaches_back:
                facts[last:] = [replace(f, value="uncertain") if f.value in DECIDED else f for f in facts[last:]]
            if state.denies_list:              # `Dilatation, hypertrophie absentes`: the whole list may be denied
                facts[start:] = [replace(f, value="uncertain") if f.value == "abnormal" else f for f in facts[start:]]
            last = len(facts)
            if understood:
                facts += found
            else:
                unrecognized.append(rc.clause_template(_EF_TOKEN.sub("#", segment)))
        if state.abandoned or (state.pending and not (statements and set(state.pending) == {"lv_function"})):
            unrecognized.append(rc.clause_template(_EF_TOKEN.sub("#", text)))
        if doubted or _mixed_state(facts[start:]):
            facts[start:] = [replace(f, value="uncertain") if f.value in DECIDED else f for f in facts[start:]]
        elif not state.labelled and any(f.value == "note" for f in facts[start:]):
            # `Graisse épicardique normale`: the state is the note's, not the structure's.
            facts[start:] = [replace(f, value="uncertain") if f.attribute == OVERALL and f.value in DECIDED else f
                             for f in facts[start:]]
    return tuple(facts), tuple(unrecognized)


def _source_fact(attribute, source, row, policy):
    """The Fact of one report value, or None when it is missing."""
    number = rc.finite(row.get(source["column"]))
    if number is None:
        return None
    number *= source.get("scale", 1)
    low, high = source["valid"]
    if not low <= number <= high:
        value = "invalid"
    elif number <= source["normal_max"]:
        value = "normal"
    elif number > source["abnormal_above"]:
        value = "abnormal"
    else:
        value = "indeterminate"
    return Fact(attribute, value, "structured", source["column"], policy.rule_version, low=number, high=number,
                unit=source["unit"])


def structured_facts(structure, row, policy):
    """The Facts of the report's measurements for one structure. `row` maps report columns to values."""
    facts = []
    if structure == "lv":
        ef = rc.finite(row.get(policy.ef["column"]))
        if ef is not None:
            low, high = policy.ef["valid"]
            value = ("invalid" if not low <= ef <= high else "normal" if ef >= policy.ef["normal_min"] else "abnormal")
            facts.append(Fact("lv_function", value, "structured", policy.ef["column"], policy.rule_version,
                              low=ef, high=ef, unit=policy.ef["unit"]))
    for attribute, measure in policy.measures.items():
        if measure["structure"] != structure:
            continue
        for source in measure["sources"]:
            fact = _source_fact(attribute, source, row, policy)
            if fact is None:
                continue
            facts.append(fact)
            if measure["combine"] == "first_available" and fact.value != "invalid":
                break
    return tuple(facts)


def report_columns(structure, policy):
    """The report columns `structured_facts` reads for a structure."""
    columns = [policy.ef["column"]] if structure == "lv" else []
    for measure in policy.measures.values():
        if measure["structure"] == structure:
            columns += [source["column"] for source in measure["sources"]]
    return columns


def _measured(attribute, facts, policy):
    """What the measurements say of an attribute: normal, abnormal, or None (missing, invalid or in the grey zone)."""
    values = [f.value for f in facts if f.attribute == attribute and f.value != "invalid"]
    if "abnormal" in values:
        return "abnormal"
    measure = policy.measures.get(attribute)
    needed = len(measure["sources"]) if measure and measure["combine"] == "every" else 1
    return "normal" if len(values) >= needed and all(v == "normal" for v in values) else None


def _settled(attribute, claims, structured, policy):
    """Whether a value over the limit stays in the measure's `mild` band and the report shows normal geometry: relative
    wall thickness in range, unrounded, and the attribute settling it normal, stated or measured. A missing wall or
    cavity settles nothing."""
    mild = policy.measures.get(attribute, {}).get("mild")
    over = [f.low for f in structured if f.attribute == attribute and f.value == "abnormal"]
    if not mild or max(over, default=0) > mild["up_to"]:
        return False
    values = {f.evidence: f.low for f in structured if f.value != "invalid"}
    wall, cavity = values.get(mild["geometry"]["wall"]), values.get(mild["geometry"]["cavity"])
    if wall is None or cavity is None or 2 * wall / cavity > mild["geometry"]["max"] + _RATIO_TOLERANCE:
        return False
    stated = {c.value for c in claims if c.attribute == mild["settled_by"]}
    return "normal" in stated or _measured(mild["settled_by"], structured, policy) == "normal"


def _resolve(attribute, claims, measured):
    """(status of an attribute, conflict or None). status: normal, abnormal, conflict, uncertain, not_assessed,
    hidden_abnormal (measured, not in the caption) or not_stated."""
    values = {c.value for c in claims}
    stated = values & set(DECIDED)
    if len(stated) == 2:
        kind = "text_structured" if any(c.source == "appended" for c in claims) else "text_text"
        return "conflict", {"attribute": attribute, "kind": kind, "text": sorted(stated), "structured": measured}
    text = next(iter(stated), None)
    if text and measured and text != measured:
        return "conflict", {"attribute": attribute, "kind": "text_structured", "text": [text],
                            "structured": measured}
    if text == "abnormal":
        return "abnormal", None
    if "uncertain" in values:
        return "uncertain", None
    if "not_assessed" in values:
        return "not_assessed", None
    if text == "normal":
        return "normal", None
    return ("hidden_abnormal" if measured == "abnormal" else "not_stated"), None


@dataclass(frozen=True)
class Assessment:
    flag: str
    facts: tuple              # text facts, then structured facts
    conflicts: tuple          # dicts: attribute, kind, text, structured
    reasons: tuple            # why the caption is not plainly normal: "kind" or "kind:attribute"
    unrecognized: tuple       # the wording that was not understood
    evidence: str             # the sources behind decided facts, for example structured+text
    boundary: bool            # a measurement in the grey zone or settled as mild, or an EF near the cutoff


def _near_cutoff(facts, policy):
    for fact in facts:
        if fact.value == "indeterminate":
            return True
        if fact.unit == policy.ef["unit"] and fact.low is not None:
            cutoff, margin = policy.ef["normal_min"], policy.ef["boundary_margin"]
            if abs(fact.low - cutoff) <= margin or abs(fact.high - cutoff) <= margin:
                return True
    return False


def assess(structure, caption, policy, structured=(), ef_status=None, stranded=0):
    """The caption's flag and the facts behind it. `structured`: from `structured_facts`, for the same structure.
    `ef_status`: from `radar_captions.lv_caption`, for an LV caption. `stranded`: from `retain_caption`; a caption that
    lost text nobody could read cannot be normal."""
    claims, unrecognized = read_caption(structure, caption, policy)
    if ef_status == "appended":
        claims = tuple(replace(c, source="appended") if c.evidence == EF_EVIDENCE else c for c in claims)
    whole = [c for c in claims if c.attribute == OVERALL]
    whole_normal = [c for c in whole if c.value == "normal"]
    required = policy.required[structure]
    attributes = {c.attribute for c in claims} | {f.attribute for f in structured} | set(required)
    attributes.discard(OVERALL)

    status, conflicts, settled = {}, [], False
    for attribute in sorted(attributes):
        # A device is a fact a whole-structure normal does not touch: `Bioprothèse sans anomalie` still has one.
        own = [c for c in claims if c.attribute == attribute] + ([] if attribute.endswith("_device") else whole_normal)
        measured = _measured(attribute, structured, policy)
        if measured == "abnormal" and _settled(attribute, claims, structured, policy):
            measured, settled = None, True
        status[attribute], conflict = _resolve(attribute, own, measured)
        if conflict:
            conflicts.append(conflict)
    if whole:
        status[OVERALL], conflict = _resolve(OVERALL, whole, None)
        if conflict:
            conflicts.append(conflict)
    if ef_status == "conflict" and status.get("lv_function") != "conflict":
        status["lv_function"] = "conflict"
        conflicts.append({"attribute": "lv_function", "kind": "text_structured", "text": ["ef_value"],
                          "structured": "ef_value"})

    reasons = []
    names = {"conflict": "conflict", "uncertain": "uncertain", "not_assessed": "not_assessed",
             "hidden_abnormal": "structured_abnormal_not_in_caption"}
    for attribute, value in status.items():
        if value in names:
            reasons.append(f"{names[value]}:{attribute}")
    if unrecognized:
        reasons.append("unrecognized_text")
    if stranded:
        reasons.append("stranded_text")
    for attribute in required:
        stated = [v for k, v in status.items() if k == attribute or k.startswith(attribute + ":")]
        said = [c for c in claims if c.attribute == attribute or c.attribute.startswith(attribute + ":")]
        if all(v in ("not_stated", "hidden_abnormal") for v in stated) or (
                said and not whole_normal and all(c.narrow for c in said)):
            reasons.append(f"missing_required:{attribute}")
    if not claims:
        reasons.append("no_understood_claim")

    flag = "abnormal" if "abnormal" in status.values() else "unknown" if reasons else "normal"
    facts = (*claims, *structured)
    evidence = "+".join(sorted({f.source for f in facts if f.value in DECIDED})) or "none"
    return Assessment(flag, facts, tuple(conflicts), tuple(reasons), unrecognized, evidence,
                      settled or _near_cutoff(facts, policy))
