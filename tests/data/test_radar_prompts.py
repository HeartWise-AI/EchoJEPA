# tests/data/test_radar_prompts.py

"""The shipped zero-shot prompt files against the caption parser and the label table.

Every prompt must be a sentence the training captions could hold and the parser reads as intended: the caption filter
keeps it whole (`retain_caption`: B-mode scope, nothing dropped or stranded), no word of it is unrecognized, and the
parser reads the task's `parser_attribute` as the prompt's side: abnormal for a positive prompt or a graded class,
normal for a negative prompt or the normal class. The tasks of `prompts_v0.yaml` must also name a label column of
`data/make_study_targets.py` with the matching task type and codes, because the evaluation refuses anything else.
This checks the prompts against the rules, not against how reports are worded.
"""

import os
import sys
import unittest

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(REPO, "data") not in sys.path:
    sys.path.append(os.path.join(REPO, "data"))
import make_study_targets as mst  # noqa: E402
import radar_captions as rc  # noqa: E402
import radar_normality as rn  # noqa: E402

from src.models.radar.routing import load_routing  # noqa: E402
from src.models.radar.zero_shot import SLOT, load_prompt_tasks  # noqa: E402

LABELLED = os.path.join(REPO, "configs", "radar", "prompts_v0.yaml")
FINDINGS = os.path.join(REPO, "configs", "radar", "prompts_v0_findings.yaml")
MODE_OF_LABEL = {"binary": "binary", "categorical": "classification", "regression": "regression"}
GRADED_NEGATIVES = {"lvh_modsev"}      # the negative set holds the mild grade: moderate or severe against the rest
NORMAL_CLASS = mst.THREE_WAY["normal"]


def specs(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["tasks"]


def about(fact, attribute):
    """Whether a parser fact is about `attribute`, one of its parts (`aorta_size:root`) or, for a structure-wide
    attribute such as `ivc`, one of its attributes (`ivc_size`)."""
    return fact.attribute == attribute or fact.attribute.startswith((attribute + ":", attribute + "_"))


class TestPromptsAgainstTheParser(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.policy = rn.load_policy()
        cls.trained = {s.name for s in load_routing().trained}

    def read(self, structure, prompt):
        """The parser's facts of one prompt, after checking that a caption could hold it and that it is understood."""
        kept, dropped, stranded = rn.retain_caption(prompt)
        self.assertEqual((kept, dropped, stranded), (prompt, 0, 0), f"the caption filter changes {prompt!r}")
        result = rn.assess(structure, kept, self.policy)
        self.assertEqual(result.unrecognized, (), f"the parser does not understand {prompt!r}")
        return result.facts

    def sides(self, structure, attribute, prompt):
        return {f.value for f in self.read(structure, prompt) if about(f, attribute)}

    def check_abnormal(self, structure, attribute, prompt):
        self.assertIn("abnormal", self.sides(structure, attribute, prompt), prompt)

    def check_normal(self, structure, attribute, prompt):
        facts = self.read(structure, prompt)
        self.assertNotIn("abnormal", {f.value for f in facts if about(f, attribute)}, prompt)
        # A device task's negatives describe a native valve: its appearance is what they state.
        stated = [f for f in facts if f.value == "normal" and (about(f, attribute) or attribute.endswith("_device"))]
        self.assertTrue(stated, f"{prompt!r} states nothing normal about {attribute}")

    def test_every_prompt_is_a_caption_the_parser_reads_as_its_side(self):
        for path in (LABELLED, FINDINGS):
            tasks = load_prompt_tasks(path)
            for name, spec in specs(path).items():
                task, attribute = tasks[name], spec["parser_attribute"]
                with self.subTest(file=os.path.basename(path), task=name):
                    self.assertIn(task.structure, self.trained)
                    if task.mode == "binary":
                        for prompt in task.label_sources:
                            self.check_abnormal(task.structure, attribute, prompt)
                        for prompt in task.negative_sources:
                            if name in GRADED_NEGATIVES:
                                self.read(task.structure, prompt)
                            else:
                                self.check_normal(task.structure, attribute, prompt)
                    elif task.mode == "categorical":
                        for prompt, value in zip(task.label_sources, task.values):
                            check = self.check_normal if value == NORMAL_CLASS else self.check_abnormal
                            check(task.structure, attribute, prompt)
                    else:
                        for template in task.label_sources:
                            low, high = task.value_range
                            for value in (low, (low + high) // 2, high):
                                self.read(task.structure, template.replace(SLOT, str(value)))
                                read = rc.find_ef(template.replace(SLOT, str(value)))
                                self.assertEqual([(s.low, s.high, s.certain) for s in read], [(value, value, True)])

    def test_graded_negatives_hold_no_higher_grade(self):
        task = load_prompt_tasks(LABELLED)["lvh_modsev"]
        for prompt in task.negative_sources:
            self.assertNotRegex(prompt, "modérée|sévère")
        for prompt in task.label_sources:
            self.assertRegex(prompt, "modérée|sévère")

    def test_no_prompt_is_in_both_sets_of_a_task(self):
        for path in (LABELLED, FINDINGS):
            for name, task in load_prompt_tasks(path).items():
                with self.subTest(task=name):
                    self.assertFalse(set(task.label_sources) & set(task.negative_sources))
                    self.assertEqual(len(set(task.label_sources)), len(task.label_sources))


class TestPromptsAgainstTheLabelTable(unittest.TestCase):

    def test_labelled_tasks_name_a_label_column_of_the_same_type_and_codes(self):
        for name, task in load_prompt_tasks(LABELLED).items():
            with self.subTest(task=name):
                self.assertEqual(mst.TASKS.get(task.target), MODE_OF_LABEL[task.mode])
                if task.mode == "categorical":
                    # every class has prompts
                    self.assertEqual(set(task.values), set(mst.CLASSES[task.target].values()))

    def test_findings_have_no_label_column_yet(self):
        """A task that gains a label column belongs in prompts_v0.yaml, where the evaluation can score it."""
        for name, task in load_prompt_tasks(FINDINGS).items():
            with self.subTest(task=name):
                self.assertNotIn(task.target, mst.TASKS)
                self.assertEqual(task.mode, "binary")


if __name__ == "__main__":
    unittest.main()
