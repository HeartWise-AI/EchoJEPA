# tools/radar/render_zero_shot_video.py

"""A short explainer video of one study's RADAR zero-shot prediction, from an `explain_<study_id>.json`.

The JSON is written by `evals.radar_frozen.zero_shot_eval --dump_study <study_id>` (`explain_header` and
`explain_task` there). It holds, per evaluation bag, the clips the model read (clip id = file path, and view), and per
task every prompt with its cosine similarity to the study's structure embedding (and logit = similarity / T), what
the rule selected, each bag's value, the pooled prediction and the label.

The video (H.264 MP4, about 30-60 s at the default durations) shows, in order:

1. a title card (study, checkpoint temperature, bags);
2. the input clips of each bag playing as a grid, with their views;
3. per task: the prompt scores of each bag as a bar chart with the selected prompts highlighted, the selection rule
   with the bag values, and the final value beside the reference label;
4. a summary card.

A JSON with `"synthetic": true` is marked "synthetic scores - demo of the format" on every frame. Clips that cannot be
read (for example a real run's paths opened on another machine) are drawn as grey tiles with their view.

Two commands, run from the repository root:

    # render a dump
    python -m tools.radar.render_zero_shot_video render --explain <out_dir>/explain_<study_id>.json --out video.mp4

    # a synthetic dump (random embeddings, the real prompt file and scoring code) over given clips, for a demo
    python -m tools.radar.render_zero_shot_video synthetic --clip A4C=/path/a.mp4 --clip PLAX=/path/b.mp4 \\
        --out synthetic.json [--label lvef=62 --label lv_systolic_function=2]

Only matplotlib, OpenCV (to read clips) and the ffmpeg binary (to encode) are used; nothing runs on a GPU.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import matplotlib
import matplotlib.patches

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SYNTHETIC_BANNER = "SYNTHETIC SCORES – demo of the format (not a model output)"
WIDTH, HEIGHT, DPI = 1280, 720, 100
COLORS = {"selected": "#d1495b", "other": "#9aa5b1", "positive": "#d1495b", "negative": "#2e86ab",
          "label": "#1b998b", "prediction": "#d1495b", "grades": ["#d1495b", "#edae49", "#2e86ab", "#1b998b"]}
DEFAULT_DURATIONS = {"title": 4.0, "clips_per_bag": 4.0, "bag_scores": 3.5, "rule": 5.0, "final": 4.0,
                     "summary": 4.0}


# ---- clips ------------------------------------------------------------------------------------------------------

def read_clip(path, max_frames=64, size=160):
    """Up to `max_frames` RGB frames [T, size, size, 3] of a video file, or None when it cannot be read."""
    try:
        import cv2
    except ImportError:
        return None
    if not path or not os.path.exists(path):
        return None
    capture = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (size, size)))
    capture.release()
    return np.stack(frames) if frames else None


# ---- frame plumbing ---------------------------------------------------------------------------------------------

def new_figure(record):
    fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI)
    fig.patch.set_facecolor("white")
    if record.get("synthetic"):
        fig.text(0.5, 0.985, SYNTHETIC_BANNER, ha="center", va="top", fontsize=13, color="white", weight="bold",
                 bbox={"facecolor": "#c0392b", "edgecolor": "none", "boxstyle": "round,pad=0.3"})
    fig.text(0.99, 0.01, f"RADAR zero-shot · study {record['study_id']}", ha="right", va="bottom", fontsize=9,
             color="#777777")
    return fig


def figure_pixels(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


class Writer:
    """Raw RGB frames piped to ffmpeg (libx264, yuv420p, playable everywhere)."""

    def __init__(self, path, fps):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is not on PATH.")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.fps, self.frames = fps, 0
        self.process = subprocess.Popen(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium",
             "-movflags", "+faststart", path], stdin=subprocess.PIPE)

    def write(self, pixels, seconds=None):
        repeat = 1 if seconds is None else max(1, round(seconds * self.fps))
        data = np.ascontiguousarray(pixels[:HEIGHT, :WIDTH]).tobytes()
        for _ in range(repeat):
            self.process.stdin.write(data)
        self.frames += repeat

    def still(self, fig, seconds):
        self.write(figure_pixels(fig), seconds)
        plt.close(fig)

    def close(self):
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError("ffmpeg failed to encode the video.")


def wrap(text, width):
    return "\n".join(textwrap.wrap(str(text), width)) or str(text)


def short(text, width=70):
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


def fmt(value, digits=1):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}"


# ---- scenes -----------------------------------------------------------------------------------------------------

def title_scene(writer, record, tasks, seconds):
    fig = new_figure(record)
    fig.text(0.5, 0.72, "What RADAR does for one study", ha="center", fontsize=30, weight="bold")
    fig.text(0.5, 0.63, f"Zero-shot prediction · study {record['study_id']}", ha="center", fontsize=18)
    steps = ("1. Clips: the study is read as {n} bags of clips (views routed to each structure)\n"
             "2. Embedding: each bag gives one embedding per structure (LV, RV, valves...)\n"
             "3. Prompts: each sentence is encoded by the caption encoder; score = cosine similarity\n"
             "4. Rule: the scores are turned into a value (selection rule), then averaged over bags\n"
             "5. Result: compared with the reference label").format(n=record["num_bags"])
    fig.text(0.5, 0.36, steps, ha="center", va="center", fontsize=14, linespacing=1.7, family="monospace")
    fig.text(0.5, 0.12, f"Tasks shown: {', '.join(tasks)}   ·   temperature T = {fmt(record['temperature'], 4)}"
             f"   ·   pooling: {record.get('pooling', '?')}", ha="center", fontsize=12, color="#555555")
    writer.still(fig, seconds)


def clips_scene(writer, record, seconds_per_bag, max_clips=12):
    cache = {}
    for bag in record["bags"]:
        clips = bag["clips"][:max_clips]
        if not clips:
            continue
        cols = min(4, len(clips))
        rows = int(np.ceil(len(clips) / cols))
        fig = new_figure(record)
        fig.text(0.5, 0.895, f"Input: bag {bag['bag'] + 1} of {record['num_bags']} · {len(bag['clips'])} clips",
                 ha="center", fontsize=20, weight="bold")
        supported = ", ".join(bag.get("structures_supported", [])) or "none"
        fig.text(0.5, 0.855, wrap(f"structures this bag can read: {supported}", 140), ha="center", va="top",
                 fontsize=11, color="#555555")
        grid = fig.add_gridspec(rows, cols, left=0.12, right=0.88, top=0.79, bottom=0.05, wspace=0.08, hspace=0.25)
        images, videos = [], []
        for k, clip in enumerate(clips):
            ax = fig.add_subplot(grid[k // cols, k % cols])
            ax.set_xticks([]), ax.set_yticks([])
            path = clip["clip_id"]
            if path not in cache:
                cache[path] = read_clip(path)
            video = cache[path]
            if video is None:
                video = np.full((1, 160, 160, 3), 200, dtype=np.uint8)
                ax.text(0.5, 0.5, "clip not readable\non this machine", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="#444444")
            images.append(ax.imshow(video[0]))
            videos.append(video)
            ax.set_title(f"{clip['view']}" + (f" · {clip['acquisition']}" if clip.get("acquisition") else ""),
                         fontsize=11, weight="bold")
        if len(bag["clips"]) > max_clips:
            fig.text(0.5, 0.02, f"(+{len(bag['clips']) - max_clips} more clips not shown)", ha="center", fontsize=10)
        for t in range(max(1, round(seconds_per_bag * writer.fps))):
            frame_index = int(t * 30 / writer.fps)            # play near the clips' native 30 fps
            for image, video in zip(images, videos):
                image.set_data(video[frame_index % len(video)])
            writer.write(figure_pixels(fig))
        plt.close(fig)


def regression_bag_figure(record, name, task, b):
    bag = task["bags"][b]
    values, sims = np.asarray(task["values"], float), np.asarray(bag["similarities"], float)
    selected = np.zeros(len(values), bool)
    selected[bag["selected"]] = True
    fig = new_figure(record)
    supported = task["bags_supported"][b]
    fig.text(0.5, 0.895, f"{name} · bag {b + 1}: score of every prompt" + ("" if supported else
             "  (bag does not support this structure: ignored)"), ha="center", fontsize=18, weight="bold")
    ax = fig.add_axes([0.07, 0.38, 0.88, 0.46])
    if len(np.unique(values)) == len(values):
        ax.bar(values, sims, width=0.85, color=np.where(selected, COLORS["selected"], COLORS["other"]))
    else:
        ax.scatter(values[~selected], sims[~selected], s=10, color=COLORS["other"], label="other prompts")
        ax.scatter(values[selected], sims[selected], s=22, color=COLORS["selected"], label="selected")
    low = sims.min() - 0.1 * (np.ptp(sims) + 1e-6)
    ax.set_ylim(low, sims.max() + 0.1 * (np.ptp(sims) + 1e-6))
    ax.set_xlabel("value named by the prompt", fontsize=12)
    ax.set_ylabel(f"cosine similarity\n(logit = sim / T, T={fmt(record['temperature'], 3)})", fontsize=11)
    ax.axvline(bag["value"], color=COLORS["prediction"], ls="--", lw=1.5)
    ax.text(bag["value"], ax.get_ylim()[1], f" bag value {fmt(bag['value'])}", color=COLORS["prediction"],
            va="top", fontsize=11, weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    order = np.argsort(-sims)[:5]
    lines = [f"#{r + 1}  sim {sims[j]:+.3f}  logit {sims[j] / record['temperature']:+.2f}   “{short(task['prompts'][j])}”"
             for r, j in enumerate(order)]
    fig.text(0.07, 0.29, f"{len(values)} prompts scored; highlighted: the top {task['top_k']} "
             f"(keep = {task['keep']}). Best five:", fontsize=12, weight="bold")
    fig.text(0.07, 0.26, "\n".join(lines), fontsize=11, va="top", family="monospace", linespacing=1.5)
    return fig


def regression_rule_figure(record, name, task):
    fig = new_figure(record)
    fig.text(0.5, 0.90, f"{name}: selection rule", ha="center", fontsize=22, weight="bold")
    fig.text(0.5, 0.82, wrap(task["rule"], 100), ha="center", va="top", fontsize=14)
    lines = []
    for b, bag in enumerate(task["bags"]):
        chosen = sorted(bag["selected_values"])
        tag = "" if task["bags_supported"][b] else "   (ignored: no routed clip)"
        lines.append(f"bag {b + 1}: top {len(chosen)} values = {', '.join(fmt(v, 0) for v in chosen)}")
        lines.append(f"        median → {fmt(bag['value'])}{tag}")
    used = [bag["value"] for bag, ok in zip(task["bags"], task["bags_supported"]) if ok]
    lines.append("")
    lines.append(f"study value = mean({', '.join(fmt(v) for v in used)}) = {fmt(task['prediction'])}")
    fig.text(0.06, 0.66, "\n".join(wrap(line, 120) for line in lines), va="top", fontsize=12.5,
             family="monospace", linespacing=1.6)
    return fig


def binary_bag_figure(record, name, task, b, limit=10):
    bag = task["bags"][b]
    sims, sides = np.asarray(bag["similarities"], float), np.asarray(task["sides"])
    fig = new_figure(record)
    fig.text(0.5, 0.895, f"{name} · bag {b + 1}: positive vs negative prompts", ha="center", fontsize=18,
             weight="bold")
    shown = []
    for side in ("positive", "negative"):
        idx = np.where(sides == side)[0]
        shown += list(idx[np.argsort(-sims[idx])][:limit])
    ax = fig.add_axes([0.42, 0.22, 0.53, 0.62])
    y = np.arange(len(shown))[::-1]
    ax.barh(y, sims[shown], color=[COLORS[sides[j]] for j in shown])
    ax.set_yticks(y, [short(task["prompts"][j], 55) for j in shown], fontsize=9)
    ax.set_xlabel(f"cosine similarity (logit = sim / T, T={fmt(record['temperature'], 3)})")
    scores = bag["set_scores"]
    for side in ("positive", "negative"):
        ax.axvline(scores[side], color=COLORS[side], ls="--", lw=1.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.04, 0.80, f"set score (+) = {scores['positive']:+.3f}\nset score (−) = {scores['negative']:+.3f}"
             f"\n\nP(finding) this bag = {fmt(bag['value'], 3)}\ncloser set: {bag['selected']}", fontsize=13,
             va="top", family="monospace", linespacing=1.6)
    return fig


def categorical_bag_figure(record, name, task, b):
    bag = task["bags"][b]
    sims = np.asarray(bag["similarities"], float)
    grades, prompt_grades = task["grades"], task["prompt_grades"]
    names = task.get("grade_names") or {}
    color = {g: COLORS["grades"][k % len(COLORS["grades"])] for k, g in enumerate(grades)}
    fig = new_figure(record)
    fig.text(0.5, 0.895, f"{name} · bag {b + 1}: every prompt, coloured by grade", ha="center", fontsize=18,
             weight="bold")
    order = np.argsort(-sims)[:16]
    ax = fig.add_axes([0.52, 0.30, 0.44, 0.54])
    y = np.arange(len(order))[::-1]
    ax.barh(y, sims[order], color=[color[prompt_grades[j]] for j in order])
    ax.set_yticks(y, [short(task["prompts"][j], 55) for j in order], fontsize=9)
    ax.set_xlabel(f"cosine similarity (logit = sim / T, T={fmt(record['temperature'], 3)})")
    ax.spines[["top", "right"]].set_visible(False)
    ax2 = fig.add_axes([0.05, 0.30, 0.16, 0.40])
    labels = [f"{fmt(g, 0)} {names.get(str(int(g)), '')}".strip() for g in grades]
    ax2.bar(range(len(grades)), bag["probabilities"], color=[color[g] for g in grades])
    ax2.set_xticks(range(len(grades)), [lab.replace(" ", "\n", 1) for lab in labels], fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.set_title("grade probabilities\n(this bag)", fontsize=11)
    ax.legend(handles=[matplotlib.patches.Patch(color=color[g], label=f"grade {lab}") for g, lab in zip(grades, labels)],
              loc="lower right", fontsize=9, frameon=False)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.text(0.07, 0.20, "grade score = similarity to the grade's mean prompt embedding: "
             + ", ".join(f"{lab} {s:+.3f}" for lab, s in zip(labels, bag["grade_scores"])), fontsize=11)
    return fig


def rule_figure(record, name, task):
    fig = new_figure(record)
    fig.text(0.5, 0.88, f"{name}: selection rule", ha="center", fontsize=22, weight="bold")
    fig.text(0.5, 0.78, wrap(task["rule"], 100), ha="center", va="top", fontsize=14)
    lines = []
    for b, bag in enumerate(task["bags"]):
        tag = "" if task["bags_supported"][b] else "  (ignored: no routed clip)"
        if task["mode"] == "binary":
            lines.append(f"bag {b + 1}: P = {fmt(bag['value'], 3)}{tag}")
        else:
            lines.append(f"bag {b + 1}: P(grades) = [{', '.join(fmt(p, 3) for p in bag['probabilities'])}]{tag}")
    if task["mode"] == "categorical":
        lines.append(f"\nmean over bags = [{', '.join(fmt(p, 3) for p in task['pooled_probabilities'])}]"
                     f"  → argmax grade {fmt(task['prediction'], 0)}")
    else:
        lines.append(f"\nmean over bags = {fmt(task['prediction'], 3)}")
    fig.text(0.10, 0.60, "\n".join(lines), va="top", fontsize=14, family="monospace", linespacing=1.7)
    return fig


def final_figure(record, name, task):
    fig = new_figure(record)
    fig.text(0.5, 0.86, f"{name}: final prediction", ha="center", fontsize=24, weight="bold")
    digits = 3 if task["mode"] == "binary" else (0 if task["mode"] == "categorical" else 1)
    grade_names = task.get("grade_names") or {}

    def show(v):
        if task["mode"] == "categorical" and v is not None:
            return f"{fmt(v, 0)} ({grade_names.get(str(int(v)), '')})".replace(" ()", "")
        return fmt(v, digits)
    unit = " %" if task.get("target") == "lvef" else ""
    fig.text(0.30, 0.55, "RADAR (zero-shot)", ha="center", fontsize=18, color=COLORS["prediction"])
    fig.text(0.30, 0.40, show(task["prediction"]) + unit, ha="center", fontsize=54, weight="bold",
             color=COLORS["prediction"])
    fig.text(0.70, 0.55, f"reference label ({task.get('target')})", ha="center", fontsize=18, color=COLORS["label"])
    fig.text(0.70, 0.40, show(task["label"]) + ("" if task["label"] is None else unit), ha="center", fontsize=54,
             weight="bold", color=COLORS["label"])
    if task["mode"] == "regression" and task["prediction"] is not None and task["label"] is not None:
        fig.text(0.5, 0.22, f"error = {task['prediction'] - task['label']:+.1f}{unit}", ha="center", fontsize=18)
    if task["mode"] == "binary":
        fig.text(0.5, 0.22, "probability of the finding; called at the run's operating threshold (0.5 by default)",
                 ha="center", fontsize=13, color="#555555")
    supported = sum(task["bags_supported"])
    fig.text(0.5, 0.12, f"averaged over {supported} of {len(task['bags_supported'])} bags that support "
             f"the {task['structure']} structure", ha="center", fontsize=13, color="#555555")
    return fig


def summary_scene(writer, record, names, seconds):
    fig = new_figure(record)
    fig.text(0.5, 0.82, "Summary", ha="center", fontsize=26, weight="bold")
    lines = []
    for name in names:
        task = record["tasks"][name]
        digits = 3 if task["mode"] == "binary" else (0 if task["mode"] == "categorical" else 1)
        lines.append(f"{name:<24} predicted {fmt(task['prediction'], digits):>7}   label {fmt(task['label'], digits):>7}")
    fig.text(0.5, 0.55, "\n".join(lines), ha="center", va="center", fontsize=16, family="monospace",
             linespacing=1.8)
    fig.text(0.5, 0.25, "No task head: every number above comes from comparing the study embedding with sentences.",
             ha="center", fontsize=13, color="#555555")
    writer.still(fig, seconds)


# ---- render -----------------------------------------------------------------------------------------------------

def choose_tasks(record, requested=None):
    """The tasks to show: those requested, else the first regression task and the first other task."""
    tasks = record["tasks"]
    if requested:
        unknown = [t for t in requested if t not in tasks]
        if unknown:
            raise ValueError(f"Not in the dump: {unknown}. Available: {sorted(tasks)}.")
        return list(requested)
    chosen = [n for n, t in tasks.items() if t["mode"] == "regression"][:1]
    chosen += [n for n, t in tasks.items() if t["mode"] != "regression"][:1]
    return chosen or list(tasks)[:2]


def render(record, out, tasks=None, fps=15, durations=None):
    """Write the video of `record` (a loaded explain JSON) to `out`; returns the number of frames."""
    d = {**DEFAULT_DURATIONS, **(durations or {})}
    names = choose_tasks(record, tasks)
    writer = Writer(out, fps)
    try:
        title_scene(writer, record, names, d["title"])
        clips_scene(writer, record, d["clips_per_bag"])
        for name in names:
            task = record["tasks"][name]
            for b in range(len(task["bags"])):
                if task["mode"] == "regression":
                    fig = regression_bag_figure(record, name, task, b)
                elif task["mode"] == "binary":
                    fig = binary_bag_figure(record, name, task, b)
                else:
                    fig = categorical_bag_figure(record, name, task, b)
                writer.still(fig, d["bag_scores"])
            rule = regression_rule_figure if task["mode"] == "regression" else rule_figure
            writer.still(rule(record, name, task), d["rule"])
            writer.still(final_figure(record, name, task), d["final"])
        summary_scene(writer, record, names, d["summary"])
    finally:
        writer.close()
    return writer.frames


# ---- synthetic dump ---------------------------------------------------------------------------------------------

def synthetic_record(clips, prompts=None, tasks=("lv_ejection_fraction", "lv_systolic_function"), labels=None,
                     num_bags=3, clips_per_bag=6, temperature=0.07, keep=0.2, dim=64, seed=0, study_id="SYNTHETIC"):
    """An explain record with the real prompt file and scoring code of `zero_shot_eval` but random embeddings.

    `clips`: [(view, path)]. Each bag draws `clips_per_bag` of them. Prompt embeddings are random unit vectors, with a
    random component shared with the study embedding that peaks at one random value for regression and favours one
    random grade for categorical tasks, so that the charts have a shape; the numbers mean nothing."""
    import torch

    from evals.radar_frozen import zero_shot_eval as zs
    from src.models.radar.zero_shot import load_prompt_tasks, regression_prompts

    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    loaded = load_prompt_tasks(prompts or zs.DEFAULT_PROMPTS)
    labels = labels or {}
    record = {"study_id": study_id, "synthetic": True, "temperature": temperature, "pooling": "mean_similarity",
              "keep": keep, "num_bags": num_bags, "bags": [], "tasks": {}}
    for b in range(num_bags):
        pick = sorted(rng.choice(len(clips), size=min(clips_per_bag, len(clips)), replace=False))
        record["bags"].append({"bag": b, "clips": [{"clip_id": clips[k][1], "view": clips[k][0],
                                                    "acquisition": "bmode"} for k in pick],
                               "structures_supported": sorted({loaded[t].structure for t in tasks})})

    def unit(x):
        return torch.nn.functional.normalize(x, dim=-1)

    for name in tasks:
        task = loaded[name]
        z = unit(torch.randn(num_bags, dim, generator=gen))
        shared = z.mean(0, keepdim=True)
        if task.mode == "regression":
            values = (list(task.values) if task.values
                      else [v for v, _ in regression_prompts(task.label_sources, *task.value_range)])
            centre, width = rng.uniform(25, 70), rng.uniform(6, 14)
            weight = torch.tensor([1.2 * np.exp(-0.5 * ((v - centre) / width) ** 2) for v in values])[:, None]
            encoded = {"values": values,
                       "prompts": unit(torch.randn(len(values), dim, generator=gen) * 0.2 + weight * shared)}
        elif task.mode == "binary":
            encoded = {"positive": unit(torch.randn(len(task.label_sources), dim, generator=gen)
                                        + 0.4 * rng.uniform() * shared),
                       "negative": unit(torch.randn(len(task.negative_sources), dim, generator=gen)
                                        + 0.4 * rng.uniform() * shared)}
        else:
            grades = sorted(set(task.values))
            favoured = rng.choice(grades)
            encoded = {"grades": grades, "sets": [
                unit(torch.randn(sum(v == g for v in task.values), dim, generator=gen) * 0.5
                     + (0.8 if g == favoured else 0.1) * shared) for g in grades]}
        has = torch.ones(num_bags, 1, dtype=torch.bool)
        per_bag = torch.stack([zs.score_task(task, z[b:b + 1], encoded, temperature, keep=keep)
                               for b in range(num_bags)])
        pooled, _ = zs.pool_bags(per_bag, has)
        if task.mode == "categorical":
            prediction = float(np.asarray(encoded["grades"], float)[pooled[0].argmax()])
        else:
            prediction = float(pooled[0])
        label = labels.get(task.target, labels.get(name))
        record["tasks"][name] = zs.explain_task(task, z, has[:, 0], encoded, temperature, "mean_similarity", keep,
                                                per_bag[:, 0], pooled[0], prediction,
                                                None if label is None else float(label))
    return zs._json(record)


GRADE_NAMES = {"lv_systolic_function": {"0": "mild", "1": "mod/sev", "2": "normal"},
               "lv_size": {"0": "mild", "1": "mod/sev", "2": "normal"}}


def add_grade_names(record):
    """The THREE_WAY grade names of prompts_v0.yaml (0 mild, 1 moderate or severe, 2 normal) for display."""
    for task in record["tasks"].values():
        if task["mode"] == "categorical" and task.get("target") in GRADE_NAMES and "grade_names" not in task:
            task["grade_names"] = GRADE_NAMES[task["target"]]
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("render", help="render an explain JSON as an MP4")
    r.add_argument("--explain", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--task", action="append", help="task to show (repeatable; default: one regression + one other)")
    r.add_argument("--fps", type=int, default=15)
    r.add_argument("--speed", type=float, default=1.0, help="multiply every scene duration by 1/speed")
    s = sub.add_parser("synthetic", help="write a synthetic explain JSON over given clips")
    s.add_argument("--clip", action="append", required=True, metavar="VIEW=PATH")
    s.add_argument("--out", required=True)
    s.add_argument("--prompts", help="prompt file (default: configs/radar/prompts_v0.yaml)")
    s.add_argument("--task", action="append", help="tasks (default: lv_ejection_fraction, lv_systolic_function)")
    s.add_argument("--label", action="append", default=[], metavar="TARGET=VALUE")
    s.add_argument("--study_id", default="SYNTHETIC")
    s.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "synthetic":
        clips = [tuple(c.split("=", 1)) for c in args.clip]
        labels = {k: float(v) for k, v in (x.split("=", 1) for x in args.label)}
        record = synthetic_record(clips, args.prompts, tuple(args.task or ("lv_ejection_fraction",
                                                                           "lv_systolic_function")),
                                  labels, seed=args.seed, study_id=args.study_id)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False, allow_nan=False)
        print(f"wrote {args.out}")
        return
    with open(args.explain, encoding="utf-8") as f:
        record = add_grade_names(json.load(f))
    durations = {k: v / args.speed for k, v in DEFAULT_DURATIONS.items()}
    frames = render(record, args.out, args.task, args.fps, durations)
    print(f"wrote {args.out}: {frames} frames, {frames / args.fps:.1f} s")


if __name__ == "__main__":
    main()
