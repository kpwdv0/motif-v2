import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from computingstuff import clean_for_json
from doingstuff import loadnotes, skyline_melody, build_training_examples_polyphonic, is_periodic

MODEL = "claude-opus-5-5"
MIDI_DIR = "maestro_subset"  # can also point at the full maestro folder, year folders work
OUT_DIR = "llm-precomputed"
WORKERS = 3  # plan limits, don't go too wild
MAX_FAILS = 6  # this many fails in a row probably means the usage limit hit
MIN_LEN = 3
MAX_LEN = 12

PROMPT = """below is the melody line of a piano piece, one note per line: index, midi pitch, start time (s), duration (s).

{melody}

find the motifs in it. a motif is a short melodic idea ({min_len} to {max_len} notes) that the composer brings
back at least twice, maybe transposed or a little changed. skip things that are just trills, scales,
arpeggio patterns or repeated notes, those aren't motifs.

for each motif give a short name, its length in notes, and the index of the first note of every
place it shows up. most important motifs first, at most 12."""

SCHEMA = {
    "type": "object",
    "properties": {
        "motifs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "length": {"type": "integer"},
                    "starts": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["name", "length", "starts"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["motifs"],
    "additionalProperties": False,
}


def find_midis(midi_dir):
    # piece ids match computingstuff ("2018_MIDI-....midi") so load_maestro_splits still works
    res = []

    for root, _, files in os.walk(midi_dir):
        for f in sorted(files):
            if not f.endswith((".midi", ".mid")):
                continue

            path = os.path.join(root, f)
            piece_id = os.path.relpath(path, midi_dir).replace(os.sep, "_")
            res.append((piece_id, path))

    return sorted(res)


def ask_claude(melody):
    # goes through claude code so it uses the claude.ai plan instead of an api key
    lines = "\n".join(f"{i} {p} {s:.2f} {e - s:.2f}" for i, (s, e, p) in enumerate(melody))
    prompt = PROMPT.format(melody=lines, min_len=MIN_LEN, max_len=MAX_LEN)

    cmd = [
        "claude", "-p", prompt,
        "--output-format", "json",
        "--json-schema", json.dumps(SCHEMA),
        "--model", MODEL,
        "--effort", "medium",
        "--tools", "",
        "--no-session-persistence",
    ]

    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}  # otherwise it bills the key
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)

    try:
        res = json.loads(p.stdout)
    except json.JSONDecodeError:
        print(f"  claude failed: {(p.stderr or p.stdout)[:200]}")
        return None

    if res.get("is_error") or not res.get("structured_output"):
        print(f"  claude failed: {str(res.get('result'))[:200]}")
        return None

    return res["structured_output"]["motifs"]


def intervals(melody, start, length):
    notes = melody[start:start + length]
    return [b[2] - a[2] for a, b in zip(notes, notes[1:])]


def check_motif(melody, m):
    """keeps the spots that really match claude's motif, plus exact repeats it missed"""
    length = m["length"]
    if not MIN_LEN <= length <= MAX_LEN:
        return None

    starts = sorted(set(s for s in m["starts"] if 0 <= s <= len(melody) - length))
    if not starts:
        return None

    ref = intervals(melody, starts[0], length)
    if is_periodic(tuple(ref)):
        return None  # trill or something

    # one interval off is fine, like when the answer comes back tonal
    close = lambda s: sum(a != b for a, b in zip(ref, intervals(melody, s, length))) <= 1
    found = [s for s in starts if close(s)]

    found += [s for s in range(len(melody) - length + 1) if intervals(melody, s, length) == ref]
    found = sorted(set(found))

    return found if len(found) >= 2 else None


fails = 0
stop = threading.Event()
lock = threading.Lock()


def do_piece(piece_id, path):
    global fails
    out_path = os.path.join(OUT_DIR, f"{piece_id}.json")

    if stop.is_set() or os.path.exists(out_path):
        return

    notes = loadnotes(path)
    melody = skyline_melody(notes)

    if len(melody) < 40:
        return  # too short for the 20/20 buffers anyway

    try:
        found = ask_claude(melody)
    except subprocess.TimeoutExpired:
        print(f"{piece_id}: timed out")
        found = None

    with lock:
        fails = 0 if found is not None else fails + 1

        if fails >= MAX_FAILS and not stop.is_set():
            print(f"\n{MAX_FAILS} fails in a row, probably hit the usage limit. stopping, rerun later to keep going")
            stop.set()

    if found is None:
        return

    motifs = {}
    lens = {}

    for m in found:
        starts = check_motif(melody, m)
        if starts is None:
            continue

        key = f"{len(motifs)}_{m['name']}"  # names can repeat
        motifs[key] = starts
        lens[key] = m["length"]

    if not motifs:
        print(f"{piece_id}: skipped, none of claude's motifs checked out")
        return

    examples = build_training_examples_polyphonic(notes, melody, motifs, num_cuts=4)
    if not examples:
        return

    res = []
    for ex in examples:
        res.append({
            "motifs": ex["motifs"],
            "motif_lens": {k: lens[k] for k in ex["motifs"]},
            "context": clean_for_json(ex["context"]),
            "target": clean_for_json(ex["target"]),
            "cut_time": float(ex["cut_time"]),
        })

    with open(out_path, "w") as f:
        json.dump(res, f)

    print(f"{piece_id}: {len(motifs)}/{len(found)} motifs kept, {len(res)} examples")


def write_index():
    # rebuilt from whatever is on disk so reruns add up
    files = sorted(f for f in os.listdir(OUT_DIR) if f.endswith(".json") and f not in ("index.json", "splits.json"))

    index = []
    splits = {}

    for i, f in enumerate(files):
        piece_id = f[:-5]

        with open(os.path.join(OUT_DIR, f)) as fh:
            num = len(json.load(fh))

        index.append({"piece_id": piece_id, "num_examples": num, "file": f})
        # for the full maestro use load_maestro_splits instead, this is for the subset
        splits[piece_id] = "validation" if i % 20 == 0 else "test" if i % 20 == 1 else "train"

    with open(os.path.join(OUT_DIR, "index.json"), "w") as f:
        json.dump(index, f, indent=2)

    with open(os.path.join(OUT_DIR, "splits.json"), "w") as f:
        json.dump(splits, f, indent=2)

    return len(index), sum(e["num_examples"] for e in index)


if __name__ == "__main__":
    midi_dir = sys.argv[1] if len(sys.argv) > 1 else MIDI_DIR
    os.makedirs(OUT_DIR, exist_ok=True)

    pieces = find_midis(midi_dir)
    print(f"{len(pieces)} pieces in {midi_dir}")

    try:
        with ThreadPoolExecutor(WORKERS) as pool:
            list(pool.map(lambda args: do_piece(*args), pieces))

    finally:
        n, total = write_index()
        print(f"\n{n} pieces, {total} examples in {OUT_DIR}")
