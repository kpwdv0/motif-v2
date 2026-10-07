import json
import os
import sys

from doingstuff import loadnotes, skyline_melody, find_motifs
from llmstuff import OUT_DIR, intervals

MAESTRO_DIR = os.path.expanduser("~/data/maestro-v3.0.0")
RULE_LEN = 7  # window_size=6 transitions


def motif_stats(melody, motifs, lens):
    if not motifs:
        return None

    covered = set()
    spots = []
    exact = []

    for key, positions in motifs.items():
        length = lens[key]
        spots.append(len(positions))

        for p in positions:
            covered.update(range(p, p + length))

        # how many repeats use the same pitch jumps as the first one
        ref = intervals(melody, positions[0], length)
        exact.append(sum(intervals(melody, p, length) == ref for p in positions) / len(positions))

    return {
        "motifs": len(motifs),
        "spots": sum(spots) / len(spots),
        "exact": sum(exact) / len(exact),
        "covered": len(covered) / len(melody),
    }


def compare_piece(entry, maestro_dir):
    # piece ids look like "2018_MIDI-....midi"
    year, filename = entry["piece_id"].split("_", 1)
    melody = skyline_melody(loadnotes(os.path.join(maestro_dir, year, filename)))

    with open(os.path.join(OUT_DIR, entry["file"])) as f:
        examples = json.load(f)

    # each example only keeps spots before its cut, so merge all 4
    claude = {}
    claude_lens = {}
    for ex in examples:
        for key, positions in ex["motifs"].items():
            claude.setdefault(key, set()).update(positions)
            claude_lens[key] = ex["motif_lens"][key]

    claude = {k: sorted(v) for k, v in claude.items()}
    rule = {str(k): v for k, v in find_motifs(melody, window_size=6).items()}

    return (motif_stats(melody, claude, claude_lens),
            motif_stats(melody, rule, {k: RULE_LEN for k in rule}))


if __name__ == "__main__":
    maestro_dir = sys.argv[1] if len(sys.argv) > 1 else MAESTRO_DIR

    with open(os.path.join(OUT_DIR, "index.json")) as f:
        index = json.load(f)

    results = [compare_piece(e, maestro_dir) for e in index]
    results = [r for r in results if r[0] and r[1]]
    print(f"{len(results)} pieces\n")

    print(f"{'':12} {'claude':>8} {'rule':>8}")
    for k in ["motifs", "spots", "exact", "covered"]:
        cl = sum(r[0][k] for r in results) / len(results)
        ru = sum(r[1][k] for r in results) / len(results)
        print(f"{k:12} {cl:8.2f} {ru:8.2f}")

    print("\nclaude's covered is only a lower bound, the saved examples just keep motifs before each cut")
