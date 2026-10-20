"""Création du split val à partir de train (VGGSound n'en fournit pas).

Le val est :
  - groupé par vidéo YouTube : 38 % des clips partagent leur vidéo source avec un autre clip, et
    tous les clips d'une même vidéo vont du même côté (le split train/test d'origine fait de même) ;
  - stratifié par classe : ~fraction des clips de train de chaque classe (au moins un groupe par classe).

Les fichiers sont déplacés de data/train/ vers data/val/ (simple renommage), et la colonne split est
mise à jour dans les manifestes et dans data/metadata.csv. La sélection est déterministe (graine fixe) :
relancer le script après une interruption termine le même split.

Usage : python -I scripts/make_val_split.py [--fraction 0.05] [--seed 0]
"""

import argparse
import collections
import csv
import random
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
MANIFESTS = DATA / "manifests"


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.rename(path)


def select_val(train: list[dict], fraction: float, seed: int) -> set[str]:
    """clip_id des clips de val : groupes ytid tirés au hasard, classe par classe."""
    groups = collections.defaultdict(list)
    for r in train:
        groups[r["ytid"]].append(r["clip_id"])
    label_of = {r["clip_id"]: r["label"] for r in train}
    # Un groupe est rattaché au label majoritaire de ses clips (90 vidéos sur 161 606 ont plusieurs labels).
    by_label = collections.defaultdict(list)
    for ytid in sorted(groups):
        labels = collections.Counter(label_of[c] for c in groups[ytid])
        by_label[min(labels, key=lambda l: (-labels[l], l))].append(ytid)
    n_clips = collections.Counter(label_of.values())

    rng = random.Random(seed)
    val = set()
    for label in sorted(by_label):
        ytids = by_label[label]
        rng.shuffle(ytids)
        target, taken = max(1, round(fraction * n_clips[label])), 0
        for ytid in ytids:
            if taken >= target:
                break
            val.update(groups[ytid])
            taken += len(groups[ytid])
    return val


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _, rows = read_csv(DATA / "metadata.csv")
    if any(r["split"] == "val" for r in rows):
        raise SystemExit("data/metadata.csv contient déjà un split val, rien à faire")
    val = select_val([r for r in rows if r["split"] == "train"], args.fraction, args.seed)

    for sub in ("frames", "audio"):
        (DATA / "val" / sub).mkdir(parents=True, exist_ok=True)
    for clip_id in sorted(val):
        for src, dst in (
            (DATA / "train" / "frames" / clip_id, DATA / "val" / "frames" / clip_id),
            (DATA / "train" / "audio" / f"{clip_id}.flac", DATA / "val" / "audio" / f"{clip_id}.flac"),
        ):
            if src.exists():  # absent si déjà déplacé par une exécution interrompue
                src.rename(dst)

    for manifest in sorted(MANIFESTS.glob("vggsound_[0-9][0-9].csv")):
        fields, mrows = read_csv(manifest)
        for r in mrows:
            if r["clip_id"] in val:
                r["split"] = "val"
        write_csv(manifest, fields, mrows)
    fields, rows = read_csv(DATA / "metadata.csv")
    for r in rows:
        if r["clip_id"] in val:
            r["split"] = "val"
    write_csv(DATA / "metadata.csv", fields, rows)

    counts = collections.Counter(r["split"] for r in rows)
    print(f"val : {len(val)} clips déplacés ; train {counts['train']}, val {counts['val']}, test {counts['test']}")


if __name__ == "__main__":
    main()
