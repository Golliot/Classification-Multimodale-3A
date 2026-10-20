"""Téléchargement et pré-traitement de VGGSound (HuggingFace : Loie/VGGSound).

Les archives vggsound_XX.tar.gz sont traitées une par une :
  1. téléchargement dans temp_data/ (reprise possible) et vérification du sha256 ;
  2. extraction des .mp4 dans temp_data/vggsound_XX/, puis suppression de l'archive.
     Certaines archives du dépôt sont corrompues (vggsound_08 contient 1,2 Mo de zéros) :
     la suite est alors récupérée en se resynchronisant après la zone abîmée ;
  3. pour chaque vidéo, en parallèle, un seul décodage ffmpeg produit :
       - les images : 1 par seconde, JPEG en résolution d'origine. Le filtre fps=1
         retient pour l'image k la trame affichée à t ~ k + 0.5 s, c'est-à-dire
         le milieu de la seconde audio [k, k+1) ;
       - l'audio : intégral, FLAC mono 16 kHz 16 bits ;
  4. écriture du manifeste data/manifests/vggsound_XX.csv, suppression des vidéos brutes.

Le split vient de vggsound.csv (train/test, il n'y a pas de val dans l'original) :
  data/{train,test}/frames/<clip_id>/{00,01,...}.jpg
  data/{train,test}/audio/<clip_id>.flac
  data/metadata.csv        index de tous les clips traités (concaténation des manifestes)
avec clip_id = <youtube_id>_<start:06d>, le nom de la vidéo d'origine.
Le split val est ensuite créé à partir de train par scripts/make_val_split.py.

Le script est reprenable : un subset dont le manifeste existe est ignoré, un clip
dont les sorties existent n'est pas retraité.

Usage : python -I scripts/prepare_vggsound.py --subsets 0      (ou 0-19, 3,5,7...)
"""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_URL = "https://huggingface.co/datasets/Loie/VGGSound"
TREE_API = "https://huggingface.co/api/datasets/Loie/VGGSound/tree/main"
N_SUBSETS = 20

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TEMP = ROOT / "temp_data"
MANIFESTS = DATA / "manifests"
LABELS_CSV = DATA / "vggsound.csv"

CLIP_RE = re.compile(r"^(?P<ytid>[A-Za-z0-9_-]{11})_(?P<start>\d{6})\.mp4$")
SPLITS = ("train", "test")
SAMPLE_RATE = 16000
JPEG_QUALITY = 2  # échelle mjpeg de ffmpeg : 2 (meilleure) à 31

MANIFEST_FIELDS = ["clip_id", "ytid", "start_s", "label", "split", "n_frames", "audio_duration_s", "subset"]
FAILURE_FIELDS = ["clip_id", "subset", "reason"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def parse_subsets(spec: str) -> list[int]:
    ids = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        ids.update(range(int(lo), int(hi or lo) + 1))
    if not ids <= set(range(N_SUBSETS)):
        raise ValueError(f"subsets hors de [0, {N_SUBSETS - 1}] : {spec}")
    return sorted(ids)


def fetch_remote_files() -> dict[str, dict]:
    """Taille et sha256 de chaque fichier du dépôt HuggingFace."""
    with urllib.request.urlopen(TREE_API, timeout=60) as r:
        entries = json.load(r)
    return {e["path"]: {"size": e["size"], "sha256": e.get("lfs", {}).get("oid")} for e in entries if e["type"] == "file"}


def load_labels() -> dict[tuple[str, int], tuple[str, str]]:
    """(youtube_id, start) -> (label, split)."""
    if not LABELS_CSV.exists():
        subprocess.run(["curl", "-L", "--fail", "-sS", "-o", str(LABELS_CSV), f"{REPO_URL}/resolve/main/vggsound.csv"], check=True)
    with open(LABELS_CSV, newline="") as f:
        return {(ytid, int(start)): (label, split) for ytid, start, label, split in csv.reader(f)}


def download(url: str, dest: Path, size: int, sha256: str | None, max_attempts: int = 20) -> None:
    """Télécharge url vers dest avec curl, en reprenant les téléchargements interrompus."""
    if dest.exists() and dest.stat().st_size > size:
        dest.unlink()
    attempt = 0
    while not (dest.exists() and dest.stat().st_size == size):
        attempt += 1
        if attempt > max_attempts:
            raise RuntimeError(f"échec du téléchargement de {url}")
        proc = subprocess.Popen(["curl", "-L", "--fail", "-sS", "-C", "-", "-o", str(dest), url])
        prev_t, prev_n = time.monotonic(), dest.stat().st_size if dest.exists() else 0
        while True:
            try:
                proc.wait(timeout=60)
                break
            except subprocess.TimeoutExpired:
                t, n = time.monotonic(), dest.stat().st_size if dest.exists() else 0
                log(f"  {n / 1e9:.2f}/{size / 1e9:.2f} Go ({n / size:.0%}), {(n - prev_n) / (t - prev_t) / 1e6:.1f} Mo/s")
                prev_t, prev_n = t, n
        if proc.returncode != 0:
            log(f"  curl a échoué (code {proc.returncode}), tentative {attempt}/{max_attempts}")
            time.sleep(10)
    if sha256:
        log("  vérification du sha256")
        with open(dest, "rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        if digest != sha256:
            dest.unlink()
            raise RuntimeError(f"sha256 invalide pour {dest.name} : {digest} != {sha256}")


def extract_members(tf: tarfile.TarFile, out_dir: Path, unexpected: list[str], written: list[str]) -> None:
    """Copie à plat les .mp4 de tf : seul le nom de fichier (validé par CLIP_RE) est conservé.

    Chaque vidéo est écrite dans un .part renommé une fois complète (puis ajoutée à written) : une
    vidéo tronquée par une corruption de l'archive reste en .part et n'est pas traitée.
    """
    for member in tf:
        if not member.isfile():
            continue
        name = os.path.basename(member.name)
        if not CLIP_RE.match(name):
            unexpected.append(member.name)
            continue
        part = out_dir / f"{name}.part"
        with tf.extractfile(member) as src, open(part, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        part.rename(out_dir / name)
        written.append(name)


def deflate_start(f) -> int:
    """Offset du flux deflate dans un fichier gzip (après l'en-tête et ses champs optionnels)."""
    head = f.read(10)
    if head[:3] != b"\x1f\x8b\x08":
        raise ValueError("pas un fichier gzip/deflate")
    flags = head[3]
    if flags & 0x04:  # FEXTRA
        f.read(int.from_bytes(f.read(2), "little"))
    for flag in (0x08, 0x10):  # FNAME, FCOMMENT : chaînes terminées par \0
        if flags & flag:
            while f.read(1) not in (b"\x00", b""):
                pass
    if flags & 0x02:  # FHCRC
        f.read(2)
    return f.tell()


def inflate(f, start: int, out_path: Path | None = None) -> int | None:
    """Décompresse le flux deflate brut à partir de l'offset start (dans out_path si donné).

    Renvoie l'offset de la première erreur de décompression, ou None si le flux se termine proprement.
    """
    d = zlib.decompressobj(-15)
    f.seek(start)
    pos = start
    with open(out_path, "wb") if out_path else open(os.devnull, "wb") as out:
        while chunk := f.read(1 << 20):
            saved = d.copy()
            try:
                out.write(d.decompress(chunk))
            except zlib.error:
                # On rejoue le bloc octet par octet pour situer l'erreur exactement.
                d = saved
                for i in range(len(chunk)):
                    try:
                        out.write(d.decompress(chunk[i : i + 1]))
                    except zlib.error:
                        return pos + i
            if d.eof:
                return None
            pos += len(chunk)
    return pos


def find_stored_chain(f, start: int, chain: int = 3) -> int | None:
    """Premier offset >= start où débutent `chain` blocs deflate « stored » valides consécutifs.

    Les vidéos étant incompressibles, gzip les range dans des blocs stored, alignés sur l'octet et
    décodables sans contexte : en-tête 0b000, puis LEN et NLEN = ~LEN sur 16 bits.
    """

    def stored_len(buf: bytes, p: int) -> int | None:
        if p + 5 > len(buf) or buf[p] & 0x07:  # bloc stored non final
            return None
        n = int.from_bytes(buf[p + 1 : p + 3], "little")
        return n if int.from_bytes(buf[p + 3 : p + 5], "little") == n ^ 0xFFFF else None

    window = 1 << 22
    pos = start
    while True:
        f.seek(pos)
        buf = f.read(window + chain * (5 + 0xFFFF))
        if len(buf) < 5:
            return None
        for p in range(min(window, len(buf))):
            q, k = p, 0
            while k < chain and (n := stored_len(buf, q)) is not None:
                q, k = q + 5 + n, k + 1
            if k == chain:
                return pos + p
        pos += window


def first_tar_header(path: Path) -> int | None:
    """Offset du premier en-tête tar valide (magic ustar et checksum corrects) dans un flux tar quelconque."""
    overlap = tarfile.BLOCKSIZE
    with open(path, "rb") as f:
        base = 0
        while buf := f.read(1 << 24):
            i = buf.find(b"ustar")
            while i != -1:
                start = base + i - 257
                if start >= 0:
                    f.seek(start)
                    try:
                        tarfile.TarInfo.frombuf(f.read(tarfile.BLOCKSIZE), "utf-8", "surrogateescape")
                        return start
                    except tarfile.HeaderError:
                        pass
                i = buf.find(b"ustar", i + 1)
            if len(buf) <= overlap:
                return None
            base += len(buf) - overlap
            f.seek(base)
    return None


def salvage_after_corruption(archive: Path, err: int, out_dir: Path, unexpected: list[str]) -> None:
    """Récupère les vidéos situées après chaque zone corrompue du flux gzip."""
    segment = out_dir.with_name(f"{out_dir.name}.segment")
    with open(archive, "rb") as f:
        while err is not None:
            start = find_stored_chain(f, err + 1)
            if start is None:
                log(f"  rien de récupérable après l'octet {err}")
                return
            log(f"  erreur gzip à l'octet {err}, reprise à l'octet {start} ({(start - err) / 1e6:.1f} Mo sautés)")
            err = inflate(f, start, segment)
            header = first_tar_header(segment)
            if header is not None:
                with open(segment, "rb") as seg:
                    seg.seek(header)
                    try:
                        with tarfile.open(fileobj=seg, mode="r:") as tf:
                            extract_members(tf, out_dir, unexpected, [])
                    except tarfile.ReadError:
                        pass  # segment tronqué par la zone corrompue suivante
            segment.unlink()


def extract_videos(archive: Path, out_dir: Path) -> int:
    """Extrait les .mp4 de l'archive dans out_dir, en contournant les éventuelles zones corrompues."""
    out_dir.mkdir(parents=True)
    unexpected, written = [], []
    with open(archive, "rb") as f:
        err = inflate(f, deflate_start(f))
    try:
        with tarfile.open(archive, "r|gz") as tf:
            extract_members(tf, out_dir, unexpected, written)
        if err is not None and written:
            # tarfile s'est arrêté sans erreur sur un en-tête mis à zéro par la corruption :
            # la fin de la dernière vidéo peut déjà se trouver dans la zone abîmée.
            (out_dir / written[-1]).unlink()
            log(f"  vidéo potentiellement corrompue, ignorée : {written[-1]}")
    except tarfile.ReadError:
        if err is None:
            raise
    if err is not None:
        salvage_after_corruption(archive, err, out_dir, unexpected)
    if unexpected:
        log(f"  {len(unexpected)} fichiers inattendus ignorés, ex. : {unexpected[:5]}")
    for part in out_dir.glob("*.part"):
        log(f"  vidéo tronquée par la corruption, ignorée : {part.name.removesuffix('.part')}")
    return sum(1 for _ in out_dir.glob("*.mp4"))


def read_flac_info(path: Path) -> tuple[int, int, int, int]:
    """(sample_rate, channels, bits_per_sample, total_samples) lus dans le bloc STREAMINFO."""
    with open(path, "rb") as f:
        head = f.read(26)
    if head[:4] != b"fLaC":
        raise ValueError(f"{path} n'est pas un fichier FLAC")
    x = int.from_bytes(head[18:26], "big")
    return x >> 44, ((x >> 41) & 0x7) + 1, ((x >> 36) & 0x1F) + 1, x & ((1 << 36) - 1)


def process_clip(ffmpeg: str, video: Path, split_dir: Path) -> tuple[int, float]:
    """Extrait images et audio d'une vidéo ; renvoie (nombre d'images, durée audio en s)."""
    clip_id = video.stem
    frames_dir = split_dir / "frames" / clip_id
    audio_path = split_dir / "audio" / f"{clip_id}.flac"
    if not (frames_dir.is_dir() and audio_path.is_file()):
        # Écriture dans des .part puis renommage : un clip interrompu n'est jamais pris pour terminé.
        tmp_frames = frames_dir.with_name(f"{clip_id}.part")
        tmp_audio = audio_path.with_name(f"{clip_id}.flac.part")
        shutil.rmtree(tmp_frames, ignore_errors=True)
        tmp_frames.mkdir(parents=True)
        cmd = [
            ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-threads", "1", "-i", str(video),
            "-map", "0:v:0", "-vf", "fps=1", "-q:v", str(JPEG_QUALITY), "-start_number", "0",
            "-f", "image2", str(tmp_frames / "%02d.jpg"),
            "-map", "0:a:0", "-af", "aresample=resampler=soxr:rematrix_maxval=1.0",
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-sample_fmt", "s16", "-c:a", "flac", "-f", "flac", str(tmp_audio),
        ]  # fmt: skip
        res = subprocess.run(cmd, capture_output=True, text=True)
        n_frames = sum(1 for _ in tmp_frames.glob("*.jpg"))
        if res.returncode != 0 or n_frames == 0 or not tmp_audio.is_file():
            shutil.rmtree(tmp_frames, ignore_errors=True)
            tmp_audio.unlink(missing_ok=True)
            err = res.stderr.strip().splitlines()
            raise RuntimeError(err[-1] if err else f"sortie incomplète ({n_frames} images)")
        shutil.rmtree(frames_dir, ignore_errors=True)
        tmp_audio.rename(audio_path)
        tmp_frames.rename(frames_dir)

    n_frames = sum(1 for _ in frames_dir.glob("*.jpg"))
    sr, channels, bits, n_samples = read_flac_info(audio_path)
    if (sr, channels, bits) != (SAMPLE_RATE, 1, 16) or n_samples == 0:
        raise RuntimeError(f"FLAC inattendu : {sr} Hz, {channels} canaux, {bits} bits, {n_samples} échantillons")
    return n_frames, n_samples / sr


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.rename(path)


def rebuild_metadata() -> None:
    rows = []
    for manifest in sorted(MANIFESTS.glob("vggsound_[0-9][0-9].csv")):
        with open(manifest, newline="") as f:
            rows.extend(csv.DictReader(f))
    write_csv(DATA / "metadata.csv", MANIFEST_FIELDS, rows)
    log(f"metadata.csv : {len(rows)} clips")


def process_subset(idx: int, ffmpeg: str, labels: dict, remote: dict, workers: int) -> None:
    name = f"vggsound_{idx:02d}"
    manifest = MANIFESTS / f"{name}.csv"
    if manifest.exists():
        log(f"{name} : déjà traité, ignoré")
        return
    videos_dir = TEMP / name
    if not videos_dir.is_dir():
        archive = TEMP / f"{name}.tar.gz"
        info = remote[archive.name]
        log(f"{name} : téléchargement ({info['size'] / 1e9:.2f} Go)")
        download(f"{REPO_URL}/resolve/main/{archive.name}", archive, info["size"], info["sha256"])
        log(f"{name} : extraction")
        tmp_dir = TEMP / f"{name}.part"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        n = extract_videos(archive, tmp_dir)
        tmp_dir.rename(videos_dir)
        archive.unlink()
        log(f"{name} : {n} vidéos extraites, archive supprimée")

    videos = sorted(videos_dir.glob("*.mp4"))
    log(f"{name} : traitement de {len(videos)} vidéos ({workers} workers)")
    rows, failures = [], []
    t0 = time.monotonic()
    with ThreadPoolExecutor(workers) as pool:
        futures = {}
        for video in videos:
            m = CLIP_RE.match(video.name)
            key = (m["ytid"], int(m["start"]))
            if key not in labels:
                failures.append({"clip_id": video.stem, "subset": name, "reason": "absent de vggsound.csv"})
                continue
            label, split = labels[key]
            row = {"clip_id": video.stem, "ytid": key[0], "start_s": key[1], "label": label, "split": split, "subset": name}
            futures[pool.submit(process_clip, ffmpeg, video, DATA / split)] = row
        for i, fut in enumerate(as_completed(futures), 1):
            row = futures[fut]
            try:
                row["n_frames"], duration = fut.result()
                row["audio_duration_s"] = f"{duration:.3f}"
                rows.append(row)
            except Exception as e:
                failures.append({"clip_id": row["clip_id"], "subset": name, "reason": str(e)})
            if i % 1000 == 0 or i == len(futures):
                log(f"  {i}/{len(futures)} vidéos, {i / (time.monotonic() - t0):.1f} vidéos/s")

    write_csv(MANIFESTS / f"{name}_failures.csv", FAILURE_FIELDS, sorted(failures, key=lambda r: r["clip_id"]))
    write_csv(manifest, MANIFEST_FIELDS, sorted(rows, key=lambda r: r["clip_id"]))
    shutil.rmtree(videos_dir)
    log(f"{name} : {len(rows)} clips traités, {len(failures)} échecs, vidéos brutes supprimées")
    rebuild_metadata()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subsets", default=f"0-{N_SUBSETS - 1}", help="ex. 0, 0-19, 3,5,7")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    ffmpeg = find_ffmpeg()
    for d in (TEMP, MANIFESTS, *(DATA / s / sub for s in SPLITS for sub in ("frames", "audio"))):
        d.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    remote = fetch_remote_files()
    log(f"ffmpeg : {ffmpeg}")
    log(f"{len(labels)} clips étiquetés dans vggsound.csv")
    for idx in parse_subsets(args.subsets):
        process_subset(idx, ffmpeg, labels, remote, args.workers)
    log("terminé")


if __name__ == "__main__":
    main()
