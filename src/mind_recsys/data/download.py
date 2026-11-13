import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

MIND_SMALL_TRAIN_URL = "https://mind201910.blob.core.windows.net/mind/MINDsmall_train.zip"
MIND_SMALL_DEV_URL = "https://mind201910.blob.core.windows.net/mind/MINDsmall_dev.zip"


def download_file(url: str, dest_path: Path, chunk_size: int = 1024 * 1024) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        return

    temp_path = dest_path.with_suffix(".tmp")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    with (
        open(temp_path, "wb") as file,
        tqdm(
            desc=f"Downloading {dest_path.name}",
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for data in response.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

    temp_path.replace(dest_path)


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)


def download_mind_small(
    target_dir: Path,
    include_dev: bool = True,
) -> dict[str, Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}

    train_dir = target_dir / "train"
    if not (train_dir / "news.tsv").exists() or not (train_dir / "behaviors.tsv").exists():
        train_zip = target_dir / "MINDsmall_train.zip"
        download_file(MIND_SMALL_TRAIN_URL, train_zip)
        extract_zip(train_zip, train_dir)
        if train_zip.exists():
            train_zip.unlink()
    results["train"] = train_dir

    if include_dev:
        dev_dir = target_dir / "dev"
        if not (dev_dir / "news.tsv").exists() or not (dev_dir / "behaviors.tsv").exists():
            dev_zip = target_dir / "MINDsmall_dev.zip"
            download_file(MIND_SMALL_DEV_URL, dev_zip)
            extract_zip(dev_zip, dev_dir)
            if dev_zip.exists():
                dev_zip.unlink()
        results["dev"] = dev_dir

    return results
