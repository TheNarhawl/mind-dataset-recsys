import os
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

MIND_SMALL_TRAIN_URLS = [
    "https://huggingface.co/datasets/gamusa/MIND/resolve/main/MINDsmall_train.zip",
    "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDsmall_train.zip",
    "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip",
]

MIND_SMALL_DEV_URLS = [
    "https://huggingface.co/datasets/gamusa/MIND/resolve/main/MINDsmall_dev.zip",
    "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDsmall_dev.zip",
    "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip",
]


def download_file(
    urls: list[str] | str,
    dest_path: Path,
    chunk_size: int = 1024 * 1024,
    token: str | None = None,
) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return

    url_list = [urls] if isinstance(urls, str) else urls

    auth_token = token or os.environ.get("HF_TOKEN")
    headers: dict[str, str] = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    last_error: Exception | None = None
    temp_path = dest_path.with_suffix(".tmp")

    for url in url_list:
        try:
            req_headers = dict(headers) if "huggingface.co" in url else {}
            response = requests.get(url, stream=True, timeout=60, headers=req_headers)
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

            # Verify it's a valid non-empty zip
            if zipfile.is_zipfile(temp_path):
                temp_path.replace(dest_path)
                return
            else:
                temp_path.unlink(missing_ok=True)
                raise ValueError(f"Downloaded file from {url} is not a valid zip archive.")

        except Exception as err:
            last_error = err
            if temp_path.exists():
                temp_path.unlink()
            continue

    if last_error:
        raise RuntimeError(
            f"Failed to download {dest_path.name} from any available mirrors. Last error: {last_error}"
        )


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)


def download_mind_small(
    target_dir: Path,
    include_dev: bool = True,
    token: str | None = None,
) -> dict[str, Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}

    train_dir = target_dir / "train"
    if not (train_dir / "news.tsv").exists() or not (train_dir / "behaviors.tsv").exists():
        train_zip = target_dir / "MINDsmall_train.zip"
        download_file(MIND_SMALL_TRAIN_URLS, train_zip, token=token)
        extract_zip(train_zip, train_dir)
        if train_zip.exists():
            train_zip.unlink()
    results["train"] = train_dir

    if include_dev:
        dev_dir = target_dir / "dev"
        if not (dev_dir / "news.tsv").exists() or not (dev_dir / "behaviors.tsv").exists():
            dev_zip = target_dir / "MINDsmall_dev.zip"
            download_file(MIND_SMALL_DEV_URLS, dev_zip, token=token)
            extract_zip(dev_zip, dev_dir)
            if dev_zip.exists():
                dev_zip.unlink()
        results["dev"] = dev_dir

    return results
