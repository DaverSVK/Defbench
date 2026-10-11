"""Optional dataset download from Kaggle.

Credentials are never stored in code or logs. Two mechanisms are supported:

* ``kagglehub`` (default). Public datasets can be downloaded anonymously; if the
  dataset requires authentication kagglehub reads ``KAGGLE_USERNAME``/``KAGGLE_KEY``
  or ``~/.kaggle/kaggle.json`` by itself - we never read or print those values.
* Manual: download the zip from the Kaggle web page and extract it into
  ``data/raw/<dataset>/`` (the parsers search recursively, nested folder names are
  not assumed).
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)


def download_kaggle_dataset(slug: str, target_dir: Path, force: bool = False) -> Path:
    """Download ``owner/dataset`` via kagglehub and copy/link it into ``target_dir``.

    Returns the directory that contains the extracted dataset.
    """
    target_dir = Path(target_dir)
    marker = target_dir / ".download_complete"
    if marker.exists() and not force:
        log.info("Dataset already present in %s (delete it or use --force to re-download)", target_dir)
        return target_dir
    try:
        import kagglehub
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("kagglehub is not installed: pip install kagglehub") from e

    log.info("Downloading Kaggle dataset %s (credentials, if needed, are read by kagglehub itself)", slug)
    try:
        cache_path = Path(kagglehub.dataset_download(slug, force_download=force))
    except Exception as e:
        # Do not echo exception text blindly: it may contain request headers in rare cases.
        msg = type(e).__name__
        raise RuntimeError(
            f"Kaggle download of {slug} failed ({msg}). If the dataset requires authentication, set "
            "KAGGLE_USERNAME/KAGGLE_KEY in the environment or place kaggle.json in ~/.kaggle, "
            "or download the archive manually and extract it into " + str(target_dir)) from None

    target_dir.mkdir(parents=True, exist_ok=True)
    # copy (not symlink) so that the raw data are self-contained and hashable
    for item in cache_path.iterdir():
        dst = target_dir / item.name
        if dst.exists():
            if force:
                shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
            else:
                continue
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)
    marker.write_text(f"source=kaggle:{slug}\ncache={cache_path.as_posix()}\n", encoding="utf-8")
    return target_dir
