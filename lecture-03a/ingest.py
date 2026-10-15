"""Build data.csv from the raw Bank Marketing dataset.

Usage:
    python ingest.py [--source path/or/url] [--out path/to/data.csv]

The source is the semicolon-separated bank-full.csv, or the UCI zip that contains it
(possibly inside a nested zip). The output is the comma-separated data.csv that
data_validation.py and train.py expect.
"""

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from config_loader import load_config
from logger import get_logger

config = load_config()
logger = get_logger("ingest")

UCI_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
RAW_FILE_NAME = "bank-full.csv"


def read_source(source: str) -> bytes:
    if source.startswith(("http://", "https://")):
        logger.info("Downloading %s", source)
        with urllib.request.urlopen(source, timeout=120) as response:
            return response.read()
    logger.info("Reading %s", source)
    return Path(source).read_bytes()


def find_raw_csv(content: bytes) -> bytes:
    """Return bank-full.csv from a plain file or a zip, searching nested zips too."""
    if not zipfile.is_zipfile(io.BytesIO(content)):
        return content
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for name in archive.namelist():
            if Path(name).name == RAW_FILE_NAME:
                return archive.read(name)
        for name in archive.namelist():
            if name.endswith(".zip"):
                try:
                    return find_raw_csv(archive.read(name))
                except FileNotFoundError:
                    continue
    raise FileNotFoundError(f"{RAW_FILE_NAME} not found in archive")


def ingest(source: str, out_path: Path) -> None:
    df = pd.read_csv(io.BytesIO(find_raw_csv(read_source(source))), sep=";")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d rows, %d columns to %s", len(df), len(df.columns), out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=str(config["paths"]["raw_data_path"]),
                        help=f"local file or URL (e.g. {UCI_URL})")
    parser.add_argument("--out", type=Path, default=config["paths"]["data_path"])
    args = parser.parse_args()
    ingest(args.source, args.out)
