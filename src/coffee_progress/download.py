"""Download the official target PrepareCoffee archive without installing MuJoCo."""

import argparse
import hashlib
import json
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Pin the official RoboCasa manifest, rather than silently changing data releases.
ROBOCASA_COMMIT = "3443cdf3dd20f9ae8ed41adba442059b506891d0"
MANIFEST_URL = (
    f"https://raw.githubusercontent.com/robocasa/robocasa/{ROBOCASA_COMMIT}/"
    "robocasa/models/assets/box_links/box_links_ds.json"
)
ARCHIVE_KEY = "target/composite/PrepareCoffee/20250812/lerobot.tar"


def download(destination: str | Path) -> Path:
    """Keep the archive and a SHA256 provenance record beside the extracted data."""
    destination = Path(destination)
    root = destination / "lerobot"
    record = destination / "download.json"
    if record.exists() and (root / "meta" / "info.json").exists():
        print(f"Already downloaded: {root}")
        return root
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite {root}; use a fresh destination")
    destination.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(MANIFEST_URL, timeout=60) as response:
        shared_url = json.load(response)[ARCHIVE_KEY]
    url = shared_url.replace("/s/", "/shared/static/") + ".tar"
    archive = destination / "lerobot.tar"
    if not archive.exists():
        partial = archive.with_suffix(".tar.part")
        digest = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as output:
            total = int(response.headers.get("Content-Length", 0))
            count = 0
            next_report = 0
            while block := response.read(1024 * 1024):
                output.write(block)
                digest.update(block)
                count += len(block)
                if count >= next_report:
                    print(f"Downloaded {count / 1e6:.0f} / {total / 1e6:.0f} MB", flush=True)
                    next_report = count + 100_000_000
            if total and count != total:
                raise OSError(f"Incomplete archive: {count} of {total} bytes")
        partial.rename(archive)
        checksum = digest.hexdigest()
    else:
        with archive.open("rb") as handle:
            checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    print("Extracting official dataset...", flush=True)
    with tarfile.open(archive) as handle:
        # Python's data filter rejects traversal, links escaping the destination,
        # devices, and unsafe permissions. Also require the expected root folder.
        for member in handle.getmembers():
            if Path(member.name).parts[0] != "lerobot":
                raise ValueError(f"Unexpected archive entry: {member.name}")
        handle.extractall(destination, filter="data")
    if not (root / "meta" / "info.json").exists():
        raise OSError("Archive did not contain a LeRobot dataset")
    record.write_text(
        json.dumps(
            {
                "robocasa_commit": ROBOCASA_COMMIT,
                "archive_key": ARCHIVE_KEY,
                "source_url": url,
                "sha256": checksum,
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Ready: {root}", flush=True)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", default="data/prepare_coffee")
    args = parser.parse_args()
    download(args.destination)


if __name__ == "__main__":
    main()
