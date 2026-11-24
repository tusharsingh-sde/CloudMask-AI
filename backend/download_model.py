"""
Download the trained CloudMask AI model checkpoint from GitHub Releases.

Usage:
    python download_model.py [OPTIONS]

Options:
    --force          Overwrite existing model checkpoint without prompting.
    --skip-existing  Exit cleanly (code 0) if model checkpoint already exists.
    --output PATH    Custom output file path (defaults to <script_dir>/cloudmask_unet_best.pt).
    --tag TAG        Download from a specific release tag instead of latest.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Optional, Tuple

# Configuration
REPO_OWNER = "tusharsingh-sde"
REPO_NAME = "CloudMask-AI"
MODEL_FILENAME = "cloudmask_unet_best.pt"
ZIP_MAGIC_BYTES = b"PK\x03\x04"


def build_github_request(url: str) -> urllib.request.Request:
    """Build a urllib Request with standard GitHub headers and optional authentication."""
    headers = {
        "User-Agent": "CloudMask-AI-Downloader/1.0",
        "Accept": "application/vnd.github.v3+json",
    }
    github_token = os.getenv("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    return urllib.request.Request(url, headers=headers)


def get_release_asset_url(tag: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """Fetch the asset download URL and release tag name from GitHub Releases API."""
    if tag:
        api_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases/tags/{tag}"
        print(f"📡 Querying GitHub Release tag '{tag}' from {REPO_OWNER}/{REPO_NAME}...")
    else:
        api_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases/latest"
        print(f"📡 Querying latest GitHub Release from {REPO_OWNER}/{REPO_NAME}...")

    req = build_github_request(api_url)

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            release_data = json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 403 and "rate limit" in str(e).lower():
            raise RuntimeError(
                f"GitHub API rate limit exceeded. Set the GITHUB_TOKEN environment variable to authenticate."
            ) from e
        raise

    release_tag = release_data.get("tag_name", "unknown")
    for asset in release_data.get("assets", []):
        if asset.get("name") == MODEL_FILENAME:
            return asset.get("browser_download_url"), release_tag

    return None, release_tag


def download_atomic_with_progress(url: str, dest_path: str):
    """Download a file atomically to a .tmp file with progress, verifying file integrity."""
    tmp_path = dest_path + ".tmp"
    print(f"⬇️  Streaming download from: {url}")
    print(f"   Destination: {dest_path}")

    req = build_github_request(url)

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            total_size_header = response.headers.get("Content-Length")
            total_size = int(total_size_header) if total_size_header else 0
            downloaded = 0
            block_size = 1024 * 1024  # 1MB chunks

            with open(tmp_path, "wb") as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        mb_down = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        print(f"\r   Progress: {mb_down:.1f}/{mb_total:.1f} MB ({pct:.1f}%)", end="", flush=True)
                    else:
                        mb_down = downloaded / (1024 * 1024)
                        print(f"\r   Downloaded: {mb_down:.1f} MB", end="", flush=True)

        print("\n🔍 Validating downloaded archive integrity...")

        # 1. Verify non-empty size
        actual_size = os.path.getsize(tmp_path)
        if actual_size == 0:
            raise ValueError("Downloaded file is empty (0 bytes).")

        if total_size > 0 and actual_size != total_size:
            raise ValueError(f"Content-Length mismatch: expected {total_size} bytes, got {actual_size} bytes.")

        # 2. Check PyTorch zip archive magic bytes (PK\x03\x04)
        with open(tmp_path, "rb") as f:
            header = f.read(4)
            if header != ZIP_MAGIC_BYTES:
                # Inspect for HTML error response
                f.seek(0)
                snippet = f.read(256).decode("utf-8", errors="replace")
                raise ValueError(
                    f"Invalid model archive (header mismatch: {header!r}). "
                    f"Possible HTML error page received:\n{snippet[:120]}..."
                )

        # 3. Atomic replacement
        os.replace(tmp_path, dest_path)
        print(f"✅ Model successfully downloaded & validated: {dest_path} ({actual_size / (1024*1024):.1f} MB)")

    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="Download pre-trained CloudMask AI model from GitHub Releases.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing model checkpoint without prompting.")
    parser.add_argument("--skip-existing", action="store_true", help="Exit cleanly (0) if checkpoint already exists.")
    parser.add_argument("--output", type=str, default=None, help="Custom destination path for model checkpoint.")
    parser.add_argument("--tag", type=str, default=None, help="Download from a specific release tag instead of latest.")
    return parser.parse_args()


def main():
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dest_path = os.path.abspath(args.output) if args.output else os.path.join(script_dir, MODEL_FILENAME)

    # Check existing file
    if os.path.exists(dest_path):
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"ℹ️  Checkpoint already exists at {dest_path} ({size_mb:.1f} MB)")

        if args.skip_existing:
            print("   --skip-existing specified. Exiting cleanly.")
            sys.exit(0)

        if not args.force:
            # Check non-interactive terminal safety (CI / Docker)
            if not sys.stdin.isatty():
                print("   Non-interactive terminal detected (CI/Docker). Skipping re-download.")
                sys.exit(0)

            try:
                choice = input("   Download and overwrite? (y/N): ").strip().lower()
                if choice != "y":
                    print("   Download cancelled.")
                    sys.exit(0)
            except (EOFError, KeyboardInterrupt):
                print("\n   Download cancelled.")
                sys.exit(0)

    try:
        url, tag = get_release_asset_url(args.tag)
    except Exception as e:
        print(f"❌ Failed to query GitHub release: {e}")
        print(f"   Visit: https://github.com/{REPO_OWNER}/{REPO_NAME}/releases")
        sys.exit(1)

    if not url:
        print(f"❌ Asset '{MODEL_FILENAME}' was not found in release tag '{tag}'.")
        print(f"   Visit: https://github.com/{REPO_OWNER}/{REPO_NAME}/releases")
        sys.exit(1)

    print(f"🏷️  Resolved GitHub Release Tag: {tag}")

    try:
        download_atomic_with_progress(url, dest_path)
    except Exception as e:
        print(f"❌ Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
