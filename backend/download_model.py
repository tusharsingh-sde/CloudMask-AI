"""
Download the trained CloudMask AI model from GitHub Releases.

Usage:
    python download_model.py

This script downloads 'cloudmask_unet_best.pt' from the latest GitHub Release
so you don't need to train the model yourself.
"""

import os
import sys
import urllib.request
import json

# ──────────── Configuration ────────────
REPO_OWNER = "tusharsingh-sde"
REPO_NAME = "CloudMask-AI"
MODEL_FILENAME = "cloudmask_unet_best.pt"

def get_latest_release_url():
    """Fetch the download URL for the model from the latest GitHub release."""
    api_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases/latest"
    
    print(f"📡 Fetching latest release from {REPO_OWNER}/{REPO_NAME}...")
    req = urllib.request.Request(api_url, headers={"Accept": "application/vnd.github.v3+json"})
    
    with urllib.request.urlopen(req) as response:
        release_data = json.loads(response.read().decode())
    
    for asset in release_data.get("assets", []):
        if asset["name"] == MODEL_FILENAME:
            return asset["browser_download_url"], release_data["tag_name"]
    
    return None, None

def download_with_progress(url, dest_path):
    """Download a file with a progress indicator."""
    print(f"⬇️  Downloading {MODEL_FILENAME}...")
    
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        block_size = 1024 * 1024  # 1MB chunks
        
        with open(dest_path, "wb") as f:
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
                    print(f"\r   {mb_down:.1f}/{mb_total:.1f} MB ({pct:.0f}%)", end="", flush=True)
    
    print(f"\n✅ Saved to: {dest_path}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dest_path = os.path.join(script_dir, MODEL_FILENAME)

    # Check if model already exists
    if os.path.exists(dest_path):
        print(f"✅ {MODEL_FILENAME} already exists ({os.path.getsize(dest_path) / (1024*1024):.1f} MB)")
        
        overwrite = input("   Download again? (y/N): ").strip().lower()
        if overwrite != "y":
            print("   Skipped.")
            return
    
    try:
        url, tag = get_latest_release_url()
    except Exception as e:
        print(f"❌ Failed to fetch release info: {e}")
        print(f"   Make sure {REPO_OWNER}/{REPO_NAME} has a public release with {MODEL_FILENAME}")
        sys.exit(1)
    
    if url is None:
        print(f"❌ No asset named '{MODEL_FILENAME}' found in the latest release.")
        print(f"   Visit: https://github.com/{REPO_OWNER}/{REPO_NAME}/releases")
        sys.exit(1)
    
    print(f"🏷️  Release: {tag}")
    
    try:
        download_with_progress(url, dest_path)
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        # Clean up partial file
        if os.path.exists(dest_path):
            os.remove(dest_path)
        sys.exit(1)

if __name__ == "__main__":
    main()
