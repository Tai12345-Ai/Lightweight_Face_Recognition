#!/usr/bin/env python3
"""Download a public OneDrive file through the Microsoft shares API.

For a shared folder, pass --remote-path with the file path inside that folder,
for example:

    python scripts/download_onedrive_checkpoint.py \
        --share-url "https://1drv.ms/u/s!..." \
        --remote-path "ms1mv3_arcface_r18_fp16/backbone.pth" \
        --output "/kaggle/working/backbone.pth"
"""

import argparse
import base64
import os
import sys
import urllib.parse
import urllib.request


def share_id_from_url(url):
    encoded = base64.urlsafe_b64encode(url.encode("utf-8")).decode("utf-8")
    return "u!" + encoded.rstrip("=")


def build_download_url(share_url, remote_path):
    share_id = share_id_from_url(share_url)
    base = f"https://api.onedrive.com/v1.0/shares/{share_id}"
    if remote_path:
        quoted_path = urllib.parse.quote(remote_path.strip("/"))
        return f"{base}/root:/{quoted_path}:/content"
    return f"{base}/root/content"


def download(url, output):
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    tmp_output = output + ".tmp"
    with urllib.request.urlopen(url) as response, open(tmp_output, "wb") as f:
        total = response.headers.get("Content-Length")
        total = int(total) if total else None
        read = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            read += len(chunk)
            if total:
                pct = read * 100.0 / total
                print(f"\rDownloaded {read / (1024 ** 2):.1f} MB ({pct:.1f}%)", end="")
            else:
                print(f"\rDownloaded {read / (1024 ** 2):.1f} MB", end="")
            sys.stdout.flush()
    print()
    os.replace(tmp_output, output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--share-url", required=True)
    parser.add_argument("--remote-path", default="")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    download_url = build_download_url(args.share_url, args.remote_path)
    print(f"Downloading from: {download_url}")
    download(download_url, args.output)
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()

