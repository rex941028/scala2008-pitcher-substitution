# -*- coding: utf-8 -*-
"""
Rebuild data/ from scratch. The repository already ships the result (data/processed/*.csv.gz and the
roster files), so this is only needed to reproduce it or to add another season.

  1. Retrosheet event files -> data/raw/YYYYeve           (rosters *.ROS are used by the models)
  2. Chadwick `cwevent`:
       Windows : Chadwick 0.10.0 binaries are downloaded to tools/chadwick
       macOS / Linux : `cwevent` must be on PATH (macOS: `brew install chadwick`;
                       Linux: build from https://github.com/chadwickbureau/chadwick)
  3. cwevent -> data/processed/events_YYYY.csv and .csv.gz
       standard fields 0-4,7-17,26-40,43,58-61,75-79,96; extended fields 0-2,4-8,13-15,17-19,45,55
  4. (optional, --paper) the paper PDF from the Internet Archive copy of D-Scholarship@Pitt

Run:  python src/data_prep.py [--years 2005 2006] [--paper]
"""
import argparse
import gzip
import os
import platform
import shutil
import subprocess
import urllib.request
import zipfile

from common import PROCESSED, RAW, ROOT

PAPER_URL = "https://web.archive.org/web/2015id_/http://d-scholarship.pitt.edu/22626/1/Scala_2008.pdf"
CHADWICK_URL = "https://github.com/chadwickbureau/chadwick/releases/download/v0.10.0/chadwick-0.10.0-win.zip"
RETRO_URL = "https://www.retrosheet.org/events/{y}eve.zip"
FIELDS = "0-4,7-17,26-40,43,58-61,75-79,96"
XFIELDS = "0-2,4-8,13-15,17-19,45,55"


def fetch(url, path):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(path, "wb") as fh:
        fh.write(r.read())


def find_cwevent():
    exe = shutil.which("cwevent")
    if exe:
        return exe
    if platform.system() == "Windows":
        tools = os.path.join(ROOT, "tools")
        fetch(CHADWICK_URL, os.path.join(tools, "chadwick.zip"))
        with zipfile.ZipFile(os.path.join(tools, "chadwick.zip")) as z:
            z.extractall(os.path.join(tools, "chadwick"))
        for cand in (os.path.join(tools, "chadwick", "chadwick", "cwevent.exe"),
                     os.path.join(tools, "chadwick", "cwevent.exe")):
            if os.path.exists(cand):
                return cand
    raise SystemExit("cwevent not found. macOS: `brew install chadwick`; Linux: build Chadwick from "
                     "https://github.com/chadwickbureau/chadwick and put cwevent on PATH.")


def gzip_copy(src):
    dst = src + ".gz"
    with open(src, "rb") as fi, open(dst, "wb") as raw, \
            gzip.GzipFile(filename=os.path.basename(src), mode="wb", fileobj=raw, compresslevel=9, mtime=0) as fo:
        shutil.copyfileobj(fi, fo)
    return dst


def main(years=(2005, 2006), paper=False):
    if paper:
        fetch(PAPER_URL, os.path.join(ROOT, "paper", "Scala_2008_ASEM.pdf"))
    exe = find_cwevent()
    os.makedirs(PROCESSED, exist_ok=True)
    for y in years:
        z = os.path.join(RAW, f"{y}eve.zip")
        fetch(RETRO_URL.format(y=y), z)
        d = os.path.join(RAW, f"{y}eve")
        with zipfile.ZipFile(z) as zz:
            zz.extractall(d)
        files = sorted(f for f in os.listdir(d) if f.upper().endswith((".EVA", ".EVN")))
        out = os.path.join(PROCESSED, f"events_{y}.csv")
        with open(out, "wb") as fh:
            subprocess.run([exe, "-q", "-n", "-y", str(y), "-f", FIELDS, "-x", XFIELDS] + files,
                           cwd=d, stdout=fh, check=True)
        print(y, "->", out, "and", gzip_copy(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2005, 2006])
    ap.add_argument("--paper", action="store_true", help="also download the paper PDF")
    a = ap.parse_args()
    main(tuple(a.years), a.paper)
