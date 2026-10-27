"""Download the MaleCNS v1.0 tables the simulator needs (~570 MB) into data/raw/.

Data: FlyEM male CNS connectome v1.0, HHMI Janelia / Google Research, CC BY 4.0.
"""
import sys
import urllib.request
from pathlib import Path

BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
FILES = [
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",               # cell types, somata (14 MB)
    "body-neurotransmitters-male-cns-v1.0.feather",                     # predicted transmitters (43 MB)
    "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather", # neuron->neuron synapse counts (508 MB)
]
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = RAW / name
        if dest.exists():
            print(f"already have {name}")
            continue
        part = dest.with_name(dest.name + ".part")

        def progress(blocks, block_size, total):
            done = blocks * block_size / 1e6
            sys.stdout.write(f"\r{name}: {done:,.0f} / {total / 1e6:,.0f} MB")
            sys.stdout.flush()

        urllib.request.urlretrieve(BASE + name, part, progress)
        part.rename(dest)
        print()


if __name__ == "__main__":
    main()
