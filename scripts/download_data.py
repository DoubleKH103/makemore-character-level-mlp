from pathlib import Path
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT = DATA_DIR / "names.txt"

DATA_DIR.mkdir(parents=True, exist_ok=True)

if OUTPUT.exists():
    print(f"Dataset already exists: {OUTPUT}")
else:
    print("Downloading names.txt...")
    with urlopen(URL) as response:
        OUTPUT.write_bytes(response.read())
    print(f"Saved dataset to {OUTPUT}")
