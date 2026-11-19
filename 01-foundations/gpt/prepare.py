"""Download the tiny-Shakespeare corpus (~1 MB) into data/."""
from pathlib import Path
import urllib.request

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUT = Path(__file__).parent / "data" / "shakespeare.txt"

if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    if OUT.exists():
        print(f"already have {OUT}")
    else:
        urllib.request.urlretrieve(URL, OUT)
        print(f"downloaded {OUT}")
    text = OUT.read_text()
    print(f"{len(text):,} characters, {len(set(text))} unique")
