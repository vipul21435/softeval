"""Download ChaosNLI into a local folder. Nothing fetched here is committed to this repository.

    python examples/chaosnli/fetch.py DATA_DIR

ChaosNLI (Nie, Zhou and Bansal, 2020) is licensed CC BY-NC 4.0. The archive link in the authors'
repository no longer serves the file, so the three JSONL files are taken from a public
preservation mirror and checked against the SHA-256 sums that mirror publishes. The small
random-baseline prediction files come from the authors' repository. The archive of saved model
predictions is requested too; when it cannot be downloaded the script says so and carries on.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

MIRROR = "https://huggingface.co/datasets/earino/chaosnli/resolve/main/raw/"
AUTHORS_REPO = "https://raw.githubusercontent.com/easonnie/ChaosNLI/master/data/prediction_samples/"
MODEL_PREDICTIONS = "https://www.dropbox.com/s/qy7uk6ajm5x6dl6/model_predictions.zip?dl=1"

DATA_FILES = {
    "chaosNLI_snli.jsonl": "99f9015ddda7d85f66a087452bc30d53974314fe27e7d589e2f41ad44bd509c1",
    "chaosNLI_mnli_m.jsonl": "8eb49b589488e7b1a9ec95fb8a864bd50f0509b4f808eed0aed3f8167617664d",
    "chaosNLI_alphanli.jsonl": "f55817ecc11a02ba3030cd7d2c567d23843456809020723abad53f4fde024c9e",
}
BASELINE_FILES = (
    "snli_random_baseline.json",
    "mnli_random_baseline.json",
    "abdnli_random_baseline.json",
)


def download(url: str, target: Path) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "softeval-example"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    target.write_bytes(data)
    return data


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    out = Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for name, expected in DATA_FILES.items():
        data = download(MIRROR + name, out / name)
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected:
            print(f"{name}: SHA-256 {digest} does not match the published {expected}")
            return 1
        print(f"{name}: {len(data)} bytes, SHA-256 matches")
    for name in BASELINE_FILES:
        data = download(AUTHORS_REPO + name, out / name)
        print(f"{name}: {len(data)} bytes")
    try:
        data = download(MODEL_PREDICTIONS, out / "model_predictions.zip")
    except (urllib.error.URLError, OSError) as exc:
        data = b""
        print(f"model_predictions.zip: not available ({exc})")
    if data[:2] == b"PK":
        print(f"model_predictions.zip: {len(data)} bytes; unzip it next to the JSONL files")
    else:
        (out / "model_predictions.zip").unlink(missing_ok=True)
        print("model_predictions.zip: the link does not serve an archive any more; skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
