import gzip
import shutil
from pathlib import Path

import fire
import requests


def download_json_dataset(target_path):
    url = "https://production-media.paperswithcode.com/about/papers-with-abstracts.json.gz"
    r = requests.get(url)
    with open(Path(target_path) / "papers-with-abstracts.json.gz", "wb") as outfile:
        outfile.write(r.content)

    with gzip.open(Path(target_path) / "papers-with-abstracts.json.gz", "rb") as f_in:
        with open(Path(target_path) / "papers-with-abstracts.json", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


if __name__ == "__main__":
    fire.Fire(download_json_dataset)
