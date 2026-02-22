import os
import glob
import gzip
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URLS = {
    "en": "http://storage.googleapis.com/books/ngrams/books/20200217/eng/eng-1-ngrams_exports.html",
    "es": "http://storage.googleapis.com/books/ngrams/books/20200217/spa/spa-1-ngrams_exports.html"
}


def download_and_process(lang: str, url: str):
    print(f"\n[INFO] Processing language: {lang}")

    out_dir = f"resources/{lang}"
    os.makedirs(out_dir, exist_ok=True)

    print("[INFO] Fetching list of files...")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    links = [li.a['href'] for li in soup.find_all("li")]

    print("[INFO] Downloading files...")
    for link in tqdm(links):
        filename = link.split("/")[-1]
        filepath = os.path.join(out_dir, filename)
        if not os.path.exists(filepath):
            r = requests.get(link)
            with open(filepath, "wb") as f:
                f.write(r.content)

    print("[INFO] Extracting frequencies...")
    matrix_path = os.path.join(out_dir, "matrix.txt")
    with open(matrix_path, "w", encoding="utf-8") as fout:
        for fname in tqdm(glob.glob(f"{out_dir}/*.gz")):
            with gzip.open(fname, "rt", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    word = parts[0]
                    freq = sum(int(x.split(",")[1]) for x in parts[1:])
                    fout.write(f"{word}\t{freq}\n")

    print("[INFO] Cleaning and aggregating...")
    with open(matrix_path, encoding="utf-8") as f:
        lines = f.readlines()

    records = [
        (str(l.split("\t")[0]).split("_")[0], int(l.split("\t")[1]))
        for l in lines if "\t" in l
    ]

    df = pd.DataFrame(records, columns=["word", "freq"])
    df_grouped = df.groupby("word").freq.sum().reset_index()

    output_file = os.path.join(out_dir, "word_freq_matrix.tsv.gz")
    df_grouped.to_csv(output_file, sep="\t", index=False, compression="gzip")

    print(f"[DONE] Saved word frequencies for {lang} → {output_file}")


if __name__ == "__main__":
    for lang, url in BASE_URLS.items():
        download_and_process(lang, url)

    print("\n[INFO] All done")
