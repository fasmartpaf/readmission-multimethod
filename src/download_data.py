"""Download the UCI Diabetes 130-US Hospitals dataset (1999-2008).

Source: UCI Machine Learning Repository (id=296), DOI: 10.24432/C5230J
Reference: Strack et al. (2014), DOI: 10.1155/2014/781670

Produces: data/diabetic_data.csv
"""
import os, socket, urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
URL = "https://archive.ics.uci.edu/static/public/296/data.csv"
OUT = os.path.join(DATA_DIR, "diabetic_data.csv")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(OUT) and os.path.getsize(OUT) > 1_000_000:
        print(f"Already present: {OUT}")
        return
    print("Downloading UCI Diabetes 130-US Hospitals dataset ...")
    socket.setdefaulttimeout(120)
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(OUT, "wb") as f:
        total = 0
        while True:
            chunk = r.read(65536)
            if not chunk:
                break
            f.write(chunk); total += len(chunk)
    print(f"Saved {total/1e6:.1f} MB to {OUT}")


if __name__ == "__main__":
    main()
