import os
import csv
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import mimetypes

CSV_PATH = "train_split.csv"      # your CSV file (columns: image_id,url)
OUTPUT_DIR = "train_images"       # folder to save images
FAILED_CSV_PATH = "failed_urls.csv"
MAX_WORKERS = 32
TIMEOUT = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_extension(url, content_type):
    if content_type:
        ext = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if ext:
            return ext
    ext = os.path.splitext(urlparse(url).path)[1]
    return ext if ext else ".jpg"

def download_image(image_id, url):
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        ext = get_extension(url, resp.headers.get("Content-Type"))
        file_path = os.path.join(OUTPUT_DIR, f"{image_id}{ext}")
        with open(file_path, "wb") as f:
            f.write(resp.content)
        return ("ok", image_id)
    except Exception as e:
        # Return enough info so we can save it later
        return ("fail", image_id, url, str(e))

def main():
    # Read CSV (skips header if present)
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        first_row = next(reader)
        if first_row and first_row[0].lower().startswith("image"):
            rows = [(r[0], r[1]) for r in reader]
        else:
            rows = [(first_row[0], first_row[1])] + [(r[0], r[1]) for r in reader]

    failures = []
    total = len(rows)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(download_image, img_id, url) for img_id, url in rows]
        for future in as_completed(futures):
            result = future.result()
            if result[0] == "ok":
                _, image_id = result
                print(f"Downloaded {image_id}")
            else:
                _, image_id, url, err = result
                print(f"Failed {image_id}: {err}")
                failures.append((image_id, url, err))

    # Write failures to CSV
    if failures:
        with open(FAILED_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_id", "url", "error"])
            writer.writerows(failures)
        print(f"\nSaved {len(failures)} failed URLs to {FAILED_CSV_PATH}")
    else:
        print("\nNo failures 🎉")

    print(f"Completed. {total - len(failures)}/{total} downloaded successfully.")

if __name__ == "__main__":
    main()
