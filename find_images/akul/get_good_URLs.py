from concurrent.futures import ThreadPoolExecutor
thread_executor = ThreadPoolExecutor(max_workers=20)
import functools
import asyncio
import aiohttp
import nest_asyncio
from tqdm import tqdm
import aiohttp
import asyncio

# Enable nested async in environments like Jupyter or ChatGPT
nest_asyncio.apply()

# Load the raw lines from the file
with open("top_lab_images_10k.csv", "r", encoding='utf8') as f:
    lines = f.readlines()

# Extract URLs from lines (assuming format: score url caption)
urls = []
for line in lines[1:]:  # Skip header
    parts = line.strip().split()
    if len(parts) >= 2:
        urls.append(parts[1])

# with open("all_urls.txt", "r", encoding="utf-8") as f:
#     lines = f.readlines()
# urls = [url for url in lines if url.startswith("http") and "." in url]
# urls = urls[:1000]
print(urls[:5])


# Set concurrency and timeout
CONCURRENT_REQUESTS = 50
TIMEOUT = 5  # seconds
import aiohttp
import asyncio
import requests

REJECT_LOG = "rejected_urls.log"

async def check_url(session, url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "*/*"
    }

    url = url.strip().strip('"').strip("'")

    # Try aiohttp first
    try:
        async with session.get(url, timeout=5, headers=headers, allow_redirects=True) as resp:
            if 200 <= resp.status < 400:
                content_type = resp.headers.get("Content-Type", "").lower()
                if "image" in content_type or url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff')):
                    return True, url
                else:
                    reason = f"Rejected content-type: {content_type}"
            else:
                reason = f"Bad status code: {resp.status}"
    except Exception as e:
        reason = f"[aiohttp] Exception: {type(e).__name__} - {str(e)}"
    else:
        # If aiohttp runs but doesn't return a valid image
        reason = "[aiohttp] Failed to verify image content"

    # Fallback to requests
    loop = asyncio.get_running_loop()
    try:
        r = await loop.run_in_executor(
            thread_executor,
            functools.partial(requests.get, url, headers=headers, timeout=5)
        )

        if 200 <= r.status_code < 400:
            content_type = r.headers.get("Content-Type", "").lower()
            if "image" in content_type or url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff')):
                return True, url
            else:
                reason = f"[requests fallback] Rejected content-type: {content_type}"
        else:
            reason = f"[requests fallback] Bad status code: {r.status_code}"
    except Exception as e:
        reason = f"[requests fallback] Exception: {type(e).__name__} - {str(e)}"

    # If both fail, log and return failure
    with open(REJECT_LOG, "a", encoding="utf-8") as log:
        log.write(f"{url} - {reason}\n")

    return False, None





# Main function to validate URLs concurrently
async def validate_urls(url_list):
    valid = []
    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [check_url(session, url) for url in url_list]
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            success, result = await future
            if success:
                valid.append(result)

    return valid

# Run the validation
valid_image_urls = asyncio.get_event_loop().run_until_complete(validate_urls(urls))

# Write valid URLs to file
with open("valid_lab_image_urls.txt", "w") as f:
    for url in valid_image_urls:
        f.write(url + "\n")
