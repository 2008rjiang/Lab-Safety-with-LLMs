import requests

url = "https://www.prepareforsuccess.org.uk/supportfiles/ukcisa_studying_science_engineering_or_technology.jpg"
headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "*/*"
}

resp = requests.get(url, headers=headers)
print(resp.status_code)
print(resp.headers.get("Content-Type"))
