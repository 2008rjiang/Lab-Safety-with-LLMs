"""
python normalize_url.py valid9.txt valid9.txt
"""


from urllib.parse import urlsplit, urlunsplit

def normalize_url(u):
    parts = urlsplit(u)
    # drop query and fragments
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
