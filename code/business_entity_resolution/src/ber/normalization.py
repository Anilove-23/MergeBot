import re
import unicodedata

from tqdm.auto import tqdm

SUFFIXES = re.compile(r"\b(?:incorporated|inc|corporation|corp|limited|ltd|llc|plc|private|pvt|gmbh|sarl|sas)\b")


def normalize(value):
    text = unicodedata.normalize("NFKC", value).casefold().replace("&", " and ")
    return " ".join(re.sub(r"[^\w\s]|_", " ", text).split())


def fold_accents(value):
    return "".join(c for c in unicodedata.normalize("NFKD", value) if not unicodedata.combining(c))


def address_tokens(value):
    return frozenset(re.findall(r"\b\w*\d\w*\b", value))


def preprocess(frame):
    frame = frame.copy()
    for raw, short in tqdm([("business_name", "name"), ("business_address", "address")],
                           desc="Normalize fields", leave=False):
        frame[short] = frame[raw].map(normalize)
        frame[f"{short}_folded"] = frame[short].map(fold_accents)
    frame["name_core"] = frame.name.map(lambda s: " ".join(sorted(SUFFIXES.sub(" ", s).split())))
    frame["country_norm"] = frame.country.map(normalize)
    frame["numbers"] = frame.address.map(address_tokens)
    return frame.reset_index(drop=True)
