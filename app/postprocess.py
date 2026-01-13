import re

MAX_LEN = 8
MIN_LEN = 3


def clean(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def smart_postprocess(detected: str) -> str:
    detected = clean(detected)

    if len(detected) < MIN_LEN:
        return detected

    chars = list(detected)

    for i, c in enumerate(chars):
        # --- ZONE 1: PREFIX ---
        if i < 2:
            if c == '0': chars[i] = 'O'
            elif c == '1': chars[i] = 'I'
            elif c == '2': chars[i] = 'Z'
            elif c == '5': chars[i] = 'S'
            elif c == '6': chars[i] = 'G'
            elif c == '8': chars[i] = 'B'
            elif c == '4': chars[i] = 'A'

        # --- ZONE 2: SUFFIX ---
        else:
            if c in {'O', 'Q', 'D'}:
                chars[i] = '0'

    result = "".join(chars)

    if len(result) > MAX_LEN:
        result = result[:MAX_LEN]

    return result
