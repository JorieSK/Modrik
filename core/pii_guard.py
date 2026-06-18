import re

# Saudi National ID (10 digits, starts with 1) or Iqama (starts with 2)
_SAUDI_ID = re.compile(r'\b[12]\d{9}\b')

# Saudi IBAN: SA + 22 digits
_IBAN = re.compile(r'\bSA\d{22}\b', re.IGNORECASE)

_PATTERNS = [
    (_SAUDI_ID, "[رقم هوية]"),
    (_IBAN,     "[رقم IBAN]"),
]


def redact_pii(text: str) -> tuple[str, int]:
    """Redact Saudi IDs and IBANs. Returns (redacted_text, count_redacted)."""
    count = 0
    for pattern, replacement in _PATTERNS:
        matches = pattern.findall(text)
        count += len(matches)
        text = pattern.sub(replacement, text)
    return text, count
