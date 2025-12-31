from typing import Dict

CHINESE_DIGITS: Dict[str, int] = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4, 
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "两": 2,
}
UNIT_MAP: Dict[str, int] = {"十": 10, "百": 100, "千": 1000}
SECTION_UNIT_MAP: Dict[str, int] = {"万": 10_000, "亿": 100_000_000}

def chinese_numeral_to_str(text: str) -> str:
    """Convert Chinese numerals (e.g., '五百八十六') into arabic string."""
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Empty Chinese numeral.")
    if cleaned.isdigit():
        return cleaned

    total = 0
    section = 0
    number = 0

    for char in cleaned:
        if char in CHINESE_DIGITS:
            number = CHINESE_DIGITS[char]
        elif char in UNIT_MAP:
            unit = UNIT_MAP[char]
            if number == 0:
                number = 1
            section += number * unit
            number = 0
        elif char in SECTION_UNIT_MAP:
            section += number
            if section == 0:
                section = 1
            total += section * SECTION_UNIT_MAP[char]
            section = 0
            number = 0
        else:
            # Skip unsupported characters silently.
            continue

    total += section + number
    return str(total)

def arabic_to_chinese(num: int) -> str:
    """Convert arabic integers into Chinese numerals (supports up to 4 digits)."""
    if num == 0:
        return "零"

    digits = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
    units = ["", "十", "百", "千"]
    result = ""
    num_str = str(num)
    length = len(num_str)
    zero_flag = False

    for idx, char in enumerate(num_str):
        digit = int(char)
        pos = length - idx - 1
        if digit == 0:
            zero_flag = True
            continue

        if zero_flag:
            result += "零"
            zero_flag = False

        if digit == 1 and pos == 1 and result == "":
            result += units[pos]
        else:
            result += digits[digit] + units[pos]

    return result or digits[0]
