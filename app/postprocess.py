import re


class PlatePostProcessor:
    def process(self, text: str) -> str:
        text = re.sub(r'[^A-Z0-9]', '', text.upper())

        # Remove garbage prefixes
        while len(text) > 7 and text[0] in ['I', 'E', 'U', 'L', 'F', '1']:
            text = text[1:]

        if len(text) < 3:
            return text

        chars = list(text)

        first_map = {
            '0': 'O', '1': 'I', '2': 'Z', '4': 'A',
            '5': 'S', '8': 'B', 'F': 'S', 'D': 'O'
        }

        rest_map = {
            'O': '0', 'Q': '0', 'L': '1', 'B': '8'
        }

        for i in range(min(2, len(chars))):
            if chars[i] in first_map:
                chars[i] = first_map[chars[i]]

        for i in range(2, len(chars)):
            if chars[i] in rest_map:
                chars[i] = rest_map[chars[i]]

        return "".join(chars[:8])
