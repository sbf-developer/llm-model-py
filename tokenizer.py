# Converts text ↔ numbers (models only understand numbers)


class CharTokenizer:
    # Builds a vocab from your text — one number per unique character

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}  # string → int (for encode)
        self.itos = {i: ch for ch, i in self.stoi.items()}  # int → string (for decode)
        self.vocab_size = len(chars)

    def encode(self, s: str) -> list[int]:
        # text → list of token ids, e.g. "hi" → [7, 8]
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        # token ids → text, e.g. [7, 8] → "hi"
        return "".join(self.itos[i] for i in ids)

    @classmethod
    def from_maps(cls, stoi: dict, itos: dict):
        # reload saved vocab from a checkpoint (for generate.py)
        tok = cls.__new__(cls)
        tok.stoi = dict(stoi)
        tok.itos = {int(k): v for k, v in itos.items()}
        tok.vocab_size = len(stoi)
        return tok

    def merge_new_chars(self, text: str) -> bool:
        # add new characters from updated data without reshuffling existing ids
        added = False
        for ch in sorted(set(text)):
            if ch not in self.stoi:
                idx = self.vocab_size
                self.stoi[ch] = idx
                self.itos[idx] = ch
                self.vocab_size += 1
                added = True
        return added
