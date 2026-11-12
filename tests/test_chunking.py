from src.chunking import fixed_chunks, recursive_chunks


class TinyTokenizer:
    model_max_length = 8
    def encode(self, text, add_special_tokens=False, truncation=False):
        return text.split()
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


def test_fixed_respects_effective_limit_and_retains_tail():
    chunks = fixed_chunks("one two three four five six seven", "d", TinyTokenizer(), 128)
    assert [c.token_count for c in chunks] == [6, 1]


def test_recursive_keeps_content():
    chunks = recursive_chunks("one two. three four five.", "d", TinyTokenizer(), 4)
    assert " ".join(c.text for c in chunks).replace("  ", " ") == "one two. three four five."
