import gutenbergpy.textget
import unicodedata

import sys
from senstore.segmenter import segment_file


def file2sents(input_path: str, output_path: str) -> None:
    sents = segment_file(input_path)
    with open(output_path, "w", encoding="utf-8") as f:
        for sent in sents:
            f.write(sent + "\n")


def to_ascii(text):
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")


def main(id_no, title):
    # This gets a book by its gutenberg id number
    raw_book = gutenbergpy.textget.get_text_by_id(id_no)  # with headers
    clean_book = gutenbergpy.textget.strip_headers(raw_book)  # without headers
    with open(f"data/{title}.txt", "w") as f:
        f.write(to_ascii(clean_book.decode("utf-8")))
    file2sents(f"data/{title}.txt", f"data/{title}_sents.txt")
    print(len(clean_book))
    print(len(raw_book))


if __name__ == "__main__":
    pass
    # main(28698, "crystal")
    # main(2600, "war_and_peace")
    # main(73425, "guermantes")
    main(31516, "theyes")
