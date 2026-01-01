import sys
import os
import gutenbergpy.textget
import unicodedata


from senstore.segmenter import segment_file
from to_sent_file import file2sents


def to_ascii(text):
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")


def main(id_no, title):
    # This gets a book by its gutenberg id number
    os.makedirs("data/", exist_ok=True)
    raw_book = gutenbergpy.textget.get_text_by_id(id_no)  # with headers
    clean_book = gutenbergpy.textget.strip_headers(raw_book)  # without headers
    with open(f"data/{title}.txt", "w") as f:
        f.write(to_ascii(clean_book.decode("utf-8")))
    file2sents(f"data/{title}.txt", f"data/{title}_sents.txt")
    print(len(clean_book))
    print(len(raw_book))


if __name__ == "__main__":
    pass
    main(28698, "crystal")
    main(2600, "war_and_peace")
    main(73425, "guermantes")
    main(31516, "the_eyes")
