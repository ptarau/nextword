import sys
import os
import gutenbergpy.textget
import unicodedata


from senstore.segmenter import segment_file
from to_sent_file import file2sents


def to_ascii(text):
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")


def fetch(id_no, title):
    # This gets a book by its gutenberg id number
    os.makedirs("data/", exist_ok=True)
    raw_book = gutenbergpy.textget.get_text_by_id(id_no)  # with headers
    clean_book = gutenbergpy.textget.strip_headers(raw_book)  # without headers
    with open(f"data/{title}.txt", "w") as f:
        f.write(to_ascii(clean_book.decode("utf-8")))
    file2sents(f"data/{title}.txt", f"data/{title}_sents.txt")
    print(len(clean_book))
    print(len(raw_book))


def main():
    pass
    # fetch(28698, "crystal")
    # fetch(2600, "war_and_peace")
    # fetch(73425, "guermantes")
    # fetch(31516, "the_eyes")
    # fetch(68283, "cthulhu")
    # fetch(7849, "trial")
    # fetch(4300, "ulysses")
    # fetch(22566, "wizard_of_oz")
    # fetch(345, "dracula")


if __name__ == "__main__":
    main()
