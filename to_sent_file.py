#!/usr/bin/env python3
import sys
from senstore.segmenter import segment_file

def file2sents(input_path: str, output_path: str) -> None:
    sents = segment_file(input_path)
    with open(output_path, "w", encoding="utf-8") as f:
        for sent in sents:
            f.write(sent + "\n")

def main(argv):
    if len(argv) != 3:
        print("Usage: file2sents_cli.py INPUT_PATH OUTPUT_PATH", file=sys.stderr)
        return 2
    file2sents(argv[1], argv[2])
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
