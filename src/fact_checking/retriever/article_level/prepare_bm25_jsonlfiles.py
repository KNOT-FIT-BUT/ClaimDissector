# Written by Martin Fajcik <martin.fajcik@vut.cz>
#
import os

from jsonlines import jsonlines

if __name__ == "__main__":
    left_bracket_replacement = "-LRB-"
    right_bracket_replacment = "-RRB-"
    title_space_replacement = "_"

    FILESPATH = ".index/FEVER_wikipages/wiki-pages"

    PREP_PATH = ".index/FEVER_wikipages/prep_anserini/wiki.jsonl"
    FILES = [f for f in os.listdir(FILESPATH) if f.endswith(".jsonl")]
    FILES = sorted(FILES, key=lambda x: int(x[:-len(".jsonl")][-3:]))
    _id = 0
    with jsonlines.open(PREP_PATH, mode="w") as writer:
        for FILEPATH in FILES:
            with jsonlines.open(os.path.join(FILESPATH, FILEPATH), "r") as reader:
                for e in reader:
                    if '' in [e['id'], e['text'], e['lines']]:
                        continue
                    processed_title = e['id'] \
                        .replace(left_bracket_replacement, "(") \
                        .replace(right_bracket_replacment, ")") \
                        .replace(title_space_replacement, " ")
                    processed_text = e['text'] \
                        .replace(left_bracket_replacement, "(") \
                        .replace(right_bracket_replacment, ")")

                    json_e = {
                        "id": _id,
                        "contents": processed_title + " \n" + processed_text
                    }
                    _id += 1
                    writer.write(json_e)
