# Written by Martin Fajcik <martin.fajcik@vut.cz>
#
import os
import sqlite3

from jsonlines import jsonlines

from ....common.utility import unicode_normalize as normalize


def read_lines(wikifile):
    return [line for line in wikifile.split("\n")]


def process_lines_encoding(wikifile):
    return [normalize(line.split('\t')[1])
                .replace(left_bracket_replacement, "(")
                .replace(right_bracket_replacment, ")")
            for line in read_lines(wikifile)
            if len(line.split('\t')) > 1 and normalize(line.split('\t')[1]) != ""]


if __name__ == "__main__":
    left_bracket_replacement = "-LRB-"
    right_bracket_replacment = "-RRB-"
    title_space_replacement = "_"

    FILESPATH = ".index/FEVER_wikipages/wiki-pages"

    DB_PATH = ".index/FEVER_wikipages/feverwiki.db"
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    FILES = [f for f in os.listdir(FILESPATH) if f.endswith(".jsonl")]
    FILES = sorted(FILES, key=lambda x: int(x[:-len(".jsonl")][-3:]))
    _id = 0
    with sqlite3.connect(DB_PATH) as db:
        db_cursor = db.cursor()
        db_cursor.execute(
            "CREATE TABLE documents (id PRIMARY KEY, document_title, document_context, lines)")
        db.commit()
        for FILEPATH in FILES:
            with jsonlines.open(os.path.join(FILESPATH, FILEPATH), "r") as reader:
                for e in reader:
                    if '' in [e['id'], e['text'], e['lines']]:
                        continue
                    # processed_title = e['id'] \
                    #     .replace(left_bracket_replacement, "(") \
                    #     .replace(right_bracket_replacment, ")") \
                    #     .replace(title_space_replacement, " ")
                    processed_text = e['text'] \
                        .replace(left_bracket_replacement, "(") \
                        .replace(right_bracket_replacment, ")")
                    processed_lines = process_lines_encoding(e['lines'])

                    json_e = {
                        "id": _id,
                        "document_title": e['id'],
                        "document_context": processed_text,
                        "lines": "\n".join(processed_lines)
                    }
                    _id += 1
                    db_cursor.execute(
                        "INSERT INTO documents VALUES (:id, :document_title, :document_context, :lines)",
                        json_e)
        db.commit()
