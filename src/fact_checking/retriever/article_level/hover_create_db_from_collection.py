# Written by Martin Fajcik <martin.fajcik@vut.cz>
#
import os
import sqlite3
import sys
from jsonlines import jsonlines
from stanza.server import CoreNLPClient
from tqdm import tqdm
from transformers import AutoTokenizer

from ....common.utility import unicode_normalize, mkdir


def read_lines(wikifile):
    return [line for line in wikifile.split("\n")]


def process_lines_encoding(sentences):
    result = []
    result_indices = []
    for i, line in enumerate(sentences):
        if line != "":
            result.append(unicode_normalize(line.replace("\n", "")))
            result_indices.append(i)
    return result, result_indices


def parse_to_sentences(text, parser):
    output = parser.annotate(text)
    sents = []
    for sent in output.sentence:
        sent = text[sent.characterOffsetBegin:sent.characterOffsetEnd]
        sents.append(sent)
    return sents


if __name__ == "__main__":
    FILEPATH = ".index/HoVer/collection.json"

    BLOCK_SIZE = int(sys.argv[1])
    wraparound = False
    reparse_sentences = True
    model_type = "microsoft/deberta-v3-base"
    print(f"BLOCK SIZE: {BLOCK_SIZE}")
    print(f"WRAPAROUND: {wraparound}")
    print(f"TOKENIZER_TYPE: {model_type}")
    print(f"REPARSE_SENTENCES: {reparse_sentences}")
    DB_PATH = f".index/HOVER_wikipages/hoverwiki_blocks_{BLOCK_SIZE}{'_wraparound' if wraparound else ''}{'_reparsed' if reparse_sentences else ''}.db"
    if reparse_sentences:
        parser = CoreNLPClient(
            annotators=['ssplit'],
            memory='4G',
            endpoint='http://localhost:9001',
            be_quiet=True,
            use_gpu=True)
        parser.start()
    mkdir(os.path.dirname(DB_PATH))
    tokenizer = AutoTokenizer.from_pretrained(model_type, cache_dir=".Transformers_cache")
    pbar = tqdm(total=5_233_330)
    _id = 0
    with sqlite3.connect(DB_PATH) as db:
        db_cursor = db.cursor()
        db_cursor.execute(
            "CREATE TABLE documents (id PRIMARY KEY, pid, document_title, lines, lines_i)")
        db.commit()
        with jsonlines.open(FILEPATH, "r") as reader:
            for e in reader:
                if '' in [e['pid'], e['title']] or e['text'] == []:
                    continue
                if reparse_sentences:
                    fulltext = "".join(e['text'])
                    fulltext = unicode_normalize(fulltext)
                    e['text'] = parse_to_sentences(fulltext, parser)
                processed_title = unicode_normalize(e['title'])

                non_empty_sentences, non_empty_sentence_ids = process_lines_encoding(e["text"])

                tokenized_title_len = len(tokenizer.tokenize(processed_title))
                accumulated_length = 0
                accumulated_lines = []
                accumulated_sentence_idx = []
                line_lengths = []
                for line, line_idx in zip(non_empty_sentences, non_empty_sentence_ids):
                    line_len = len(tokenizer.tokenize(line))
                    line_lengths.append(line_len)
                    # if this line is super big (larger than block size)
                    if tokenized_title_len + line_len > BLOCK_SIZE:
                        # write what was in the buffer, if anything
                        if accumulated_lines:
                            json_e = {
                                "id": _id,
                                "pid": e['pid'],
                                "document_title": e['title'],
                                "lines": "\n".join(accumulated_lines),
                                "lines_i": "|".join(accumulated_sentence_idx)
                            }
                            _id += 1
                            pbar.update()
                            db_cursor.execute(
                                "INSERT INTO documents VALUES (:id, :pid, :document_title, :lines, :lines_i)",
                                json_e)
                            accumulated_lines = []
                            accumulated_sentence_idx = []
                            accumulated_length = 0

                        # write out the big line
                        json_e = {
                            "id": _id,
                            "pid": e['pid'],
                            "document_title": e['title'],
                            "lines": line,
                            "lines_i": str(line_idx)
                        }
                        _id += 1
                        pbar.update()
                        db_cursor.execute(
                            "INSERT INTO documents VALUES (:id, :pid, :document_title, :lines, :lines_i)",
                            json_e)
                        continue

                    # if this line is not longer than block size
                    accumulated_length += line_len
                    # check if accumulating this line would cross block size
                    if tokenized_title_len + accumulated_length > BLOCK_SIZE:
                        # if it would, write out the accumulated lines so far except for this one
                        json_e = {
                            "id": _id,
                            "pid": e['pid'],
                            "document_title": e['title'],
                            "lines": "\n".join(accumulated_lines),
                            "lines_i": "|".join(accumulated_sentence_idx)
                        }
                        _id += 1
                        pbar.update()
                        db_cursor.execute(
                            "INSERT INTO documents VALUES (:id, :pid, :document_title, :lines, :lines_i)",
                            json_e)

                        # and initialize accumulated set with this line
                        accumulated_lines = [line]
                        accumulated_sentence_idx = [str(line_idx)]
                        accumulated_length = line_len
                    else:
                        # otherwise add line to accumulated lines
                        accumulated_lines.append(line)
                        accumulated_sentence_idx.append(str(line_idx))

                # Write out last sentences for document
                if accumulated_length > 0:
                    assert len(accumulated_sentence_idx) > 0 and len(accumulated_lines) > 0
                    if wraparound:
                        # in wrap around version
                        # we pad block with starting sequences again
                        last_block_len = accumulated_length
                        for i, l in enumerate(line_lengths):
                            # do not repeat same sentence twice
                            if str(i) == accumulated_sentence_idx[0] or \
                                    tokenized_title_len + last_block_len + l > BLOCK_SIZE:  # last block must be lesser than BLOCK_SIZE too
                                break
                            accumulated_lines.append(non_empty_sentences[i])
                            accumulated_sentence_idx.append(non_empty_sentence_ids[i])
                            last_block_len += l

                    json_e = {
                        "id": _id,
                        "pid": e['pid'],
                        "document_title": e['title'],
                        "lines": "\n".join(accumulated_lines),
                        "lines_i": "|".join(accumulated_sentence_idx)
                    }
                    _id += 1
                    pbar.update()
                    db_cursor.execute(
                        "INSERT INTO documents VALUES (:id, :pid, :document_title, :lines, :lines_i)",
                        json_e)

                    # and initialize accumulated set with this line
                    accumulated_lines = []
                    accumulated_sentence_idx = []
                    accumulated_length = 0
        db.commit()
        db_cursor.execute("CREATE INDEX titleretspeedup ON documents(document_title)")
        db_cursor.execute("CREATE INDEX pidretspeedup ON documents(pid)")
        if reparse_sentences:
            parser.stop()
