# Written by Martin Fajcik <martin.fajcik@vut.cz>
#
import os
import pickle

from jsonlines import jsonlines
from tqdm import tqdm

from src.common.db import PassageDB
from ....common.utility import unicode_normalize

"""
Example of processed data
{"id": "Quercus_arkansana", 
"text": "Quercus arkansana -LRB- also called Arkansas oak -RRB- is a species of plant in the beech family . 
It is native to the southeastern United States -LRB- eastern Texas , southern Arkansas , Louisiana , Mississippi , Alabama , 
Georgia , and the Florida Panhandle -RRB- .   Quercus arkansana is a deciduous tree up to 15 meters -LRB- 50 feet -RRB- tall . 
Bark is black . Leaves are sometimes unlobed , sometimes with 2 or 3 shallow lobes .   It is threatened by use of its habitat 
for pine plantations , clearing of land , and diebacks that may be caused by drought . It is also susceptible to introgression 
with commoner species ", "lines": "0\tQuercus arkansana -LRB- also called Arkansas oak -RRB- is a species of plant in the beech
 family .\tQuercus\tQuercus\tplant\tplant\tbeech family\tFagaceae\tArkansas\tArkansas\n1\tIt is native to the southeastern 
 United States -LRB- eastern Texas , southern Arkansas , Louisiana , Mississippi , Alabama , Georgia , and the Florida Panhandle
  -RRB- .\tTexas\tTexas\tArkansas\tArkansas\tLouisiana\tLouisiana\tMississippi\tMississippi\tAlabama\tAlabama\tGeorgia\tGeorgia
   (U.S. state)\tFlorida Panhandle\tFlorida Panhandle\n2\t\n3\t\n4\tQuercus arkansana is a deciduous tree up to 15 meters
    -LRB- 50 feet -RRB- tall .\tQuercus\tQuercus\n5\tBark is black .\n6\tLeaves are sometimes unlobed , sometimes with 2
     or 3 shallow lobes .\n7\t\n8\t\n9\tIt is threatened by use of its habitat for pine plantations , clearing of land ,
      and diebacks that may be caused by drought .\n10\tIt is also susceptible to introgression with commoner species
      \tintrogression\tintrogression\n11\t"}
"""


def process_title_rev(title):
    # Borrowed from
    # https://github.com/dominiksinsaarland/domlin_fever/blob/9c05fc9949472f6a3eb31a816bf8d9dd680cae69/src_legacy/domlin/sentence_retrieval_part_2.py#L15
    title = title.replace("(", "-LRB-")
    title = title.replace(")", "-RRB-")
    title = title.replace(":", "-COLON-")
    title = title.replace(" ", "_")
    return title


def extract_entities(lines):
    """
    Extracts the non-empty sentences and their numbers of the "lines" field in
    a JSON object from a FEVER wiki-pages JSONL file.
    """
    entities = dict()

    sentence_index = 0
    for line in lines.split('\n'):
        tokens = line.split('\t')
        if not tokens[0].isnumeric() or int(tokens[0]) != sentence_index:
            # skip non-sentences, caused by unexpected \n's
            continue
        else:
            hyperlink_tokens = [t for t in tokens[2:] if len(t) > 0]
            if len(hyperlink_tokens) % 2 == 0:
                entities[int(tokens[0])] = [process_title_rev(e) for e in hyperlink_tokens[1::2]]
            sentence_index += 1

    return entities


if __name__ == "__main__":
    FILESPATH = ".index/FEVER_wikipages/wiki-pages"
    DB = ".index/FEVER_wikipages/feverwiki_blocks_500_wraparound.db"
    db = PassageDB(db_path=DB)
    all_titles = db.get_doc_titles()

    hyperlink_dictionary = dict()
    FILES = [f for f in os.listdir(FILESPATH) if f.endswith(".jsonl")]
    FILES = sorted(FILES, key=lambda x: int(x[:-len(".jsonl")][-3:]))
    itr = tqdm(FILES)
    for FILEPATH in itr:
        with jsonlines.open(os.path.join(FILESPATH, FILEPATH), "r") as reader:
            for e in reader:
                if '' in [e['id'], e['text'], e['lines']]:
                    continue
                e['id'] = unicode_normalize(e['id'])
                e['text'] = unicode_normalize(e['text'])
                e['lines'] = unicode_normalize(e['lines'])
                entities_per_sentence = extract_entities(e['lines'])
                assert e['id'] in all_titles
                for sent_id, entities in entities_per_sentence.items():
                    if len(entities) < 1:
                        continue
                    _filtered_entities = set()
                    for ent in entities:
                        if ent in all_titles:
                            _filtered_entities.add(ent)
                        elif ent.capitalize() in all_titles:
                            _filtered_entities.add(ent.capitalize())
                    entities = _filtered_entities

                    hyperlink_dictionary[(e['id'], sent_id)] = entities
    with open("fever_wiki_entities_per_sentence.pkl", "wb") as wf:
        pickle.dump(hyperlink_dictionary, wf)
