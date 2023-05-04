# Written by Martin Fajcik <martin.fajcik@vut.cz>
#
from stanza.server import CoreNLPClient

with CoreNLPClient(
        annotators=['ssplit'],
        memory='4G',
        endpoint='http://localhost:9001',
        be_quiet=True,
        use_gpu=True) as corenlp:
    def parse_to_sentences(s):
        output = corenlp.annotate(s)
        para_sents = []
        for sent in output.sentence:
            sent = s[sent.characterOffsetBegin:sent.characterOffsetEnd]
            para_sents.append(sent)
        return para_sents


    TEXT = "Project Timberwind aimed to develop nuclear thermal rockets. Initial funding by the Strategic Defense Initiative (\"Star Wars\") from 1987 through 1991 totaled $139 million (then-year)." \
           " The proposed rocket was later expanded into a larger design after the project was transferred to the Air Force Space Nuclear Thermal Propulsion (SNTP) program and underwent an audit in 1992 due to concerns raised by Steven Aftergood." \
           " This special access program provided the motivation for starting the FAS Government Secrecy project." \
           " Convicted spy Stewart Nozette was found to be on the master access list for the TIMBER WIND project."
    print(parse_to_sentences(TEXT))
