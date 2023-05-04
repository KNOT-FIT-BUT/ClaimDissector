# Written by Martin Fajcik <martin.fajcik@vut.cz>
#
from transformers import AutoTokenizer, PreTrainedTokenizer, SLOW_TO_FAST_CONVERTERS

from .convert_slow_tokenizer import DebertaV2Converter


def init_tokenizer(config, decoder_prefix_extra_special_tokens=None,
                   tokenizer_class=AutoTokenizer) -> PreTrainedTokenizer:
    if str(tokenizer_class) not in SLOW_TO_FAST_CONVERTERS:
        tokenizer_class_name = tokenizer_class.__name__
        # Patch converter for DebertaV2
        if tokenizer_class_name.startswith("DebertaV2Tokenizer"):
            SLOW_TO_FAST_CONVERTERS["DebertaV2Tokenizer"] = DebertaV2Converter

    """
    Creates tokenizer and add special tokens into it
    """
    reader_tokenizer = tokenizer_class.from_pretrained(config["verifier_tokenizer_type"],
                                                       cache_dir=config["transformers_cache"])
    reader_tokenizer.claim_special_token = '<claim>'
    reader_tokenizer.title_special_token = '<title>'
    reader_tokenizer.passage_special_token = '<passage>'
    reader_tokenizer.sentence_special_token = '<sentence>'

    if decoder_prefix_extra_special_tokens is not None:
        reader_tokenizer.decoder_prefix = [f"<decoder_prefixlm_token_{i}>" for i in
                                           range(decoder_prefix_extra_special_tokens)]
    else:
        reader_tokenizer.decoder_prefix = []
    reader_tokenizer.add_tokens(
        [reader_tokenizer.claim_special_token,
         reader_tokenizer.passage_special_token,
         reader_tokenizer.title_special_token,
         reader_tokenizer.sentence_special_token] +
        reader_tokenizer.decoder_prefix,
        special_tokens=True)

    verified_tokenized = reader_tokenizer.tokenize('true')
    refuted_tokenized = reader_tokenizer.tokenize('false')
    no_info_tokenized = reader_tokenizer.tokenize('unknown')

    # make sure all are single token expressions
    assert len(verified_tokenized) == len(refuted_tokenized) == len(no_info_tokenized) == 1

    reader_tokenizer.verified_token = verified_tokenized[0]
    reader_tokenizer.refuted_token = refuted_tokenized[0]
    reader_tokenizer.noinfo_token = no_info_tokenized[0]
    return reader_tokenizer
