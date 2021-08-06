#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Readers for the pke module."""

import os
import sys
import json
import logging
import xml.etree.ElementTree as etree
import spacy
from spacy import language
from nltk import sent_tokenize, word_tokenize
from spacy.util import working_dir

from data_structures import Document

if sys.platform == 'win32':
    from eunjeon import Mecab
else:
    from konlpy.tag import Mecab

class Reader(object):
    def read(self, path):
        raise NotImplementedError


class MinimalCoreNLPReader(Reader):
    """Minimal CoreNLP XML Parser."""

    def __init__(self):
        self.parser = etree.XMLParser()

    def read(self, path, **kwargs):
        sentences = []
        tree = etree.parse(path, self.parser)
        for sentence in tree.iterfind('./document/sentences/sentence'):
            # get the character offsets
            starts = [int(u.text) for u in
                      sentence.iterfind("tokens/token/CharacterOffsetBegin")]
            ends = [int(u.text) for u in
                    sentence.iterfind("tokens/token/CharacterOffsetEnd")]
            sentences.append({
                "words": [u.text for u in
                          sentence.iterfind("tokens/token/word")],
                "lemmas": [u.text for u in
                           sentence.iterfind("tokens/token/lemma")],
                "POS": [u.text for u in sentence.iterfind("tokens/token/POS")],
                "char_offsets": [(starts[k], ends[k]) for k in
                                 range(len(starts))]
            })
            sentences[-1].update(sentence.attrib)

        doc = Document.from_sentences(sentences, input_file=path, **kwargs)

        return doc


# FIX
def fix_spacy_for_french(nlp):
    """Fixes https://github.com/boudinfl/pke/issues/115.
    For some special tokenisation cases, spacy do not assign a `pos` field.
    Taken from https://github.com/explosion/spaCy/issues/5179.
    """
    from spacy.symbols import TAG
    if nlp.lang != 'fr':
        # Only fix french model
        return nlp
    if '' not in [t.pos_ for t in nlp('est-ce')]:
        # If the bug does not happen do nothing
        return nlp
    rules = nlp.Defaults.tokenizer_exceptions

    for orth, token_dicts in rules.items():
        for token_dict in token_dicts:
            if TAG in token_dict:
                del token_dict[TAG]
    try:
        nlp.tokenizer = nlp.Defaults.create_tokenizer(nlp)  # this property assignment flushes the cache
    except Exception as e:
        # There was a problem fallback on using `pos = token.pos_ or token.tag_`
        ()
    return nlp


def list_linked_spacy_models():
    """ Read SPACY/data and return a list of link_name """
    spacy_data = os.path.join(spacy.info(silent=True)['Location'], 'data')
    linked = [d for d in os.listdir(spacy_data) if os.path.islink(os.path.join(spacy_data, d))]
    # linked = [os.path.join(spacy_data, d) for d in os.listdir(spacy_data)]
    # linked = {os.readlink(d): os.path.basename(d) for d in linked if os.path.islink(d)}
    return linked


def list_downloaded_spacy_models():
    """ Scan PYTHONPATH to find spacy models """
    models = []
    # For each directory in PYTHONPATH
    paths = [p for p in sys.path if os.path.isdir(p)]
    for site_package_dir in paths:
        # For each module
        modules = [os.path.join(site_package_dir, m) for m in os.listdir(site_package_dir)]
        modules = [m for m in modules if os.path.isdir(m)]
        for module_dir in modules:
            if 'meta.json' in os.listdir(module_dir):
                # Ensure the package we're in is a spacy model
                meta_path = os.path.join(module_dir, 'meta.json')
                with open(meta_path) as f:
                    meta = json.load(f)
                if meta.get('parent_package', '') == 'spacy':
                    models.append(module_dir)
    return models


def str2spacy(model):
    if int(spacy.__version__.split('.')[0]) < 3:
        downloaded_models = [os.path.basename(m) for m in list_downloaded_spacy_models()]
        links = list_linked_spacy_models()
    else:
        # As of spacy v3, links do not exist anymore and it is simpler to get a list of
        # downloaded models
        downloaded_models = list(spacy.info()['pipelines'])
        links = []
    filtered_downloaded = [m for m in downloaded_models if m[:2] == model]
    if model in downloaded_models + links:
        # Check whether `model` is the name of a model/link
        return model
    elif filtered_downloaded:
        # Check whether `model` is a lang code and corresponds to a downloaded model
        return filtered_downloaded[0]
    else:
        # Return asked model to have an informative error.
        return model


class RawTextReader(Reader):
    """Reader for raw text."""

    def __init__(self, language=None):
        """Constructor for RawTextReader.
        Args:
            language (str): language of text to process.
        """

        self.language = language

        if language is None:
            self.language = 'en'

    def read(self, text, **kwargs):
        """Read the input file and use spacy to pre-process.
        Spacy model selection: By default this function will load the spacy
        model that is closest to the `language` parameter ('fr' language will
        load the spacy model linked to 'fr' or any 'fr_core_web_*' available
        model). In order to select the model that will be used please provide a
        preloaded model via the `spacy_model` parameter, or link the model you
        wish to use to the corresponding language code
        `python3 -m spacy link spacy_model lang_code`.
        Args:
            text (str): raw text to pre-process.
            max_length (int): maximum number of characters in a single text for
                spacy (for spacy<3 compatibility, as of spacy v3 long texts
                should be splitted in smaller portions), default to
                1,000,000 characters (1mb).
            spacy_model (model): an already loaded spacy model.
        """

        spacy_model = kwargs.get('spacy_model', None)
        sentences = []

        if self.language != 'ko':
            if spacy_model is None:
                try:
                    spacy_model = spacy.load(str2spacy(self.language),
                                            disable=['ner', 'textcat', 'parser'])
                except OSError:
                    logging.warning('No spacy model for \'{}\' language.'.format(self.language))
                    logging.warning('Falling back to using english model. There might '
                        'be tokenization and postagging errors. A list of available '
                        'spacy model is available at https://spacy.io/models.'.format(
                            self.language))
                    spacy_model = spacy.load(str2spacy('en'),
                                            disable=['ner', 'textcat', 'parser'])

                if int(spacy.__version__.split('.')[0]) < 3:
                    sentencizer = spacy_model.create_pipe('sentencizer')
                else:
                    sentencizer = 'sentencizer'
                spacy_model.add_pipe(sentencizer)

                if 'max_length' in kwargs and kwargs['max_length']:
                    spacy_model.max_length = kwargs['max_length']

            spacy_model = fix_spacy_for_french(spacy_model)
            spacy_doc = spacy_model(text)

            # sentences = []
            for sentence_id, sentence in enumerate(spacy_doc.sents):
                sentences.append({
                    "words": [token.text for token in sentence],
                    "lemmas": [token.lemma_ for token in sentence],
                    # FIX : This is a fallback if `fix_spacy_for_french` does not work
                    "POS": [token.pos_ or token.tag_ for token in sentence],
                    "char_offsets": [(token.idx, token.idx + len(token.text))
                                    for token in sentence]
                })

        else:
            tag_set = {'NNP':'NOUN','NNG':'NOUN','NNB':'AUX','NNBC':'AUX','NR':'NUM','NP':'PROPN',
            'VV':'VERB','VA':'ADJ','VX':'PART','VCP':'PART','VCN':'PART',
            'MM':'DET','MAG':'ADV','MAJ':'CONJ','IC':'X',
            'JKS':'PART','JKC':'PART','JKG':'PART','JKO':'PART','JKB':'PART','JKV':'PART','JKQ':'PART','JC':'PART','JX':'PART',
            'EP':'PART','EF':'PART','EC':'PART','ETN':'PART','ETM':'PART',
            'XPN':'PART','XSN':'PART','XSV':'PART','XSA':'PART','XR':'PART',
            'SF':'PUNCT','SE':'PUNCT','SSO':'PUNCT','SSC':'PUNCT','SC':'PUNCT','SY':'PUNCT',
            'SH':'NOUN','SL':'NOUN','SN':'NUM'}

            spacy_doc = Mecab()
            text = text.replace('·',',').replace('+',',')
            for sentence_id, sentence in enumerate(sent_tokenize(text)):
                words_list, lemmas_list, pos_list, char_offsets_list = [], [], [], []
                offset = 0
                
                for word_idx, words in enumerate(word_tokenize(sentence)):
                    pos_by_word = spacy_doc.pos(words+' .')[:-1]

                    # Forward Inflect suspected
                    if pos_by_word[0][1] in ('NNP','NNG'):
                        if word_idx < len(word_tokenize(sentence)): # Inflected into multiple PoS
                            pos_by_context = spacy_doc.pos(' '.join(word_tokenize(sentence)[word_idx:word_idx+2]))[:2]  # PoS using following the word
                            if pos_by_context[0][0] != pos_by_word[0][0]:
                                spacy_pos = pos_by_context
                            else:
                                spacy_pos = pos_by_word
                        else:
                            spacy_pos = pos_by_word
                    else:
                        spacy_pos = pos_by_word

                    # print(words,spacy_pos)

                    # # Abnormal character detection
                    # if spacy_pos[0][1] == 'NNG':
                    #     try:
                    #         abnormal_offset = [words.find(pos[0]) for pos in spacy_pos].index(0)
                    #     except ValueError:
                    #         abnormal_offset = [words.find(pos[0][0]) for pos in spacy_pos].index(0)
                    #     spacy_pos = spacy_pos[abnormal_offset:]

                    # Backward Inflect suspected
                    # if len(spacy_pos[0][0])==1 and spacy_pos[0][1]=='NNG' and pos_list:
                    if spacy_pos[0][1]=='NNG' and pos_list:
                        if pos_list[-1]=='NNG':
                            pos_with_last = spacy_doc.pos(words_list[-1]+' '+words)
                            try:
                                last_offset = [(pos[1]=='NNB' and len(pos[0])==1) for pos in pos_with_last].index(True)
                                spacy_pos = pos_with_last[last_offset:]
                            except ValueError:
                                pass
                            # last_offset = [pos[0].find(spacy_pos[0][0]) for pos in pos_with_last].index(0)
                            # spacy_pos = pos_with_last[last_offset:]

                    # print(words,spacy_pos)

                    # Subjective Nouns (Noun + descriptives)
                    if len(spacy_pos)>1 and spacy_pos[0][1] in ('NNP','NNG','SL'):
                        try:
                            pos_offset = [pos[1] not in ('NNP','NNG','SL') for pos in spacy_pos].index(True)
                            aux_offset = sum([len(word[0]) for word in spacy_pos[:pos_offset]])
                            # aux_offset = words.find(spacy_pos[pos_offset-1][0][-1])+1
                            # print(words,pos_offset,aux_offset)
                        except ValueError:
                            pos_offset = 1
                            aux_offset = len(words)
                    else:
                        pos_offset = 1
                        aux_offset = len(words)

                    words_list.append(words[:aux_offset])
                    # lemmas_list.append(spacy_pos[0][0])

                    # Number
                    if len(spacy_pos)>1 and spacy_pos[0][1]=='SN': #and spacy_pos[-1][1]=='SN':
                        # words_list.append(words)
                        lemmas_list.append(words)
                    else:
                        # words_list.append(words[:aux_offset])
                        lemmas_list.append(spacy_pos[0][0])

                    # if len(words)==1 and spacy_pos[0][1].startswith('N'):   # Dependent Nouns
                    if len(spacy_pos[0][0])==1 and spacy_pos[0][1].startswith('N'):   # Dependent Nouns
                        pos_list.append('NNB')
                    elif len(spacy_pos)>1 and spacy_pos[0][1]=='SN': #and spacy_pos[-1][1]=='SN': # Number
                        pos_list.append('SN')
                    else:
                        pos_list.append(spacy_pos[0][1].replace('SL','NNP'))

                    if aux_offset < len(words):
                        words_list.append(words[aux_offset:])
                        lemmas_list.append(spacy_pos[pos_offset][0])
                        pos_list.append(spacy_pos[pos_offset][1].replace('SL','NNP'))


                    # words_list.append(words)
                    # lemmas_list.append(spacy_pos[0][0])
                    # if len(words)==1 and spacy_pos[0][1].startswith('N'):
                    #     pos_list.append('NNB')
                    # else:
                    #     pos_list.append(spacy_pos[0][1].replace('SL','NNP'))

                    # for idx, (word, pos) in enumerate(spacy_pos):
                    #     # print(idx,(word,pos))
                    #     if idx == 0:    # First element is added automatically
                    #         words_list.append(word)
                    #         lemmas_list.append(word)
                    #         if len(word)==1 and not pos.startswith('S'):
                    #             tag = 'NNB' # Arbitrary tagging
                    #         elif pos == 'SL':
                    #             tag = 'NNP'
                    #         else:
                    #             tag = pos

                    #         if idx == len(spacy_pos)-1:
                    #             pos_list.append(tag)

                    #     elif pos in ('NNP','NNG'):  # Compound Nouns
                    #         words_list[-1] += word
                    #         lemmas_list[-1] += word

                    #         if idx == len(spacy_pos)-1:
                    #             pos_list.append(tag)

                    #     else:
                    #         pos_list.append(tag)
                    #         if tag.startswith('VV'):
                    #             lemmas_list[-1] += '다'
                    #             words_list[-1] = words
                    #         elif tag in ('NNP','NNG'):
                    #             words_list.append(word)
                    #             lemmas_list.append(word)
                    #             pos_list.append(pos)
                    #         else:
                    #             words_list[-1] = words
                    #             lemmas_list[-1] = words
                    #         break
                
                for word in words_list:
                    if char_offsets_list:
                        offset = char_offsets_list[-1][1] + 1
                    char_offsets_list.append((offset,offset+len(word)))

                pos_list = [tag_set[pos] if tag_set.get(pos) else 'X' for pos in pos_list]

                sentences.append({
                    "words": [word for word in words_list],
                    "lemmas": [lemma for lemma in lemmas_list],
                    "POS": [pos for pos in pos_list],
                    "char_offsets": [offset for offset in char_offsets_list]
                })

                # print(sentences)

        doc = Document.from_sentences(
            sentences, input_file=kwargs.get('input_file', None), **kwargs)

        return doc