from stanfordcorenlp import StanfordCoreNLP


def _consume_os(tags):
    ## reference: https://github.com/explosion/spaCy/blob/c7d53348d7c0474852dc5ebe5794f2816ef7eb01/spacy/gold.pyx
    while tags and tags[0] == 'O':
        yield tags.pop(0)


def _consume_ent(tags):
    if not tags:
        return []
    tag = tags.pop(0)
    target_in = 'I' + tag[1:]
    target_last = 'L' + tag[1:]
    length = 1
    while tags and tags[0] in {target_in, target_last}:
        length += 1
        tags.pop(0)
    label = tag[2:]
    if length == 1:
        return ['U-' + label]
    else:
        start = 'B-' + label
        end = 'L-' + label
        middle = ['I-%s' % label for _ in range(1, length - 1)]
        return [start] + middle + [end]
    
def iob_to_biluo(tags):
    out = []
    curr_label = None
    tags = list(tags)
    while tags:
        out.extend(_consume_os(tags))
        out.extend(_consume_ent(tags))
    return out

def read_conll_ner(input_path):
    ## reference: https://github.com/explosion/spaCy/blob/master/spacy/cli/converters/conll_ner2json.py
    text = open(input_path,'r', encoding='utf-8').read()
    i = 0
    delimit_docs = '-DOCSTART- -X- O O'
    output_docs = []
    for doc in text.strip().split(delimit_docs):
        doc = doc.strip()
        if not doc:
            continue
        output_doc = []
        for sent in doc.split('\n\n'):
            sent = sent.strip()
            if not sent:
                continue
            lines = [line.strip() for line in sent.split('\n') if line.strip()]
            words, tags, chunks, iob_ents = zip(*[line.split() for line in lines])
            biluo_ents = iob_to_biluo(iob_ents)
            output_doc.append({'tokens': [
                {'orth': w, 'tag': tag, 'ner': ent} for (w, tag, ent) in
                zip(words, tags, biluo_ents)
            ]})
        output_docs.append({
            'id': len(output_docs),
            'paragraphs': [{'sentences': output_doc}]
        })
        output_doc = []
    return output_docs


nlp = StanfordCoreNLP('../NLP/stanford-corenlp-full-2018-02-27') 

test_data = read_conll_ner('../NLP/Chapter-5-NER/CoNLL - 2003/en/test.txt')

print(test_data[0]['paragraphs'][0]['sentences'][1])

tokens = [token['orth'] for token in test_data[0]['paragraphs'][0]['sentences'][1]['tokens']]
sentence = ' '.join(tokens)

a = nlp.relation(sentence)

print(a['sentences'][0].keys())


