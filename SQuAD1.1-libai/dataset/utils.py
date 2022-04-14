import numpy as np
import json
from tqdm import tqdm


def get_format_text_and_word_offset(text):

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    
    for c in text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)
    return doc_tokens, char_to_word_offset

def load_examples(data_dir):
    with open(data_dir, "r") as f:
        raw_data = json.loads(f.read())
        data = raw_data['data']
    
    examples = []
    for i in range(len(data)):
        paragraphs = data[i]['paragraphs']
        for j in range(len(paragraphs)):
            context = paragraphs[j]['context']
            context_tokens, word_offset = get_format_text_and_word_offset(context)
            qas = paragraphs[j]['qas']
            for k in range(len(qas)):
                question_text = qas[k]['question']
                qas_id = qas[k]['id']
                answer_offset = qas[k]['answers'][0]['answer_start']
                orig_answer_text = qas[k]['answers'][0]['text']
                answer_length = len(orig_answer_text)
                start_position = word_offset[answer_offset]
                end_position = word_offset[answer_offset + answer_length - 1]

                # check whether the position match the answer text
                actual_text = " ".join(context_tokens[start_position:(end_position + 1)])
                cleaned_answer_text = " ".join(orig_answer_text.strip().split())
                if actual_text.find(cleaned_answer_text) == -1:
                    continue
                
                examples.append([
                    qas_id,
                    question_text,
                    orig_answer_text,
                    " ".join(context_tokens),
                    start_position,
                    end_position
                ])
    return examples

def improve_answer_span(context_tokens, answer_tokens, start_position, end_position):
    new_end = None
    for i in range(start_position, len(context_tokens)):
        if context_tokens[i] != answer_tokens[0]:
            continue
        for j in range(len(answer_tokens)):
            if answer_tokens[j] != context_tokens[i + j]:
                break
            new_end = i + j
        if new_end - i + 1 == len(answer_tokens):
            return i, new_end
    return start_position, end_position

def pad_sequence(sequence_ids, max_length, pad_id):
    num_pad = max_length - len(sequence_ids)
    sequence_ids = sequence_ids + num_pad * [pad_id]
    return sequence_ids

def make_padding_mask(q_ids, kv_ids, pad_id):
    q = (np.array(q_ids) != pad_id).reshape(-1, 1)
    kv = (np.array(kv_ids) != pad_id).reshape(1, -1)
    padding_mask = (q * kv).astype(float)
    return padding_mask


if __name__ == "__main__":
    from libai.tokenizer import BertTokenizer

    tokenizer = BertTokenizer("/home/zhuangyulin/vocabs/bert-large-uncased-vocab.txt", do_lower_case=True)
    examples = load_examples("/home/zhuangyulin/datasets/SQuAD/train-v1.1.json")
