import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
import jieba
import json

from utils.dataset import MyDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]


def yield_tokens(dataset, language):
    lans = {"en": 0, "cn": 1}
    tokenizer = {}
    tokenizer["en"] = get_tokenizer("basic_english")
    tokenizer["cn"] = get_tokenizer(jieba.lcut)
    for pair in dataset:
        yield tokenizer[language](pair[lans[language]])


def make_vocab(dataset):
    vocab_en = build_vocab_from_iterator(yield_tokens(dataset, "en"), min_freq=2)
    vocab_cn = build_vocab_from_iterator(yield_tokens(dataset, "cn"), min_freq=2)
    vocab = {}
    vocab["en"] = build_vocab_from_iterator(
        [vocab_en.get_itos()], specials=special_symbols, special_first=True
    )
    vocab["en"].set_default_index(UNK_IDX)
    vocab["cn"] = build_vocab_from_iterator(
        [vocab_cn.get_itos()], specials=special_symbols, special_first=True
    )
    vocab["cn"].set_default_index(UNK_IDX)
    return vocab


def save_vocab(vocab, vocab_en_path, vocab_cn_path):
    with open(vocab_en_path, "w") as vocab_en_file:
        json.dump(vocab["en"].get_itos(), vocab_en_file)
    with open(vocab_cn_path, "w") as vocab_cn_file:
        json.dump(vocab["cn"].get_itos(), vocab_cn_file)


def load_vocab(vocab_en_path, vocab_cn_path):
    vocab = {}
    with open(vocab_en_path, "r") as vocab_en_file:
        itos_en = json.load(vocab_en_file)
        vocab["en"] = build_vocab_from_iterator(
            [itos_en], specials=special_symbols, special_first=True
        )
        vocab["en"].set_default_index(UNK_IDX)
    with open(vocab_cn_path, "r") as vocab_cn_file:
        itos_cn = json.load(vocab_cn_file)
        vocab["cn"] = build_vocab_from_iterator(
            [itos_cn], specials=special_symbols, special_first=True
        )
        vocab["cn"].set_default_index(UNK_IDX)
    return vocab


def ids2tensor(token_ids):
    return torch.cat(
        (torch.tensor([BOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX]))
    )


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


def get_text2tensor(vocab):
    text2tensor = {}
    tokenizer = {}
    tokenizer["en"] = get_tokenizer("basic_english")
    tokenizer["cn"] = get_tokenizer(jieba.lcut)
    for lan in ["en", "cn"]:
        text2tensor[lan] = sequential_transforms(tokenizer[lan], vocab[lan], ids2tensor)
    return text2tensor


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
