import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
import json

from utils.dataset import Couplets

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]


# 从数据集获取词源
def yield_tokens(dataset):
    tokenizer = get_tokenizer(None)
    for item in dataset:
        tokens = tokenizer(item[0]) + tokenizer(item[1])
        yield tokens


# 没有词表文件时，重新构建词表
def make_vocab(dataset):
    # 先从数据集中构建原始词表 vocab_raw
    vocab_raw = build_vocab_from_iterator(yield_tokens(dataset), min_freq=1)
    # 再从 vocab_raw 中构建新的词表（为了统一不同情况下使用的词表相同）
    vocab = build_vocab_from_iterator(
        [vocab_raw.get_itos()], specials=special_symbols, special_first=True
    )
    vocab.set_default_index(UNK_IDX)
    return vocab


# 存储词表
def save_vocab(vocab, vocab_path):
    with open(vocab_path, "w") as vocab_file:
        json.dump(vocab.get_itos(), vocab_file)


# 有词表文件时，读取词表
def load_vocab(vocab_path):
    with open(vocab_path, "r") as vocab_file:
        itos = json.load(vocab_file)
        vocab = build_vocab_from_iterator(
            [itos], specials=special_symbols, special_first=True
        )
        vocab.set_default_index(UNK_IDX)
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
    tokenizer = get_tokenizer(None)
    text2tensor = sequential_transforms(tokenizer, vocab, ids2tensor)
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
