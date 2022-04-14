import os.path
import torch
import torchtext
from torchtext.data.utils import get_tokenizer

from utils.config import parseArgs
from utils.model import Seq2SeqTransformer
from utils.dataset import MyDataset
from utils.dataprocess import *

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


def infer(model, vocab, sentence):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text2tensor = get_text2tensor(vocab)

    # Encode
    src = text2tensor["en"](sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    src = src.to(device)
    src_mask = src_mask.to(device)
    memory = model.encode(src, src_mask)
    memory = memory.to(device)

    # Decode
    ys = torch.ones(1, 1).fill_(BOS_IDX).type(torch.long).to(device)
    for i in range(num_tokens + 5 - 1):
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(
            device
        )
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break

    # Format
    tgt_tokens = ys.flatten()
    return (
        " ".join(vocab["cn"].lookup_tokens(list(tgt_tokens.cpu().numpy())))
        .replace("<bos>", "")
        .replace("<eos>", "")
        .replace(" ", "")
    )


if __name__ == "__main__":
    args = parseArgs()
    print("Loading Vocab...")
    vocab = {}
    if os.path.exists(args.vocab_en_path) and os.path.exists(args.vocab_cn_path):
        vocab = load_vocab(args.vocab_en_path, args.vocab_cn_path)
    else:
        vocab = make_vocab(MyDataset(args.train_path))
        save_vocab(vocab, args.vocab_en_path, args.vocab_cn_path)

    print("Loading Model...")
    model = Seq2SeqTransformer(
        src_vocab_size=len(vocab["en"]),
        tgt_vocab_size=len(vocab["cn"]),
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        emb_size=args.emb_size,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
    model = torch.load(args.load_path)
    model.eval()

    sentence = input("Please enter an English sentence:\n")
    result = infer(model, vocab, sentence)
    print("The translated sentence is:\n" + result)
