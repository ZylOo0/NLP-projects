import os.path
import torch
import torchtext
from torchtext.data.utils import get_tokenizer

from utils.config import parseArgs
from utils.model import Seq2SeqTransformer
from utils.dataset import Couplets
from utils.dataprocess import *

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

class Inferer:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading Vocab...")
        if os.path.exists(args.vocab_path):
            self.vocab = load_vocab(args.vocab_path)
        else:
            self.vocab = make_vocab(Couplets(args.train_path))
            save_vocab(self.vocab, args.vocab_path)
        
        print("Loading Model...")
        self.model = torch.load(args.load_path)
        self.model.eval()
    
    def infer(self, sentence):
        # Encode
        sentence = " ".join([token for token in sentence])
        text2tensor = get_text2tensor(self.vocab)
        src = text2tensor(sentence).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        src = src.to(self.device)
        src_mask = src_mask.to(self.device)
        memory = self.model.encode(src, src_mask)
        memory = memory.to(self.device)

        # Decode
        ys = torch.ones(1, 1).fill_(BOS_IDX).type(torch.long).to(self.device)
        for i in range(num_tokens + 5 - 1):
            tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(self.device)
            out = self.model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = self.model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == EOS_IDX:
                break
        tgt_tokens = ys.flatten()
        return (
            "".join(self.vocab.lookup_tokens(list(tgt_tokens.cpu().numpy())))
            .replace("<bos>", "")
            .replace("<eos>", "")
        )

if __name__ == "__main__":
    args = parseArgs()
    inferer = Inferer(args)
    
    sentence = input("上联：\n")
    result = inferer.infer(sentence)
    print("下联：\n" + result)