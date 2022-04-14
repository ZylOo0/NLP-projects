import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torchtext
import os.path
from timeit import default_timer as timer
from tqdm import tqdm

from utils.config import parseArgs
from utils.model import Seq2SeqTransformer
from utils.dataset import Couplets
from utils.dataprocess import *

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


class Trainer:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading Dataset...")
        self.train_set = Couplets(args.train_path)
        self.val_set = Couplets(args.val_path)

        print("Loading Vocab...")
        if os.path.exists(args.vocab_path):
            self.vocab = load_vocab(args.vocab_path)
        else:
            self.vocab = make_vocab(self.train_set)
            save_vocab(self.vocab, args.vocab_path)
        self.text2tensor = get_text2tensor(self.vocab)

        print("Initializing Model...")
        self.model = Seq2SeqTransformer(
            src_vocab_size=len(self.vocab),
            tgt_vocab_size=len(self.vocab),
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            emb_size=args.emb_size,
            nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
        )
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.model = self.model.to(self.device)

        print("Initializing Training Parameters...")
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.lr, betas=args.betas, eps=args.eps
        )
        self.save_path = args.save_path

    def collate_fn(self, batch):
        src_batch, tgt_batch = [], []
        for sample in batch:
            src_batch.append(self.text2tensor(sample[0].rstrip("\n")))
            tgt_batch.append(self.text2tensor(sample[1].rstrip("\n")))
        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
        return src_batch, tgt_batch

    def train_epoch(self):
        self.model.train()
        losses = 0
        train_dataloader = DataLoader(
            self.train_set, batch_size=self.batch_size, collate_fn=self.collate_fn
        )
        for src, tgt in tqdm(train_dataloader):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            tgt_input = tgt[:-1, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                src, tgt_input
            )
            logits = self.model(
                src,
                tgt_input,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                src_padding_mask,
            )
            self.optimizer.zero_grad()
            tgt_out = tgt[1:, :]
            loss = self.loss_fn(
                logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1)
            )
            loss.backward()
            self.optimizer.step()
            losses += loss.item()
        return losses / len(train_dataloader)

    def evaluate(self):
        self.model.eval()
        losses = 0
        val_dataloader = DataLoader(
            self.val_set, batch_size=self.batch_size, collate_fn=self.collate_fn
        )
        for src, tgt in val_dataloader:
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            tgt_input = tgt[:-1, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                src, tgt_input
            )
            logits = self.model(
                src,
                tgt_input,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                src_padding_mask,
            )
            tgt_out = tgt[1:, :]
            loss = self.loss_fn(
                logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1)
            )
            losses += loss.item()

        return losses / len(val_dataloader)

    def train_and_eval(self):
        best_model = self.model
        min_val_loss = 1e9
        for epoch in range(1, self.num_epochs + 1):
            print(f"Training Epoch {epoch}...")
            start_time = timer()
            train_loss = self.train_epoch()
            end_time = timer()
            val_loss = self.evaluate()
            print(
                f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, Epoch time: {(end_time - start_time):.3f}s"
            )
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_model = self.model
        print(f"Min Val Loss: {min_val_loss:.3f}")
        torch.save(best_model, self.save_path)
        print("Best Model Saved.")


if __name__ == "__main__":
    args = parseArgs()
    trainer = Trainer(args)
    trainer.train_and_eval()
