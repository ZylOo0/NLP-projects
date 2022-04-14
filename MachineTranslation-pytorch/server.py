from flask import Flask, render_template, request, url_for
import torch
import os

from infer import infer
from utils.config import parseArgs
from utils.model import Seq2SeqTransformer
from utils.dataset import MyDataset
from utils.dataprocess import *

app = Flask(__name__)
app.config["SECRET_KEY"] = "12345"
base_dir = os.environ.get("BASE_DIR", "")

args = parseArgs()
vocab = {}
if os.path.exists(args.vocab_en_path) and os.path.exists(args.vocab_cn_path):
    vocab = load_vocab(args.vocab_en_path, args.vocab_cn_path)
else:
    vocab = make_vocab(MyDataset(args.train_path))
    save_vocab(vocab, args.vocab_en_path, args.vocab_cn_path)
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


@app.route(f"{base_dir}/v1/index", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("home.html")
    if request.method == "POST":
        if request.form["sentence"] == "":
            return ""
        result = infer(model, vocab, request.form["sentence"])
        return result

if __name__ == "__main__":
    app.run("0.0.0.0")
