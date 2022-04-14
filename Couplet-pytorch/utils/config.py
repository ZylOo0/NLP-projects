import argparse

DATASET_PATH = "/dataset/1564ee80/v1/couplets"


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default=f"{DATASET_PATH}/train")
    parser.add_argument("--val_path", type=str, default=f"{DATASET_PATH}/test")
    parser.add_argument("--vocab_path", type=str, default="/workspace/utils/vocab.json")
    parser.add_argument("--save_path", type=str, default="/workspace/weights/l6-2.pth")
    parser.add_argument("--load_path", type=str, default="/workspace/weights/l6-1.pth")

    parser.add_argument("--num_encoder_layers", type=int, default=6)
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--emb_size", type=int, default=512)
    parser.add_argument("--dim_feedforward", type=int, default=512)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.98))
    parser.add_argument("--eps", type=float, default=1e-9)

    args = parser.parse_args()
    return args
