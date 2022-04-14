from omegaconf import OmegaConf

from libai.config import get_config
from libai.config import LazyCall
from libai.data.build import build_nlp_test_loader, build_nlp_train_loader
from libai.tokenizer import BertTokenizer

from modeling.model import ModelForSquad

tokenization = get_config("common/data/bert_dataset.py").tokenization
optim = get_config("common/optim.py").optim
model_cfg = get_config("common/models/bert.py").cfg
graph = get_config("common/models/graph.py").graph
train = get_config("common/train.py").train

tokenization.tokenizer = LazyCall(BertTokenizer)(
    vocab_file="/home/zhuangyulin/vocabs/bert-large-uncased-vocab.txt",
    do_lower_case=True,
)


