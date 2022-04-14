from tqdm import tqdm
import oneflow as flow
from oneflow.utils.data import Dataset

from libai.data.structures import DistTensorData, Instance

from utils import load_examples, improve_answer_span, pad_sequence, make_padding_mask

class SquadDataset(Dataset):
    def __init__(
        self,
        data_dir,
        tokenizer,
        max_seq_length = 512,
        max_question_length = 128,
        doc_stride = 128,
    ):
        self.tokenizer = tokenizer
        self.vocab = self.tokenizer.get_vocab()
        cls_id = self.vocab[self.tokenizer.cls_token]
        sep_id = self.vocab[self.tokenizer.sep_token]
        pad_id = self.vocab[self.tokenizer.pad_token]

        self.max_seq_length = max_seq_length
        self.max_question_length = max_question_length
        self.doc_stride = doc_stride

        self.features = []
        examples = load_examples(data_dir)
        for example in tqdm(examples):
            question_tokens = self.tokenizer.tokenize(example[1])
            if len(question_tokens) > self.max_question_length:
                question_tokens = question_tokens[:self.max_question_length]
            question_ids = self.tokenizer.convert_tokens_to_ids(question_tokens)
            question_ids = [cls_id] + question_ids + [sep_id]
            context_tokens = self.tokenizer.tokenize(example[3])
            context_ids = self.tokenizer.convert_tokens_to_ids(context_tokens)
            answer_tokens = self.tokenizer.tokenize(example[2])
            start_position, end_position = example[4], example[5]
            start_position, end_position = improve_answer_span(context_tokens, answer_tokens, start_position, end_position)

            rest_length = self.max_seq_length - len(question_ids)
            context_length = len(context_ids)
            if context_length + 1 <= rest_length:  # 不需要滑动窗口
                input_ids = question_ids + context_ids + [sep_id]
                input_ids = pad_sequence(input_ids, self.max_seq_length, pad_id)
                attention_mask = make_padding_mask(input_ids, input_ids, pad_id)
                start_position += len(question_ids)
                end_position += len(question_ids)
                self.features.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'start_position': start_position,
                    'end_position': end_position,
                })
            else:  # 需要滑动窗口
                continue  # TODO
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, i):
        feature = self.features[i]
        return Instance(
            input_ids = DistTensorData(flow.tensor(feature['input_ids'], dtype=flow.long)),
            attention_mask = DistTensorData(flow.tensor(feature['attention_mask'], dtype=flow.long)),
            start_position = DistTensorData(flow.tensor(feature['start_position'], dtype=flow.long)),
            end_position = DistTensorData(flow.tensor(feature['end_position'], dtype=flow.long)),
        )


if __name__ == "__main__":
    from libai.tokenizer import BertTokenizer
    tokenizer = BertTokenizer("/home/zhuangyulin/vocabs/bert-large-uncased-vocab.txt", do_lower_case=True)
    dataset = SquadDataset(data_dir="/home/zhuangyulin/datasets/SQuAD/train-v1.1.json", tokenizer=tokenizer)
    print(len(dataset))
