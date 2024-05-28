from torch.utils.data import Dataset
import re


class TranslateDataset(Dataset):
    def __init__(self, source_tokenizer, target_tokenizer, source_data=None, target_data=None, source_max_seq_len=256, target_max_seq_len=256, phase="train"):
        self.source_data = source_data
        self.target_data = target_data
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_max_seq_len = source_max_seq_len
        self.target_max_seq_len = target_max_seq_len
        self.phase = phase
    
    def preprocess_seq(self, seq):
        seq = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(seq))
        seq = re.sub(r"[ ]+", " ", seq)
        seq = re.sub(r"\!+", "!", seq)
        seq = re.sub(r"\,+", ",", seq)
        seq = re.sub(r"\?+", "?", seq)
        seq = seq.lower()
        return seq

    def convert_line_uncased(self, tokenizer, text, max_seq_len):
        tokens = tokenizer.tokenize(text)[:max_seq_len-2]
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        tokens += [tokenizer.pad_token]*(max_seq_len - len(tokens))
        token_idx = tokenizer.convert_tokens_to_ids(tokens)
        return tokens, token_idx
    
    def __len__(self):
        return len(self.source_data)
    
    def __getitem__(self, index):
        # source_seq, source_idx = self.convert_line_uncased(
        #     tokenizer=self.source_tokenizer, 
        #     text=self.preprocess_seq(self.source_data[index]), 
        #     max_seq_len=self.source_max_seq_len
        # )
        # target_seq, target_idx = self.convert_line_uncased(
        #     tokenizer=self.target_tokenizer, 
        #     text=self.preprocess_seq(self.target_data[index]), 
        #     max_seq_len=self.target_max_seq_len
        # )
        
        source = self.source_tokenizer(
                text=self.preprocess_seq(self.source_data[index]),
                padding="max_length", 
                max_length=self.source_max_seq_len, 
                truncation=True, 
                return_tensors="pt"
            )
        
        if self.phase == "train":
            target = self.target_tokenizer(
                text=self.preprocess_seq(self.target_data[index]),
                padding="max_length",
                max_length=self.target_max_seq_len,
                truncation=True,
                return_tensors="pt"
            )

            return {
                "source_seq": self.source_data[index],
                "source_ids": source["input_ids"][0],
                "target_seq": self.target_data[index],
                "target_ids": target["input_ids"][0],
                }
        else:
            return {
                "source_seq": self.source_data[index],
                "source_ids": source["input_ids"][0],
            }