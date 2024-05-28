import torch
import time
import torch.nn.functional as F
from transformers import AutoTokenizer

from utils import configs
from models import Transformer


def load_model_tokenizer(configs):
    """
    This function will load model and tokenizer from pretrained model and tokenizer
    """
    device = torch.device(configs["device"])
    source_tokenizer = AutoTokenizer.from_pretrained(configs["source_tokenizer"])
    target_tokenizer = AutoTokenizer.from_pretrained(configs["target_tokenizer"])

    # Load model Transformer
    model = Transformer(
        source_vocab_size=source_tokenizer.vocab_size,
        target_vocab_size=target_tokenizer.vocab_size,
        embedding_dim=configs["embedding_dim"],
        source_max_seq_len=configs["source_max_seq_len"],
        target_max_seq_len=configs["target_max_seq_len"],
        num_layers=configs["n_layers"],
        num_heads=configs["n_heads"],
        dropout=configs["dropout"]
    )
    model.load_state_dict(torch.load(configs["model_path"], map_location='cpu'))
    model.eval()
    model.to(device)
    print(f"Done load model on the {device} device")
    return model, source_tokenizer, target_tokenizer


def translate(model, sentence, source_tokenizer, target_tokenizer,
              source_max_seq_len=256,target_max_seq_len=256,
              device=torch.device("cpu")):
    """
    This funciton will translate give a source sentence and return target sentence using beam search
    """
    # Convert source sentence to tensor
    # <pad>0, <unk>100, <cls>101, <sep>102, <mask>103
    source_tokens = source_tokenizer.encode(sentence)[:source_max_seq_len]
    source_tensor = torch.tensor(source_tokens).unsqueeze(0).to(device)
    # Create source sentence mask
    source_mask = model.make_source_mask(source_tensor, source_tokenizer.pad_token_id).to(device)
    # Feed forward Encoder
    encoder_output = model.encoder.forward(source_tensor, source_mask)
    # Initialize indexes and socores list
    indexes = [target_tokenizer.cls_token_id]
    scores = []
    # Start decoding
    for _ in range(target_max_seq_len):
        input_token = torch.tensor([indexes]).to(device)
        # Create mask
        target_mask = model.make_target_mask(input_token).to(device)
        # Decoder forward pass
        pred = model.decoder.forward(input_token, encoder_output, source_mask, target_mask)
        # Forward to linear classify token in vocab and Softmax
        pred = F.softmax(model.final_linear(pred), dim=-1)
        # Get tail predict token
        value, index = pred[:, -1, :].max(dim=-1)

        indexes.append(index.item())
        scores.append(value.item())

        if index == target_tokenizer.sep_token_id:
            break

    # Convert target sentence from tokens to string
    target_sentence = target_tokenizer.decode(indexes, skip_special_tokens=True)
    return target_sentence


def main():
    # Translate a sentence
    sentence = "My family is very poor, I had to go through hard life when I was young, now I have a better life."
    print("--- English input sentence:", sentence)
    print("--- Translating...")
    device = torch.device(configs["device"])
    model, source_tokenizer, target_tokenizer = load_model_tokenizer(configs)
    st = time.time()
    trans_sen = translate(
        model=model,
        sentence=sentence, 
        source_tokenizer=source_tokenizer, 
        target_tokenizer=target_tokenizer, 
        source_max_seq_len=configs["source_max_seq_len"],
        target_max_seq_len=configs["target_max_seq_len"],
        device=device
    )
    end = time.time()
    print("--- Sentences translated into Chinese:", trans_sen)
    print(f"--- Time: {end-st} (s)")

if __name__ == "__main__":
    main()