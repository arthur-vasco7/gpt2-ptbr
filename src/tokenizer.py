import os, re, tiktoken
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

def train(train_path, out_dir, vocab_size=16000):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=['<|endoftext|>'], show_progress=True)

    tokenizer.train([train_path], trainer)

    tokenizer.save(os.path.join(out_dir, 'tokenizer.json'))
    tokenizer.model.save(out_dir)

    return tokenizer

def fertility(text, encode_fn):
    clean_text = text.replace("<|endoftext|>", " ")
    num_words = len(clean_text.split())
    num_tokens = len(encode_fn(text))
    fertility = num_tokens / num_words
    return num_words, num_tokens, fertility

def main():
    train_path = "data/train.txt"
    val_path = "data/val.txt"
    ref_tok_path = "data/tokenizer_16k/tokenizer.json"
    out_dir = "my_tokenizer_16k"

    my_tok = train(train_path, out_dir, vocab_size=16000)
    ref_tok = Tokenizer.from_file(ref_tok_path)
    gpt2_tok = tiktoken.get_encoding("gpt2")

    print(f"ID <|endoftext|> (Próprio)    : {my_tok.token_to_id('<|endoftext|>')}")
    print(f"ID <|endoftext|> (Referência) : {ref_tok.token_to_id('<|endoftext|>')}")

    for split_name, path in [("TREINO (train.txt)", train_path), ("VALIDAÇÃO (val.txt)", val_path)]:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        print(f"\n=== Fertilidade em {split_name} ===")
        for label, encode_fn in [
            ("Meu BPE 16k", lambda t: my_tok.encode(t).ids),
            ("Referência BPE 16k", lambda t: ref_tok.encode(t).ids),
            ("GPT-2 (tiktoken)", lambda t: gpt2_tok.encode(t, allowed_special={"<|endoftext|>"})),
        ]:
            words, tokens, fert = fertility(text, encode_fn)
            print(f"{label:<20} | Palavras: {words:,} | Tokens: {tokens:,} | Fertilidade: {fert:.4f} tokens/palavra")

    palavras_teste = [
        "computação",
        "científico",
        "metodológicas",
        "dá-lo",
        "analisá-las",
        "desenvolveram",
        "implementaríamos",
    ]

    print("\n=== Comparação de Segmentação ===")
    for palavra in palavras_teste:
        my_pieces = [my_tok.decode([tid]) for tid in my_tok.encode(palavra).ids]
        ref_pieces = [ref_tok.decode([tid]) for tid in ref_tok.encode(palavra).ids]
        gpt2_pieces = [gpt2_tok.decode([tid]) for tid in gpt2_tok.encode(palavra)]

        print(f"\nPalavra: '{palavra}'")
        print(f"  Meu BPE 16k : {my_pieces} ({len(my_pieces)} tokens)")
        print(f"  Referência  : {ref_pieces} ({len(ref_pieces)} tokens)")
        print(f"  GPT-2       : {gpt2_pieces} ({len(gpt2_pieces)} tokens)")

if __name__ == "__main__":
    main()
