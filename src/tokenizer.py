# rasbt
from collections import Counter, deque
from functools import lru_cache

class BPETokenizer:
    def __init__ (self, max_tokens):
        self.vocab = {}
        self.inverse_vocab = {}
        self.bpe_merges = {}

    def train(self, text, vocab_size, allowed_special={'<|endoftext|>'}):

        # Substitui espaços vazios por "Ġ"
        processed_text = []
        for i, char in enumerate(text):
            if char == ' ' and i != 0:
                processed_text.append('Ġ')
            if char != ' ':
                processed_text.append(char)
        processed_text = ''.join(processed_text)

        # Primeiros 256 caracteres ASCII
        unique_chars = [chr(i) for i in range(256)]
        # Outros caracteres fora dos 256, mas que estejam no texto recebido
        unique_chars.extend(char for char in sorted(set(processed_text)) if char not in unique_chars)
        # Garantir que o caractere de espaço esteja incluído
        if 'Ġ' not in unique_chars:
            unique_chars.append('Ġ')

        # Criação dos vocabulários
        self.vocab = {i: char for i, char in enumerate(unique_chars)}
        self.inverse_vocab = {char: i for i, char in self.vocab.items()}

        # Adiciona os tokens especiais passados
        if allowed_special:
            for token in allowed_special:
                if token not in self.inverse_vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token
                    self.inverse_vocab[token] = new_id

        # Tokeniza o texto processado em ids
        token_ids = [self.inverse_vocab[char] for char in processed_text]

        # Passos do BPE (encontra e substitui os pares frequentes) repetidamente
        for new_id in range(len(self.vocab), vocab_size):
            if len(token_ids) < 2:
                break
            pair_id = self.find_freq_pair(token_ids, mode="most")
            if pair_id is None:  # Fim do treinamento (acabaram os pares)
                break
            
            updated = self.replace_pair(token_ids, pair_id, new_id)
            if updated == token_ids:
                break

            token_ids = updated
            self.bpe_merges[pair_id] = new_id

        # Adiciona os tokens formados na execução dos passos bpe no vocabulário
        for (p0, p1), new_id in self.bpe_merges.items():
            merged_token = self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id

    def encode(self, text):
        tokens = []
        
        words = text.replace("\n", " \n ").split()  # faz com que '\n' seja um token separado

        for i, word in enumerate(words):
            if i > 0 and not word.startswith("\n"):
                tokens.append("Ġ" + word)  # Devolve os espaços que o split() removeu
            else:
                tokens.append(word)  # '\n' e primeira palavra

        token_ids = []
        for token in tokens:
            if token in self.inverse_vocab: # Palavras que já estão no vocabulário
                token_id = self.inverse_vocab[token]
                token_ids.append(token_id)
            else: # Palavras novas, que precisam ser quebradas
                sub_token_ids = self.tokenize_with_bpe(token)
                token_ids.extend(sub_token_ids)

        return token_ids

    def tokenize_with_bpe(self, token):
        token_ids = [self.inverse_vocab.get(char, None) for char in token]
        if None in token_ids:
            missing_chars = [char for char, tid in zip(token, token_ids) if tid is None]
            raise ValueError(f"Characters not found in vocab: {missing_chars}")

        can_merge = True
        while can_merge and len(token_ids) > 1:
            can_merge = False
            new_tokens = []
            i = 0
            while i < len(token_ids) - 1:
                pair = (token_ids[i], token_ids[i + 1])
                if pair in self.bpe_merges:
                    merged_token_id = self.bpe_merges[pair]
                    new_tokens.append(merged_token_id)
                    # Uncomment for educational purposes:
                    # print(f"Merged pair {pair} -> {merged_token_id} ('{self.vocab[merged_token_id]}')")
                    i += 2  # Skip the next token as it's merged
                    can_merge = True
                else:
                    new_tokens.append(token_ids[i])
                    i += 1
            if i < len(token_ids):
                new_tokens.append(token_ids[i])
            token_ids = new_tokens

        return token_ids

    def decode(self, token_ids):
        decoded_string = ""
        for token_id in token_ids:
            if token_id not in self.vocab:
                raise ValueError(f"Token ID {token_id} not found in vocab.")
            token = self.vocab[token_id]
            if token.startswith("Ġ"):
                # Desfaz o token de espaço para espaço
                decoded_string += " " + token[1:]
            else:
                decoded_string += token
        return decoded_string

    # Funções auxiliares
    @lru_cache(maxsize=None)
    def get_special_token_id(self, token):
        return self.inverse_vocab.get(token, None)

    @staticmethod
    def find_freq_pair(token_ids, mode="most"):
        if(len(token_ids) < 2):
            return None
        pairs = Counter(zip(token_ids, token_ids[1:]))
        if not pairs:
            return None

        if mode == "most":
            return max(pairs.items(), key=lambda x: x[1])[0]
        elif mode == "least":
            return min(pairs.items(), key=lambda x: x[1])[0]
        else:
            raise ValueError("Invalid mode. Choose 'most' or 'least'.")

    @staticmethod
    def replace_pair(token_ids, pair_id, new_id):
        dq = deque(token_ids)
        replaced = []

        while dq:
            current = dq.popleft()
            if dq and (current, dq[0]) == pair_id:
                replaced.append(new_id)
                # Remove the 2nd token of the pair, 1st was already removed
                dq.popleft()
            else:
                replaced.append(current)

        return replaced
