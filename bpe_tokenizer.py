import transformers
import torch
from tqdm import tqdm
from collections import OrderedDict
import re
import pickle

device = torch.device('mps' if torch.mps.is_available() else 'cpu')
print(f'Using device: {device}')

class BPETokenizer():
    def __init__(self):
        self.b2i = OrderedDict()
        self.i2b = OrderedDict()
        self.next_id = 0
        self.special_token_b2i = dict()
        self.special_token_i2b = dict()

    def _pair_stats(self, tokens, stats) -> None:
        for i in range(len(tokens) - 1):
            new_token = tokens[i] + tokens[i+1]
            if new_token not in stats:
                stats[new_token] = 0
            stats[new_token] += 1

    def _merge_pair(self, tokens, new_token) -> list:
        merged_tokens = list()
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] + tokens[i+1] == new_token:
                merged_tokens.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                merged_tokens.append(tokens[i])
                i += 1
        return merged_tokens

    def add_special_tokens(self, special_tokens):
        for token in special_tokens:
            if token not in self.special_token_b2i:
                self.special_token_b2i[token] = self.next_id
                self.special_token_i2b[self.next_id] = token
                self.next_id += 1


    def train(self, text_list, vocab_size):
        # byte is the most basic token
        # initialize vocab
        for i in range(256):
            self.b2i[bytes([i])] = i
        self.next_id = 256

        # text to byte
        tokens_list = list()
        for text in text_list:
            tokens = [bytes([b]) for b in text.encode('utf-8')]
            tokens_list.append(tokens)

        progress = tqdm(total=vocab_size-256)
        while True:
            # quit when get enough tokens
            if self.next_id >= vocab_size:
                break

            # find adjacent token pairs frequency
            stats = dict()
            for tokens in tokens_list:
                self._pair_stats(tokens, stats)

            # no more adjacent token pairs
            if not stats:
                break

            # combine the adjacent token pairs with highest frequency
            # add them to the vocab as new tokens
            new_tokens = max(stats, key=stats.get)
            new_tokens_list = list()
            for tokens in tokens_list:
                new_tokens_list.append(self._merge_pair(tokens, new_tokens))
            tokens_list = new_tokens_list

            # add the new token to the vocab
            self.b2i[new_tokens] = self.next_id
            self.next_id += 1

            # udpate progress bar
            progress.update(1)
        
        self.i2b = {v: k for k, v in self.b2i.items()}

    def vocab_size(self):
        return self.next_id
    
    def vocab(self):
        v = dict()
        v.update(self.i2b)
        v.update({id: token.encode('utf-8') for id, token in self.special_token_i2b.items()})
        return self.b2i

    def save(self, path):
        with open(path, 'wb') as fp:
            fp.write(pickle.dumps((self.b2i, self.i2b, self.next_id)))

    def load(self, path):
        with open(path, 'rb') as fp:
            self.b2i, self.i2b, self.next_id = pickle.loads(fp.read())
        self.i2b = {v: k for k, v in self.b2i.items()}
        self.special_token_i2b = {v: k for k, v in self.special_token_b2i.items()}

    def encode(self, text):
        # separate special tokens
        pattern = '(' + '|' .join([re.escape(tok) for tok in self.special_token_b2i]) + ')'
        splits = re.split(pattern, text)  # ['<|im_start|>', 'user', '<||>']

        # result of encoding
        enc_ids = list()
        enc_tok = list()
        for sub_text in splits:
            if sub_text in self.special_token_b2i:
                enc_ids.append(self.special_token_b2i[sub_text])
                enc_tok.append(sub_text.encode('utf-8'))
            else:
                tokens = [bytes([b]) for b in sub_text.encode('utf-8')]
                while True:
                    stats = dict()
                    self._pair_stats(tokens, stats)

                    # choose the pair with smallest id to merge
                    new_token = None
                    for merge_token in stats:
                        if merge_token in self.b2i and (new_token is None or self.b2i[merge_token] < self.b2i[new_token]):
                            new_token = merge_token
                    
                    # no available pair to merge
                    if new_token is None:
                        break

                    # merge the pair
                    tokens = self._merge_pair(tokens, new_token)
                enc_ids.extend([self.b2i[token] for token in tokens])
                enc_tok.extend(tokens)
        return enc_ids, enc_tok

    def decode(self, ids):
        bytes_list = list()
        for id in ids:
            if id in self.i2b:
                bytes_list.append(self.i2b[id].decode('utf-8'))
            else:
                bytes_list.append(self.i2b[id])
        return b''.join(bytes_list).decode('utf-8', errors='replace')



if __name__ == '__main__':
    # load data
    zh = open('dataset/zh.txt', 'r').read()
    en = open('dataset/en.txt', 'r').read()

    # train tokenizer
    tokenizer = BPETokenizer()
    tokenizer.train(text_list=[zh, en], vocab_size=10000)

    # add special tokens
    tokenizer.add_special_tokens(['<|im_start|>', '<|im_end|>', '<|EOT|>', '<|padding|>'])

    # save
    tokenizer.save('tokenizer.bin')

    # 
    tokenizer = BPETokenizer()
    tokenizer.load('tokenizer.bin')
    print(f'vocab size: {tokenizer.vocab_size()}')

    # encoding
    ids, tokens = tokenizer.encode('<|im_start|>system\nyou are a helpful assistant\n<|im_end|>\n<|im_start|>user\n今天的天气\n<im_end>')
    print('Encoded:', ids, tokens)

    # decoding
    text = tokenizer.decode(ids)
    print('Decoded:', text)

    # print dictionary
    print('Vocab:', tokenizer.vocab())
