import torchtext

MAX_VOCAB_SIZE = 50002


def get_data(path, batch_size, bptt_len, device=-1):
    # 构建词表
    TEXT = torchtext.data.Field(lower=True)
    train, val = torchtext.datasets.LanguageModelingDataset.splits(path=path,
                                                                   train="train.txt",
                                                                   validation="valid.txt", text_field=TEXT)
    TEXT.build_vocab(train, max_size=MAX_VOCAB_SIZE)
    # 构建train、val的迭代器（list）
    train_iter, val_iter = torchtext.data.BPTTIterator.splits(
        (train, val), batch_size=batch_size, device=device, bptt_len=bptt_len,
        repeat=False, shuffle=True)
    return train_iter, val_iter, len(TEXT.vocab)
