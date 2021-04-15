import torch
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from collections import Counter
from torchtext.vocab import Vocab
import io
import torch.nn as nn
import os
import gc
import subprocess
import time


def get_gpu_memory_map():
  if torch.cuda.is_available():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])

    return float(result)
  return 0.0


def save_model(model, optimizer, epoch, save_path="checkpoint"):
    os.makedirs(save_path, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        # "scheduler_state_dict": scheduler.state_dict(),
    }, os.path.join(save_path, "seq2seq.model.ep{}".format(epoch)))


def load_model(model, optimizer, save_path, device):
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"] + 1


def build_vocab(filepath):
    counter = Counter()
    for path in filepath:
        with io.open(path, encoding="utf8") as f:
            for string_ in f:
                counter.update(string_.split())
    return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])


def data_process(filepaths, vocab):
    raw_de_iter = iter(io.open(filepaths[0], encoding="utf8"))
    raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
    data = []
    for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
        if len(raw_de) > 64 or len(raw_en) > 64:
            continue
        de_tensor_ = torch.tensor([vocab[token] for token in raw_de],
                                  dtype=torch.long)
        en_tensor_ = torch.tensor([vocab[token] for token in raw_en],
                                  dtype=torch.long)
        data.append((de_tensor_, en_tensor_))
    return data


def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model: nn.Module,
          iterator: torch.utils.data.DataLoader,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float, device):
    model.train()

    epoch_loss = 0

    for _, (src, trg) in enumerate(iterator):
        str_output = f"LOOP: ({_}) | BASE STATE {get_gpu_memory_map()}\n"
        # print("LOOP: (%s) | BASE STATE" % _, get_gpu_memory_map())

        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()

        output = model(src, trg)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

        # print("LOOP: (%s) | PEAK STATE" % _, get_gpu_memory_map())
        str_output += f"LOOP: ({_}) | PEAK STATE {get_gpu_memory_map()}\n"

        del src
        del trg
        del output

        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(0.5)

        # print("LOOP: (%s) | OUT  STATE" % _, get_gpu_memory_map())
        str_output += f"LOOP: ({_}) | OUT STATE {get_gpu_memory_map()}\n"

        if _ % 1000 == 0:
            # print(str_output)
            print(f'\tIter: {_ + 1:02} | loss={epoch_loss / (_ + 1)}')
            print('-' * 29)

    return epoch_loss / len(iterator)


def evaluate(model: nn.Module,
             iterator: torch.utils.data.DataLoader,
             criterion: nn.Module, device):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for _, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg, 0)  # turn off teacher forcing

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

            del src
            del trg
            del output

            gc.collect()
            torch.cuda.empty_cache()

    return epoch_loss / len(iterator)


def epoch_time(start_time,
               end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs