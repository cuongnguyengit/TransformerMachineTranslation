from utils import *
import torch
from model import *
from hparams import *
import math
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 256

train_filepaths1 = ['./data/train_src1.txt', './data/train_tgt1.txt']
train_filepaths2 = ['./data/train_src2.txt', './data/train_tgt2.txt']

vocab = build_vocab(train_filepaths1)

print(f'Batch size is {BATCH_SIZE}')
print(f'length vocab is {len(vocab)}')

PAD_IDX = vocab.stoi['<pad>']
EOS_IDX = vocab.stoi['<eos>']
BOS_IDX = vocab.stoi['<bos>']

def generate_batch(data_batch):
  de_batch, en_batch = [], []
  for (de_item, en_item) in data_batch:
    de_batch.append(torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
    en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
  de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
  en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
  return de_batch, en_batch

train_data1 = data_process(train_filepaths1, vocab)
train_data2 = data_process(train_filepaths2, vocab)
train_data = []
train_data.extend(train_data1)
train_data.extend(train_data2)

print(f"Training Data: {len(train_data1)} + {len(train_data2)} = {len(train_data)}")

train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)

train_core_iter = DataLoader(train_data1, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)

INPUT_DIM = OUTPUT_DIM = len(vocab)

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)

attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)

dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)

model.apply(init_weights)

optimizer = optim.Adam(model.parameters())

print(f'The model has {count_parameters(model):,} trainable parameters')

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

N_EPOCHS = 300
CLIP = 1

best_valid_loss = float('inf')
try:
    start_epoch = load_model(model, optimizer, '/content/drive/MyDrive/checkpoint/seq2seq.model.ep4', device)
    print(f'Resume Training with {start_epoch}')
except:
    print('Start new Training')
    start_epoch = 0

for epoch in range(start_epoch, N_EPOCHS):
    print(f'Epoch: {epoch + 1:02}')
    start_time = time.time()

    if epoch >= 50:
        train_loss = train(model, train_core_iter, optimizer, criterion, CLIP, device)
    else:
        train_loss = train(model, train_iter, optimizer, criterion, CLIP, device)
    # valid_loss = evaluate(model, valid_iter, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'\tEpoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print("*" * 30)
    # print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    if (epoch + 1) % checkpoint_per_epoch == 0:
        print(f'Save model at {epoch}')
        save_model(model, optimizer, epoch, '/content/drive/MyDrive/checkpoint')

# test_loss = evaluate(model, test_iter, criterion)
# print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')