from tqdm import tqdm
from training_func import train_model, evaluate
from model import KWS_model
import wandb
import torch
import torch.nn as nn
import torch
import torchaudio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = KWS_model()
model.to(device)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

CLIP = 1
N_EPOCHS = 7

wandb.init(project="trus_va_kws")
wandb.watch(model, log="all")

for epoch in tqdm(range(1, N_EPOCHS + 1)):
    train_loss, train_acc, train_fr, train_fa = train_model(model, train_dataloader, optimizer, criterion, CLIP)
    test_loss, test_acc, test_fr, test_fa = evaluate(model, test_dataloader, criterion)

    wandb.log({"learning_rate" : 0.001,
               "model" : 'kws_attention',
               "optimizer" : 'Adam',
               "num_workers" : 1,
               "train_loss": train_loss,
               "train_accuracy": train_acc,
               "train_fr" : train_fr,
               "train_fa" : train_fa,
               "test_loss": test_loss,
               "test_accuracy": test_acc,
               "train_fr" : train_fr,
               "train_fa" : train_fa
               })
torch.save({'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
                }, 'kws_model_1')
