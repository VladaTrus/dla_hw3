import torch
import torchaudio
from model import KWS_model
from dataset import transform_test
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = KWS_model()
model.load_state_dict(torch.load('kws_model_1')['model_state_dict'])
model.to(device)
model.eval()

wav, _ = torchaudio.load("test_stream.wav")
spec = transform_test(wav).to(device)
probs = model.inference(spec.unsqueeze(0))
plt.figure(figsize=(15, 8))
plt.title('probs for sheila')
plt.ylabel('prob')
plt.grid()
plt.plot(probs)
plt.savefig('test_wav.png')
plt.show()
