import argparse, json, random
import torch
import torch.nn.functional as F
from core import *

def train(epochs=20,lr=0.0003):
    rows=load_rows()
    if len(rows)<2: raise ValueError('Record at least 2 labeled samples (many hours recommended for quality)')
    random.seed(42);random.shuffle(rows)
    vocab=vocab_from_rows(rows)
    model=ScratchTTS(len(vocab)); opt=torch.optim.Adam(model.parameters(),lr=lr)
    DATA.mkdir(exist_ok=True); CKPT.parent.mkdir(exist_ok=True)
    # Explicit duration heuristic: equal frames per character. This is a weak baseline.
    for epoch in range(epochs):
        model.train();losses=[]
        for row in rows:
            audio=read_wav(AUDIO/row['file']); target=mel_spec(audio)
            tokens=encode(row['text'],vocab)
            if not len(tokens): continue
            pred=model(tokens,target.shape[-1]); loss=F.l1_loss(pred,target)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            losses.append(float(loss.item()))
        value=sum(losses)/max(1,len(losses))
        torch.save({'state':model.state_dict(),'vocab':vocab,'frames_per_char':float(sum(mel_spec(read_wav(AUDIO/r['file'])).shape[-1] for r in rows)/sum(max(1,len(normalize(r['text']))) for r in rows)),'epoch':epoch+1},CKPT)
        with (DATA/'training_log.jsonl').open('a') as f: f.write(json.dumps({'epoch':epoch+1,'loss':value})+'\n')
        print(f'epoch {epoch+1}/{epochs} loss={value:.4f}',flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--epochs',type=int,default=20);args=a.parse_args();train(args.epochs)
