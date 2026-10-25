import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from core import normalize,vocab_from_rows,encode,ScratchTTS

def test_model_shape():
    rows=[{'text':'നമസ്കാരം'}]
    vocab=vocab_from_rows(rows)
    out=ScratchTTS(len(vocab))(encode(rows[0]['text'],vocab),120)
    assert out.shape==(80,120)

def test_normalize():assert normalize('  ഹലോ   ലോകം  ')=='ഹലോ ലോകം'
