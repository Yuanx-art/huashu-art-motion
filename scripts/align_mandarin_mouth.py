#!/usr/bin/env python3
# /// script
# dependencies = ["numpy", "scipy", "onnxruntime", "requests", "pypinyin", "allosaurus"]
# ///
"""Align supplied Chinese text to WAV with Mandarin CTC, then infer mouth cues.

Character and phoneme boundaries use separate acoustic CTC models. Without a
phone model, fail explicitly rather than silently substitute vowel heuristics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import math
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from pypinyin import pinyin, Style

MODEL = "onnx-community/wav2vec2-large-xlsr-53-chinese-zh-cn-ONNX"

INITIAL_IPA = dict(zip(
    'b p m f d t n l g k h j q x zh ch sh r z c s'.split(),
    'p pʰ m f t tʰ n l k kʰ x tɕ tɕʰ ɕ tʂ tʂʰ ʂ ɹ ts tsʰ s'.split()))
FINAL_IPA = {
    'a':'a', 'o':'o', 'e':'ɤ', 'i':'i', 'u':'u', 'v':'y', 'ü':'y',
    'ai':'a i', 'ei':'e i', 'ao':'a u', 'ou':'o u',
    'an':'a n', 'en':'ə n', 'ang':'a ŋ', 'eng':'ə ŋ', 'ong':'u ŋ',
    'ia':'j a', 'ie':'j e', 'iao':'j a u', 'iu':'j o u',
    'ian':'j e n', 'in':'i n', 'iang':'j a ŋ', 'ing':'i ŋ', 'iong':'j u ŋ',
    'ua':'w a', 'uo':'w o', 'uai':'w a i', 'ui':'w e i',
    'uan':'w a n', 'un':'w ə n', 'uang':'w a ŋ', 'ueng':'w ə ŋ',
    've':'ɥ e', 'üe':'ɥ e', 'van':'ɥ e n', 'üan':'ɥ e n',
    'vn':'y n', 'ün':'y n', 'er':'ə ɹ',
}


def ipa_for(char):
    initial = pinyin(char, style=Style.INITIALS, strict=False)[0][0]
    final = pinyin(char, style=Style.FINALS, strict=True)[0][0]
    if final == 'i' and initial in ('zh', 'ch', 'sh', 'r', 'z', 'c', 's'):
        phones = ['ɹ̩']
    else:
        if final not in FINAL_IPA:
            raise ValueError(f'Unsupported pronunciation {char}: {initial}/{final}')
        phones = FINAL_IPA[final].split()
    if initial in INITIAL_IPA:
        phones.insert(0, INITIAL_IPA[initial])
    return phones


def phone_viseme(phone):
    if phone in ('p', 'pʰ', 'm'): return 'MBP'
    if phone in ('f', 'v'): return 'FV'
    if phone in ('u', 'w', 'y', 'ɥ'): return 'U'
    if phone == 'o': return 'O'
    if phone == 'a': return 'A'
    if phone in ('i', 'j'): return 'I'
    if phone in ('e', 'ə', 'ɤ', 'ɹ̩'): return 'E'
    return None  # tongue consonants inherit adjacent vowel, not a lip closure


def align_phones(audio_path, text, model_path):
    from argparse import Namespace
    import torch
    from allosaurus.audio import read_audio
    from allosaurus.pm.factory import read_pm
    from allosaurus.am.factory import read_am
    from allosaurus.lm.factory import read_lm
    from allosaurus.am.utils import move_to_tensor
    model_path = Path(model_path)
    if not (model_path / 'model.pt').is_file():
        raise ValueError('Missing Allosaurus phoneme model: see references/口型同步.md')
    cfg = Namespace(model=model_path.name, device_id=-1, lang='cmn', approximate=False, prior=None)
    torch.set_num_threads(4)
    pm, am, lm = read_pm(model_path, cfg), read_am(model_path, cfg), read_lm(model_path, cfg)
    feats = pm.compute(read_audio(audio_path))
    f, n = move_to_tensor([feats[None, :], np.array([len(feats)], np.int32)], -1)
    am.eval()
    with torch.no_grad():
        logits = am(f, n)[0].numpy().astype(np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    target = [(ch, phone) for ch in text for phone in ipa_for(ch)]
    inventory = lm.unit.unit_to_id
    missing = [phone for _, phone in target if phone not in inventory]
    if missing:
        raise ValueError(f'Phonemes absent from acoustic model: {missing}')
    tokens = [inventory[phone] for _, phone in target]
    spans = force_align(logp, tokens, 0)
    phones = []
    for (ch, phone), token, (start, end) in zip(target, tokens, spans):
        phones.append({'char':ch, 'phoneme':phone, 'start':start*cfg.window_shift,
                       'end':end*cfg.window_shift,
                       'confidence':float(np.exp(logp[start:end,token]).mean()),
                       'timing':'acoustic_phoneme_ctc_forced_alignment'})
    for left, right in zip(phones, phones[1:]):
        if right['start']-left['end'] < .18:
            mid=(left['end']+right['start'])/2
            left['end']=right['start']=mid
    for i, phone in enumerate(phones):
        mouth = phone_viseme(phone['phoneme'])
        if mouth is None:
            mouth=next((phone_viseme(p['phoneme']) for p in phones[i+1:]
                        if p['char']==phone['char'] and phone_viseme(p['phoneme'])),None)
            if mouth is None:
                mouth=phone_viseme(phones[max(0,i-1)]['phoneme']) or 'E'
        phone['viseme']=mouth
        phone['start']=round(phone['start'],4);phone['end']=round(phone['end'],4)
    return phones


def force_align(logp, tokens, blank):
    """CTC Viterbi alignment, including repeated-character blank constraints."""
    labels = [blank]
    for token in tokens:
        labels += [token, blank]
    labels = np.asarray(labels)
    n, states = len(logp), len(labels)
    if not tokens or n < len(tokens):
        raise ValueError("Empty transcript or insufficient audio frames")
    scores = np.full(states, -np.inf)
    scores[0] = logp[0, blank]
    scores[1] = logp[0, tokens[0]]
    back = np.zeros((n, states), np.int8)
    skip = (labels != blank) & (labels != np.roll(labels, 2))
    skip[:2] = False
    for t in range(1, n):
        choices = np.stack([scores, np.r_[-np.inf, scores[:-1]],
                            np.r_[-np.inf, -np.inf, scores[:-2]]])
        choices[2, ~skip] = -np.inf
        step = choices.argmax(axis=0)
        back[t] = step
        scores = choices[step, np.arange(states)] + logp[t, labels]
    state = states - 1 if scores[-1] > scores[-2] else states - 2
    if not np.isfinite(scores[state]):
        raise ValueError("Transcript cannot be aligned to this audio")
    path = np.empty(n, np.int32)
    for t in range(n - 1, -1, -1):
        path[t] = state
        state -= int(back[t, state])
    spans = []
    for i in range(len(tokens)):
        frames = np.flatnonzero(path == 2 * i + 1)
        if not len(frames):
            raise ValueError(f"Unaligned transcript character {i}")
        spans.append((int(frames[0]), int(frames[-1]) + 1))
    return spans


def get_model(cache):
    import requests
    cache.mkdir(parents=True, exist_ok=True)
    files = ["onnx/model_quantized.onnx", "vocab.json", "config.json", "preprocessor_config.json"]
    for name in files:
        dst = cache / name
        if dst.is_file():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading verified HTTPS model file: {name}", flush=True)
        url = f"https://huggingface.co/{MODEL}/resolve/main/{name}"
        with requests.get(url, stream=True, timeout=(30, 120),
                          verify="/etc/ssl/certs/ca-certificates.crt") as r:
            r.raise_for_status()
            tmp = dst.with_suffix(dst.suffix + ".partial")
            with tmp.open("wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
            tmp.replace(dst)
    return cache / files[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="/workspace/.cache/mandarin-mouth-model")
    ap.add_argument("--phone-model", default="/workspace/.cache/allosaurus/uni2005")
    args = ap.parse_args()
    if re.search(r"[A-Za-z0-9]", args.text):
        ap.error("Expand numbers and foreign words into confirmed Chinese pronunciation first")
    text = re.sub(r"[^\u4e00-\u9fff]", "", args.text)
    if not text:
        ap.error("Need Chinese characters in transcript")
    cache = Path(args.cache)
    model = get_model(cache)
    import onnxruntime as ort
    rate, samples = wavfile.read(args.audio)
    scale = np.iinfo(samples.dtype).max if np.issubdtype(samples.dtype, np.integer) else 1
    audio = samples.astype(np.float32) / scale
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = resample_poly(audio, 16000 // math.gcd(rate, 16000), rate // math.gcd(rate, 16000))
    duration = len(audio) / 16000
    if duration > 20:
        ap.error("Align short utterances (<=20s); split long narration first")
    normalized = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    session = ort.InferenceSession(str(model), sess_options=opts, providers=["CPUExecutionProvider"])
    feed = {"input_values": normalized[None, :]}
    if any(i.name == "attention_mask" for i in session.get_inputs()):
        feed["attention_mask"] = np.ones((1, len(audio)), np.int64)
    logits = session.run(None, feed)[0][0].astype(np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    vocab = json.loads((cache / "vocab.json").read_text())
    config = json.loads((cache / "config.json").read_text())
    blank = config.get("pad_token_id", vocab.get("<pad>", 0))
    unknown = [ch for ch in text if ch not in vocab]
    if unknown:
        raise ValueError(f"Characters missing from acoustic vocabulary: {unknown}")
    tokens = [vocab[ch] for ch in text]
    spans = force_align(logp, tokens, blank)
    # Wav2vec2 receptive field: 400 samples, frame stride 320 samples.
    stride = math.prod(config.get("conv_stride", [5, 2, 2, 2, 2, 2, 2])) / 16000
    segments = [{"char": ch, "start": a * stride, "end": b * stride,
                 "confidence": float(np.exp(logp[a:b, token]).mean())}
                for ch, token, (a, b) in zip(text, tokens, spans)]
    for left, right in zip(segments, segments[1:]):
        gap = right["start"] - left["end"]
        if gap < .20:
            middle = (left["end"] + right["start"]) / 2
            left["end"] = right["start"] = middle
    # Consonant closure precedes the recognized syllable nucleus.
    for s in segments:
        initial = pinyin(s["char"], style=Style.INITIALS, strict=False)[0][0]
        final = pinyin(s["char"], style=Style.FINALS, strict=True)[0][0]
        s.update(initial=initial, final=final)
    cues = align_phones(args.audio, text, args.phone_model)
    decoded = []
    inv = {v: k for k, v in vocab.items()}
    previous = None
    for token in logp.argmax(axis=1):
        if token != previous and token != blank:
            decoded.append(inv.get(int(token), "?"))
        previous = token
    result = {"text": args.text, "duration": duration, "model": MODEL,
              "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
              "character_timing": "acoustic_ctc_forced_alignment",
              "phoneme_timing": "acoustic_phoneme_ctc_forced_alignment",
              "phoneme_model": "Allosaurus uni2005 universal phone acoustic model",
              "greedy_transcript": "".join(decoded), "characters": segments, "cues": cues,
              "review_required": [q for q in cues if q["confidence"] < .05]}
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
