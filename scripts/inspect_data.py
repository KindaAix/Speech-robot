"""Quick script to validate dataset/data.jsonl entries and check that audio files are readable."""
import json
import os
import soundfile as sf
from tqdm import tqdm

DATA_JSONL = os.path.join('dataset', 'data.jsonl')

def main():
    if not os.path.exists(DATA_JSONL):
        print('Missing', DATA_JSONL)
        return
    total = 0
    bad = 0
    with open(DATA_JSONL, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='Checking entries'):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
                af = obj.get('audio_filepath')
                txt = obj.get('text')
                if not af or not txt:
                    raise ValueError('missing fields')
                if not os.path.exists(af):
                    print(f'Missing audio file: {af}')
                    bad += 1
                    continue
                # try reading header
                with sf.SoundFile(af) as snd:
                    _ = snd.samplerate
            except Exception as e:
                print('Bad entry:', e)
                bad += 1
    print(f'Total entries: {total}, Bad entries: {bad}')

if __name__ == '__main__':
    main()
