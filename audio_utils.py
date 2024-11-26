import os
import librosa
import numpy as np
from tqdm import tqdm

def load_audio_files(basePath: str,filepaths: list, class_1_speakers: list, name: str):
    audio_files = {
        'class_0': [],
        'class_1': [],
        'test_class_0': [],
        'test_class_1': []
        }
    test_set = False
    for filepath in filepaths:
        absPath = os.path.join(basePath, filepath)
        for file in os.listdir(absPath):
            if file.split('_')[1] == name:
                test_set = True

            audio, _ = librosa.load(os.path.join(absPath, file),sr=None)
            trimmed, _ = librosa.effects.trim(audio, top_db=20)

            if(file.split('_')[0] in class_1_speakers):
                if test_set:
                    audio_files['test_class_1'].append(trimmed)
                else: 
                    audio_files['class_1'].append(trimmed)
            else:
                if test_set:
                    audio_files['test_class_0'].append(trimmed)
                else:
                    audio_files['class_0'].append(trimmed)
            test_set = False

    return audio_files

def split_audio(sample_rate: str, audio,duration = 5.0, ):
    samples_per_segment = duration * sample_rate
    segments = len(audio)
    return np.array_split(audio, int(segments/samples_per_segment))

def time_shift(sample_rate, audio, max_shift = 2.5):
    shift = np.random.uniform(-max_shift, max_shift)
    shift_samples = int(shift * sample_rate)
    return np.roll(audio, shift_samples)

def generate_spectrograms(audio: dict[str, list], sample_rate: str , class_label, duration = 5.0, max_shift = 2.5, shifts = None):
    spectrograms = []
    for au in tqdm(audio[class_label],desc=f"Processing audio files of {class_label}"):
        to_split = [au]
        if shifts is not None:
            for _ in range(shifts):
                to_split.append(time_shift(sample_rate, au, max_shift))
                
        for sp in to_split:
            segments = split_audio(sample_rate, sp, duration)

            for segment in segments:
                S = librosa.feature.melspectrogram(y=segment, sr=sample_rate)
                S_dB = librosa.power_to_db(S, ref=np.max)
                spectrograms.append(S_dB)

    return spectrograms
