import os
import librosa
import numpy as np
from tqdm import tqdm

def load_audio_files(filepath: str, class_1_speakers: list):
    audio_files = {
        'class_0': [],
        'class_1': []
        }
    for file in os.listdir(filepath):
        audio, _ = librosa.load(os.path.join(filepath, file),sr=None)
        trimmed, _ = librosa.effects.trim(audio, top_db=20)
        if(file.split('_')[0] in class_1_speakers):
            audio_files['class_1'].append(trimmed)
        else:
            audio_files['class_0'].append(trimmed)
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
