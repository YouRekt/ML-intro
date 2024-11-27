import os
import time
import librosa
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import gc

def load_audio_files(basePath: str,filepaths: list, class_1_speakers: list, name: str, sample_rate: int):
    i = 0
    batch_size = 10
    test_set = False
    for filepath in filepaths:
        absPath = os.path.join(basePath, filepath)
        files = os.listdir(absPath)
        for idx in range(0, len(files), batch_size):
            files = np.concatenate((files, files))
            batch = files[idx:idx + batch_size]
            for file in tqdm(batch, desc=f"Processing batch {idx//batch_size + 1} of {absPath}"):
                test_set = file.split('_')[1] == name
                
                
                audio, _ = librosa.load(os.path.join(absPath, file),sr=None)
                trimmed, _ = librosa.effects.trim(audio, top_db=20)
                del audio
                
                class_prefix = 'test_' if test_set else ''
                class_num = '1' if file.split('_')[0] in class_1_speakers else '0'
                class_label = f"{class_prefix}class_{class_num}"
                
                shifts = 1 if class_num == '1' else None
                i = generate_spectogram(trimmed, class_label, sample_rate, shifts=shifts, ind=i)
            
                i += 1
                del trimmed
        gc.collect()

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

def generate_spectogram(audio, class_label: str, sample_rate: int, duration = 5.0, max_shift = 2.5, shifts = None, ind = 0):
    to_split = [audio]
    if shifts is not None:
        for _ in range(shifts):
            to_split.append(time_shift(sample_rate, audio, max_shift))
            
    for sp in to_split:
        segments = split_audio(sample_rate, sp, duration)

        for segment in segments:
            S = librosa.feature.melspectrogram(y=segment, sr=sample_rate)
            S_dB = librosa.power_to_db(S, ref=np.max)
            del S
            plt.figure(figsize=(2, 2))
            plt.ioff()  # Disable interactive mode
            librosa.display.specshow(S_dB, sr=sample_rate, x_axis='time', y_axis='mel')
            # plt.colorbar(format='%+2.0f dB')
            # plt.title(class_label)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join('spectrograms', class_label, f'{class_label}_{ind}.png'), bbox_inches='tight', pad_inches=0)
            ind += 1
            plt.close('all')
            del segment
        del segments
    del to_split
    return ind