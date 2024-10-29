import os
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import skew, kurtosis

class DAPSAnalyzer:
    def __init__(self, data_path):
        """
        Initialize the DAPS dataset analyzer
        
        Parameters:
        data_path (str): Path to the DAPS dataset root directory
        """
        self.data_path = data_path
        self.sample_rate = 16000  # Default sample rate for analysis
        
    def parse_filename(self, filename):
        """
        Parse DAPS filename to extract metadata
        
        Parameters:
        filename (str): Name of the audio file
        
        Returns:
        dict: Dictionary containing parsed metadata
        """
        parts = filename.replace('.wav', '').split('_')
        
        metadata = {
            'filename': filename,
            'speaker': parts[0],
            'script': parts[1]
        }
        
        # Handle different filename formats
        if len(parts) == 3:  # Studio recordings (e.g., f1_script1_clean.wav)
            metadata.update({
                'recording_type': 'studio',
                'version': parts[2],
                'device': 'studio',
                'environment': 'studio'
            })
        elif len(parts) == 4:  # Device recordings (e.g., f1_script1_ipad_office1.wav)
            metadata.update({
                'recording_type': 'device',
                'device': parts[2],
                'environment': parts[3]
            })
            
        return metadata

    def extract_audio_features(self, audio_path):
        """
        Extract various audio features from a single file
        
        Parameters:
        audio_path (str): Path to audio file
        
        Returns:
        dict: Dictionary containing extracted features
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Basic statistics
            duration = librosa.get_duration(y=y, sr=sr)
            rms = np.sqrt(np.mean(y**2))
            
            # Spectral features
            spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spec_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            
            # MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_means = np.mean(mfccs, axis=1)
            
            # Zero crossing rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            
            return {
                'duration': duration,
                'rms_energy': rms,
                'spectral_centroid': spec_centroid,
                'spectral_bandwidth': spec_bandwidth,
                'spectral_rolloff': spec_rolloff,
                'zero_crossing_rate': zcr,
                'mfcc_mean': mfcc_means.mean(),
                'mfcc_std': mfcc_means.std()
            }
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return None

    def analyze_dataset(self):
        """
        Analyze the entire dataset and create a DataFrame with features
        
        Returns:
        pd.DataFrame: DataFrame containing features for all audio files
        """
        results = []
        
        # Walk through the dataset directory
        for root, dirs, files in os.walk(self.data_path):
            for file in tqdm(files, desc="Processing audio files"):
                if file.endswith('.wav'):
                    try:
                        # Get file metadata
                        metadata = self.parse_filename(file)
                        
                        # Extract audio features
                        features = self.extract_audio_features(os.path.join(root, file))
                        
                        if features:
                            # Combine metadata and features
                            entry = {**metadata, **features}
                            results.append(entry)
                            
                    except Exception as e:
                        print(f"Error processing {file}: {str(e)}")
                        continue
        
        df = pd.DataFrame(results)
        
        # Print basic dataset info
        print("\nDataset Summary:")
        print(f"Total files processed: {len(df)}")
        print("\nRecording types found:")
        print(df['recording_type'].value_counts())
        print("\nDevices found:")
        print(df['device'].value_counts())
        print("\nEnvironments found:")
        print(df['environment'].value_counts())
        
        return df

    def generate_reports(self, df):
        """
        Generate various analysis reports and visualizations
        
        Parameters:
        df (pd.DataFrame): DataFrame containing audio features
        """
        # Ensure we have data to analyze
        if len(df) == 0:
            print("No data available for analysis")
            return
            
        # 1. Basic statistics by recording type
        print("\nBasic Statistics by Recording Type:")
        type_stats = df.groupby('recording_type').agg({
            'duration': ['mean', 'std'],
            'rms_energy': ['mean', 'std'],
            'spectral_centroid': ['mean', 'std']
        }).round(3)
        print(type_stats)
        
        # 2. Create visualizations
        plt.figure(figsize=(15, 10))
        
        # RMS Energy by recording type
        plt.subplot(2, 2, 1)
        sns.boxplot(data=df, x='recording_type', y='rms_energy')
        plt.title('RMS Energy by Recording Type')
        plt.xticks(rotation=45)
        
        # Spectral centroid by recording type
        plt.subplot(2, 2, 2)
        sns.boxplot(data=df, x='recording_type', y='spectral_centroid')
        plt.title('Spectral Centroid by Recording Type')
        plt.xticks(rotation=45)
        
        # MFCC distribution
        plt.subplot(2, 2, 3)
        sns.scatterplot(data=df, x='mfcc_mean', y='mfcc_std', hue='recording_type')
        plt.title('MFCC Distribution by Recording Type')
        
        # Duration distribution
        plt.subplot(2, 2, 4)
        sns.histplot(data=df, x='duration', hue='recording_type', multiple="stack")
        plt.title('Duration Distribution by Recording Type')
        
        plt.tight_layout()
        plt.show()
        
        # 3. Device-specific analysis (only for device recordings)
        device_df = df[df['recording_type'] == 'device']
        if len(device_df) > 0:
            print("\nDevice Recording Statistics:")
            device_stats = device_df.groupby(['device', 'environment']).agg({
                'duration': ['mean', 'std'],
                'rms_energy': ['mean', 'std'],
                'spectral_centroid': ['mean', 'std']
            }).round(3)
            print(device_stats)
            
            # Additional device-specific visualization
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=device_df, x='device', y='rms_energy', hue='environment')
            plt.title('RMS Energy by Device and Environment')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

def main():
    # Replace with your actual DAPS dataset path
    data_path = "./daps"
    
    print("Starting DAPS dataset analysis...")
    analyzer = DAPSAnalyzer(data_path)
    
    try:
        df = analyzer.analyze_dataset()
        analyzer.generate_reports(df)
    except Exception as e:
        print(f"An error occurred during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()