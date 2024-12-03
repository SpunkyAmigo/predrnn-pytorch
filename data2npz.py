from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

dataset_dir_path = "/Users/mubeenqaiser/Documents/Datasets/UCF101/"

# CSV files with columns: clip_name, clip_path, label
# Make sure to join clip_path with dataset_dir_path to get the full path
train_csv_path = Path(dataset_dir_path) / "train.csv"
test_csv_path = Path(dataset_dir_path) / "test.csv"
val_csv_path = Path(dataset_dir_path) / "val.csv"

def process_video(video_path, target_size=(64, 64)):
    """Process a video file into a sequence of normalized frames."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize
        frame = cv2.resize(frame, target_size)
        
        # Normalize to [0,1]
        frame = frame.astype(np.float32) / 255.0
        
        frames.append(frame)
        
    cap.release()
    return np.array(frames)

def create_npz_dataset(csv_path, output_path, seq_length=20):
    """Create npz dataset from video clips listed in CSV."""
    df = pd.read_csv(csv_path)

    categories = ["HorseRiding", "PullUps", "PushUps", "BenchPress", "WallPushups"]
    df = df[df['label'].isin(categories)]
    
    # Initialize lists to store sequences and metadata
    all_frames = []
    clips = []
    
    current_position = 0

    df['clip_path'] = dataset_dir_path + df['clip_path']
    
    print(f"Processing {len(df)} videos...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_path = row['clip_path']
        frames = process_video(video_path)
        
        # Skip if video is too short
        if len(frames) < seq_length:
            print(f"Skipping {video_path} - too short ({len(frames)} frames)")
            continue
            
        # Take first seq_length frames
        frames = frames[:seq_length]
        
        # Add channel dimension
        frames = np.expand_dims(frames, axis=-1)
        
        # Store all frames
        all_frames.append(frames)
        
        # Store clip metadata
        clips.append([
            [current_position, seq_length//2],  # Input clip info
            [current_position + seq_length//2, seq_length//2]  # Output clip info
        ])
        
        current_position += seq_length
    
    # Convert to numpy arrays
    all_frames = np.concatenate(all_frames, axis=0)
    clips = np.array(clips)
    
    # Split into input and output
    input_raw_data = all_frames.copy()
    output_raw_data = all_frames.copy()
    
    print("\nDataset statistics:")
    print(f"All frames shape: {all_frames.shape}")
    print(f"Input raw data shape: {input_raw_data.shape}")
    print(f"Output raw data shape: {output_raw_data.shape}")
    print(f"Clips shape: {clips.shape}")
    print(f"First few clips:\n{clips[:3]}")
    
    # Save as npz
    np.savez(output_path,
             input_raw_data=input_raw_data,
             output_raw_data=output_raw_data,
             clips=clips,
             dims=np.array([[64, 64, 1], [64, 64, 1]]))
    
    print(f"\nDataset saved to {output_path}")

def main():
    # Create output directory if it doesn't exist
    output_dir = dataset_dir_path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process training data
    print("Processing training data...")
    create_npz_dataset(
        train_csv_path,
        Path(output_dir) / "ucf101-train.npz"
    )
    
    # Process validation data
    print("Processing validation data...")
    create_npz_dataset(
        val_csv_path,
        Path(output_dir) / "ucf101-val.npz"
    )
    
    # Process test data
    print("Processing test data...")
    create_npz_dataset(
        test_csv_path,
        Path(output_dir) / "ucf101-test.npz"
    )

if __name__ == "__main__":
    main()




