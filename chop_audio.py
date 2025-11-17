#!/usr/bin/env python3
import librosa
import soundfile as sf
import numpy as np
import argparse
import os
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor

def split_by_downbeats(audio_file, output_base_dir, prefix="section", bars_per_section=4):
    """Split audio into sections based on downbeats (groups of bars)"""
    print("Detecting downbeats with madmom...")
    
    # Create subdirectory named after the file
    output_dir = os.path.join(output_base_dir, prefix)
    os.makedirs(output_dir, exist_ok=True)
    
    # Use madmom's downbeat detector
    proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    act = RNNDownBeatProcessor()(audio_file)
    beats = proc(act)
    
    # beats is array of [time, position] where position 1 = downbeat (start of bar)
    downbeat_times = beats[beats[:, 1] == 1, 0]
    
    print(f"Detected {len(downbeat_times)} bars")
    print(f"Grouping into sections of {bars_per_section} bars each...")
    
    # Load audio
    y, sr = librosa.load(audio_file, sr=None)
    
    # Group downbeats into sections (e.g., every 4 bars = one section)
    section_times = downbeat_times[::bars_per_section]
    section_samples = librosa.time_to_samples(section_times, sr=sr)
    
    print(f"Creating {len(section_samples)} sections")
    
    # Export each section
    for i in range(len(section_samples) - 1):
        start = section_samples[i]
        end = section_samples[i + 1]
        section_audio = y[start:end]
        
        section_duration = (end - start) / sr
        output_file = os.path.join(output_dir, f"{prefix}_{i:03d}.wav")
        sf.write(output_file, section_audio, sr)
        print(f"Exported {output_file} (duration: {section_duration:.2f}s)")
    
    # Last section
    if len(section_samples) > 0:
        last_section = y[section_samples[-1]:]
        output_file = os.path.join(output_dir, f"{prefix}_{len(section_samples)-1:03d}.wav")
        sf.write(output_file, last_section, sr)
        duration = len(last_section) / sr
        print(f"Exported {output_file} (duration: {duration:.2f}s)")

def split_by_bars(audio_file, output_base_dir, prefix="bar"):
    """Split audio into individual bars using madmom's downbeat detection"""
    print("Detecting downbeats with madmom...")
    
    # Create subdirectory named after the file
    output_dir = os.path.join(output_base_dir, prefix)
    os.makedirs(output_dir, exist_ok=True)
    
    # Use madmom's downbeat detector
    proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    act = RNNDownBeatProcessor()(audio_file)
    beats = proc(act)
    
    # beats is array of [time, position] where position 1 = downbeat
    downbeat_times = beats[beats[:, 1] == 1, 0]
    
    print(f"Detected {len(downbeat_times)} bars")
    
    # Load audio
    y, sr = librosa.load(audio_file, sr=None)
    
    # Convert times to samples
    bar_samples = librosa.time_to_samples(downbeat_times, sr=sr)
    
    # Export each bar
    for i in range(len(bar_samples) - 1):
        start = bar_samples[i]
        end = bar_samples[i + 1]
        bar_audio = y[start:end]
        
        output_file = os.path.join(output_dir, f"{prefix}_{i:03d}.wav")
        sf.write(output_file, bar_audio, sr)
        print(f"Exported {output_file}")
    
    # Last bar
    if len(bar_samples) > 0:
        last_bar = y[bar_samples[-1]:]
        output_file = os.path.join(output_dir, f"{prefix}_{len(bar_samples)-1:03d}.wav")
        sf.write(output_file, last_bar, sr)
        print(f"Exported {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Chop audio using madmom beat detection')
    parser.add_argument('input_file', help='Input audio file')
    parser.add_argument('output_dir', help='Output directory for chunks')
    parser.add_argument('--bars', action='store_true',
                       help='Split into individual bars instead of sections')
    parser.add_argument('--bars-per-section', type=int, default=4,
                       help='Number of bars per section (default: 4)')
    parser.add_argument('--prefix', default=None, 
                       help='Prefix for output files (default: input filename without extension)')
    
    args = parser.parse_args()
    
    # If no prefix provided, use the input filename without extension
    if args.prefix is None:
        args.prefix = os.path.splitext(os.path.basename(args.input_file))[0]
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.bars:
        split_by_bars(args.input_file, args.output_dir, args.prefix)
    else:
        split_by_downbeats(args.input_file, args.output_dir, args.prefix, args.bars_per_section)
    
    print(f"\nDone! Chunks saved to {args.output_dir}")

if __name__ == "__main__":
    main()
