"""
Module 2: Feature Engineering (3.3.2) - Main Run Script

Feature Engineering

Extract 10 features from segmented sessions (behavioral, cognitive, temporal engagement)
"""

import os
import sys
import argparse
from feature_extraction import extract_all_features

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Module 2: Feature Engineering (3.3.2)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--dialog_folder', required=True,
                       help='Folder containing segmented dialog CSV files (from Module 1)')
    parser.add_argument('--class_time_file', required=True,
                       help='Path to class_time_range_by_school.csv')
    parser.add_argument('--class_schedule_file', required=True,
                       help='Path to class schedule CSV file')
    parser.add_argument('--class_info_file', default='',
                       help='Path to class info CSV file (optional, for filtering classes)')
    parser.add_argument('--output_file', default='extracted_features.csv',
                       help='Output CSV file name (default: extracted_features.csv)')
    parser.add_argument('--output_folder', default=None,
                       help='Output folder for histograms and stats (default: same as dialog_folder)')
    parser.add_argument('--no_histograms', action='store_true',
                       help='Skip histogram plotting')
    
    return parser.parse_args()


def main():
    """Execute feature extraction workflow."""
    args = parse_arguments()
    
    # Validate input folder
    if not os.path.exists(args.dialog_folder):
        print(f"Error: Dialog folder does not exist: {args.dialog_folder}")
        sys.exit(1)
    
    # Validate required files
    if not os.path.exists(args.class_time_file):
        print(f"Error: Class time file does not exist: {args.class_time_file}")
        sys.exit(1)
    
    if not os.path.exists(args.class_schedule_file):
        print(f"Error: Class schedule file does not exist: {args.class_schedule_file}")
        sys.exit(1)
    
    # Validate optional class_info_file if provided
    if args.class_info_file and not os.path.exists(args.class_info_file):
        print(f"Warning: Class info file does not exist: {args.class_info_file}")
        print("Continuing without class filtering...")
        args.class_info_file = ''
    
    # Set output folder
    if args.output_folder is None:
        output_folder = args.dialog_folder
    else:
        output_folder = args.output_folder
        os.makedirs(output_folder, exist_ok=True)
    
    print("=" * 50)
    print("Module 2: Feature Engineering (3.3.2)")
    print("=" * 50)
    print(f"Dialog folder:        {args.dialog_folder}")
    print(f"Class time file:      {args.class_time_file}")
    print(f"Class schedule file:  {args.class_schedule_file}")
    print(f"Class info file:      {args.class_info_file or 'Not provided (no filtering)'}")
    print(f"Output file:          {args.output_file}")
    print(f"Output folder:        {output_folder}")
    print("=" * 50)
    
    # Extract features
    print("\nExtracting features from all dialog files...")
    features_df = extract_all_features(
        dialog_folder=args.dialog_folder,
        class_time_file=args.class_time_file,
        class_schedule_file=args.class_schedule_file,
        school_info_file=None,
        class_info_file=args.class_info_file if args.class_info_file else None,
        final_week_file=None,
        plot_histograms=not args.no_histograms,
        output_root=output_folder
    )
    
    if features_df.empty:
        print("Error: No features extracted!")
        sys.exit(1)
    
    # Save features
    output_path = os.path.join(output_folder, args.output_file)
    features_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nFeatures saved to: {output_path}")
    print(f"Total samples: {len(features_df):,}")
    print(f"Total features: {len(features_df.columns)}")
    
    print("\n" + "=" * 50)
    print("Feature Extraction Complete!")
    print("=" * 50)
    
    return features_df


if __name__ == '__main__':
    main()

