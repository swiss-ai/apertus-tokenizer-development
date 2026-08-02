#!/bin/bash

# Script to merge tokenized datasets based on config file
# Usage: ./merge_script.sh configs_apertus_v2/<config>.cfg

CONFIG_FILE="$1"

if [ -z "$CONFIG_FILE" ]; then
  echo "Usage: $0 <config-file>"
  exit 1
fi

# Check if the file exists, then load it
if [ -f "$CONFIG_FILE" ]; then
  source "$CONFIG_FILE"
else
  echo "Error: Config file $CONFIG_FILE not found."
  exit 1
fi

# Set paths based on config
DUMPS_FOLDER=$PATH_TO_PREPROCESSING_METADATA/dumps
COMPLETED_DUMPS_FOLDER=$PATH_TO_PREPROCESSING_METADATA/completed-dumps
TOKENIZED_DATASET_FOLDER=$PATH_TO_OUTPUT_FOLDER/$TOKENIZER_NAME/$DATASET_NAME
MERGED_OUTPUT_FOLDER=$PATH_TO_OUTPUT_FOLDER/$TOKENIZER_NAME/${DATASET_NAME}_merged
MERGED_DATA_FOLDER=$MERGED_OUTPUT_FOLDER/data
MERGED_COMPLETED_DUMPS_FOLDER=$MERGED_OUTPUT_FOLDER/completed_dumps
PATH_TO_SLURM_LOGGING_DIR=$PATH_TO_OUTPUT_FOLDER/logs/slurm_logs_merge

# Check if MERGED_DUMPS_UPPER_SIZE_BOUND_GB is set
if [ -z "$MERGED_DUMPS_UPPER_SIZE_BOUND_GB" ]; then
  echo "Error: MERGED_DUMPS_UPPER_SIZE_BOUND_GB not set in config file."
  exit 1
fi

# Convert GB to bytes for calculations
UPPER_BOUND_BYTES=$((MERGED_DUMPS_UPPER_SIZE_BOUND_GB * 1024 * 1024 * 1024))

echo "=== Merge Script Configuration ==="
echo "Config file: $CONFIG_FILE"
echo "Dataset: $DATASET_NAME"
echo "Tokenizer: $TOKENIZER_NAME"
echo "Tokenized dataset folder: $TOKENIZED_DATASET_FOLDER"
echo "Merged output folder: $MERGED_OUTPUT_FOLDER"
echo "Upper bound per bucket: ${MERGED_DUMPS_UPPER_SIZE_BOUND_GB} GB"
echo "=================================="

# Check if dumps folder is not empty
if [ ! -d "$DUMPS_FOLDER" ]; then
  echo "Error: Dumps folder not found: $DUMPS_FOLDER"
  exit 1
fi

# Check if dumps are still being processed (not all moved to completed-dumps)
if [ "$(ls -A $DUMPS_FOLDER)" ]; then
  echo "Warning: Dumps folder is not empty. Some dumps may still be processing."
  echo "Files remaining in dumps folder:"
  ls -1 "$DUMPS_FOLDER"
  
  # Ask user if they want to continue
  read -p "Do you want to continue merging only completed dumps? (y/n): " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Exiting. Please wait for all dumps to complete before merging."
    exit 0
  fi
fi

# Check if there are any completed dumps
if [ ! -d "$COMPLETED_DUMPS_FOLDER" ] || [ ! "$(ls -A $COMPLETED_DUMPS_FOLDER)" ]; then
  echo "Error: No completed dumps found in $COMPLETED_DUMPS_FOLDER"
  echo "Nothing to merge. Please run tokenization first."
  exit 1
fi

# Check if tokenized dataset folder exists
if [ ! -d "$TOKENIZED_DATASET_FOLDER" ]; then
  echo "Error: Tokenized dataset folder not found: $TOKENIZED_DATASET_FOLDER"
  exit 1
fi

# Create merged output folders
mkdir -p $MERGED_DATA_FOLDER
mkdir -p $MERGED_COMPLETED_DUMPS_FOLDER
mkdir -p $PATH_TO_SLURM_LOGGING_DIR

# Compute dump sizes and create buckets
echo "Computing dump sizes and creating buckets..."

# Create a temporary file to store dump sizes
TEMP_DUMP_SIZES=$(mktemp)

# Get list of completed dumps from the completed-dumps folder
for paths_file in "$COMPLETED_DUMPS_FOLDER"/*; do
  dump_id=$(basename "$paths_file" | grep -oP '(?<=paths_file_)\d+(?=\.txt)')
  
  if [ -z "$dump_id" ]; then
    echo "Warning: Could not extract dump ID from $paths_file, skipping"
    continue
  fi
  
  # Calculate size of all .bin and .idx files in the dump folder
  dump_folder="$TOKENIZED_DATASET_FOLDER/dump-$dump_id"
  
  if [ ! -d "$dump_folder" ]; then
    echo "Warning: Dump folder not found: $dump_folder, skipping dump $dump_id"
    continue
  fi
  
  # Sum up sizes of all .bin and .idx files
  # Try Linux stat first, then macOS stat
  dump_size=$(find "$dump_folder" -type f \( -name "*.bin" -o -name "*.idx" \) -exec stat -c%s {} + 2>/dev/null | awk '{s+=$1} END {print s}')
  if [ -z "$dump_size" ]; then
    dump_size=$(find "$dump_folder" -type f \( -name "*.bin" -o -name "*.idx" \) -exec stat -f%z {} + 2>/dev/null | awk '{s+=$1} END {print s}')
  fi
  
  # Handle case where no files found
  if [ -z "$dump_size" ] || [ "$dump_size" = "0" ]; then
    echo "Warning: No .bin/.idx files found in $dump_folder, skipping dump $dump_id"
    continue
  fi
  
  echo "$dump_id $dump_size" >> "$TEMP_DUMP_SIZES"
done

# Sort by dump ID to maintain order
sort -n -k1 "$TEMP_DUMP_SIZES" -o "$TEMP_DUMP_SIZES"

# Check if we found any dumps
if [ ! -s "$TEMP_DUMP_SIZES" ]; then
  echo "Error: No valid dumps found to merge."
  rm "$TEMP_DUMP_SIZES"
  exit 1
fi

# Display summary of dumps found
total_dumps=$(wc -l < "$TEMP_DUMP_SIZES")
total_size=$(awk '{sum+=$2} END {print sum}' "$TEMP_DUMP_SIZES")
total_size_gb=$(echo "scale=2; $total_size / 1024 / 1024 / 1024" | bc)
echo "Found $total_dumps completed dumps with total size: ${total_size_gb} GB"

# Create buckets
echo "Creating buckets with upper bound of $MERGED_DUMPS_UPPER_SIZE_BOUND_GB GB..."

bucket_num=0
current_bucket_size=0
current_bucket_dumps=()

BUCKET_JOBS=$(mktemp)

while read -r dump_id dump_size; do
  # If adding this dump exceeds the limit and current bucket is not empty, finalize current bucket
  if [ $current_bucket_size -gt 0 ] && [ $((current_bucket_size + dump_size)) -gt $UPPER_BOUND_BYTES ]; then
    # Finalize current bucket
    echo "Bucket $bucket_num: ${#current_bucket_dumps[@]} dumps, size: $(echo "scale=2; $current_bucket_size / 1024 / 1024 / 1024" | bc) GB"
    echo "${current_bucket_dumps[@]}" >> "$BUCKET_JOBS"
    
    # Start new bucket
    bucket_num=$((bucket_num + 1))
    current_bucket_size=0
    current_bucket_dumps=()
  fi
  
  # Add to current bucket
  current_bucket_dumps+=("$dump_id")
  current_bucket_size=$((current_bucket_size + dump_size))
  
done < "$TEMP_DUMP_SIZES"

# Finalize last bucket if not empty
if [ ${#current_bucket_dumps[@]} -gt 0 ]; then
  echo "Bucket $bucket_num: ${#current_bucket_dumps[@]} dumps, size: $(echo "scale=2; $current_bucket_size / 1024 / 1024 / 1024" | bc) GB"
  echo "${current_bucket_dumps[@]}" >> "$BUCKET_JOBS"
fi

# Submit SLURM jobs for each bucket
echo "Submitting SLURM jobs for merging..."

RES_OPT=""
if [ -n "${RESERVATION:-}" ] || [ -n "${reservation:-}" ]; then
  VAL="${RESERVATION:-$reservation}"
  RES_OPT="--reservation=${VAL}"
fi

bucket_num=0
while read -r bucket_dumps; do
  # Collect dump directories in order
  DUMP_DIRS=()
  for dump_id in $bucket_dumps; do
    dump_folder="$TOKENIZED_DATASET_FOLDER/dump-$dump_id"
    if [ ! -d "$dump_folder" ]; then
      echo "Warning: Dump folder not found: $dump_folder"
      continue
    fi
    DUMP_DIRS+=("$dump_folder")
  done
  
  # Skip if no valid dump directories found
  if [ ${#DUMP_DIRS[@]} -eq 0 ]; then
    echo "Warning: No valid dump directories for bucket $bucket_num, skipping"
    bucket_num=$((bucket_num + 1))
    continue
  fi
  
  # Create the output prefix for this bucket
  OUTPUT_PREFIX="$MERGED_DATA_FOLDER/bucket_$bucket_num"
  
  # Prepare list of dump IDs for this bucket (to pass to merge.sh)
  DUMP_IDS_STR=$(echo $bucket_dumps | tr ' ' ',')
  
  # Submit SLURM job with multiple input directories
  echo "Submitting bucket $bucket_num with dumps: $bucket_dumps (${#DUMP_DIRS[@]} directories)"
  
  sbatch $RES_OPT --partition=$PARTITION --account=$ACCOUNT --nodes=1 --cpus-per-task=72 --time=1:00:00 $NO_REQUEUE --job-name=merge-$DATASET_NAME-bucket-$bucket_num --output=$PATH_TO_SLURM_LOGGING_DIR/merge-%x-%j.out --error=$PATH_TO_SLURM_LOGGING_DIR/merge-%x-%j.err merge.sh $OUTPUT_PREFIX "$DUMP_IDS_STR" "$COMPLETED_DUMPS_FOLDER" "$MERGED_COMPLETED_DUMPS_FOLDER" ${DUMP_DIRS[@]}
  
  bucket_num=$((bucket_num + 1))
done < "$BUCKET_JOBS"

# Cleanup
rm "$TEMP_DUMP_SIZES"
rm "$BUCKET_JOBS"

total_buckets=$bucket_num
echo ""
echo "=== Merge Summary ==="
echo "Total dumps processed: $total_dumps"
echo "Total size: ${total_size_gb} GB"
echo "Number of buckets created: $total_buckets"
echo "Upper bound per bucket: ${MERGED_DUMPS_UPPER_SIZE_BOUND_GB} GB"
echo "====================="
echo ""
echo "Submitted $total_buckets merge job(s)."
echo "Output will be in: $MERGED_OUTPUT_FOLDER"
echo "  Data files: $MERGED_DATA_FOLDER/bucket_*.bin and bucket_*.idx"
echo "  Parquet file mapping: $MERGED_COMPLETED_DUMPS_FOLDER/bucket_*.txt"
echo ""
echo "Monitor jobs with: squeue -u \$USER --name=merge-$DATASET_NAME-bucket-*"
echo "View logs at: $PATH_TO_SLURM_LOGGING_DIR/merge-*.out"
