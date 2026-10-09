#!/bin/bash

# ⚠️ WARNING ⚠️
# Make sure to prepare the dumps before tokenizing the data!
# Check scripts/tokenization/prepare_dumps.py
# ⚠️ WARNING ⚠️

# Parse args: accept config file and optional --dont_compute_dumps flag
DONT_COMPUTE_DUMPS=0
CONFIG_FILE=""
for arg in "$@"; do
  if [ "$arg" = "--dont_compute_dumps" ]; then
    DONT_COMPUTE_DUMPS=1
  elif [ -z "$CONFIG_FILE" ]; then
    CONFIG_FILE="$arg"
  fi
done

if [ -z "$CONFIG_FILE" ]; then
  echo "Usage: $0 <config-file> [--dont_compute_dumps]"
  exit 1
fi

# Check if the file exists, then load it
if [ -f "$CONFIG_FILE" ]; then
  source "$CONFIG_FILE"
else
  echo "Error: Config file $CONFIG_FILE not found."
  exit 1
fi

RES_OPT=""
if [ -n "${RESERVATION:-}" ] || [ -n "${reservation:-}" ]; then
  VAL="${RESERVATION:-$reservation}"
  RES_OPT="--reservation=${VAL}"
fi

CSV_RESULTS_FILE=$PATH_TO_PREPROCESSING_METADATA/tokenize-$TOKENIZER_NAME-$DATASET_NAME.csv # Used later by tokenize.sh
PATH_TO_DATATROVE_LOGGING_DIR=$PATH_TO_OUTPUT_FOLDER/logs/datatrove_logs                    # Where datatrove logs are stored
PATH_TO_SLURM_LOGGING_DIR=$PATH_TO_OUTPUT_FOLDER/logs/slurm_logs                            # Where slurm logs are stored
DATASET_OUTPUT_FOLDER_NAME=$PATH_TO_OUTPUT_FOLDER/$TOKENIZER_NAME/$DATASET_NAME             # Where tokenized data is stored
TOKENIZER_BATCH_SIZE=${TOKENIZER_BATCH_SIZE:-10000}
TOKENIZER_BATCH_BYTES=${TOKENIZER_BATCH_BYTES:-33554432}
TOKENIZER_WORKERS_OVERRIDE=${TOKENIZER_WORKERS:-}
TOKENIZER_THREADS_OVERRIDE=${TOKENIZER_THREADS:-}

if [ "$DONT_COMPUTE_DUMPS" -eq 0 ]; then
  mkdir -p $PATH_TO_PREPROCESSING_METADATA/completed-dumps #used later by tokenize.sh
  mkdir -p $PATH_TO_SLURM_LOGGING_DIR
  mkdir -p $DATASET_OUTPUT_FOLDER_NAME
  ln -sfn $DATASET_OUTPUT_FOLDER_NAME $PATH_TO_PREPROCESSING_METADATA/tokenized-dir-link

  # Create dumps
  srun --environment=./env.toml --partition=debug --account=$ACCOUNT --job-name=dumps_prep --export=ALL bash -lc "python3 prepare_dumps.py --dataset-folder '${PATH_TO_RAW_DATASET}' --preprocessing-metadata-folder '${PATH_TO_PREPROCESSING_METADATA}' --n-dumps '${DUMPS_NUMBER}'"
else
  echo "Skipping dump creation and directory setup due to --dont_compute_dumps flag"
fi

echo "slurm_job_id,node,start,end,paths_file,output_folder,dataset_total_size,processed_total_size,number_of_workers_per_node,time,bw,total_tokens_processed,throughput (Million Tokens/Second/Node)" >$CSV_RESULTS_FILE
# Iterate through all dumps paths files
for paths_file in "$PATH_TO_PREPROCESSING_METADATA/dumps"/*; do
  dump=$(grep -oP '(?<=paths_file_)\d+(?=\.txt)' <<<$paths_file)
  output_folder=$DATASET_OUTPUT_FOLDER_NAME/dump-$dump
  logging_dir=$PATH_TO_DATATROVE_LOGGING_DIR/$TOKENIZER_NAME/$DATASET_NAME/dump-$dump
  file_count=$(awk 'NF { count++ } END { print count + 0 }' "$paths_file")
  if (( file_count < 1 )); then
    echo "Error: paths file $paths_file contains no input files."
    exit 1
  fi

  tokenizer_tasks=$NUMBER_OF_DATATROVE_TASKS
  if (( tokenizer_tasks > file_count )); then
    tokenizer_tasks=$file_count
  fi

  if [[ -n $TOKENIZER_WORKERS_OVERRIDE ]]; then
    tokenizer_workers=$TOKENIZER_WORKERS_OVERRIDE
  else
    tokenizer_workers=$tokenizer_tasks
    if (( tokenizer_workers > 32 )); then
      tokenizer_workers=32
    fi
  fi

  effective_workers=$tokenizer_workers
  if (( effective_workers < 1 )); then
    effective_workers=$tokenizer_tasks
    if (( effective_workers > 32 )); then
      effective_workers=32
    fi
  fi
  if (( effective_workers > tokenizer_tasks )); then
    effective_workers=$tokenizer_tasks
  fi
  if (( effective_workers < 1 )); then
    effective_workers=1
  fi

  if [[ -n $TOKENIZER_THREADS_OVERRIDE ]]; then
    tokenizer_threads=$TOKENIZER_THREADS_OVERRIDE
  else
    tokenizer_threads=$((CPUS_PER_TASK / effective_workers))
    if (( tokenizer_threads > 144 )); then
      tokenizer_threads=144
    fi
  fi
  if (( tokenizer_threads < 1 )); then
    tokenizer_threads=1
  fi

  sbatch $RES_OPT --partition=$PARTITION --account=$ACCOUNT --nodes=$NODES --gres=gpu:$GPUS --time=$TIME --cpus-per-task=$CPUS_PER_TASK $NO_REQUEUE --job-name=tokenize-$DATASET_NAME-dump-$dump --output=$PATH_TO_SLURM_LOGGING_DIR/R-%x-%j.out --error=$PATH_TO_SLURM_LOGGING_DIR/R-%x-%j.err tokenize.sh $PATH_TO_PREPROCESSING_METADATA/raw-dataset-link $output_folder $TOKENIZER $logging_dir $CSV_RESULTS_FILE $paths_file $tokenizer_tasks $COLUMN_KEY $REHYDRATE_FLAG $EXTENSION $TOKENIZER_BATCH_SIZE $TOKENIZER_BATCH_BYTES $tokenizer_workers $tokenizer_threads
done
