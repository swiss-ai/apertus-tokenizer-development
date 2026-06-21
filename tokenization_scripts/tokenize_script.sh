#!/bin/bash

# ⚠️ WARNING ⚠️
# Make sure to prepare the dumps before tokenizing the data!
# Check scripts/tokenization/prepare_dumps.py
# ⚠️ WARNING ⚠️

CONFIG_FILE=$1

# Check if the file exists, then load it
if [ -f "$CONFIG_FILE" ]; then
  source "$CONFIG_FILE"
else
  echo "Error: Config file $CONFIG_FILE not found."
  exit 1
fi

REHYDRATE=False # Set to True or False
if [ "$REHYDRATE" = "True" ]; then
  REHYDRATE_FLAG="--rehydrate"
else
  REHYDRATE_FLAG=""
fi

CSV_RESULTS_FILE=$PATH_TO_PREPROCESSING_METADATA/tokenize-$TOKENIZER_NAME-$DATASET_NAME.csv # Used later by tokenize.sh
PATH_TO_DATATROVE_LOGGING_DIR=$PATH_TO_OUTPUT_FOLDER/logs/datatrove_logs                    # Where datatrove logs are stored
PATH_TO_SLURM_LOGGING_DIR=$PATH_TO_OUTPUT_FOLDER/logs/slurm_logs                            # Where slurm logs are stored
DATASET_OUTPUT_FOLDER_NAME=$PATH_TO_OUTPUT_FOLDER/$TOKENIZER_NAME/$DATASET_NAME             # Where tokenized data is stored

mkdir -p $PATH_TO_PREPROCESSING_METADATA/completed-dumps #used later by tokenize.sh
mkdir -p $PATH_TO_SLURM_LOGGING_DIR
mkdir -p $DATASET_OUTPUT_FOLDER_NAME
ln -sfn $DATASET_OUTPUT_FOLDER_NAME $PATH_TO_PREPROCESSING_METADATA/tokenized-dir-link

# Create dumps
sbatch --wait --environment=./env.toml --partition=$PARTITION --account=$ACCOUNT --job-name=dumps_prep --wrap="python3 prepare_dumps.py --dataset-folder '${PATH_TO_RAW_DATASET}' --preprocessing-metadata-folder '${PATH_TO_PREPROCESSING_METADATA}' --n-dumps '${DUMPS_NUMBER}'"

echo "slurm_job_id,node,start,end,paths_file,output_folder,dataset_total_size,processed_total_size,number_of_workers_per_node,time,bw,total_tokens_processed,throughput (Million Tokens/Second/Node)" >$CSV_RESULTS_FILE
# Iterate through all dumps paths files
for paths_file in "$PATH_TO_PREPROCESSING_METADATA/dumps"/*; do
  dump=$(grep -oP '(?<=paths_file_)\d+(?=\.txt)' <<<$paths_file)
  output_folder=$DATASET_OUTPUT_FOLDER_NAME/dump-$dump
  logging_dir=$PATH_TO_DATATROVE_LOGGING_DIR/$TOKENIZER_NAME/$DATASET_NAME/dump-$dump
  sbatch --partition=$PARTITION --account=$ACCOUNT --nodes=$NODES --gres=gpu:$GPUS --time=$TIME --cpus-per-task=$CPUS_PER_TASK $NO_REQUEUE --job-name=tokenize-$DATASET_NAME-dump-$dump --output=$PATH_TO_SLURM_LOGGING_DIR/R-%x-%j.out --error=$PATH_TO_SLURM_LOGGING_DIR/R-%x-%j.err tokenize.sh $PATH_TO_PREPROCESSING_METADATA/raw-dataset-link $output_folder $TOKENIZER $logging_dir $CSV_RESULTS_FILE $paths_file $NUMBER_OF_DATATROVE_TASKS $COLUMN_KEY $REHYDRATE_FLAG
done
