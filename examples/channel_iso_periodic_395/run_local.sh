#!/bin/bash

# -----------------------------------------------------------------------------
# CHAPSim2 Execution Script
# This script runs CHAPSim2 with options for processors and memory checking via Valgrind.
# -----------------------------------------------------------------------------

# Define the base directory for CHAPSim
chap_sim_dir="../CHAPSim2"

# Define base filename with timestamp
timestamp=$(date +'%Y-%m-%d_%H.%M')
base_filename="output_chapsim2_${timestamp}.log"
output_file="$base_filename"

# Ensure unique output filename if it exists
count=2
while [ -e "$output_file" ]; do
    output_file="${base_filename%.*}_$count.${base_filename##*.}"
    count=$((count + 1))
done

# -----------------------------------------------------------------------------
# Step 1: Create necessary directories
# -----------------------------------------------------------------------------
# Check if '0_src' directory exists, if not, create it
if [ ! -d "0_src" ]; then
    mkdir "0_src" || { echo "Error: Failed to create '0_src' directory"; exit 1; }
    echo "'0_src' directory created."
fi

# Create a timestamped directory for the source files
src_dir="src_${timestamp}"
cp -r "$chap_sim_dir/src" "0_src/$src_dir" || { echo "Error: Failed to copy source files"; exit 1; }
echo "Source files copied to '0_src/$src_dir'."

# -----------------------------------------------------------------------------
# Step 2: Ask user for processor count
# -----------------------------------------------------------------------------
# Default to 1 if no input is provided
read -p "Enter the number of processors (default is 1): " num_processors
num_processors=${num_processors:-1}  # Default to 1 if empty

# Validate input: Ensure num_processors is a positive integer
if ! [[ "$num_processors" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid input. Using default value: 1"
    num_processors=1
fi

# -----------------------------------------------------------------------------
# Step 3: Ask user if Valgrind should be enabled
# -----------------------------------------------------------------------------
read -p "Enable Valgrind? (y/n, default is n): " use_valgrind
use_valgrind=${use_valgrind:-n}  # Default to 'n' if empty

# -----------------------------------------------------------------------------
# Step 4: Run CHAPSim normally or with Valgrind
# -----------------------------------------------------------------------------
echo "Running CHAPSim with $num_processors process(es)..."
nohup mpirun -np "$num_processors" "$chap_sim_dir/bin/CHAPSim" > "$output_file" 2>&1 &


# Capture the PID of the running mpirun process
mpirun_pid=$!

# Check if Valgrind is enabled
if [ "$use_valgrind" = "y" ]; then
    echo "Running with Valgrind..."
    # Run Valgrind with the actual PID in the log filename
    mpirun -np "$num_processors" valgrind --track-origins=yes --leak-check=full --show-leak-kinds=all \
        --tool=memcheck --error-limit=no --verbose "$chap_sim_dir/bin/CHAPSim" 2> "valgrind_output_${mpirun_pid}.log" &
    echo "Valgrind output will be saved in valgrind_output_${mpirun_pid}.log"
fi

echo "Command is running in the background. Output will be saved to $output_file."
