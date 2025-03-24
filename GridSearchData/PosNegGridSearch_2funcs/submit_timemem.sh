#!/bin/bash
# Usage: ./fsubmit.sh <num_cores> <total_tasks>
#cd /Users/nt625/Documents/GitHub/ACSPaperSep2024/PAPERREVIEWS/NEWCODE/DoublePositive

# NUM_CORES=32
# TOTAL_TASKS=1000
# start_time=$(date +%s)

# for (( i=0; i<TOTAL_TASKS; i++ )); do
#   echo "Running task $i"
#   (
#     python code1_posneg_optimisedsolve.py "$i" > /dev/null 2>&1
#   ) &

#   # Run up to NUM_CORES in parallel
#   if (( $((i % NUM_CORES)) == $((NUM_CORES - 1)) )); then
#     wait  # Wait for current batch of processes to finish
#     end_time=$(date +%s)
#     elapsed_time=$((end_time - start_time))
#     echo "Elapsed time: $elapsed_time seconds"
#   fi
# done

# wait  # Ensure any remaining processes finish
# echo "All tasks are done."


#!/bin/bash
# Usage: ./farm_out.sh <num_cores> <total_tasks>

NUM_CORES=32
TOTAL_TASKS=1000

start_time=$(date +%s)
total_memory_usage=0
total_time=0

# Function to measure memory and time for a single task
measure_task() {
  task_index=$1
  start_task_time=$(date +%s)
  
  # Start the task and capture its PID
  python code1_posneg_optimisedsolve.py "$task_index" > /dev/null 2>&1 &
  task_pid=$!
  
  # Wait for the task to finish
  wait $task_pid
  
  # Measure task's memory usage and running time
  end_task_time=$(date +%s)
  task_runtime=$((end_task_time - start_task_time))
  
  # Use ps to measure memory usage of the task in KB
  task_memory_kb=$(ps -o rss= -p $task_pid)
  
  # Convert memory from KB to bytes (1 KB = 1024 bytes)
  task_memory_bytes=$((task_memory_kb * 1024))
  
  # Add to total memory and time usage
  total_memory_usage=$((total_memory_usage + task_memory_bytes))
  total_time=$((total_time + task_runtime))
  
  echo "Task $task_index completed in $task_runtime seconds using $task_memory_bytes bytes"
}

# Loop through all tasks
for (( i=0; i<TOTAL_TASKS; i++ )); do
  echo "Starting task $i"
  
  # Run the task in the background
  measure_task $i &
  
  # Ensure not exceeding NUM_CORES running tasks at once
  if (( $((i % NUM_CORES)) == $((NUM_CORES - 1)) )); then
    wait  # Wait for the current batch of processes to finish
    echo "Batch finished."
  fi
done

wait  # Ensure any remaining processes finish

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))

# Print total time and memory usage
echo "Total execution time: $elapsed_time seconds"
echo "Total memory usage: $total_memory_usage bytes"
echo "All tasks are done."
