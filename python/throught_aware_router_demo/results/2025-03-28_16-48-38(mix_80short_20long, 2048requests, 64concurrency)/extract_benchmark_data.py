import os

# Function to extract the relevant portion from the benchmark.log
def extract_benchmark_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Find the section starting from "============ Serving Benchmark Result ============"
    start_index = None
    for i, line in enumerate(lines):
        if "============ Serving Benchmark Result ============" in line:
            start_index = i
            break
    
    if start_index is None:
        return ""  # No benchmark data found
    
    # Extract the relevant part of the file
    return ''.join(lines[start_index:])

# Function to extract Worker tokens info from router_pow_2.log
def extract_worker_tokens(file_path):
    worker_info = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Read the last lines and look for worker information
    for line in reversed(lines):
        if "Worker: http://localhost:" in line and "Tokens:" in line:
            worker_info.append(line.strip())
            if len(worker_info) == 2:  # We only need the last two worker entries
                break
    
    return '\n'.join(reversed(worker_info))  # Return the two worker entries in correct order

# Function to process all experiments in the pow_2 directory
def process_experiments(directory):
    output_file = os.path.join(directory, 'test_pow_2.txt')
    
    # Open the output file in append mode
    with open(output_file, 'a') as outfile:
        # Loop through each experiment directory
        for experiment_dir in os.listdir(directory):
            experiment_path = os.path.join(directory, experiment_dir)
            
            if os.path.isdir(experiment_path):
                benchmark_log_path = os.path.join(experiment_path, 'benchmark.log')
                router_log_path = os.path.join(experiment_path, 'router_pow_2.log')
                
                if os.path.exists(benchmark_log_path) and os.path.exists(router_log_path):
                    # Extract worker info from router_pow_2.log
                    worker_info = extract_worker_tokens(router_log_path)
                    # Extract benchmark data from benchmark.log
                    benchmark_data = extract_benchmark_data(benchmark_log_path)
                    
                    # If benchmark data and worker info were found, write them to the output file
                    if benchmark_data:
                        outfile.write(f"Experiment: {experiment_dir}\n")
                        outfile.write(f"Worker Info:\n{worker_info}\n")
                        outfile.write(benchmark_data)
                        outfile.write("\n\n")

# Main execution
if __name__ == "__main__":
    # Set the directory path where 'pow_2' is located
    base_directory = "./"  # Replace this with the actual path
    
    pow_2_directory = os.path.join(base_directory, 'pow_2')
    
    if os.path.exists(pow_2_directory):
        process_experiments(pow_2_directory)
        print(f"Benchmark data has been written to {os.path.join(pow_2_directory, 'test_pow_2.txt')}")
    else:
        print(f"Directory '{pow_2_directory}' not found.")
