import subprocess

def run_instance(start_index, end_index):
    subprocess.Popen(['python', 'your_script.py', '--start', str(start_index), '--end', str(end_index)])

total_cpgs = 27517 #44417 # 27517 > 0.5  # 263497 Total number of CpG sites
batch_size = 1000  # Number of CpGs per batch
parallel_processes = 35  # Number of instances to run in parallel


# Generate batches
batches = [(i, min(i + batch_size - 1, total_cpgs - 1)) for i in range(0, total_cpgs, batch_size)]
print(batches)
print(len(batches) / parallel_processes)


for i in range(0, len(batches), parallel_processes):
    processes = []
    for j in range(parallel_processes):
        if i + j < len(batches):
            start_index, end_index = batches[i + j]
            print(f"Starting process for CpGs {start_index} to {end_index}")
            p = subprocess.Popen(['python', 'MethPrediction.Model.py', '--start', str(start_index), '--end', str(end_index)])
            processes.append(p)
    # Wait for all processes to finish
    for p in processes:
        p.wait()
