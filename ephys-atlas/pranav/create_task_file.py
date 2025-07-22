import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("/mnt/sdceph/users/prai1/data/projects/psychedlics")



def create_task_file(inp_file, outp_file, run_program_path="Runprogram.sh"):
    # Read the CSV file

    if isinstance(inp_file, pd.DataFrame):
        df = inp_file
    elif isinstance(inp_file, (str,Path)):
        df = pd.read_csv(inp_file)
    else:
        raise ValueError(f"Invalid input type: {type(inp_file)}")
    
    # Create the task file
    task_file_path = outp_file
    
    # Generate command lines for each row
    with open(task_file_path, 'w') as f:
        for _, row in df.iterrows():
            command = f"source {run_program_path} --pid {row['pid']} --eid {row['eid']} --probe_name {row['probe_name']} --start_time {row['t_start']} --duration {row['duration']}\n"
            f.write(command)
    
    print(f"Task file created at: {task_file_path}")

if __name__ == "__main__":
    inp_file = OUTPUT_DIR / 'psychedlics_snippets_df.csv'
    outp_file = OUTPUT_DIR / 'Full_Task_file'
    create_task_file(inp_file, outp_file, run_program_path="/mnt/home/prai1/projects/psychedlics/Runprogram.sh")