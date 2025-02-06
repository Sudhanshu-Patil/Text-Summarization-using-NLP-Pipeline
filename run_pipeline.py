import subprocess

def run_script(script_name, timeout=300):
    try:
        process = subprocess.Popen(['python', script_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=timeout)
        if process.returncode != 0:
            print(f"Error running {script_name}:")
            print(stderr)
            exit(1)
        else:
            print(f"Successfully ran {script_name}")
            print(stdout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        print(f"Timeout expired for {script_name}")
        print(stderr)

scripts = [
    'preprocess.py',
    'augment_data.py',
    'split_data.py',
    'train_model.py',
    'ner_extraction.py',
    'combine_extraction.py',
    'extract_entities_for_snippets.py',
    'evaluate_model.py'
]

for script in scripts:
    run_script(script)
