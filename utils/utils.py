import os


def init():
    # Specify the file path
    file_path = '/Users/welcome/.zprofile.sh'
    # Read the file content
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse and set environment variables
    for line in lines:
        # Remove any leading/trailing whitespaces
        line = line.strip()

        # Skip empty lines or comments
        if not line or line.startswith('#'):
            continue

        # Remove 'export' and split into key-value pair
        if line.startswith('export '):
            line = line[len('export '):]  # Remove 'export ' part

        # Split the line at the first '=' and strip spaces
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()  # Remove leading/trailing spaces from the key
            value = value.strip().strip('"')  # Remove spaces and quotes from value
            os.environ[key] = value  # Set the environment variable
