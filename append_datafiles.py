import os

def append_text_files(directory, output_filename):
    # Ensure the directory exists
    if not os.path.isdir(directory):
        print(f"The directory '{directory}' does not exist.")
        return
    
    # Create the full path for the output file
    output_file_path = os.path.join(directory, output_filename)
    
    # Open the output file in write mode
    with open(output_file_path, 'w') as outfile:
        # Iterate over all files in the directory
        print(sorted(os.listdir(directory)))
        for filename in sorted(os.listdir(directory)):
            # Check if the file is a text file
            if filename.endswith('.txt') and filename.startswith("k"):
                file_path = os.path.join(directory, filename)
                # Open the text file in read mode
                with open(file_path, 'r') as infile:
                    # Read the contents of the file and write to the output file
                    outfile.write(infile.read())
                    # Optionally add a newline character after each file's content
                    outfile.write('\n')
    
    print(f"All text files have been appended into '{output_file_path}'")

# Example usage
if __name__ == "__main__":
    # Prompt the user for the directory and output filename
    directory = "/home/tin/FER/Diplomski/4.semestar/Diplomski rad/code/act_inf_logs/ex8-precisions/k/data/"
    output_filename = "combined.txt"
    
    # Append the text files
    append_text_files(directory, output_filename)

