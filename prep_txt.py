import os

root_dir = "."
output_file = "python_files.txt"

with open(output_file, "w") as f:
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                if filename == "prep_txt.py" or filename == "util.py":
                    continue
                file_path = os.path.join(dirpath, filename)
                f.write(f"File: {filename}\n")
                f.write(f"Path: {file_path}\n\n")
                with open(file_path, "r", encoding="utf-8") as py_file:
                    f.write(py_file.read())
                    f.write("\n\n")
