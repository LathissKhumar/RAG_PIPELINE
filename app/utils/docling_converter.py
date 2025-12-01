import subprocess
import os

def convert_pdf_to_markdown(pdf_path: str, output_dir: str) -> str:
    # Use docling CLI, redirect stdout to markdown file
    md_filename = os.path.splitext(os.path.basename(pdf_path))[0] + ".md"
    md_path = os.path.join(output_dir, md_filename)
    with open(md_path, "w") as md_file:
        subprocess.run(["docling", pdf_path], stdout=md_file, check=True)
    return md_path
