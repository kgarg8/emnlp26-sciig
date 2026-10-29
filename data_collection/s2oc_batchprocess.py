import os
import glob
import time
from doc2json.grobid2json.process_pdf import process_pdf_file

INPUT_DIR = 'iclr_2025_pdfs'
TEMP_DIR = 'temp'
OUTPUT_DIR = 'ICLR_2025_s2orc_json_files'

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

start_time = time.time()

for pdf_file in glob.glob(os.path.join(INPUT_DIR, '*.pdf')):
    print(f"Processing {pdf_file} ...")
    try:
        output_file = process_pdf_file(pdf_file, TEMP_DIR, OUTPUT_DIR)
        print(f"Created: {output_file}")
    except Exception as e:
        print(f"Error processing {pdf_file}: {e}")

print("Total runtime: %.3f seconds" % (time.time() - start_time))
