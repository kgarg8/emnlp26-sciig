import json
import ijson

INPUT_FILE     = "Preprocessed/Preprocessed_ICLR_2025_s2orc.json"
OUTPUT_PREFIX  = "Preprocessed/ICLR_2025_s2orc_Split"
CHUNK_SIZES    = (1000, 1000, 1000)  # first three files
NUM_PARTITIONS = len(CHUNK_SIZES) + 1

def split_stream(input_path, output_prefix, chunk_sizes):
    # open output files and write opening '['
    outs = [open(f"{output_prefix}_{i+1}.json", "w") for i in range(len(chunk_sizes)+1)]
    for f in outs:
        f.write("[\n")

    counts = [0] * len(outs)
    with open(input_path, "r") as inf:
        for idx, obj in enumerate(ijson.items(inf, "item")):
            # decide which partition this record belongs to
            if idx < chunk_sizes[0]:
                part = 0
            elif idx < sum(chunk_sizes[:2]):
                part = 1
            elif idx < sum(chunk_sizes[:3]):
                part = 2
            else:
                part = 3

            # comma‑separate if not first in that file
            if counts[part]:
                outs[part].write(",\n")
            json.dump(obj, outs[part])
            counts[part] += 1

    # close JSON arrays and files
    for f in outs:
        f.write("\n]\n")
        f.close()

if __name__ == "__main__":
    split_stream(INPUT_FILE, OUTPUT_PREFIX, CHUNK_SIZES)
    print("Done splitting into {} files.".format(NUM_PARTITIONS))