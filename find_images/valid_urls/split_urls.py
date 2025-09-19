#!/usr/bin/env python3

def split_urls(input_file, chunk_size=800, output_prefix="valid"):
    """
    Split a text file of URLs into multiple files with a specified number of URLs each.

    Args:
        input_file (str): Path to the input file containing one URL per line.
        chunk_size (int): Number of URLs per split file.
        output_prefix (str): Prefix for the output files. Each file will be named {output_prefix}{n}.txt.
    """
    # Read all URLs, stripping whitespace and ignoring empty lines
    with open(input_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]

    total_urls = len(urls)
    # Calculate how many files are needed
    num_files = (total_urls + chunk_size - 1) // chunk_size

    for i in range(num_files):
        start_index = i * chunk_size
        end_index = start_index + chunk_size
        chunk = urls[start_index:end_index]
        output_filename = f"{output_prefix}{i+1}.txt"
        # Write the chunk to the output file
        with open(output_filename, 'w', encoding='utf-8') as out_f:
            for url in chunk:
                out_f.write(url + "\n")
        print(f"Wrote {len(chunk)} URLs to {output_filename}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split a URL list into multiple files.")
    parser.add_argument(
        "input_file",
        help="Path to the input txt file with one URL per line"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=800,
        help="Number of URLs per output file"
    )
    parser.add_argument(
        "--output_prefix",
        default="valid",
        help="Prefix for output files (e.g., 'valid' gives valid1.txt, valid2.txt, etc.)"
    )
    args = parser.parse_args()
    split_urls(args.input_file, args.chunk_size, args.output_prefix)
