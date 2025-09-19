import re
import argparse

def filter_unique_lines(input_file, output_file):
    """
    Read the input_file line by line, extract the URL on each line,
    drop any line whose URL was seen before, and write the rest (with numbers)
    to output_file.
    """
    url_pattern = re.compile(r'(https?://\S+)')
    seen = set()

    with open(input_file, 'r', encoding='utf-8') as f, \
         open(output_file, 'w', encoding='utf-8') as fw:

        for line in f:
            match = url_pattern.search(line)
            if match:
                url = match.group(1)
                if url in seen:
                    # duplicate URL → skip entire line
                    continue
                seen.add(url)
            # either no URL found or URL is new → keep line
            fw.write(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Remove lines with duplicate URLs (keeping the original numbering).'
    )
    parser.add_argument(
        'input_file', help='Path to the input text file with numbered lines and URLs'
    )
    parser.add_argument(
        'output_file', help='Path to save the filtered lines'
    )
    args = parser.parse_args()

    filter_unique_lines(args.input_file, args.output_file)
