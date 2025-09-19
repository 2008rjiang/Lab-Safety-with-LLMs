
##### Read the input_file, extract URLs, filter out those containing 'logo', and write the rest to output_file.
"""
import re

def filter_links(input_file, output_file):

    # Regex pattern to match HTTP/HTTPS URLs
    url_pattern = re.compile(r'(https?://\S+)')
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Find all URLs in the text
    links = url_pattern.findall(text)

    # Filter out URLs containing 'logo' (case-insensitive)
    filtered = [link for link in links if 'logo' not in link.lower()]

    # Write filtered URLs to the output file, one per line
    with open(output_file, 'w', encoding='utf-8') as fw:
        for link in filtered:
            fw.write(link + '\n')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Filter out URLs containing "logo" from a text file.'
    )
    parser.add_argument(
        'input_file', help='Path to the input text file containing URLs or text with URLs'
    )
    parser.add_argument(
        'output_file', help='Path to save the filtered URLs'
    )
    args = parser.parse_args()

    filter_links(args.input_file, args.output_file)
"""


##### Read the input_file line by line, drop any line containing the forbidden_substring,and write the remaining lines unchanged to output_file.
    
import argparse

def filter_lines(input_file, output_file, forbidden_substring="slide"):

    forbidden_substring = forbidden_substring.lower()
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            if forbidden_substring not in line.lower():
                fout.write(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Remove any lines containing a given substring (default "logo") from a text file.'
    )
    parser.add_argument(
        'input_file',
        help='Path to the input text file'
    )
    parser.add_argument(
        'output_file',
        help='Path to save the filtered text'
    )
    parser.add_argument(
        '--substring', '-s',
        default='logo',
        help='Substring to filter out (case-insensitive). Default: "logo".'
    )
    args = parser.parse_args()

    filter_lines(args.input_file, args.output_file, args.substring)
