import argparse

def print_lines_reversed(input_file, output_file):
    """Read all lines from input_file and print them in reverse order."""
    with open(input_file, 'r', encoding='utf-8') as f, open(output_file, 'w', encoding='utf-8') as fout:
        lines = f.readlines()
        for line in reversed(lines):
            fout.write(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Print the lines of a text file from last to first.'
    )
    parser.add_argument(
        'input_file',
        help='Path to the text file you want to read'
    )
    parser.add_argument(
        'output_file',
        help='Path to save the filtered text'
    )
    args = parser.parse_args()

    print_lines_reversed(args.input_file, args.output_file)
