from xml.dom.minidom import parse 
import re
import os

# The name of the input and output directories
input_directory_name = "us_code_xml"
output_directory_name = "us_code_parsed"

# Recursive function to parse textual data from xml
def parse_text(nodelist):
    final_text = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            final_text.append(node.data)
        elif len(node.childNodes) > 0:
            restricted_childNodes = []
            for childNode in node.childNodes:
                if childNode.nodeType != node.TEXT_NODE:
                    if childNode.tagName in ('note', 'sourceCredit'):
                        continue
                restricted_childNodes.append(childNode)
            inside_text = parse_text(restricted_childNodes)
            final_text.extend(inside_text)
    return ''.join(final_text)

# Get the file names in the directory
file_names = os.listdir(input_directory_name)

# Loop on files
for file_name in file_names:
    # Open the file
    with open(f"{input_directory_name}/{file_name}") as file:
        document = parse(file)

    # Get the textual data
    parsed_text = parse_text(document.getElementsByTagName("section"))
    # Remove extra "end of line"
    parsed_text = re.sub("\n{2,}", "\n\n", parsed_text)

    # Write the resulting file
    with open(f"{output_directory_name}/{file_name[:-4]}.txt", "w") as file:
        file.write(parsed_text)