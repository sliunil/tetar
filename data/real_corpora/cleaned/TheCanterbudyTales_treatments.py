import re

input_file_path = "../raw/poetry/GeoffreyChaucer_TheCanterburyTales.txt"
output_file_path = "poetry/GeoffreyChaucer_TheCanterburyTales_pre.txt"

with open(input_file_path) as file:
    lines = file.readlines()
    
output_text = ""
writing_on = True
for line in lines:
    if re.match("^\\d", line.strip()) or \
       re.match("^HEADING", line.strip()) or \
       re.match("^QUOTATION", line.strip()) or \
       re.match("^COLOPHON", line.strip()):
        writing_on = False
    if line.strip() == "":
        writing_on = True
    if not re.match("^\[", line.strip()):
        if writing_on:
            start_of_digit = re.search("(\d|\))*$", line).start()
            output_text += line[:start_of_digit-1] + "\n"
        
with open(output_file_path, "w") as file:
    file.write(output_text)
    
