from xml.dom.minidom import parse, parseString 

with open("raw/law/us_code_xml/usc01.xml") as file:
    document = parse(file)
    
content = document.getElementsByTagName("content")

content[0].childNodes[1].childNodes[0].data
content[0].childNodes[3].childNodes[0].data