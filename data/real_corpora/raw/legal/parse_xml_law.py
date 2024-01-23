from xml.dom.minidom import parse 

with open("raw/law/us_code_xml/usc01.xml") as file:
    document = parse(file)

sections = document.getElementsByTagName("section")

def getText(nodelist):
    rc = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
        elif len(node.childNodes) > 0:
            ic = getText(node.childNodes)
            rc.extend(ic)
    return ''.join(rc)

text = getText(sections)
print(text)