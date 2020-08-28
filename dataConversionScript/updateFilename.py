#! /usr/bin/env python
import os
import xml.dom.minidom as md 


def main():
    currDir = os.path.split(os.getcwd())[1]
    #print(currDir)
    #xmlFile = md.parse("test.xml")
    #print(xmlFile.getElementsByTagName('filename')[ 0 ].firstChild.nodeValue)
    #xmlFile.getElementsByTagName( "filename" )[ 0 ].childNodes[ 0 ].nodeValue = "testProva.jpg" 
    #path = xmlFile.getElementsByTagName('path')[ 0 ].firstChild.nodeValue
    #print(path)
    #print(os.path.dirname(path)+"/testprova.jpg")
    #xmlFile.getElementsByTagName('path')[ 0 ].firstChild.nodeValue = os.path.dirname(path)+"/testprova.jpg"
    #with open( "test.xml", "w" ) as fs:
    #    fs.write( xmlFile.toxml() ) 
    #    fs.close()  
    for count, fileFullName in enumerate(os.listdir('.')): 
        extension = os.path.splitext(fileFullName)[1]
        fileName = os.path.splitext(fileFullName)[0]
        if extension == ".xml":
            xmlFile = md.parse(fileFullName)
            tagFileName = xmlFile.getElementsByTagName('filename')[ 0 ].firstChild.nodeValue
            tagFileName = currDir + "_" + tagFileName
            xmlFile.getElementsByTagName('filename')[ 0 ].firstChild.nodeValue = tagFileName       
            fullPath = xmlFile.getElementsByTagName('path')[ 0 ].firstChild.nodeValue
            xmlFile.getElementsByTagName('path')[ 0 ].firstChild.nodeValue = os.path.dirname(fullPath) + "/" + tagFileName
            with open( fileFullName, "w" ) as fs:
                fs.write( xmlFile.toxml() ) 
                fs.close() 
        if extension == ".xml" or extension == ".jpg":
            oldPath = './' + fileFullName
            newPath = './' + currDir + '_' + fileName + extension
            os.rename(oldPath, newPath)
            print(newPath)


if __name__ == "__main__":
    main()