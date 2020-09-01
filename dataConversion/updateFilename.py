#! /usr/bin/env python
import os
import xml.dom.minidom as md 
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        "Concat image names and folder name. Update also names in the xml files.")
    parser.add_argument(
        "--input", "-i", help="Path to data folder.", required=True, type=str
    )
    args, _ = parser.parse_known_args()
    return args



def main():
    args = parse_args()
    workDir = args.input
    if os.path.isdir(workDir):
        print("Converting files in: " + workDir)
    else:
        print("The input directory does not exist")
    for count, fileFullName in enumerate(os.listdir(workDir)): 
        fileFullPath = workDir + "/" + fileFullName 
        extension = os.path.splitext(fileFullPath)[1]
        fileName = os.path.splitext(fileFullPath)[0]
        if extension == ".xml":
            xmlFile = md.parse(fileFullPath)
            tagFileName = xmlFile.getElementsByTagName('filename')[ 0 ].firstChild.nodeValue
            tagFileName = os.path.basename(workDir) + "_" + tagFileName
            xmlFile.getElementsByTagName('filename')[ 0 ].firstChild.nodeValue = tagFileName       
            fullPath = xmlFile.getElementsByTagName('path')[ 0 ].firstChild.nodeValue
            xmlFile.getElementsByTagName('path')[ 0 ].firstChild.nodeValue = os.path.dirname(fullPath) + "/" + tagFileName
            with open( workDir + "/" + fileFullName, "w" ) as fs:
                fs.write( xmlFile.toxml() ) 
                fs.close() 
        if extension == ".xml" or extension == ".jpg":
            oldPath = workDir +"/"+ fileFullName
            newPath = workDir + "/"+os.path.basename(workDir)+'_' + fileFullName
            os.rename(oldPath, newPath)

if __name__ == "__main__":
    main()