#This code does the pre-processing as discused in README. 
#Replace xxx in the path as required.
#Each of this files usually contains destination folder (for dataset), and the destination (or multiple ones in case of dataset split), these should be adjusted as required. 

#Face & Landmark extraction, this code doesn't use GPU so it can take a fair amount of time (it's efficient on the CPU- this allows tasks in the background).
python "XXX/Pre-Processing/Face & Landmark Extraction.py"

#Re-sizing the images to 224x224 as required by EfficientNetB0
python "XXX/Pre-Processing/Re-sizing.py"

#Adjust the JSON metadata files to resizing & moving to same folder as images.
python "XXX/Pre-Processing/Adjusting JSON.py"

#Splitting dataset into training/validation/testing (80%,10%,10%), while making sure frames related to one video will stay in same folder.
python "XXX/Pre-Processing/Dataset Split.py"

#Previous split has a mistake in naming convention, so this file moves associated JSON files to the previously moved frames.
python "XXX/Pre-PRocessing/Dataset Split FIX.py"
