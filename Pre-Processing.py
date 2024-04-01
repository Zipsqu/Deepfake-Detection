# Both of these codes are contating variables that should be adjusted (such as destination/output directory).
# Path to these codes should be adjusted as required (specify XXX).

#Face & Landmark extraction, this code uses multi-processing so it is quiet heavy on CPU.
python "XXX/Pre-Processing/Face & Landmark Extraction.py"

#Splitting dataset into training/validation/testing (80%,10%,10%), while making sure frames related to one video will stay in same folder.
python "XXX/Pre-Processing/Dataset Split.py"
