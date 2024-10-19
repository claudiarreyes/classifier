# classifier

Aim: to classify the reliability of Δν measurements using neural networks

Inputs required:

1. Background corrected, long cadence power spectra, files should end in '.psd.bgcorr'
2. File containing pipeline results of estimated Δν and ν_{max}, in coloumns named "numax(gaus)" and "dnu", respectively. Column named "file" should correspond to the names of the PSD files, but ending on '.psd' (not '.psd.bgcorr')

Outputs:

1. histogram_NAMEDATA.png
2. results_NAMEDATA.png
3. NAMEDATA_ALL_N1_GOODDNU_N2_FP_N3

Where *NAMEDATA* is the name of the folder containing the power spectra, *N1* is the number of stars (or PSD) with dnu and numax results from the pipeline, *N2* is the number of dnu with probabilities exceeding the threshold imposed, and *N3* is the number of false positives identified because their numax/dnu values stray too much from the empyrical relation.

Files 1 and 2 are useful graphical representations of the results.
File 3 is the table with the calculated probabilities by the neural network.
  
===============

Author: Claudia Reyes

Dates: 2020-2024

===============

Published: Monthly Notices of the Royal Astronomical Society, Volume 511, Issue 4, April 2022, Pages 5578–5596

Title: Vetting asteroseismic Δν measurements using neural networks

Authors: Claudia Reyes, Dennis Stello, Marc Hon, Joel C Zinn

DOI: https://doi.org/10.1093/mnras/stac445

===============

MODULES
=======

Preferred python version:
  
* **3.8**

Followind modules are used, and normally part of basic  python environment:
  
* **numpy**
* **matplotlib**
* **pandas**
* **os**
* **re**
* **math**
* **scipy**


Special modules **cv2** and **sklearn** likely need to be installed with the commands:
````
  > pip install opencv-python
  > pip install scikit-learn 
````

HOW TO USE
==========

Download the project folder to your machine and:

1. Open Dnu_classifier.py using a text editor
2. Locate the lines between "INPUT" and "INPUT END" 
3. Replace the paths with the corresponding path of the user machine
4. If needed, replace the value of "threshold" with a value between 0 and 1.
5. Save changes to Dnu_classifier.py
6. run the classifier from terminal using the command "python Dnu_classifier.py" this may take between 2 and 10 minutes depending on the amount of data.
