# classifier

Aim: to classify the reliability of Δν measurements using neural networks

===============

Author: Claudia Reyes
Dates: 2020-2014

===============

Published: Monthly Notices of the Royal Astronomical Society, Volume 511, Issue 4, April 2022, Pages 5578–5596
Title: Vetting asteroseismic Δν measurements using neural networks
Authors: Claudia Reyes, Dennis Stello, Marc Hon, Joel C Zinn
DOI: https://doi.org/10.1093/mnras/stac445

===============

MODULES
=======

- Preferred python version: 
* 3.8 *

- Followind modules are used, and normally part of basic  python environment:
* numpy *
* matplotlib *
* pandas *
* os *
* re *
* math *
* scipy *

- Special modules used, likely need to be installed with the commands:
> pip install opencv-python
> pip install scikit-learn 


HOW TO USE
==========

1. Open Dnu_classifier.py using a text editor
2. Locate the lines between "INPUT" and "INPUT END" 
3. Replace the paths with the corresponding path of the user machine
4. If needed, replace the value of "threshold" with a number between 0 and 1.
5. Save changes to Dnu_classifier.py
6. run the classifier from terminal using the command "python Dnu_classifier.py" this may take between 2 and 10 minutes depending on the amount of data.
