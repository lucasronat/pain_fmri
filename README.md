# pain_fmri
Pain expectation effect on pain perception in healthy adults: an fMRI study

This study aims to investigate brain activity in contexts of thermal pain at different intensities, depending on whether visual cues indicate the intensity of the upcoming stimulation. The goal is to assess changes in brain activity during pain perception based on whether the visual cues are congruent or incongruent.

Experimental conditions:

“LOW” cue – “LOW” stimulation
‘LOW’ cue – “MODERATE” stimulation
“MODERATE” cue – “MODERATE” stimulation
‘HIGH’ cue – “MODERATE” stimulation
“HIGH” cue – “HIGH” stimulation

Statistical analysis

GLM analyses (first- and second-level) are performed using the Python package nilearn. 

The analysis script (.py) includes extracting the events files, creating design matrices, setting up analysis contrasts, followed by first-level GLMs (within-subject), with multiple figure records and the creation of .html files, as well as the second-level GLM (group analysis).

The .sh file contains the configurable parameters for the .py script and allows you to specify the desired contrast, the statistical threshold, the type of statistical correction, and the segmentation atlas used to assign anatomical regions to the activation peaks of the significant clusters derived from the GLMs.
