#!/bin/bash

python glm_nilearn_tedana.py --contrast "pain" --contrast_name "painMainFx" --alpha 0.001 --height_control "fdr" --atlas_name "AAL3"

#contrasts = {
#     'pain': array([1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), seuil fdr 0.001
#     'cue': array([0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.]), seuil fdr 0.001
#     'painHighMinusLow': array([-1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.]), seuil fdr 0.001
#     'cueHighMinusLow': array([ 0.,  0.,  0.,  0.,  0., -1.,  0.,  1.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.]), seuil fdr 0.001
#     'hypoalgesia': array([ 0.,  1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.]), seuil fpr 0.001
#     'hyperalgesia': array([ 0.,  0., -1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.]), seuil fpr 0.001
#     'expectation': array([0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])}, seuil fpr 0.001
#}

#    "painMainFx": np.asarray([1,1,1,1,1,0,0,0]), seuil fdr 0.001
#    "painRegression": np.asarray([1,2,2,2,3,0,0,0])/10., seuil fdr 0.001
#    "cueRegression": np.asarray([0,0,0,0,0,1,2,3])/6., seuil fdr 0.001
#    "painHighMinusLow": np.asarray([-1,0,0,0,1,0,0,0]), seuil fdr 0.001
#    "cueHighMinusLow": np.asarray([0,0,0,0,0,-1,0,1]), seuil fdr 0.001
#    "hypoalgesia": np.asarray([0,1,-1,0,0,0,0,0]), seuil fpr 0.001
#    "hyperalgesia": np.asarray([0,0,-1,1,0,0,0,0]), seuil fpr 0.001
#    "hyperalgesia-hypoalgesia": np.asarray([0,-1,0,1,0,0,0,0]), seuil fpr 0.001
#    "incongruence-congruence": np.asarray([0,1,0,1,0,0,0,0]), seuil fpr 0.001
#}

#alpha : 0.05 or 0.001

#height_control : "fpr" or "fdr"

#atlas_name : "AAL3"

#Rolls, E. T., Huang, C. C., Lin, C. P., Feng, J., & Joliot, M. Automated anatomical labelling atlas 3.  Neuroimage, 2020, 206, 116189, doi:10.1016/j.neuroimage.2019.116189
#45 cortical and subcortical labels (amygdala, caudate, putamen, pallidum, thalamus (AAL)
# + orbitofrontal and cerebellar parcellation (AAL2)
# + parcellation of the thalamic nuclei, anterior cingulaite, accumbens, VTA, SN, red nucleus, locus coeruleus, raphe (AAL3)
