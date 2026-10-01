# About this model and its explanations

The heatmap (Grad-CAM) shows which image regions increased the model's score for the predicted class; for this architecture its values add up exactly to the model's score. It shows what the model used, not where disease is. [NeuroEdge]

Finer-CAM compares the predicted class with the runner-up class: red regions support the prediction, blue regions support the runner-up. [Finer-CAM]

The lesion outline comes from a separate segmentation network trained on expert tumor masks; it localises the lesion but does not explain the classification. [NeuroEdge]

For pituitary tumors, the classifier's evidence lies largely in surrounding anatomy rather than on the tumor itself; the lesion outline is the better guide to tumor location. [NeuroEdge]

On a held-out test set the model reached about 95% accuracy; on images from other hospitals accuracy fell to about 86%, so results on new data need expert review. [NeuroEdge]

The 95% prediction set lists every class the calibrated model cannot rule out at that confidence level; a set with more than one class, or an empty set, means the result is uncertain. [Conformal]

References: [NeuroEdge] docs/FINDINGS.md. [Finer-CAM] Zhang Z et al. Finer-CAM. CVPR 2025. [Conformal] Sadinle M, Lei J, Wasserman L. Least ambiguous set-valued classifiers with bounded error levels. JASA 2019.
