# About this model and its explanations

The heatmap (Grad-CAM) shows which image regions increased the model's score for the predicted class; for this architecture its values add up exactly to the model's score. The model weights its evidence by the detected lesion area, so the map is lesion-weighted evidence: it shows what the model used, not a diagnosis by itself. [NeuroEdge]

The lesion outline comes from the segmentation network trained on expert tumor masks; it localises the lesion, and the classifier gathers its evidence mainly inside that region. [NeuroEdge]

On held-out test tumors, about 73% of the Grad-CAM evidence lies inside the expert tumor outline and the strongest point falls on the tumor in 99% of cases; glioma maps are the least focused (about 64%). [NeuroEdge]

On a held-out test set the model reached about 95% accuracy; on images from other hospitals accuracy fell to about 87%, so results on new data need expert review. [NeuroEdge]

The 95% prediction set lists every class the calibrated model cannot rule out at that confidence level; a set with more than one class, or an empty set, means the result is uncertain. On the held-out test set these sets contained the true class about 91% of the time, slightly below the 95% target, and least often for glioma. [Conformal] [NeuroEdge]

References: [NeuroEdge] docs/FINDINGS.md. [Conformal] Sadinle M, Lei J, Wasserman L. Least ambiguous set-valued classifiers with bounded error levels. JASA 2019.
