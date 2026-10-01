# No tumor (model output)

"No tumor" means the model found no evidence for glioma, meningioma or pituitary tumor in this single slice. A single 2D slice cannot exclude a lesion elsewhere in the brain. [NeuroEdge]

The model was trained only on these four classes; other conditions such as stroke, hemorrhage, infection or demyelination are outside its scope and may be misclassified. [NeuroEdge]

In the training data, "no tumor" images were acquired with more varied MRI sequences than tumor images; the model may partly rely on overall image appearance, so this result is less reliable for non-T1 sequences. [NeuroEdge]

References: [NeuroEdge] NeuroEdge model card and findings log (docs/FINDINGS.md).
