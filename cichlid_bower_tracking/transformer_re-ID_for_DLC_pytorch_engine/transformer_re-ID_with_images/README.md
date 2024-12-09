## Experiment: transformer re-ID "with images"

In this experiment, each triplet is augmented with image crops of the anchor, positive, negative fish assemblies before being passed into a transformer re-ID model. Hence, this experiment is now referred to as re-ID "with images." This approach was first proposed and tested by Adam Thomas (summer 2024) on DeepLabCut’s TensorFlow engine and replicated by Thuan Nguyen on DeepLabCut’s newer PyTorch engine. (The TensorFlow engine is being deprecated.) Please contact these team members for further clarifications.

The credit for the `behavior_detection` package and the majority of the code in the `training_transformer_reID_with_images.py` script goes to Adam Thomas, as the versions here were adapted from his original code. Adam initially wrote the code for this experiment on DeepLabCut’s TensorFlow engine, and the code can be found on Adam’s repo: https://github.com/athomas125/cichlid-behavior-detection/tree/main/behavior_detection 
