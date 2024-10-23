# Step 1: train a baseline intent descriptor model via cross-encoder T5
 - Input: query + positive document
 - Output: intent
 - Modify from traning_GQR.py
 
 #  Tips for training:
 Manually checking arguments:
- _v3.py: load-checkpoint
- _v3.sh: warmup
- _v3.sh: model_output_path


# model_name:
- google/flan-t5-base (pretrained)
- t5-base (intent model)
- t5-large (intent model)
- t5-base-original (pretrained)
- t5-large-original (pretrained)