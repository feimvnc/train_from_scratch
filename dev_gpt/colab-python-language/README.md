## A Program to Train GPT Model

**The script works on Google Colab.
Python packages must be update / changed to allow training on local CPU machine, and there could be issues related to torch or numpy, and extra time of debugging and training are needed.**

## Open Google Colab
https://colab.research.google.com/
Open Content tab, upload files 

## How to Run

```bash
# Click Terminal tab at bottom left on colab homepage
# mkdir dev_gpt
# Upload files from local to the folder
# Execute below commands

# Create venv, skipped due to error
# python -m venv .venv
# source .venv/bin/activate

# Install dependencies: 
pip install -r requirements.txt

# Start training: 
python train.py
# Wait for the loss to drop (it might take a few minutes on CPU).

# Generate code: 
python generate.py
```

## Training Output on Colab, content folder files

/content/dev_gpt# python train.py                                                   
Running on cuda
Loading dataset from HuggingFace...
Repo card metadata block was not found. Setting CardData to empty.
Resolving data files: 100%|██████████████████████████| 53/53 [00:03<00:00, 15.33it/s]
Tokenizing data...
Dataset size: 22677655 tokens
Model Parameters: 30.01M
Starting training...
Step 0: Train Loss 10.6553
Step 100: Train Loss 4.3400
Step 200: Train Loss 4.1780
Step 300: Train Loss 3.6359
Step 400: Train Loss 3.4837
Step 500: Train Loss 3.0928
Step 600: Train Loss 3.0464
Step 700: Train Loss 3.1099
Step 800: Train Loss 2.8002
Step 900: Train Loss 2.8326
Step 1000: Train Loss 2.5336
Step 1100: Train Loss 2.6611
Step 1200: Train Loss 2.6129
Step 1300: Train Loss 2.4803
Step 1400: Train Loss 2.6020
Step 1500: Train Loss 2.5746
Step 1600: Train Loss 2.5826
Step 1700: Train Loss 2.4734
Step 1800: Train Loss 2.6028
Step 1900: Train Loss 2.2561
Step 2000: Train Loss 2.2771
Step 2100: Train Loss 2.2656
Step 2200: Train Loss 2.0084
Step 2300: Train Loss 2.1346
Step 2400: Train Loss 2.3297
Step 2500: Train Loss 2.0589
Step 2600: Train Loss 1.9075
Step 2700: Train Loss 2.0376
Step 2800: Train Loss 2.0519
Step 2900: Train Loss 1.9790
Step 3000: Train Loss 2.0297
Step 3100: Train Loss 1.9769
Step 3200: Train Loss 1.9792
Step 3300: Train Loss 1.9320
Step 3400: Train Loss 1.9622
Step 3500: Train Loss 1.7953

Step 3600: Train Loss 1.7965
Step 3700: Train Loss 1.8448
Step 3800: Train Loss 1.9005
Step 3900: Train Loss 1.8374
Step 4000: Train Loss 2.0265
Step 4100: Train Loss 1.7395
Step 4200: Train Loss 1.7962
Step 4300: Train Loss 1.8470
Step 4400: Train Loss 1.7545
Step 4500: Train Loss 1.7243
Step 4600: Train Loss 1.6944
Step 4700: Train Loss 1.8533
Step 4800: Train Loss 1.7719
Step 4900: Train Loss 1.8116
Step 0: Train Loss 10.6553
Step 100: Train Loss 4.3400
Step 200: Train Loss 4.1780
Step 300: Train Loss 3.6359
Step 400: Train Loss 3.4837
Step 500: Train Loss 3.0928
Step 600: Train Loss 3.0464
Step 700: Train Loss 3.1099
Step 800: Train Loss 2.8002
Step 900: Train Loss 2.8326
Step 1000: Train Loss 2.5336
Step 1100: Train Loss 2.6611
Step 1200: Train Loss 2.6129
Step 1300: Train Loss 2.4803
Step 1400: Train Loss 2.6020
Step 1500: Train Loss 2.5746
Step 1600: Train Loss 2.5826
Step 1700: Train Loss 2.4734
Step 1800: Train Loss 2.6028
Step 1900: Train Loss 2.2561
Step 2000: Train Loss 2.2771
Step 2100: Train Loss 2.2656
Step 2200: Train Loss 2.0084
Step 2300: Train Loss 2.1346
Step 2400: Train Loss 2.3297
Step 2500: Train Loss 2.0589
Step 2600: Train Loss 1.9075
Step 2700: Train Loss 2.0376
Step 2800: Train Loss 2.0519
Step 2900: Train Loss 1.9790
Step 3000: Train Loss 2.0297
Step 3100: Train Loss 1.9769
Step 3200: Train Loss 1.9792
Step 3300: Train Loss 1.9320
Step 3400: Train Loss 1.9622
Step 3500: Train Loss 1.7953

Step 3600: Train Loss 1.7965
Step 3700: Train Loss 1.8448
Step 3800: Train Loss 1.9005
Step 3900: Train Loss 1.8374
Step 4000: Train Loss 2.0265
Step 4100: Train Loss 1.7395
Step 4200: Train Loss 1.7962
Step 4300: Train Loss 1.8470
Step 4400: Train Loss 1.7545
Step 4500: Train Loss 1.7243
Step 4600: Train Loss 1.6944
Step 4700: Train Loss 1.8533
Step 4800: Train Loss 1.7719
Step 4900: Train Loss 1.8116
Training complete. Model saved to dev_gpt_model.pt

/content/dev_gpt# 
/content/
/content/dev_gpt# python generate.py 
def calculate_sum(a, b):
        W = len(b) == 2
        W.set_xs(a,b)
       x.set_module(a,mask, masked)
        assert N.version == 4
       tol = newRscore(y, b)
       ok = set(x)

/content/dev_gpt# python generate.py 
def calculate_sum(a, b):
      # Collect distance of numpy-equal
    for x in x[1, 0]]):
        assert_equal(co_sarray, b._ bytearray, [0, 1], dtype=dtype=dtype)
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unrecognence geometry interface in an Ansiblerefundefined value.
/content/dev_gpt# python generate.py 
def calculate_sum(a, b):
     return dup_washout(a,c)


def Draw new_block_log2():
   """Use gtest._log2d_insert` base components for this model implementation."""

    for add_no_memory_logers in ['csd_log2d_sqlad_lookups']:
       a = opts.diffcompute_warning(name, f)
    
/content/dev_gpt# python generate.py 
def calculate_sum(a, b):
       """
       Given a active integer from the sum along the axes. This age is
       ``b``.

    For example, find precision(a, b):
        returns a Nakernel12352. It is Magic x250
        mask directly.

    >>> mean(1, a Decimal, MANY)
   