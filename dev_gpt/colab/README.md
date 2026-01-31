## A Program to Train GPT Model

**The script works on Google Colab.
Python packages must be update / changed to allow training on local CPU machine, and there could be issues related to torch or numpy, and extra time of debugging and training are needed.**

## Open Google Colab
https://colab.research.google.com/
Open Content tab, upload files 

## How to Run

```bash
# Create venv
python -m venv .venv
source .venv/bin/activate

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
README.md: 10.5kB [00:00, 29.4MB/s]
Tokenizing data...
Dataset size: 497745 tokens
Model Parameters: 30.01M
Starting training...
Step 0: Train Loss 10.8746
Step 100: Train Loss 6.0370
Step 200: Train Loss 5.5291
Step 300: Train Loss 5.0158
...
Step 1000: Train Loss 3.1157
Step 1100: Train Loss 2.9041
Step 1200: Train Loss 2.7833
...
Step 2300: Train Loss 0.9252
Step 2400: Train Loss 0.7240
Step 3200: Train Loss 0.2267
Step 3200: Train Loss 0.2267
...
Step 3000: Train Loss 0.2771
Step 3100: Train Loss 0.2573
Step 3200: Train Loss 0.2267
...
Step 4000: Train Loss 0.0870
Step 4100: Train Loss 0.0731
Step 4200: Train Loss 0.0790
...
Step 4700: Train Loss 0.0520
Step 4800: Train Loss 0.0548
Step 4900: Train Loss 0.0426
Training complete. Model saved to dev_gpt_model.pt

/content/dev_gpt# ls -ltr
total 117312
drwxr-xr-x 5 root root      4096 Jan 31 06:18 venv312
-rw-r--r-- 1 root root        58 Jan 31 06:32 requirements.txt
-rw-r--r-- 1 root root      4569 Jan 31 06:33 model.py
-rw-r--r-- 1 root root      1543 Jan 31 06:33 train.py
-rw-r--r-- 1 root root      1085 Jan 31 06:33 generate.py
-rw-r--r-- 1 root root      1316 Jan 31 06:36 config.py
-rw-r--r-- 1 root root      2117 Jan 31 06:36 data.py
drwxr-xr-x 2 root root      4096 Jan 31 06:36 __pycache__
-rw-r--r-- 1 root root 120082859 Jan 31 07:22 dev_gpt_model.pt

/content/dev_gpt# python generate.py 
def calculate_sum(a, b):
    
 The <unk> are similar to the construction in which are shown in the store , such as " <unk> and <unk> " , master of violence , either <unk> hard card after which had trioxide and lithium was subsequently carried out at the site . It is raised in 169 @-@ 970 out of water spored in rural areas . The enzyme <unk> , which are cooler than now doesStep 2100: Train Loss 1.1947

/content/dev_gpt# python generate.py 
def calculate_sum(a, b):
    
 The <unk> are similar to the construction in which are shown in the store , such as " <unk> and <unk> " , master of violence , either <unk> hard card after which had trioxide and lithium was subsequently carried out at the site . It is raised in 169 @-@ 970 out of water spored in rural areas . The enzyme <unk> , which are cooler than now does not seen as a bridge using <unk> . 
 The lyric is

/content/dev_gpt# python generate.py 
def calculate_sum(a, b):
    
 The ribry Laboratory of the City of Sarnabhumi Airport , like <unk> and <unk> flesh <unk> are treated differently by the post here . The city 's pounds ( <unk> and relatively small Meanwhile , and <unk> is served by the City , the destination as a " Try " . Large @-@ Daily gripping <unk> the smuggling of petrochemical hospital , the site makes other venues as highly likely to be found in its own pilot
/content/dev_gpt# 
/content/dev_gpt# python generate.py 
def calculate_sum(a, b):
    
 The <unk> are an individual in the grant to which reveal a special zones in the institution into terminal <unk> and <unk> and <unk> . The varying availability of Ottoman @-@ based on architecture were issued in Kent , designed to survive the use of light formations in 1941 around 1800 . This was put in dynamic and released in Finland . In the equipment to reports of construction of newly formed by King Andrew Pudding barracks for the area . FITs , whose book was
