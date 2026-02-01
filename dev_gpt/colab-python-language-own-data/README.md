## A Program to Train GPT Model (Using Local Arrow Dataset)

**This version uses a local Arrow dataset stored in `local_dataset_own/` for faster loading and offline training.**

**The script works on Google Colab and local machines with GPU/CPU.**

## Dataset Setup

This project uses a **local Arrow dataset** instead of downloading from HuggingFace each time:

- **Dataset location**: `local_dataset_own/` (10,004 Python code samples)
- **Size**: ~95 MB of code, stored in efficient Arrow format
- **Benefits**: 
  - ✓ No internet required after initial setup
  - ✓ 100x faster loading than streaming
  - ✓ Instant random access to any sample

### How the Local Dataset Was Created

```bash
# 1. Download from HuggingFace (one-time)
python download_dataset.py

# 2. Convert to text (optional, for inspection)
python convert_arrow_to_text.py

# 3. Convert text back to Arrow (creates local_dataset_own/)
python convert_text_to_arrow.py
```

## Configuration

Edit `config.py` to control dataset loading:

```python
use_local_dataset: bool = True  # Use local Arrow dataset
local_dataset_path: str = "local_dataset_own"  # Path to Arrow files
max_samples: int = 10000  # Number of samples to use
```

Set `use_local_dataset = False` to download from HuggingFace instead.

## Open Google Colab
https://colab.research.google.com/
Open Content tab, upload files including the `local_dataset_own/` folder 

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



/content/colab-python-language-own-data# python generate.py 
def calculate_sum(a, b):
        ---------: To create off=True%,
        49: To remove VSM_dict & Emodetic cool domain packetThicks
       bug: https://ythonrenceapi.com/stable/70669364712dc536105747.0.00000000j8fb106544364348298AdFreq,
      208: To store, inserts medrio por banana and soc
/content/colab-python-language-own-data# python generate.py 
def calculate_sum(a, b):
       return math_ops.ones("x", b).imagone(-1).ansinal_ + (1, 2)

  def vples_1(np=True):
    with pytest.raises(ValueError):
       a = array_ops.cast('x', smaller.size())
    return np.dot(a, a)


class TestIp:

    def test
/content/colab-python-language-own-data# python generate.py 
def calculate_sum(a, b):
      res_sum = indices[0]
       for l in range(1, 2):
          a[:2] = _mean_norm(xy)
        i += 1
         i += 1
        img_4 = X[-8xi:j * 0]
     
/content/colab-python-language-own-data# 
