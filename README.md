# 2026-PULS-Master
Classification of Positive Unlabeled data under Label Shift

## Setup Instructions

Follow these steps to set up the environment and install all necessary dependencies.

### 1. Clone the Repository

Clone this repository.

```bash
git clone https://github.com/izabelatelejko/2026-PULS-Master.git
cd 2026-PULS-Master
```

### 2. Create a Conda Environment

Create a new Conda environment named puls with Python 3.10.

```bash
conda create -y --name=puls python=3.10
```

### 3. Activate the Environment

```bash
conda activate puls
```

### 4. Install PyTorch with GPU Support

To use GPU acceleration, install PyTorch with CUDA support. For most GPUs (CUDA 11.8):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For newer GPUs (e.g. RTX 50 series), use the nightly build with CUDA 12.8:

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 5. Install Project Dependencies

Install the required dependencies using pip.

```bash
pip install -r requirements.txt
```

### 6. Install the Project

Install the project in editable mode.

```bash
pip install -e .
```
