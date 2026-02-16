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

### 4. Install CUDA and PyTorch with GPU Support

To use GPU acceleration, install the required CUDA toolkit and PyTorch with CUDA 11.8 support. 

```bash
conda install cudatoolkit=11.8 pytorch-cuda=11.8 -c nvidia -c pytorch -c conda-forge
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
