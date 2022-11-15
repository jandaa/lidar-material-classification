# Install dependencies
sudo apt-get install \
    build-essential \
    python3-dev

# Create virtual environment
python3.8 -m venv .venv
source ~/.bashrc
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install pytorch with CUDA version 11.3
pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu113

# Install all other dependencies
pip install -r setup/requirements.txt
pip install git+https://github.com/leddartech/pioneer.das.api.git