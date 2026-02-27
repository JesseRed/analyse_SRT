HSMM Environment Setup (pyhsmm on Python 3.10)

This document describes how to reproducibly set up a working pyhsmm environment without touching the main Python 3.11 project environment.

The main project remains unchanged.
HSMM runs in a dedicated virtual environment: .venv_hsmm.

Overview

Why this setup?

pyhsmm is not compatible with modern Python toolchains out-of-the-box.

It requires:

Python 3.10

pip < 24

numpy 1.26.x

manual patching of legacy imports

We isolate everything in .venv_hsmm.

This keeps the main analysis environment clean and stable.

1️⃣ System Requirements (Ubuntu / WSL)

Install system build dependencies:

sudo apt-get update
sudo apt-get install -y \
  build-essential \
  gfortran \
  libopenblas-dev \
  liblapack-dev \
  libeigen3-dev \
  make \
  libssl-dev \
  zlib1g-dev \
  libbz2-dev \
  libreadline-dev \
  libsqlite3-dev \
  curl \
  llvm \
  libncursesw5-dev \
  xz-utils \
  tk-dev \
  libxml2-dev \
  libxmlsec1-dev \
  libffi-dev \
  liblzma-dev
2️⃣ Install Python 3.10 via pyenv

We do NOT use conda.

Install pyenv
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
git clone https://github.com/pyenv/pyenv-virtualenv.git ~/.pyenv/plugins/pyenv-virtualenv

Add to ~/.bashrc:

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

Reload shell:

exec "$SHELL"

Verify:

pyenv --version
Install Python 3.10
pyenv install 3.10.14
3️⃣ Create HSMM Virtual Environment

From the project root:

cd ~/Code/analyse_SRT
~/.pyenv/versions/3.10.14/bin/python -m venv .venv_hsmm
source .venv_hsmm/bin/activate

Confirm:

python -V

Should show:

Python 3.10.x
4️⃣ Install Pinned Python Dependencies

Install controlled versions:

pip install pip==23.3.2
pip install setuptools<70 wheel
pip install numpy==1.26.4
pip install scipy matplotlib requests future cython nose

Why?

pip ≥ 24 breaks legacy setup.py builds

numpy ≥ 2.0 breaks pybasicbayes

numpy 1.26.4 is stable with Python 3.10

5️⃣ Install pyhsmm (Offline Eigen Fix)

pyhsmm tries to download Eigen from GitLab.
We avoid that by linking system Eigen.

rm -rf /tmp/pyhsmm_build
git clone --recursive https://github.com/mattjj/pyhsmm.git /tmp/pyhsmm_build
cd /tmp/pyhsmm_build

mkdir -p deps
ln -sf /usr/include/eigen3/Eigen deps/Eigen

Install using legacy build:

pip install --no-build-isolation --no-use-pep517 .
6️⃣ Apply Compatibility Patches

After installation, run:

cd ~/Code/analyse_SRT
source .venv_hsmm/bin/activate
python tools/patch_pyhsmm_stack.py

This automatically fixes:

scipy.misc.logsumexp → scipy.special.logsumexp

numpy.core.umath_tests.inner1d → safe NumPy einsum replacement

legacy np.Inf usage (if present)

7️⃣ Test Installation
python -c "import pyhsmm; print('pyhsmm import ok')"

If successful:

pyhsmm import ok
8️⃣ Usage

To activate HSMM environment:

source .venv_hsmm/bin/activate

To return to main environment:

deactivate
9️⃣ Important Notes

Do NOT upgrade pip in .venv_hsmm.

Do NOT upgrade numpy beyond 1.26.x.

The main project .venv remains fully independent.

10️⃣ Reproducibility Strategy

The repository contains:

requirements_hsmm_py310.txt

tools/patch_pyhsmm_stack.py

This setup document

This guarantees deterministic setup on:

New workstation

WSL instance

Remote Linux server

Final Architecture
analyse_SRT/
│
├── .venv               → Python 3.11 (main project)
├── .venv_hsmm          → Python 3.10 (pyhsmm only)
│
├── tools/
│   └── patch_pyhsmm_stack.py
│
└── docs/
    └── HSMM_SETUP.md

You now have:

Clean separation

Stable legacy compatibility

Fully reproducible HSMM pipeline

No contamination of your main environment