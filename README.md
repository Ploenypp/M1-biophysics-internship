# 🔬 M1-biophysics-internship
Python packages and pipelines developed for my M1 biophysics internship involving gel mechanics characterization supervised by Dr. Jonathan Fouchard and Dr. Aline Stedman

## 📌 Project Overview
This repository provides image analysis, mechanical data processing, and multi-source data visualization. 

### Key Features
* **Modular Codebase** : Standardized `src/` Python package layout for reusability across scripts and Jupyter Notebooks
* **OS-Agnostic Paths** : Built with Python’s `pathlib` for native path resolution and seamless execution across Windows, macOS, and Linux environments
* **Image Processing** : OpenCV-based pipeline configured to handle multi-page/high-bit-depth TIFF images cleanly
* **Traceable Structured Data Export** : Multi-sheet Excel exports that preserve original raw data alongside computed metrics, eliminating dependency on in-memory Python state

## 📁 Repository Architecture
```text
M1-biophysics-internship/
├── data/                        # Example mock data
├── notebooks/                   # Jupyter Notebook demonstrating pipelines
├── src/
│   └── analysis_tools/          # Core package source code
│       ├── image_analysis.py
│       ├── poisson_ratio.py
│       └── stress_strain.py
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py                     # Package setup configuration
```


## 💻 Installation & Setup
### 1. Clone repository
```bash
git clone [https://github.com/Ploenypp/M1-biophysics-internship.git](https://github.com/Ploenypp/M1-biophysics-internship.git)
cd M1-biophysics-internship
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS & Linux
source venv/bin/activate
```

### 3. Install Dependencies and Local Package analysis_tools
```bash
pip install -r requirements.txt
pip install -e .
```

## 🧬 Acknowledgements
### Supervisors
* **Dr. Jonathan Fouchard** - Coordinated Development of Muscle and Connective Tissues
* **Dr. Aline Stedman** - Morphogenesis of the Vertebrate Brain

### Affiliations
* **Research Unit** : Development, Adaptation, Aging (Dev2A)
* **Institution** : Institut Biologie Paris-Seine (IBPS)
* **University** : Sorbonne University

