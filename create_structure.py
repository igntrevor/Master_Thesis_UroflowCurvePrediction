import os

# The Project Map
structure = {
    "configs/data": ["local.yaml"],
    "configs/model": ["rf.yaml"],
    "configs/sweep": [],
    "data/raw": [],
    "data/processed": [],
    "notebooks": [],
    "results/figures": [],
    "results/tables": [],
    "scripts": ["01_segment.py", "02_benchmark.py"],
    "src": ["__init__.py", "data.py", "features.py", "analysis.py"],
    "thesis/chapters": ["01_intro.tex", "02_methods.tex", "03_results.tex"],
    "thesis/figures": [], # We will link this logically
    ".github/workflows": []
}

files = {
    "configs/config.yaml": "# Global Config\nseed: 42",
    "README.md": "# Master's Thesis: Flow Prediction\n\n## Abstract\n...",
    ".gitignore": "*.wav\n__pycache__/\n.env\ndata/\noutputs/\n.DS_Store\n",
    "requirements.txt": "numpy\npandas\nlibrosa\nhydra-core\nscikit-learn\nmatplotlib\n",
    "thesis/main.tex": "\\documentclass{article}\n\\begin{document}\nHello World\n\\end{document}"
}

def create_project():
    print("🚀 Initializing Research Repository...")
    
    # 1. Create Directories
    for folder, children in structure.items():
        os.makedirs(folder, exist_ok=True)
        for child in children:
            # If it looks like a file (has extension), create it
            if "." in child:
                with open(os.path.join(folder, child), 'w') as f:
                    pass
    
    # 2. Create Root Files
    for filename, content in files.items():
        with open(filename, 'w') as f:
            f.write(content)
            
    print("✅ Structure created successfully!")
    print("👉 Next step: Open this folder in VS Code.")

if __name__ == "__main__":
    create_project()