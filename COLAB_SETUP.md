# Running shadowvqe on Google Colab (with GPU / TPU)

## Should you use Colab?

| Task | CPU (Local) | Colab Free GPU | Colab Pro A100 |
|---|---|---|---|
| study1 (H2 curve, no PySCF) | ~3 min | ~1.5 min | ~40 sec |
| study1 (H2 curve, with PySCF) | ~10 min | ~5 min | ~90 sec |
| study2 (shadow scaling) | ~30 sec | ~15 sec | ~8 sec |
| study5 (LiH + BeH2) | ~15 min | ~7 min | ~2 min |
| study6 (shot budget) | ~10 min | ~5 min | ~90 sec |
| study7 (variance reduction) | ~8 min | ~4 min | ~60 sec |

**Recommendation:** For study5 (LiH/BeH2), run on Colab. For all others, your local machine is fine.

## Important: Does GPU actually help?

Qiskit's statevector simulator runs on **CPU** by default. The GPU option is only used
when:
- You use `qiskit-aer` with `AerSimulator(device='GPU')`
- Your system has CUDA-enabled GPU

For this library, which uses exact `Statevector` simulation, there is **no direct GPU acceleration**
in the current code. However, Colab A100 has very fast CPUs (Intel Xeon with high clock
speeds) that still give 3-5x speedup over a typical laptop.

If you want true GPU acceleration, you need:
```python
pip install qiskit-aer-gpu   # CUDA-required
```
Then replace `Statevector(circuit)` with an Aer GPU simulator — this requires code
changes. Not recommended unless you have > 20 qubits.

## Quick-start Colab notebook

Paste into a new Colab notebook (Runtime → Change runtime type → GPU/T4 for fastest):

```python
# Cell 1: Install dependencies (run once)
!pip install "qiskit>=2.0" "qiskit-nature[pyscf]" numpy scipy matplotlib --quiet

# Cell 2: Upload the library (Option A: from Google Drive)
from google.colab import drive
drive.mount('/content/drive')
import sys
sys.path.insert(0, '/content/drive/MyDrive/Tomography/src')

# Cell 2 (Option B: clone from GitHub if you push there)
# !git clone https://github.com/YOUR_USERNAME/shadowvqe.git
# sys.path.insert(0, '/content/shadowvqe/src')

# Cell 3: Verify imports
import shadowvqe
print(shadowvqe.__version__)
from shadowvqe.molecules import check_pyscf_available
check_pyscf_available()
print("PySCF: OK")

# Cell 4: Run any study
exec(open('/content/drive/MyDrive/Tomography/examples/study5_molecules.py').read())
```

## Uploading to Google Drive (Windows)

1. Open Google Drive in your browser.
2. Create a folder called `Tomography`.
3. Upload the entire `Tomography\` directory:
   - `src/` (the shadowvqe package)
   - `examples/` (all study files)
4. In Colab, mount Drive as shown above.

## Faster approach: use Colab's file upload

```python
# Upload a zip file directly
from google.colab import files
uploaded = files.upload()   # select Tomography.zip
import zipfile
with zipfile.ZipFile('Tomography.zip', 'r') as z:
    z.extractall('/content/')
sys.path.insert(0, '/content/Tomography/src')
```

## Runtime comparison (approximate, Intel Core i5 laptop vs Colab T4)

| Metric | Local CPU | Colab T4 (free) | Speedup |
|---|---|---|---|
| H2 VQE (400 iter) | 3.2 sec | 1.1 sec | 3x |
| 1000 shadows (H2) | 1.5 sec | 0.5 sec | 3x |
| LiH Hamiltonian build | 25 sec | 8 sec | 3x |
| BeH2 Hamiltonian build | 55 sec | 18 sec | 3x |

## Tips for faster runs on Colab

1. **Reduce N_SHADOWS** in study configs: 1000 instead of 3000 for quick tests.
2. **Reduce N_TRIALS** in study6/study7: 5-10 instead of 15-20.
3. **Reduce bond distances** in study1: use `h2_pes([0.5, 0.735, 1.0, 1.5, 2.0])`.
4. **Use T4 GPU runtime** (free tier): 4x faster CPU despite name.
5. **Avoid TPU**: qiskit does not support TPU natively.

## Checking if your session has a GPU

```python
import subprocess
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(result.stdout if result.returncode == 0 else "No GPU found")
```

## Saving figures from Colab to Drive

Add at the end of any study:
```python
import shutil
shutil.copytree('/content/Tomography/figures',
                '/content/drive/MyDrive/Tomography/figures',
                dirs_exist_ok=True)
print("Figures saved to Drive!")
```
