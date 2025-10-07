Overview:
This script solves the CFI dispersion relation. This repo also contains a script for plotting. 

Requiements:

  Tools-

  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt

  Files- 

  Post-merger snapshot files - model_rl*_orthonormal.h5
  interpolate_table.py ( NuLib opacities interpolation script)

Run:

  python3 disp.py --scan-slice (--mono) (--axis=) (--index=) (--stride=) (--point)    

  [The command line arguments inside brackets are optional]  


LICENSE: MIT  