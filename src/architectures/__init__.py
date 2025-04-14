import os
import sys

# Aggiungi la directory principale del progetto al sys.path
# Poiché architectures è in src/, dobbiamo risalire di due livelli per arrivare alla root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)