#!/usr/bin/env python3
"""
Quick script to update all benchmark plotting code to use professional visualizations.
"""

import re
from pathlib import Path

examples_dir = Path("examples")

updates = {
    "benchmark_molecular_vqe.py": {
        "import_add": "from shadowvqe.visualization_research import plot_vqe_shadow_comparison_research",
        "description": "Update molecular VQE plotting",
    },
    "benchmark_heisenberg_measurement_cost.py": {
        "import_add": "from shadowvqe.visualization_research import plot_measurement_cost_research",
        "description": "Update Heisenberg measurement cost plotting",
    },
    "benchmark_adaptive_shadow_variance.py": {
        "import_add": "from shadowvqe.visualization_research import plot_variance_reduction_research",
        "description": "Update variance reduction plotting",
    },
    "benchmark_fmo_energy_assembly.py": {
        "import_add": "from shadowvqe.visualization_research import plot_fmo_scaling_research",
        "description": "Update FMO scaling plotting",
    },
}

for filename, info in updates.items():
    filepath = examples_dir / filename
    if not filepath.exists():
        print(f"SKIP: {filename} not found")
        continue

    print(f"Updating: {filename}")
    content = filepath.read_text()

    # Add import if not present
    import_line = info["import_add"]
    if import_line not in content and "visualization_research" not in content:
        # Find the last shadowvqe import and add after it
        match = re.search(r'(from shadowvqe\..+\n)', content)
        if match:
            pos = match.end()
            content = content[:pos] + import_line + "\n" + content[pos:]

    filepath.write_text(content)
    print(f"  OK: {info['description']}")

print("\nAll visualization imports added. Now update plotting sections manually or with sed.")
