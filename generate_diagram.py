# generate_diagram.py
import sys
import json
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate(spec_json):
    """Generates a diagram and prints the base64 string to stdout."""
    fig = None
    try:
        spec = json.loads(spec_json)
        if spec.get('type') != 'graph':
            return

        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        data = spec.get('data', [])

        # Only plot if valid data is present
        if not data or not all(isinstance(p, list) and len(p) == 2 for p in data):
            return

        x_vals = [p[0] for p in data]
        y_vals = [p[1] for p in data]
        ax.plot(x_vals, y_vals, marker='o', linestyle='-')
        ax.set_title(spec.get('title', ''))
        ax.set_xlabel(spec.get('x_label', ''))
        ax.set_ylabel(spec.get('y_label', ''))
        ax.grid(True)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        print(base64.b64encode(buf.read()).decode('utf-8'))

    except Exception:
        pass
    finally:
        if fig is not None:
            plt.close(fig)

if __name__ == '__main__':
    # The first argument from the command line will be the JSON string
    if len(sys.argv) > 1:
        generate(sys.argv[1])