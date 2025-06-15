import json

with open('NOTEBOOK.ipynb', 'r', encoding='utf-8') as infile:
    notebook = json.load(infile)

# Ensure metadata and widgets are present
if "widgets" not in notebook.setdefault("metadata", {}):
    notebook["metadata"]["widgets"] = {}
notebook["metadata"]["widgets"]["state"] = {}  # add state

with open('NOTEBOOK.ipynb', 'w', encoding='utf-8') as outfile:
    json.dump(notebook, outfile, indent=2)

print("Metadata 'state' added.")
