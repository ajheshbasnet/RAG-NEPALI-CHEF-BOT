import json

with open('NOTEBOOK.ipynb', 'r', encoding='utf-8') as infile:
    notebook = json.load(infile)

if "widgets" in notebook.setdefault("metadata", {}):
    notebook["metadata"]["widgets"]["state"] = {}
else:
    notebook["metadata"]["widgets"] = {"state": {}}

with open('NOTEBOOK.ipynb', 'w') as outfile:
    json.dump(notebook, outfile, indent=2)

print("Added 'state' to metadata.")
