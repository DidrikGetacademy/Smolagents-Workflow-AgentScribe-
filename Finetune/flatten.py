import json

# Input file: must be a JSON array like [ {messages: [...]}, {messages: [...]}, ... ]
with open("multi_example.json", "r") as infile:
    data = json.load(infile)

# Output as valid .jsonl
with open("flattened_examples.jsonl", "w") as outfile:
    for example in data:
        jsonl_line = json.dumps(example, separators=(",", ":"))  # flatten into 1 line
        outfile.write(jsonl_line + "\n")
