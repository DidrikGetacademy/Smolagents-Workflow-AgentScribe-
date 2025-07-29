import re

def find_emails(lines):
    emails = []

    for line in lines:
        match = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", line)
        if match:
            emails.extend(match)

    return emails

# Eksempeldata
text_lines = [
    "Contact me at test@example.com for details.",
    "My backup is: me.second@mail.co.uk",
    "No email here!",
    "Fake: just@wrong",
]

# Kjør funksjonen
results = find_emails(text_lines)
print("Found emails:", results)
