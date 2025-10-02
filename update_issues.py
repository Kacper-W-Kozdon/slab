import os

from dotenv import load_dotenv
from requests import Request, Session

load_dotenv()

todo_body = []

with open("TODO.md", "r") as todo_file:
    for line in todo_file:
        todo_body.append(line)

todo_body_text = "".join(todo_body)

token = os.getenv("TOKEN")

url = "https://api.github.com/repos/Kacper-W-Kozdon/slab/issues/1"
data = todo_body_text
headers = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {token}",
    "X-GitHub-Api-Version": "2022-11-28",
}

session = Session()

req = Request("GET", url, headers=headers)
prepped = req.prepare()
resp = session.send(prepped)

if resp.json().get("id") is not None:
    pass
else:
    pass
