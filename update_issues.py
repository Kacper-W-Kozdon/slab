import copy
import os

import requests.sessions
from dotenv import load_dotenv
from requests import ConnectionError, Request, Session

load_dotenv()

todo_body = []

with open("TODO.md", "r") as todo_file:
    for line in todo_file:
        line = line.replace("\n", "<br>")
        todo_body.append(line)

todo_body_text = "".join(todo_body)

token = os.getenv("TOKEN")

url = "https://api.github.com/repos/Kacper-W-Kozdon/slab/issues/1"
markdown_url = "https://api.github.com/markdown"

authorize_headers: dict[str, str] = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {token}",
    "X-GitHub-Api-Version": "2022-11-28",
}
markdown_headers: dict[str, str] = {
    "X-GitHub-Api-Version": "2022-11-28",
    "Accept": "text/html",
}

markdown_data = todo_body_text

session = Session()


def markdown_body(
    url: str, headers: dict[str, str], session: requests.sessions.Session, data: str
) -> str:
    data = '{"text": ""}'
    data = data.replace('"text": ""', f'"text": "{data}"')

    markdown = Request("POST", url, headers=headers, data=data)
    prepped_markdown = markdown.prepare()
    resp = session.send(prepped_markdown)
    print(resp.status_code)
    return resp.content


print(markdown_body(markdown_url, markdown_headers, session, markdown_data))


def update_TODO(
    url: str, headers: dict[str, str], session: requests.sessions.Session, body: str
) -> None:
    print("---Updating TODO---\n")
    auth_headers = copy.copy(headers)
    test_body = "**TEST1**<br>-[ ] TEST2"
    data = '{"title": "TODO", "body": "", "state": "open"}'
    data = data.replace('"body": ""', f'"body": "{test_body}"')
    print(data)

    patch = Request("PATCH", url, headers=auth_headers, data=data)
    prepped_patch = patch.prepare()
    resp = session.send(prepped_patch)
    if str(resp.status_code) != "200":
        raise ConnectionError(
            f"Failed to update TODO with {resp.status_code=}.",
            request=prepped_patch,
            response=resp,
        )

    print("SUCCESS")


def create_TODO(
    url: str, headers: dict[str, str], session: requests.sessions.Session, body: str
) -> None:
    print("---Creating TODO---\n")
    auth_headers = copy.copy(headers)
    data = {"title": "TODO", "body": body, "state": "open"}

    create = Request("POST", url, headers=auth_headers, data=data)
    prepped_create = create.prepare()
    resp = session.send(prepped_create)
    if str(resp.status_code) != "200":
        raise ConnectionError(
            f"Failed to create TODO with {resp.status_code=}.",
            request=prepped_create,
            response=resp,
        )

    print("SUCCESS")


get = Request("GET", url, headers=authorize_headers)
prepped = get.prepare()
response = session.send(prepped)

if response.json().get("id") is not None:
    update_TODO(url, authorize_headers, session, todo_body_text)
else:
    create_TODO(url, authorize_headers, session, todo_body_text)
