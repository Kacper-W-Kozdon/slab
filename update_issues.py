import copy
import os
import pathlib

import requests.sessions
from dotenv import load_dotenv
from requests import ConnectionError, Request, Session

load_dotenv()

todo_body = []

root = pathlib.Path(__file__).parent.resolve()
print(root)

with open(f"{root}\\TODO.md", "r") as todo_file:
    for line in todo_file:
        line = line.replace("\n", "")
        line = line.replace("\\", "")
        line = line.replace('"', "'")
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
    "Authorization": f"Bearer {token}",
}

markdown_data = todo_body

session = Session()


def markdown_body(
    url: str,
    headers: dict[str, str],
    session: requests.sessions.Session,
    body_list: list[str],
) -> str:
    ret = ""
    for body in body_list:
        data = '{"text": ""}'
        # print(body)
        data = data.replace('"text": ""', f'"text": "{body}"')

        markdown = Request("POST", url, headers=headers, data=data)
        prepped_markdown = markdown.prepare()
        resp = session.send(prepped_markdown)
        ret += resp.text
    ret = ret.replace("\n", "")
    ret = ret.replace('"', "'")
    return ret


print(markdown_body(markdown_url, markdown_headers, session, markdown_data))


def update_TODO(
    url: str, headers: dict[str, str], session: requests.sessions.Session, body: str
) -> None:
    print("---Updating TODO---\n")
    auth_headers = copy.copy(headers)
    # test_body = "<div><h1 class='heading-element'>SLAB Project</h1></div>"
    data = '{"title": "TODO", "body": "", "state": "open"}'
    data = data.replace('"body": ""', f'"body": "{body}"')
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
    data = '{"title": "TODO", "body": "", "state": "open"}'
    data = data.replace('"body": ""', f'"body": "{body}"')

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
    body = markdown_body(markdown_url, markdown_headers, session, markdown_data)
    update_TODO(url, authorize_headers, session, body)
else:
    body = markdown_body(markdown_url, markdown_headers, session, markdown_data)
    create_TODO(url, authorize_headers, session, body)
