import os
import subprocess
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/drive.file"]

base_dir = Path("Data")
CLIENT_SECRET_FILE = base_dir / "client_secret_992814957758.apps.googleusercontent.json"
TOKEN_FILE = base_dir / "token.json"
UPLOAD_FOLDER_ID = "1Q5To9zpCuXB8-6fqjpRNYLc6RElQExwU"


def create_archive(archive_name: str) -> Path:
    archive_filename = base_dir / f"{archive_name}.zip"
    cmd = f"zip -r {archive_filename} Data/seg-study.db Data/studies runs outputs.txt"
    subprocess.run(cmd.split(), cwd=base_dir.parent)
    return archive_filename


def upload_to_drive(upload_file: Path):
    """Uploads a file to Google Drive."""
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CLIENT_SECRET_FILE), SCOPES
            )
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())

    try:
        service = build("drive", "v3", credentials=creds)

        file_metadata = {"name": upload_file.name, "parents": [UPLOAD_FOLDER_ID]}
        media = MediaFileUpload(str(upload_file), resumable=True)

        file = (
            service.files()
            .create(body=file_metadata, media_body=media, fields="id")
            .execute()
        )
        print(f"File ID: {file.get('id')}")

    except HttpError as error:
        print(f"An error occurred: {error}")


def upload_experiment(study_name: str):
    upload_file = create_archive(study_name)
    if not CLIENT_SECRET_FILE.exists():
        print(f"Error: Credentials file not found at {CLIENT_SECRET_FILE}")
        print(
            "Please make sure your client_secret.json file is in the correct location."
        )
    else:
        upload_to_drive(upload_file)


if __name__ == "__main__":
    upload_experiment("study-0501-191549")
