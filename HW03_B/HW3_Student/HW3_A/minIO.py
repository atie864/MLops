import os
import subprocess
from dotenv import load_dotenv

load_dotenv()

result = subprocess.run(
    ["bash", "scripts/01_upload_to_minio.sh"],
    env=os.environ,  # passes all loaded env vars to the subprocess
    capture_output=True,
    text=True
)
print(result.stdout)
print(result.stderr)