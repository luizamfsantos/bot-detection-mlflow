import logging
import os
import subprocess
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s:%(levelname)s]: %(message)s"
)


def setup_environment():
    logging.info("Setting up development environment...")

    # Create virtual environment
    subprocess.run([sys.executable, "-m", "venv", "env"])

    # Determine platform-specific paths for Python and pip
    if sys.platform == "win32":
        venv_python = os.path.join("env", "Scripts", "python.exe")
        venv_pip = os.path.join("env", "Scripts", "pip.exe")
    else:
        venv_python = "env/bin/python"
        venv_pip = "env/bin/pip"

    # Ensure pip is up-to-date
    subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip"])

    # Install dependencies
    requirements_file = (
        "requirements-dev.txt"
        if os.path.exists("requirements-dev.txt")
        else "requirements.txt"
    )
    subprocess.run([venv_pip, "install", "-r", requirements_file])

    # Install pre-commit hooks if available
    if os.path.exists(".pre-commit-config.yaml"):
        subprocess.run([venv_python, "-m", "pip", "install", "pre-commit"])
        subprocess.run([venv_python, "-m", "pre-commit", "install"])

    if sys.platform == "win32":
        activate_cmd = os.path.join("env", "Scripts", "activate")
    else:
        activate_cmd = "source env/bin/activate"

    logging.info("Development environment setup complete!")
    logging.info(f"To activate the environment, run: {activate_cmd}")


if __name__ == "__main__":
    setup_environment()
