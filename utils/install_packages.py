import subprocess
import sys


def install(packages):
    """Install packages using pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    error_list = []

    for package in packages:
        package = package.strip()
        if not package or package.startswith("#"):
            continue
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}. Continuing...")
            error_list.append(package)

    if error_list:
        print(f"Failed to install the following packages: {error_list}")


if __name__ == "__main__":
    with open("requirements.txt", "r") as f:
        packages = f.readlines()
    install(packages)
