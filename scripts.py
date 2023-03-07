import subprocess


def test() -> None:
    """
    Run all unittests. Equivalent to:
    `poetry run python -u -m unittest discover`
    """
    subprocess.Popen(["python", "-m", "pytest", "--import-mode=importlib"])


def lint() -> None:
    """
    Run all unittests. Equivalent to:
    `poetry run python -u -m unittest discover`
    """
    subprocess.Popen(["pylint", ".", " --rcfile=.pylintrc"])
