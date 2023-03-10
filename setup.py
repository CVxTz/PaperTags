import os

from setuptools import setup

path = os.path.abspath(os.path.dirname(__file__))

try:
    with open(os.path.join(path, "requirements.txt"), encoding="utf-8") as f:
        REQUIRED = f.read().split("\n")
except FileNotFoundError:
    REQUIRED = []

setup(
    name="paper_tags",
    version="0.0.1",
    author="Youness MANSAR",
    author_email="mansaryounessecp@gmail.com",
    description="nlp",
    license="GNU",
    keywords="nlp",
    url="https://github.com/CVxTz/PaperTags",
    packages=["paper_tags"],
    classifiers=[
        "Topic :: Utilities",
    ],
    include_package_data=True,
    install_requires=REQUIRED,
)
