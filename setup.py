
from setuptools import setup, find_packages

setup(
    name = "webclassification",
    version = 1.0,
    author = "Ankita De",
    author_email = "de.ankita8@gmail.com",
    install_requires = ["scipy", "numpy", "scikit-learn","textblob","nltk"],
    packages = find_packages()
)
