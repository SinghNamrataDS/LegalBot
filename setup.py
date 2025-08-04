from setuptools import setup,find_packages
 
with open("requirements.txt") as f:
    requirements = f.read().splitlines()
 
setup(
    name="Legalbot based on RAG",
    version="0.1",
    author="Namrata",
    packages=find_packages(),
    install_requires = requirements,
)
 