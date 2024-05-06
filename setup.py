from setuptools import setup, find_packages
with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="MEGraphAU",
    version="v0.0.1",
    description="""A tool to perdict face action unit (FAU) and emotion from video.""",
    long_description_content_type="text/markdown",
    author="Andreas Susanto",
    packages=find_packages(include=["*"]),
    install_requires=required,
    license="MIT",
    url="https://github.com/Andreas-UI/ME-GraphAU-Video.git",
    python_requires=">=3.10",
)
