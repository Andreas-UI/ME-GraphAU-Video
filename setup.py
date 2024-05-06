from setuptools import setup, find_packages
with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="MEGraphAU",
    version="v0.0.1",
    description="""A tool to perdict face action unit (FAU) and emotion from video.""",
    long_description_content_type="text/markdown",
    author="Andreas Susanto",
    packages=find_packages(),
    include_package_data=True,
    package_data={'': [
        'MEGraphAU/OpenGraphAU/config/hybrid_config.yaml',
        'MEGraphAU/OpenGraphAU/checkpoints/checkpoints.txt',
        'MEGraphAU/config/BP4D_config.yaml',
        'MEGraphAU/config/DISFA_config.yaml']},
    install_requires=required,
    license="MIT",
    url="https://github.com/Andreas-UI/ME-GraphAU-Video.git",
    python_requires=">=3.10",
)
