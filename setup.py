from setuptools import setup, find_packages

setup(
    name="protein_folding_ai",
    version="0.2.0",
    description="Protein folding simulation with Rosetta REF2015 energy and Monte Carlo sampling.",
    author="protein-folding-ai contributors",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.9",
    install_requires=["numpy>=1.24"],
    extras_require={"dev": ["pytest>=7.0"]},
)
