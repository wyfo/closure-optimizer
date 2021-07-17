from setuptools import find_packages, setup

with open("README.md") as f:
    README = f.read()

setup(
    name="closure-optimizer",
    version="0.1",
    url="https://github.com/wyfo/code-optimizer",
    author="Joseph Perez",
    author_email="joperez@hotmail.fr",
    license="MIT",
    packages=find_packages(include=["closure_optimizer"]),
    package_data={"closure_optimizer": ["py.typed"]},
    description="Optimize Python closures: branch pruning, loop unrolling, function inlining, etc.",  # noqa: E501
    long_description=README,
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
