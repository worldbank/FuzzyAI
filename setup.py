from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="deduplix",
    version="0.1.0",
    author="World Bank",
    author_email="portfoliointelligence@worldbankgroup.org",
    description="Fuzzy AI — Entity deduplication and matching with LLM validation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/worldbank/fuzzy-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "rapidfuzz>=2.0.0",
        "networkx>=2.6.0",
        "tqdm>=4.60.0",
        "click>=8.0.0",
        "pyyaml>=5.4.0",
        "pydantic>=1.10.0",
    ],
    extras_require={
        "llm": ["openai>=0.27.0"],
        "spark": ["pyspark>=3.0.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.900",
        ],
    },
    entry_points={
        "console_scripts": [
            "deduplix=deduplix.cli:cli",
        ],
    },
)
