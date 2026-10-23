from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

EXTRAS = {
    "blocking": ["scikit-learn>=1.3", "scipy>=1.10", "sparse_dot_topn>=1.1"],
    "phonetic": ["metaphone>=0.6"],
    "minhash": ["datasketch>=1.6"],
    "embeddings": ["sentence-transformers>=2.7", "hnswlib>=0.8"],
    "normalize": ["anyascii>=0.3", "tldextract>=5"],
    "llm": ["openai>=1.0", "anthropic>=0.30"],
    "eval": ["scikit-learn>=1.3", "psutil"],
    "spark": ["pyspark>=3.0.0"],
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=3.0.0",
        "pyarrow",
        "black>=22.0.0",
        "flake8>=4.0.0",
        "mypy>=0.900",
    ],
}
EXTRAS["all"] = sorted({dep for k, deps in EXTRAS.items() if k not in ("dev", "spark") for dep in deps})

setup(
    name="deduplix",
    version="0.2.0",
    author="World Bank",
    author_email="portfoliointelligence@worldbankgroup.org",
    description="Fuzzy AI — Entity deduplication and matching with LLM validation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/worldbank/FuzzyAI",
    packages=find_packages(),
    package_data={"deduplix.normalize.tables": ["*.yaml"], "deduplix.llm": ["prompts/*.j2"],
                  "deduplix": ["templates/*.yaml"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "rapidfuzz>=3.6.0",
        "networkx>=2.8",
        "pydantic>=2.0.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.60.0",
        "click>=8.0.0",
        "jinja2>=3.0",
    ],
    extras_require=EXTRAS,
    entry_points={
        "console_scripts": [
            "deduplix=deduplix.cli:cli",
            "fuzzy-ai=deduplix.cli:cli",
        ],
    },
)
