# Intelligent Training Data Generation for Local Representation Models

This project implements a system for automatically generating and processing training data to support fine-tuning of local representation models, with a focus on the Qwen 2.5 series.

## Overview

The system provides:
- Automated generation of high-quality response pairs for domain-specific scenarios
- Data processing pipeline with robust quality control
- Comprehensive tracing and logging for data lineage
- Multi-language support
- Verification tools for model evaluation

## Features

- **Automated Data Generation**: Create diverse, high-quality training pairs
- **Robust Processing Pipeline**: Clean, filter, and augment generated data
- **Quality Assurance**: Validate data against domain-specific criteria
- **Tracing System**: Full data lineage and provenance tracking
- **Multilingual Support**: Generate data in multiple languages

## Getting Started

```bash
# Clone the repository
git clone https://github.com/YifeiSheng/intelligent-training-data.git

# Install dependencies
pip install -r requirements.txt

# Run data generation and data process
python src/examples/generate_data.py --model Qwen/Qwen2.5-7B-Instruct --domain finance --size 100
```

## Documentation

See the [Design Document](docs/design_document.md) for detailed information about the system architecture and implementation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
