#!/usr/bin/env python3
"""
Example script for generating a training dataset for a specific domain.
"""
import os
import argparse
import json
from datetime import datetime
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_generation.generator import DataGenerator
from src.data_processing.processor import DataProcessor
from src.utils.logging import setup_logger

# Set up logger
logger = setup_logger("generate_dataset", "logs/generate_dataset.log")

def main():
    """Main function to generate and process a dataset."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate training dataset for a specific domain")
    parser.add_argument("--domain", type=str, default="finance", 
                       choices=["finance", "healthcare", "legal", "tech", "general"],
                       help="Domain to generate data for")
    parser.add_argument("--size", type=int, default=100, help="Number of examples to generate")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model to use for generation")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--config", type=str, default=None, help="Configuration file path")
    parser.add_argument("--augment", action="store_true", help="Whether to augment the dataset")
    args = parser.parse_args()
    
    # Create timestamp for output file if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"data/generated/{args.domain}_{timestamp}.json"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    logger.info(f"Starting dataset generation for domain '{args.domain}' with {args.size} examples")
    logger.info(f"Using model: {args.model}")
    logger.info(f"Output will be saved to: {args.output}")
    
    try:
        # Initialize generator
        generator = DataGenerator(model_name=args.model, config_path=args.config)
        
        # Generate raw dataset
        logger.info("Generating raw dataset...")
        raw_dataset = generator.generate_dataset(
            domain=args.domain,
            size=args.size
        )
        
        # Save raw dataset
        raw_output = args.output.replace(".json", "_raw.json")
        with open(raw_output, 'w') as f:
            json.dump(raw_dataset, f, indent=2)
        logger.info(f"Raw dataset saved to: {raw_output}")
        
        # Initialize processor
        processor = DataProcessor(config_path=args.config)
        
        # Process dataset
        logger.info("Processing dataset...")
        processed_dataset = processor.process_dataset(raw_dataset, filter_invalid=True)
        
        # Augment dataset if requested
        if args.augment:
            logger.info("Augmenting dataset...")
            processed_dataset = processor.augment_dataset(processed_dataset, augmentation_factor=2)
        
        # Save processed dataset
        with open(args.output, 'w') as f:
            json.dump(processed_dataset, f, indent=2)
        logger.info(f"Processed dataset saved to: {args.output}")
        
        # Print summary
        logger.info(f"Dataset generation complete:")
        logger.info(f"  - Raw examples: {len(raw_dataset)}")
        logger.info(f"  - Processed examples: {len(processed_dataset)}")
        logger.info(f"  - Augmentation: {'Yes' if args.augment else 'No'}")
        
    except Exception as e:
        logger.error(f"Error during dataset generation: {e}")
        raise

if __name__ == "__main__":
    main()
