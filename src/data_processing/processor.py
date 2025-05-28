import json
import re
import copy
from datetime import datetime
from typing import Dict, List, Any, Callable, Optional

from ..utils.logging import setup_logger

logger = setup_logger("data_processor")

class DataProcessor:
    """Process and enhance generated data pairs."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the data processor.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = {}
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        
        # Initialize processing pipeline
        self.pipeline = [
            self._clean_text,
            self._validate_content,
            self._add_quality_score,
            self._tag_entities
        ]
    
    def _clean_text(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean text by removing extra whitespace, fixing formatting, etc.
        
        Args:
            data_item: Data item to process
            
        Returns:
            Processed data item
        """
        result = copy.deepcopy(data_item)
        
        # Clean input text
        if "input" in result:
            # Remove extra whitespace
            result["input"] = re.sub(r'\s+', ' ', result["input"]).strip()
        
        # Clean response text
        if "response" in result:
            # Remove extra whitespace
            result["response"] = re.sub(r'\s+', ' ', result["response"]).strip()
            # Fix common formatting issues
            result["response"] = re.sub(r'\*\*(.*?)\*\*', r'\1', result["response"])
        
        # Add processing step to trace
        self._add_trace_step(result, "clean_text")
        
        return result
    
    def _validate_content(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate content against domain rules.
        
        Args:
            data_item: Data item to validate
            
        Returns:
            Validated data item with validation results
        """
        result = copy.deepcopy(data_item)
        
        domain = result.get("domain", "general")
        response = result.get("response", "")
        
        # Initialize validation results
        if "metadata" not in result:
            result["metadata"] = {}
        
        result["metadata"]["validation"] = {
            "is_valid": True,
            "issues": []
        }
        
        # Check minimum length
        min_length = self.config.get("validation", {}).get("min_length", 50)
        if len(response) < min_length:
            result["metadata"]["validation"]["is_valid"] = False
            result["metadata"]["validation"]["issues"].append(
                f"Response too short ({len(response)} chars, minimum {min_length})"
            )
        
        # Check for prohibited content
        prohibited = self.config.get("domains", {}).get(domain, {}).get("prohibited_content", [])
        for term in prohibited:
            if term.lower() in response.lower():
                result["metadata"]["validation"]["is_valid"] = False
                result["metadata"]["validation"]["issues"].append(
                    f"Contains prohibited content: '{term}'"
                )
        
        # Add processing step to trace
        self._add_trace_step(result, "validate_content")
        
        return result
    
    def _add_quality_score(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a simple quality score based on heuristics.
        
        Args:
            data_item: Data item to score
            
        Returns:
            Data item with quality score
        """
        result = copy.deepcopy(data_item)
        
        response = result.get("response", "")
        
        if "metadata" not in result:
            result["metadata"] = {}
        
        # Base score
        score = 0.5
        
        # Length-based adjustment
        if len(response) > 200:
            score += 0.1
        if len(response) > 500:
            score += 0.1
        
        # Structure-based adjustment
        if re.search(r'\d+\.\s', response):  # Numbered points
            score += 0.05
        
        # Complexity-based adjustment
        avg_word_length = sum(len(word) for word in response.split()) / max(1, len(response.split()))
        if avg_word_length > 5:
            score += 0.05
        
        # Validation-based adjustment
        if result.get("metadata", {}).get("validation", {}).get("is_valid", True):
            score += 0.2
        else:
            score -= 0.3
        
        # Clamp score to [0, 1]
        score = max(0.0, min(1.0, score))
        
        # Add score to metadata
        result["metadata"]["quality_score"] = round(score, 2)
        
        # Add processing step to trace
        self._add_trace_step(result, "add_quality_score", {"score": score})
        
        return result
    
    def _tag_entities(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simple entity recognition and tagging.
        
        Args:
            data_item: Data item to process
            
        Returns:
            Data item with entity tags
        """
        result = copy.deepcopy(data_item)
        
        if "metadata" not in result:
            result["metadata"] = {}
        
        # Simple pattern matching for entity recognition
        entities = {
            "monetary_value": r'\$\d+(?:\.\d+)?|\d+\s(?:dollars|USD|EUR|GBP)',
            "percentage": r'\d+(?:\.\d+)?\s?%',
            "date": r'\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}',
            "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            "url": r'https?://[^\s]+'
        }
        
        found_entities = {}
        
        for entity_type, pattern in entities.items():
            matches = re.findall(pattern, result.get("response", ""))
            if matches:
                found_entities[entity_type] = matches
        
        if found_entities:
            result["metadata"]["entities"] = found_entities
        
        # Add processing step to trace
        self._add_trace_step(result, "tag_entities", {"entity_count": len(found_entities)})
        
        return result
    
    def _add_trace_step(self, data_item: Dict[str, Any], step_name: str, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a processing step to the trace.
        
        Args:
            data_item: Data item to update
            step_name: Name of the processing step
            params: Optional parameters for the step
        """
        if "trace" not in data_item:
            data_item["trace"] = {
                "parent_id": None,
                "processing_steps": []
            }
        
        data_item["trace"]["processing_steps"].append({
            "name": step_name,
            "timestamp": datetime.now().isoformat(),
            "params": params or {}
        })
    
    def process_item(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single data item through the entire pipeline.
        
        Args:
            data_item: Data item to process
            
        Returns:
            Processed data item
        """
        result = copy.deepcopy(data_item)
        
        for process_func in self.pipeline:
            result = process_func(result)
        
        return result
    
    def process_dataset(self, dataset: List[Dict[str, Any]], filter_invalid: bool = True) -> List[Dict[str, Any]]:
        """
        Process an entire dataset.
        
        Args:
            dataset: List of data items to process
            filter_invalid: Whether to filter out invalid items
            
        Returns:
            Processed dataset
        """
        logger.info(f"Processing dataset with {len(dataset)} items")
        
        processed_dataset = []
        
        for i, item in enumerate(dataset):
            if i % 100 == 0 and i > 0:
                logger.info(f"Processed {i}/{len(dataset)} items")
            
            processed_item = self.process_item(item)
            
            # Filter out invalid items if requested
            if filter_invalid and not processed_item.get("metadata", {}).get("validation", {}).get("is_valid", True):
                continue
            
            processed_dataset.append(processed_item)
        
        logger.info(f"Processing complete. {len(processed_dataset)}/{len(dataset)} items retained.")
        
        return processed_dataset
    
    def augment_dataset(self, dataset: List[Dict[str, Any]], augmentation_factor: int = 2) -> List[Dict[str, Any]]:
        """
        Augment dataset with variations.
        
        Args:
            dataset: Original dataset
            augmentation_factor: How many variations to create per item
            
        Returns:
            Augmented dataset
        """
        import uuid
        
        logger.info(f"Augmenting dataset with factor {augmentation_factor}")
        
        augmented_dataset = copy.deepcopy(dataset)
        original_size = len(dataset)
        
        for i, item in enumerate(dataset):
            if i % 100 == 0 and i > 0:
                logger.info(f"Augmented {i}/{original_size} items")
            
            for j in range(augmentation_factor - 1):  # -1 because we already have the original
                # Create a variation by modifying the response slightly
                variation = copy.deepcopy(item)
                
                # Assign new ID
                variation["id"] = str(uuid.uuid4())
                
                # Update created_at
                variation["created_at"] = datetime.now().isoformat()
                
                # Set parent ID to original
                if "trace" not in variation:
                    variation["trace"] = {}
                variation["trace"]["parent_id"] = item["id"]
                
                # Add augmentation trace step
                self._add_trace_step(variation, "augmentation", {
                    "original_id": item["id"],
                    "variation_number": j + 1
                })
                
                # Simple text augmentation (in a real system, this would be more sophisticated)
                if "response" in variation:
                    response = variation["response"]
                    
                    # Swap synonyms (simplified example)
                    replacements = [
                        ("important", "crucial"),
                        ("good", "beneficial"),
                        ("bad", "detrimental"),
                        ("big", "large"),
                        ("small", "minor")
                    ]
                    
                    for original, replacement in replacements:
                        response = re.sub(r'\b' + original + r'\b', replacement, response, count=1)
                    
                    variation["response"] = response
                    
                    # Update metadata
                    if "metadata" not in variation:
                        variation["metadata"] = {}
                    variation["metadata"]["augmented"] = True
                    variation["metadata"]["augmentation_method"] = "synonym_replacement"
                
                # Process the augmented item
                processed_variation = self.process_item(variation)
                augmented_dataset.append(processed_variation)
        
        logger.info(f"Augmentation complete. Dataset size increased from {original_size} to {len(augmented_dataset)}")
        
        return augmented_dataset
