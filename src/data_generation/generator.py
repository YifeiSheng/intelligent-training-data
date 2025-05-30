import os
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.config import load_config
from ..utils.logging import setup_logger

logger = setup_logger("data_generator")

class DataGenerator:
    """Main class for generating training data pairs."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", config_path: Optional[str] = None):
        """
        Initialize the data generator with a specific model.
        
        Args:
            model_name: Name of the model to use for generation
            config_path: Path to configuration file
        """
        self.model_name = model_name
        self.config_path = config_path
        
        # Load configuration if provided
        self.config = self._load_config(config_path) if config_path else {}
        
        # Detect if using a VL (Vision-Language) model
        is_vl_model = "vl" in model_name.lower() or "vision" in model_name.lower()
        
        # Load the appropriate model based on type
        try:
            if is_vl_model:
                from transformers import AutoModelForVision2Seq, AutoProcessor
                self.tokenizer = AutoProcessor.from_pretrained(model_name)
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    device_map="auto",
                    **self.config.get("model_params", {})
                )
            else:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    **self.config.get("model_params", {})
                )
            logger.info(f"Successfully loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Attempting to load a compatible model variant...")
            
            # If Qwen2.5 VL fails, try falling back to Qwen2
            if "qwen2.5" in model_name.lower() and "vl" in model_name.lower():
                fallback_model = model_name.replace("2.5-vl", "2").replace("2.5_vl", "2")
                logger.info(f"Falling back to: {fallback_model}")
                
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                self.model = AutoModelForCausalLM.from_pretrained(
                    fallback_model,
                    device_map="auto",
                    **self.config.get("model_params", {})
                )
                logger.info(f"Successfully loaded fallback model: {fallback_model}")
        
        # Load domain-specific templates and rules
        self.templates = self._load_templates()
        self.domain_rules = self._load_domain_rules()
        
        logger.info(f"Loaded {len(self.templates)} templates and {len(self.domain_rules)} domain rules")
    
    def _load_templates(self) -> List[Dict[str, Any]]:
        """Load templates from configuration."""
        template_path = self.config.get("template_path", "config/templates.json")
        try:
            if os.path.exists(template_path):
                with open(template_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Template file {template_path} not found. Using default templates.")
                return [
                    {
                        "domain": "finance",
                        "template": "As a financial advisor, how would you recommend {client_type} to {action} given {situation}?",
                        "parameters": {
                            "client_type": ["individual investor", "small business", "retiree"],
                            "action": ["invest", "save", "plan for retirement"],
                            "situation": ["market volatility", "low interest rates", "economic uncertainty"]
                        }
                    },
                    {
                        "domain": "healthcare",
                        "template": "What advice would you give to {patient_type} regarding {health_concern} considering their {patient_condition}?",
                        "parameters": {
                            "patient_type": ["elderly patients", "young adults", "children", "pregnant women"],
                            "health_concern": ["preventive care", "chronic disease management", "medication adherence", "nutrition"],
                            "patient_condition": ["diabetes", "hypertension", "obesity", "limited mobility"]
                        }
                    },
                    {
                        "domain": "healthcare",
                        "template": "How should healthcare providers approach {procedure} for patients with {condition}?",
                        "parameters": {
                            "procedure": ["screening", "diagnosis", "treatment planning", "follow-up care"],
                            "condition": ["chronic heart disease", "autoimmune disorders", "mental health issues", "respiratory conditions"]
                        }
                    },
                    {
                        "domain": "healthcare",
                        "template": "What are best practices for {healthcare_role} when dealing with {scenario} in {setting}?",
                        "parameters": {
                            "healthcare_role": ["nurses", "primary care physicians", "specialists", "caregivers"],
                            "scenario": ["emergency situations", "preventive care visits", "telehealth consultations", "patient education"],
                            "setting": ["hospitals", "outpatient clinics", "long-term care facilities", "home care"]
                        }
                    }
                ]
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
            return []
    
    def _load_domain_rules(self) -> Dict[str, Any]:
        """Load domain-specific rules."""
        rules_path = self.config.get("rules_path", "config/domain_rules.json")
        try:
            if os.path.exists(rules_path):
                with open(rules_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Domain rules file {rules_path} not found. Using default rules.")
                return {
                    "finance": {
                        "required_entities": ["monetary_value", "financial_instrument"],
                        "prohibited_content": ["exact_predictions", "guaranteed_returns"]
                    }
                }
        except Exception as e:
            logger.error(f"Error loading domain rules: {e}")
            return {}
    
    def generate_prompt(self, domain: str, template_idx: Optional[int] = None) -> Dict[str, str]:
        """
        Generate a prompt based on templates.
        
        Args:
            domain: The domain to generate for
            template_idx: Specific template index to use (random if None)
            
        Returns:
            Dictionary with generated prompt and metadata
        """
        # Filter templates by domain
        domain_templates = [t for t in self.templates if t["domain"] == domain]
        if not domain_templates:
            logger.warning(f"No templates found for domain {domain}")
            return {"error": f"No templates found for domain {domain}"}
        
        # Select template
        import random
        template = domain_templates[template_idx] if template_idx is not None else random.choice(domain_templates)
        
        # Fill template with parameters
        prompt_text = template["template"]
        params_used = {}
        
        for param_name, options in template["parameters"].items():
            selected_value = random.choice(options)
            prompt_text = prompt_text.replace(f"{{{param_name}}}", selected_value)
            params_used[param_name] = selected_value
        
        return {
            "prompt": prompt_text,
            "domain": domain,
            "template_id": self.templates.index(template),
            "parameters": params_used,
            "created_at": datetime.now().isoformat()
        }
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response using the model.
        
        Args:
            prompt: Input prompt to generate a response for
            
        Returns:
            Generated response text
        """
        try:
            # Format prompt for the model
            model_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # Generate response
            inputs = self.tokenizer(model_prompt, return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(
                inputs.input_ids,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            
            # Decode and clean up response
            response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
            
            # Extract just the assistant's response
            response = response.split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0].strip()
            
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def create_data_pair(self, domain: str) -> Dict[str, Any]:
        """
        Create a complete data pair (prompt + response) with metadata.
        
        Args:
            domain: Domain to generate data for
            
        Returns:
            Complete data pair as a dictionary
        """
        # Generate prompt
        prompt_data = self.generate_prompt(domain)
        if "error" in prompt_data:
            return prompt_data
        
        prompt = prompt_data["prompt"]
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Create complete data item
        data_pair = {
            "id": str(uuid.uuid4()),
            "created_at": datetime.now().isoformat(),
            "domain": domain,
            "input": prompt,
            "response": response,
            "metadata": {
                "generation_method": "template_based",
                "generator_version": "1.0",
                "model_used": self.model_name,
                "template_id": prompt_data["template_id"],
                "parameters": prompt_data["parameters"]
            },
            "trace": {
                "parent_id": None,
                "processing_steps": [
                    {
                        "name": "initial_generation",
                        "timestamp": datetime.now().isoformat(),
                        "params": {}
                    }
                ]
            }
        }
        
        return data_pair
    
    def generate_dataset(self, domain: str, size: int, output_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate a complete dataset of the specified size.
        
        Args:
            domain: Domain to generate data for
            size: Number of examples to generate
            output_path: Path to save the dataset (optional)
            
        Returns:
            List of generated data pairs
        """
        logger.info(f"Generating dataset with {size} examples for domain '{domain}'")
        
        dataset = []
        for i in range(size):
            if i % 10 == 0:
                logger.info(f"Generated {i}/{size} examples")
            
            data_pair = self.create_data_pair(domain)
            if "error" not in data_pair:
                dataset.append(data_pair)
        
        # Save dataset if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(dataset, f, indent=2)
            logger.info(f"Dataset saved to {output_path}")
        
        return dataset
