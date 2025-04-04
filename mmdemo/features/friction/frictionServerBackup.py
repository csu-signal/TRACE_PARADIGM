# ssh traceteam@tarski.cs.colostate.edu
# cd fact_server
# conda activate frictionEnv
# /home/traceteam/anaconda3/envs/frictionEnv/bin/python /home/traceteam/fact_server/friction_server.pys

import os
import sys
import socket
import re
import random
import numpy as nps
import torch
import pickle
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import pandas as pd

# Hugging Face Libraries
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from peft import AutoPeftModelForCausalLM, LoraConfig, PeftModel

@dataclass
class FrictionMetrics:
    """Metrics for generated friction statements"""
    nll: float
    predictive_entropy: float
    mutual_information: float
    perplexity: float
    conditional_entropy: float

@dataclass
class FrictionOutputInterface:
    """
    Interface for friction generation output in collaborative weight estimation task.
    
    Attributes:
        friction_statement (str): 
            Main friction statement to be displayed/spoken.
            Example: "Are we sure about comparing these blocks without considering their volume?"
            
        task_state (str): 
            Current state of the weight estimation task.
            Hidden from UI but useful for debugging.
            Example: "Red (10g) and Blue blocks compared, Yellow block pending"
            
        belief_state (str): 
            Participants' current beliefs about weights.
            Helps explain friction but may not need display.
            Example: "P1 believes yellow is heaviest, P2 uncertain about blue"
            
        rationale (str): 
            Reasoning behind the friction intervention.
            Could be shown as tooltip/explanation.
            Example: "Participants are making assumptions without evidence"
            
        metrics (Optional[FrictionMetrics]): 
            Model's generation metrics including confidence.
            Useful for debugging and demo insights.
    """
    
    friction_statement: str
    task_state: str
    belief_state: str
    rationale: str
    raw_generation: str

    metrics: Optional[FrictionMetrics] = None

    def to_dict(self):
        return asdict(self)  # Converts the object into a dictionary

class FrictionInference:  
    def __init__(self, model_path: str):
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.tags_to_parse = ["friction", "rationale", "t", "b"]   
            print("Loading base model...")
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )

            # Merge and unload adapter weights
            print("Merging and unloading adapter weights...",  self.model)
            self.model = self.model.merge_and_unload() 
            self.model = self.model.to(self.device)  # Move the model to the GPU device
            print("showing merged lora model",  self.model)
    
            # Load tokenizer
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.tokenizer.pad_token = "<|reserved_special_token_0|>"
            self.tokenizer.padding_side = 'right'

    def parse_generation(self, text: str) -> Dict[str, str]:
        """Parse generated text to extract components"""
        parsed = {tag: [] for tag in self.tags_to_parse}
        
        for tag in self.tags_to_parse:
            pattern = f"<{tag}>(.*?)</{tag}>"
            matches = re.findall(pattern, text, re.DOTALL)
            parsed[tag].extend(matches)
            
        # Handle friction tag specially
        if not parsed["friction"]:
            parsed["friction"] = [self._extract_friction(text)]
            
        return {k: " ".join(v).strip() for k, v in parsed.items()}
    
    def _extract_friction(self, text: str) -> str:
        """Extract friction statement when tags are missing"""
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text.strip())
        if len(sentences) >= 3:
            return f"{sentences[0]} {sentences[-2]} {sentences[-1]}"
        return " ".join(sentences)

    def compute_metrics(self, output_ids: torch.Tensor, scores: List[torch.Tensor], prompt_length: int) -> FrictionMetrics:
        """Compute generation metrics (fixed device handling)"""
        with torch.no_grad():
            # Ensure all tensors are on the same device
            logits = torch.stack(scores, dim=0).to(self.device)
            output_ids = output_ids.to(self.device)
            
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)
            
            # Get generated token probabilities
            token_ids = output_ids[prompt_length:]
            probs = probs[:, 0, :]  # Take first sequence
            token_probs = probs[torch.arange(len(token_ids), device=self.device), token_ids]
            
            # Calculate metrics
            nll = -torch.sum(torch.log(token_probs)) / len(token_ids)
            predictive_entropy = -torch.sum(probs * log_probs, dim=-1).mean()
            conditional_entropy = -torch.mean(torch.log(token_probs))
            mutual_information = max(predictive_entropy - conditional_entropy, 0.0)
            perplexity = torch.exp(nll)
        
            return FrictionMetrics(
                nll=nll.item(),
                predictive_entropy=predictive_entropy.item(),
                mutual_information=mutual_information.item(),
                perplexity=perplexity.item(),
                conditional_entropy=conditional_entropy.item()
            )

    def generate_friction(self, dialogue_history: str, seed: int = 42) -> dict:
        # Format prompt
        torch.manual_seed(seed)
        system_prompt_rm = (
        "You are an expert in collaborative task analysis and personality-driven communication. Think step by step. "
        "Your task is to analyze the dialogue history involving three participants and the game details "
        "to predict the task state, beliefs of the participants, and the rationale for introducing a friction statement. "
        "Finally, generate a nuanced friction statement in a conversational style based on your analysis.\n\n"
        "1. Predict the task-related context and enclose it between the markers `<t>` and `</t>`.\n\n"
        "2. Predict the belief-related context for the participants and enclose it between the markers `<b>` and `</b>`.\n\n"
        "3. Provide a rationale for why a friction statement is needed. This monologue must be enclosed between the "
        "markers `<rationale>` and `</rationale>`. Base your reasoning on evidence from the dialogue, focusing on elements such as:\n"
        "- Incorrect assumptions\n"
        "- False beliefs\n"
        "- Rash decisions\n"
        "- Missing evidence.\n\n"
        "4. Generate the friction statement, ensuring it is enclosed between the markers `<friction>` and `</friction>`. "
        "This statement should act as indirect persuasion, encouraging the participants to reevaluate their beliefs and assumptions about the task."
        )

        friction_definition_game_definition_prompt_rm = (
        "The game is called 'Game of Weights,' where participants (P1, P2, and P3) determine the weights of colored blocks. "
        "Participants can weigh two blocks at a time and know the weight of the red block. "
        "They must deduce the weights of other blocks. "
        "The dialogue history is provided below:"
        )
 
        # Final formatted prompt for the LLM
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            {system_prompt_rm} 

            {friction_definition_game_definition_prompt_rm}

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            {dialogue_history}

            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            ### Assistant:"""
 
        # Generate response
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)  
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                return_dict_in_generate=True,
                output_scores=True
            )
            generated_ids = outputs.sequences[0]
            generated_ids = generated_ids.to(self.device)
            generated_text = self.tokenizer.decode(
                outputs.sequences[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            scores = outputs.scores  # Already on correct device, don't move to list

            metrics = self.compute_metrics(generated_ids, scores, inputs['input_ids'].shape[1]) # probably not needed
            parsed = self.parse_generation(generated_text)
            #print("\nGenerated text:", generated_text)  # Debug print
            
            # Parse components
            task_state = self._extract_tag(generated_text, "t")
            belief_state = self._extract_tag(generated_text, "b")
            rationale = self._extract_tag(generated_text, "rationale")
            friction = self._extract_tag(generated_text, "friction")
            print("task state:", task_state)
            print("belief state:", belief_state)
            print("rationale:", rationale)
            print("friction:", friction)
            print("metrics", metrics, "\n")

            return FrictionOutputInterface(
                friction_statement=friction,
                task_state=task_state,
                belief_state=belief_state,
                rationale=rationale,
                metrics=metrics,
                raw_generation=generated_text) 
            
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            return None
        
    def _extract_tag(self, text: str, tag: str) -> str:
        import re
        pattern = f"<{tag}>(.*?)</{tag}>"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[0].strip() if matches else ""

def start_server(friction_detector: FrictionInference):
    HOST = '129.82.138.15'  # Standard loopback interface address (localhost)
    PORT = 65432        # Port to listen on (non-privileged ports are > 1023)
    friction_list = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {HOST}:{PORT}")

        while True:
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                while True:
                    try:
                        data = conn.recv(2048)
                        if not data:
                            break
                        print("Received Data Length:" + str(len(data)))
                        transcriptions = data.decode()
                        print(f"Transcriptions:\n{transcriptions}")
                        print("\nGenerating friction for dialogue...")
                        result = friction_detector.generate_friction(transcriptions)
                        returnString = ''
                        if result is not None:
                            if result.friction_statement != '':
                                returnString += "Friction: " + result.friction_statement
                                if result.rationale != '':  
                                    returnString += "r*Rationale" + result.rationale 
                            else:
                                conn.sendall(str.encode("No Friction", 'utf-8')) 
                                break
                            returnString = returnString.replace("’","'")
                            conn.sendall(str.encode(returnString, 'utf-8')) 
                        else:
                            conn.sendall(str.encode("No Friction", 'utf-8'))
                    except ConnectionResetError as e:
                        print(f"Connection with {addr} was reset: {e}")
                        break
                    except Exception as e:
                        print(f"An error occurred: {e}")
                        break
    friction_list_df = pd.Dataframe(friction_list)
    friction_list_df.to_csv("friction_list_df.csv")


if __name__ == "__main__":
    print("Initializing friction detector...")
    start_server(FrictionInference("Abhijnan/friction_sft_allsamples_weights_instruct")) #this is the lora model id on huggingface (SFT model)
    