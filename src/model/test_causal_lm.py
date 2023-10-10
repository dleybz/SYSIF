import pytest
import torch
from causal_lm import CausalLanguageModel

# Initialize the model for testing
model_name = "gpt2"
lm = CausalLanguageModel(model_name, device="cuda" if torch.cuda.is_available() else "cpu")

# Define test functions

def test_forward_pass():
    input_text = "This is a test sentence for forward pass."
    output = lm.forward_pass(input_text)
    assert isinstance(output, dict)  # Adjust this based on the actual return type
    # Add more assertions as needed

def test_get_output_probability():
    input_text = "This is a test sentence for output probability."
    probs = lm.get_output_probability(input_text)
    assert isinstance(probs, torch.Tensor)
    # Add more assertions as needed

def test_get_attention_maps():
    input_text = "This is a test sentence for attention maps."
    attentions = lm.get_attention_maps(input_text)
    assert isinstance(attentions, list)
    # Add more assertions as needed

def test_get_activation_values():
    input_text = "This is a test sentence for activation values."
    activations = lm.get_activation_values(input_text)
    assert isinstance(activations, list)
    # Add more assertions as needed

def test_get_intermediate_representation():
    layer_id = 11  # Change this to the desired layer for testing
    input_text = "This is a test sentence for intermediate representation."
    intermediate = lm.get_intermediate_representation(input_text, layer_id)
    assert isinstance(intermediate, torch.Tensor)
    # Add more assertions as needed
