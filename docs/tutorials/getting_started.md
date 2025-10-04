#!/usr/bin/env python3
"""
Getting Started Tutorial - Jupyter Notebook Version
Run this as a Python script or convert to notebook format
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
import matplotlib.pyplot as plt
from config import ModelConfig, VisualizationConfig
from models import TransformerEncoder
from positional_encoding import get_positional_encoding
from visualization import AttentionVisualizer, EncodingPlotter
from utils.tokenizer import SimpleTokenizer

def tutorial_setup():
    """Initial setup for the tutorial"""
    print("🧠 Positional Encoding Visualizer - Getting Started Tutorial")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Check device availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    return device

def step1_basic_configuration():
    """Step 1: Create basic model configuration"""
    print("\n📋 Step 1: Basic Configuration")
    
    # Create basic model configuration
    config = ModelConfig(
        d_model=128,
        n_heads=8,
        n_layers=4,
        max_seq_len=64,
        encoding_type="sinusoidal"
    )
    
    print(f"Model dimension: {config.d_model}")
    print(f"Number of heads: {config.n_heads}")
    print(f"Encoding type: {config.encoding_type}")
    
    return config

def step2_create_positional_encoding(config):
    """Step 2: Create and visualize positional encoding"""
    print("\n🔢 Step 2: Create Positional Encoding")
    
    # Get positional encoding
    encoding = get_positional_encoding(config)
    
    # Generate encoding for sequence length 32
    seq_len = 32
    encoding_output = encoding.forward(seq_len, config.d_model)
    
    print(f"Encoding shape: {encoding_output.shape}")
    print(f"Encoding range: [{encoding_output.min():.3f}, {encoding_output.max():.3f}]")
    
    # Create visualization
    viz_config = VisualizationConfig()
    plotter = EncodingPlotter(viz_config)
    
    # Plot sinusoidal patterns
    if hasattr(encoding, 'analyze_frequency_components'):
        fig = plotter.plot_sinusoidal_patterns(encoding, seq_len=seq_len)
        plt.savefig('tutorial_encoding_patterns.png', dpi=300, bbox_inches='tight')
        print("✅ Saved encoding patterns to 'tutorial_encoding_patterns.png'")
    
    return encoding, encoding_output

def step3_create_transformer_model(config):
    """Step 3: Create transformer model"""
    print("\n🤖 Step 3: Create Transformer Model")
    
    # Create transformer model
    model = TransformerEncoder(config)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model

def step4_analyze_attention_patterns(model, config):
    """Step 4: Analyze attention patterns"""
    print("\n👁️ Step 4: Analyze Attention Patterns")
    
    # Create sample input
    tokenizer = SimpleTokenizer()
    sample_text = "The quick brown fox jumps over the lazy dog"
    tokens = tokenizer.tokenize(sample_text)
    
    # Simple encoding (just use token indices)
    input_ids = torch.tensor([list(range(len(tokens)))]).long()
    
    print(f"Input text: {sample_text}")
    print(f"Tokens: {tokens}")
    print(f"Input shape: {input_ids.shape}")
    
    # Forward pass with visualization storage
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, store_visualizations=True)
    
    attention_weights = outputs['attention_weights']
    print(f"Number of layers with attention: {len(attention_weights)}")
    
    # Visualize first layer, first head
    if attention_weights:
        viz_config = VisualizationConfig()
        visualizer = AttentionVisualizer(viz_config)
        
        # Create attention heatmap
        fig = visualizer.visualize_attention_matrix(
            attention_weights[0],  # First layer
            tokens=tokens[:input_ids.shape[1]],
            head_idx=0,
            title="Attention Pattern - Layer 0, Head 0"
        )
        
        plt.savefig('tutorial_attention_pattern.png', dpi=300, bbox_inches='tight')
        print("✅ Saved attention pattern to 'tutorial_attention_pattern.png'")
        
        # Analyze attention patterns
        analysis = visualizer.analyze_attention_patterns(attention_weights[0])
        print(f"Average attention entropy: {analysis['entropy']['mean_entropy']:.3f}")
        print(f"Attention pattern type: {analysis.get('pattern_type', 'unknown')}")
    
    return attention_weights

def step5_compare_encoding_methods():
    """Step 5: Compare different encoding methods"""
    print("\n⚖️ Step 5: Compare Encoding Methods")
    
    encoding_types = ['sinusoidal', 'learned', 'rope']
    seq_len = 32
    d_model = 128
    
    encoding_matrices = {}
    
    for encoding_type in encoding_types:
        try:
            config = ModelConfig(
                d_model=d_model,
                encoding_type=encoding_type
            )
            
            encoding = get_positional_encoding(config)
            
            if hasattr(encoding, 'forward'):
                if encoding_type == 'rope':
                    # RoPE returns dict with cos/sin
                    rope_output = encoding.forward(seq_len, d_model)
                    if isinstance(rope_output, dict):
                        # Combine cos and sin for visualization
                        encoding_matrix = torch.cat([
                            rope_output['cos'].squeeze(0),
                            rope_output['sin'].squeeze(0)
                        ], dim=-1)
                    else:
                        encoding_matrix = rope_output.squeeze(0)
                else:
                    encoding_matrix = encoding.forward(seq_len, d_model).squeeze(0)
                
                encoding_matrices[encoding_type] = encoding_matrix
                print(f"✅ Created {encoding_type} encoding: {encoding_matrix.shape}")
        
        except Exception as e:
            print(f"⚠️ Failed to create {encoding_type} encoding: {e}")
    
    # Create comparison plot
    if len(encoding_matrices) >= 2:
        fig, axes = plt.subplots(1, len(encoding_matrices), figsize=(15, 5))
        if len(encoding_matrices) == 1:
            axes = [axes]
        
        for idx, (name, matrix) in enumerate(encoding_matrices.items()):
            im = axes[idx].imshow(matrix.T, aspect='auto', cmap='RdBu')
            axes[idx].set_title(f'{name.title()} Encoding')
            axes[idx].set_xlabel('Position')
            axes[idx].set_ylabel('Dimension')
            plt.colorbar(im, ax=axes[idx])
        
        plt.tight_layout()
        plt.savefig('tutorial_encoding_comparison.png', dpi=300, bbox_inches='tight')
        print("✅ Saved encoding comparison to 'tutorial_encoding_comparison.png'")
    
    return encoding_matrices

def step6_performance_analysis():
    """Step 6: Basic performance analysis"""
    print("\n📊 Step 6: Performance Analysis")
    
    from utils.performance_profiler import PerformanceProfiler
    
    profiler = PerformanceProfiler()
    
    # Profile encoding generation
    sequence_lengths = [16, 32, 64, 128]
    results = {}
    
    for seq_len in sequence_lengths:
        config = ModelConfig(d_model=128, encoding_type='sinusoidal')
        encoding = get_positional_encoding(config)
        
        # Profile the forward pass
        with profiler.profile(f"encoding_seq_{seq_len}"):
            _ = encoding.forward(seq_len, 128)
        
        if profiler.results:
            results[seq_len] = profiler.results[-1].execution_time
    
    # Display results
    print("Encoding generation times:")
    for seq_len, time_taken in results.items():
        print(f"  Sequence length {seq_len}: {time_taken*1000:.2f} ms")
    
    return results

def step7_save_configuration(config):
    """Step 7: Save configuration for later use"""
    print("\n💾 Step 7: Save Configuration")
    
    import json
    
    # Convert config to dict
    config_dict = {
        'd_model': config.d_model,
        'n_heads': config.n_heads,
        'n_layers': config.n_layers,
        'max_seq_len': config.max_seq_len,
        'encoding_type': config.encoding_type,
        'dropout': config.dropout
    }
    
    # Save to JSON
    with open('tutorial_config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print("✅ Saved configuration to 'tutorial_config.json'")
    
    # Load and verify
    with open('tutorial_config.json', 'r') as f:
        loaded_config = json.load(f)
    
    print("Configuration saved and verified:")
    for key, value in loaded_config.items():
        print(f"  {key}: {value}")

def tutorial_cleanup():
    """Clean up tutorial resources"""
    print("\n🧹 Tutorial Cleanup")
    
    # Close all matplotlib figures
    plt.close('all')
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("✅ Cleanup completed")

def main():
    """Run the complete getting started tutorial"""
    
    try:
        # Setup
        device = tutorial_setup()
        
        # Step 1: Configuration
        config = step1_basic_configuration()
        
        # Step 2: Positional Encoding
        encoding, encoding_output = step2_create_positional_encoding(config)
        
        # Step 3: Transformer Model
        model = step3_create_transformer_model(config)
        
        # Step 4: Attention Analysis
        attention_weights = step4_analyze_attention_patterns(model, config)
        
        # Step 5: Encoding Comparison
        encoding_matrices = step5_compare_encoding_methods()
        
        # Step 6: Performance Analysis
        performance_results = step6_performance_analysis()
        
        # Step 7: Save Configuration
        step7_save_configuration(config)
        
        # Summary
        print("\n🎉 Tutorial Completed Successfully!")
        print("Generated files:")
        print("  - tutorial_encoding_patterns.png")
        print("  - tutorial_attention_pattern.png") 
        print("  - tutorial_encoding_comparison.png")
        print("  - tutorial_config.json")
        
        print("\n🚀 Next Steps:")
        print("  - Try the advanced_usage.py tutorial")
        print("  - Experiment with different encoding types")
        print("  - Run the interactive dashboard")
        print("  - Explore customization options")
        
    except Exception as e:
        print(f"❌ Tutorial failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        tutorial_cleanup()

if __name__ == "__main__":
    main()
