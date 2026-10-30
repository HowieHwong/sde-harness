#!/usr/bin/env python3
"""
Example script for single molecule elucidation using the Spectrum Elucidator Toolkit.

This script demonstrates how to use the toolkit to elucidate a single molecule
from its NMR spectrum through iterative LLM prompting.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import ConfigManager
from src.data_utils import MolecularDataLoader
from src.llm_interface import LLMInterface
from src.elucidation_engine import ElucidationEngine, ElucidationConfig
from src.visualization import ElucidationVisualizer


def main():
    """Main function for single molecule elucidation."""
    
    print("=" * 60)
    print("SPECTRUM ELUCIDATOR TOOLKIT - SINGLE MOLECULE EXAMPLE")
    print("=" * 60)
    
    # Initialize configuration
    print("\n1. Initializing configuration...")
    config_manager = ConfigManager()
    
    # Load environment variables (for API key)
    config_manager.get_env_config()
    
    # Check if API key is available
    if not config_manager.get_llm_config().api_key:
        print("ERROR: OpenAI API key not found!")
        print("Please set the OPENAI_API_KEY environment variable or update config.json")
        print("Example: export OPENAI_API_KEY='your_api_key_here'")
        return
    
    # Validate configuration
    if not config_manager.validate_config():
        print("Configuration validation failed!")
        return
    
    print("✓ Configuration loaded and validated")
    
    # Initialize data loader
    print("\n2. Loading molecular data...")
    data_config = config_manager.get_data_config()
    data_loader = MolecularDataLoader(data_config.data_path)
    
    # Get dataset info
    dataset_info = data_loader.get_dataset_info()
    print(f"✓ Loaded {dataset_info['total_molecules']} molecules")
    print(f"✓ Available functional groups: {len(dataset_info['functional_groups'])}")
    
    # Initialize LLM interface
    print("\n3. Initializing LLM interface...")
    llm_config = config_manager.get_llm_config()
    llm_interface = LLMInterface(
        api_key=llm_config.api_key,
        model=llm_config.model,
        max_tokens=llm_config.max_tokens
    )
    print(f"✓ LLM interface initialized with model: {llm_config.model}")
    
    # Initialize elucidation engine
    print("\n4. Initializing elucidation engine...")
    elucidation_config = config_manager.get_elucidation_config()
    engine = ElucidationEngine(
        data_loader=data_loader,
        llm_interface=llm_interface,
        config=elucidation_config
    )
    print("✓ Elucidation engine initialized")
    
    # Initialize visualizer
    print("\n5. Initializing visualizer...")
    viz_config = config_manager.get_visualization_config()
    visualizer = ElucidationVisualizer(style=viz_config.style)
    print("✓ Visualizer initialized")
    
    # Select target molecule
    print("\n6. Selecting target molecule...")
    
    # You can either specify a molecule ID or use a random one
    target_molecule_id = "5_99"  # Example molecule ID
    
    # Check if molecule exists
    target_molecule = data_loader.get_molecule_by_id(target_molecule_id)
    if not target_molecule:
        print(f"ERROR: Molecule {target_molecule_id} not found!")
        print("Using random molecule instead...")
        target_molecule = data_loader.get_random_molecule()
        target_molecule_id = target_molecule['molecule_id']
    
    print(f"✓ Target molecule: {target_molecule_id}")
    print(f"  SMILES: {target_molecule['SMILES']}")
    print(f"  H-NMR: {target_molecule['H_NMR'][:100]}...")
    
    # Perform elucidation
    print("\n7. Starting molecular structure elucidation...")
    print(f"   Max iterations: {elucidation_config.max_iterations}")
    print(f"   Similarity threshold: {elucidation_config.similarity_threshold}")
    print(f"   Temperature: {elucidation_config.temperature}")
    
    try:
        result = engine.elucidate_molecule(target_molecule_id)
        
        print("\n8. Elucidation completed!")
        print(f"   Success: {result.success}")
        print(f"   Final similarity: {result.final_similarity:.3f}")
        print(f"   Total iterations: {result.total_iterations}")
        print(f"   Execution time: {result.execution_time:.2f} seconds")
        
        if result.final_smiles:
            print(f"   Final SMILES: {result.final_smiles}")
        
        # Generate summary report
        print("\n9. Generating summary report...")
        summary = engine.get_elucidation_summary(result)
        print(f"   Best similarity: {summary['best_similarity']:.3f}")
        print(f"   Improvement rate: {summary['improvement_rate']:.3f}")
        
        # Create visualizations
        print("\n10. Creating visualizations...")
        
        # Similarity progression plot
        fig1 = visualizer.plot_similarity_progression(
            result, 
            save_path=f"results/{target_molecule_id}_similarity_progression.png",
            show_plot=viz_config.show_plots
        )
        
        # SMILES evolution plot
        fig2 = visualizer.plot_smiles_evolution(
            result,
            save_path=f"results/{target_molecule_id}_smiles_evolution.png",
            show_plot=viz_config.show_plots
        )
        
        # Summary report
        report = visualizer.create_summary_report(
            result,
            save_path=f"results/{target_molecule_id}_summary.txt"
        )
        
        print("✓ Visualizations and report created")
        
        # Save detailed results
        print("\n11. Saving detailed results...")
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Save result object as JSON
        import json
        from dataclasses import asdict
        
        result_file = results_dir / f"{target_molecule_id}_detailed_results.json"
        with open(result_file, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        
        print(f"✓ Detailed results saved to: {result_file}")
        
        print("\n" + "=" * 60)
        print("ELUCIDATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: Elucidation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print final summary
    print("\nFINAL SUMMARY:")
    print(f"Target Molecule: {target_molecule_id}")
    print(f"Status: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Final NMR Similarity: {result.final_similarity:.3f}")
    print(f"Best SMILES: {result.final_smiles or 'None'}")
    print(f"Total Iterations: {result.total_iterations}")
    print(f"Execution Time: {result.execution_time:.2f} seconds")
    
    if result.success:
        print("\n🎉 Congratulations! The target similarity threshold was reached.")
    else:
        print("\n⚠️  Elucidation incomplete. Consider:")
        print("   - Increasing max iterations")
        print("   - Adjusting temperature parameter")
        print("   - Providing additional molecular context")


if __name__ == "__main__":
    main()
