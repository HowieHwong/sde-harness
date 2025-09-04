#!/usr/bin/env python3
"""
Lightweight example script for molecular structure elucidation using the Spectrum Elucidator Toolkit.

This script demonstrates basic functionality without requiring heavy dependencies like:
- No Selenium required
- RDKit (for molecular validation)
- EasyOCR (for image processing)

It focuses on the core elucidation logic and LLM-based NMR prediction.
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
    """Main function for lightweight elucidation demonstration."""
    
    print("=" * 70)
    print("SPECTRUM ELUCIDATOR TOOLKIT - LIGHTWEIGHT EXAMPLE")
    print("=" * 70)
    
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
    
    # Update configuration for lightweight mode
    print("\n2. Configuring lightweight mode...")
    config_manager.update_elucidation_config(
        use_nmr_predictor=False,  # Disable NMR predictor to avoid heavy dependencies
        max_iterations=5,          # Reduce iterations for faster testing
        similarity_threshold=0.6,  # Lower threshold for easier success
        log_level="DEBUG"          # Enable debug logging to see similarity calculation details
    )
    
    # Validate configuration
    if not config_manager.validate_config():
        print("Configuration validation failed!")
        return
    
    print("✓ Configuration loaded and validated (lightweight mode)")
    
    # Initialize data loader
    print("\n3. Loading molecular data...")
    data_config = config_manager.get_data_config()
    data_loader = MolecularDataLoader(data_config.data_path)
    
    # Get dataset info
    dataset_info = data_loader.get_dataset_info()
    print(f"✓ Loaded {dataset_info['total_molecules']} molecules")
    
    # Initialize LLM interface
    print("\n4. Initializing LLM interface...")
    llm_config = config_manager.get_llm_config()
    llm_interface = LLMInterface(
        api_key=llm_config.api_key,
        model=llm_config.model,
        max_tokens=llm_config.max_tokens
    )
    print(f"✓ LLM interface initialized with model: {llm_config.model}")
    
    # Initialize elucidation engine (without NMR predictor)
    print("\n5. Initializing elucidation engine...")
    elucidation_config = config_manager.get_elucidation_config()
    engine = ElucidationEngine(
        data_loader=data_loader,
        llm_interface=llm_interface,
        config=elucidation_config,
        nmr_predictor=None  # No NMR predictor in lightweight mode
    )
    print("✓ Elucidation engine initialized (database lookup mode)")
    
    # Initialize visualizer
    print("\n6. Initializing visualizer...")
    viz_config = config_manager.get_visualization_config()
    visualizer = ElucidationVisualizer(style=viz_config.style)
    print("✓ Visualizer initialized")
    
    # Select target molecule
    print("\n7. Selecting target molecule...")
    
    # Use a simple molecule for testing
    target_molecule_id = "5_99"  # Use a known molecule ID
    
    # Check if molecule exists
    target_molecule = data_loader.get_molecule_by_id(target_molecule_id)
    if not target_molecule:
        print(f"ERROR: Molecule {target_molecule_id} not found!")
        # Prefer the first available ID from dataset
        info = data_loader.get_dataset_info()
        fallback_id = None
        if info and info.get('sample_molecules'):
            fallback_id = info['sample_molecules'][0]
        if fallback_id:
            print(f"Using first available molecule instead: {fallback_id}")
            target_molecule_id = fallback_id
            target_molecule = data_loader.get_molecule_by_id(target_molecule_id)
        if not target_molecule:
            print("Using random molecule instead...")
            target_molecule = data_loader.get_random_molecule()
            if target_molecule:
                target_molecule_id = target_molecule['molecule_id']
        # Final fallback: use first row directly if still None
        if not target_molecule:
            if getattr(data_loader, 'data', None) is not None and not data_loader.data.empty:
                row = data_loader.data.iloc[0].to_dict()
                target_molecule = row
                target_molecule_id = str(row.get('molecule_id', 'ROW_0')).strip()
                print(f"Using dataset first row as fallback: {target_molecule_id}")
            else:
                raise RuntimeError("No molecules available in dataset after fallback attempts.")
    
    print(f"✓ Target molecule: {target_molecule_id}")
    print(f"  SMILES: {target_molecule['SMILES']}")
    if target_molecule.get('H_NMR'):
        print(f"  H-NMR: {target_molecule['H_NMR'][:100]}...")
    if target_molecule.get('C_NMR'):
        print(f"  C-NMR: {target_molecule['C_NMR'][:100]}...")
    
    # Perform elucidation
    print("\n8. Starting lightweight molecular structure elucidation...")
    print(f"   Max iterations: {elucidation_config.max_iterations}")
    print(f"   Similarity threshold: {elucidation_config.similarity_threshold}")
    print(f"   Temperature: {elucidation_config.temperature}")
    print(f"   NMR predictor enabled: {elucidation_config.use_nmr_predictor}")
    print(f"   Mode: Database lookup only (lightweight)")
    
    # Initialize result variable
    result = None
    
    try:
        result = engine.elucidate_molecule(target_molecule_id)
        
        print("\n9. Elucidation completed!")
        print(f"    Success: {result.success}")
        print(f"    Final similarity: {result.final_similarity:.3f}")
        print(f"    Total iterations: {result.total_iterations}")
        print(f"    Execution time: {result.execution_time:.2f} seconds")
        print(f"    NMR predictor used: {result.metadata.get('nmr_prediction_used', False)}")
        
        if result.final_smiles:
            print(f"    Final SMILES: {result.final_smiles}")
        
        # Generate summary report
        print("\n10. Generating summary report...")
        summary = engine.get_elucidation_summary(result)
        print(f"    Best similarity: {summary['best_similarity']:.3f}")
        print(f"    Improvement rate: {summary['improvement_rate']:.3f}")
        print(f"    NMR prediction used: {summary['nmr_prediction_used']}")
        
        # Show prediction methods used
        prediction_methods = set(step['prediction_method'] for step in summary['similarity_progression'])
        print(f"    Prediction methods used: {', '.join(prediction_methods)}")
        
        # Create visualizations
        print("\n11. Creating visualizations...")
        
        # Similarity progression plot
        fig1 = visualizer.plot_similarity_progression(
            result, 
            save_path=f"results/{target_molecule_id}_lightweight_progression.png",
            show_plot=viz_config.show_plots
        )
        
        # SMILES evolution plot
        fig2 = visualizer.plot_smiles_evolution(
            result,
            save_path=f"results/{target_molecule_id}_lightweight_evolution.png",
            show_plot=viz_config.show_plots
        )
        
        # Summary report
        report = visualizer.create_summary_report(
            result,
            save_path=f"results/{target_molecule_id}_lightweight_summary.txt"
        )
        
        print("✓ Visualizations and report created")
        
        # Save detailed results
        print("\n12. Saving detailed results...")
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Save result object as JSON
        import json
        from dataclasses import asdict
        
        result_file = results_dir / f"{target_molecule_id}_lightweight_results.json"
        with open(result_file, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        
        print(f"✓ Results saved to: {result_file}")
        
        print("\n" + "=" * 70)
        print("LIGHTWEIGHT ELUCIDATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nERROR: Lightweight elucidation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print final summary
    if result is not None:
        print("\nLIGHTWEIGHT ELUCIDATION SUMMARY:")
        print(f"Target Molecule: {target_molecule_id}")
        print(f"Status: {'SUCCESS' if result.success else 'FAILED'}")
        print(f"Final NMR Similarity: {result.final_similarity:.3f}")
        print(f"Best SMILES: {result.final_smiles or 'None'}")
        print(f"Total Iterations: {result.total_iterations}")
        print(f"Execution Time: {result.execution_time:.2f} seconds")
        print(f"NMR Predictor Used: {result.metadata.get('nmr_prediction_used', False)}")
        
        # Show NMR data availability
        if result.target_c_nmr:
            print(f"Target C-NMR Available: Yes")
        if target_molecule.get('H_NMR'):
            print(f"Target H-NMR Available: Yes")
        
        if result.success:
            print("\n🎉 Congratulations! The target similarity threshold was reached.")
            print("The lightweight elucidation system successfully completed!")
        else:
            print("\n⚠️  Elucidation incomplete. Consider:")
            print("   - Increasing max iterations")
            print("   - Adjusting temperature parameter")
            print("   - Lowering similarity threshold")
            print("   - Providing additional molecular context")
    else:
        print("\n❌ Elucidation failed - no results to summarize")
    
    print("\n🔬 Key Features Demonstrated:")
    print("   - Core elucidation logic without heavy dependencies")
    print("   - Database-based similarity calculation")
    print("   - LLM-driven structure generation and refinement")
    print("   - Iterative improvement with history feedback")
    print("   - Comprehensive result tracking and visualization")
    
    print("\n💡 To enable advanced features, install optional dependencies:")
    print("   - NMRShiftDB automation used for NMR predictions")
    print("   - rdkit-pypi: For advanced molecular validation")
    print("   - Then run: python example_enhanced_elucidation.py")


if __name__ == "__main__":
    main()
