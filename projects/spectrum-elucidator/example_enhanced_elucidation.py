#!/usr/bin/env python3
"""
Enhanced example script for molecular structure elucidation using the Spectrum Elucidator Toolkit.

This script demonstrates the advanced features including:
- NMR prediction using web scraping and LLM fallback
- Enhanced similarity calculation using C-NMR and H-NMR
- Configurable NMR tolerance and preferences
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import ConfigManager
from src.data_utils import MolecularDataLoader
from src.llm_interface import LLMInterface
from src.elucidation_engine import ElucidationEngine, ElucidationConfig
from src.visualization import ElucidationVisualizer
from src.nmr_predictor import NMRPredictor


def setup_nmr_predictor(config_manager: ConfigManager, llm_interface: LLMInterface) -> NMRPredictor:
    """
    Setup NMR predictor with configuration.
    
    Args:
        config_manager: Configuration manager instance
        llm_interface: LLM interface instance
        
    Returns:
        Configured NMR predictor instance
    """
    nmr_config = config_manager.get_nmr_predictor_config()
    
    # Get API key for LLM fallback
    api_key = config_manager.get_llm_config().api_key
    
    predictor = NMRPredictor(
        openai_api_key=api_key,
        headless=nmr_config.headless_browser,
        timeout=nmr_config.web_timeout
    )
    
    print(f"✓ NMR Predictor initialized:")
    print(f"  - Web scraping: {nmr_config.use_web_scraping}")
    print(f"  - LLM fallback: {nmr_config.use_llm_fallback}")
    print(f"  - Headless browser: {nmr_config.headless_browser}")
    print(f"  - Web timeout: {nmr_config.web_timeout}s")
    
    return predictor


def test_nmr_prediction(nmr_predictor: NMRPredictor, test_smiles: str):
    """
    Test NMR prediction functionality.
    
    Args:
        nmr_predictor: NMR predictor instance
        test_smiles: Test SMILES string
    """
    print(f"\n🧪 Testing NMR prediction for: {test_smiles}")
    
    # Get molecular formula
    formula = nmr_predictor.get_molecular_formula(test_smiles)
    print(f"  Molecular formula: {formula}")
    
    # Test web scraping
    print("  Testing web scraping...")
    start_time = time.time()
    c_nmr_records, h_nmr_records = nmr_predictor.get_nmr_from_web(test_smiles)
    web_time = time.time() - start_time
    
    print(f"    Web scraping completed in {web_time:.2f}s")
    print(f"    C-NMR records: {len(c_nmr_records)} peaks")
    print(f"    H-NMR records: {len(h_nmr_records)} peaks")
    
    if c_nmr_records:
        formatted_c = nmr_predictor.format_nmr_for_comparison(c_nmr_records, "C")
        print(f"    Formatted C-NMR: {formatted_c[:100]}...")
    
    if h_nmr_records:
        formatted_h = nmr_predictor.format_nmr_for_comparison(h_nmr_records, "H")
        print(f"    Formatted H-NMR: {formatted_h[:100]}...")
    
    return c_nmr_records, h_nmr_records


def main():
    """Main function for enhanced elucidation demonstration."""
    
    print("=" * 70)
    print("SPECTRUM ELUCIDATOR TOOLKIT - ENHANCED NMR PREDICTION EXAMPLE")
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
    
    # Initialize LLM interface
    print("\n3. Initializing LLM interface...")
    llm_config = config_manager.get_llm_config()
    llm_interface = LLMInterface(
        api_key=llm_config.api_key,
        model=llm_config.model,
        max_tokens=llm_config.max_tokens
    )
    print(f"✓ LLM interface initialized with model: {llm_config.model}")
    
    # Initialize NMR predictor
    print("\n4. Initializing NMR predictor...")
    nmr_predictor = setup_nmr_predictor(config_manager, llm_interface)
    
    # Test NMR prediction functionality
    print("\n5. Testing NMR prediction functionality...")
    test_smiles = "CCCCC1=CC=CC=C1"  # Pentylbenzene
    c_nmr_records, h_nmr_records = test_nmr_prediction(nmr_predictor, test_smiles)
    
    # Initialize elucidation engine with NMR predictor
    print("\n6. Initializing elucidation engine...")
    elucidation_config = config_manager.get_elucidation_config()
    engine = ElucidationEngine(
        data_loader=data_loader,
        llm_interface=llm_interface,
        config=elucidation_config,
        nmr_predictor=nmr_predictor
    )
    print("✓ Elucidation engine initialized with NMR predictor")
    
    # Initialize visualizer
    print("\n7. Initializing visualizer...")
    viz_config = config_manager.get_visualization_config()
    visualizer = ElucidationVisualizer(style=viz_config.style)
    print("✓ Visualizer initialized")
    
    # Select target molecule
    print("\n8. Selecting target molecule...")
    
    # Look for a molecule with both H-NMR and C-NMR data
    target_molecule_id = None
    for molecule_id in dataset_info['sample_molecules']:
        molecule = data_loader.get_molecule_by_id(molecule_id)
        if molecule and molecule.get('H_NMR') and molecule.get('C_NMR'):
            target_molecule_id = molecule_id
            break
    
    if not target_molecule_id:
        print("No molecule with both H-NMR and C-NMR data found, using first available...")
        target_molecule_id = dataset_info['sample_molecules'][0]
    
    # Check if molecule exists
    target_molecule = data_loader.get_molecule_by_id(target_molecule_id)
    if not target_molecule:
        print(f"ERROR: Molecule {target_molecule_id} not found!")
        print("Using random molecule instead...")
        target_molecule = data_loader.get_random_molecule()
        target_molecule_id = target_molecule['molecule_id']
    
    print(f"✓ Target molecule: {target_molecule_id}")
    print(f"  SMILES: {target_molecule['SMILES']}")
    if target_molecule.get('H_NMR'):
        print(f"  H-NMR: {target_molecule['H_NMR'][:100]}...")
    if target_molecule.get('C_NMR'):
        print(f"  C-NMR: {target_molecule['C_NMR'][:100]}...")
    
    # Perform elucidation
    print("\n9. Starting enhanced molecular structure elucidation...")
    print(f"   Max iterations: {elucidation_config.max_iterations}")
    print(f"   Similarity threshold: {elucidation_config.similarity_threshold}")
    print(f"   Temperature: {elucidation_config.temperature}")
    print(f"   NMR predictor enabled: {elucidation_config.use_nmr_predictor}")
    print(f"   NMR tolerance: ±{elucidation_config.nmr_tolerance} ppm")
    print(f"   Prefer C-NMR: {elucidation_config.prefer_c_nmr}")
    
    try:
        result = engine.elucidate_molecule(target_molecule_id)
        
        print("\n10. Elucidation completed!")
        print(f"    Success: {result.success}")
        print(f"    Final similarity: {result.final_similarity:.3f}")
        print(f"    Total iterations: {result.total_iterations}")
        print(f"    Execution time: {result.execution_time:.2f} seconds")
        print(f"    NMR predictor used: {result.metadata.get('nmr_prediction_used', False)}")
        
        if result.final_smiles:
            print(f"    Final SMILES: {result.final_smiles}")
        
        # Generate summary report
        print("\n11. Generating enhanced summary report...")
        summary = engine.get_elucidation_summary(result)
        print(f"    Best similarity: {summary['best_similarity']:.3f}")
        print(f"    Improvement rate: {summary['improvement_rate']:.3f}")
        print(f"    NMR prediction used: {summary['nmr_prediction_used']}")
        
        # Show prediction methods used
        prediction_methods = set(step['prediction_method'] for step in summary['similarity_progression'])
        print(f"    Prediction methods used: {', '.join(prediction_methods)}")
        
        # Create enhanced visualizations
        print("\n12. Creating enhanced visualizations...")
        
        # Similarity progression plot
        fig1 = visualizer.plot_similarity_progression(
            result, 
            save_path=f"results/{target_molecule_id}_enhanced_progression.png",
            show_plot=viz_config.show_plots
        )
        
        # SMILES evolution plot
        fig2 = visualizer.plot_smiles_evolution(
            result,
            save_path=f"results/{target_molecule_id}_enhanced_evolution.png",
            show_plot=viz_config.show_plots
        )
        
        # Enhanced summary report
        report = visualizer.create_summary_report(
            result,
            save_path=f"results/{target_molecule_id}_enhanced_summary.txt"
        )
        
        print("✓ Enhanced visualizations and report created")
        
        # Save detailed results
        print("\n13. Saving detailed results...")
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Save result object as JSON
        import json
        from dataclasses import asdict
        
        result_file = results_dir / f"{target_molecule_id}_enhanced_results.json"
        with open(result_file, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        
        print(f"✓ Enhanced results saved to: {result_file}")
        
        print("\n" + "=" * 70)
        print("ENHANCED ELUCIDATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nERROR: Enhanced elucidation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print final enhanced summary
    print("\nENHANCED ELUCIDATION SUMMARY:")
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
        print("The enhanced NMR prediction system successfully guided the elucidation!")
    else:
        print("\n⚠️  Elucidation incomplete. Consider:")
        print("   - Increasing max iterations")
        print("   - Adjusting temperature parameter")
        print("   - Adjusting NMR tolerance")
        print("   - Providing additional molecular context")
    
    print("\n🔬 Key Features Demonstrated:")
    print("   - Web scraping from NMRDB for NMR prediction")
    print("   - LLM fallback when web scraping fails")
    print("   - Enhanced similarity calculation using both H-NMR and C-NMR")
    print("   - Configurable NMR tolerance and preferences")
    print("   - Comprehensive tracking of prediction methods used")


if __name__ == "__main__":
    main()
