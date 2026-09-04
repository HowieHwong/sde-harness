#!/usr/bin/env python3
"""
Example script for batch molecule elucidation using the Spectrum Elucidator Toolkit.

This script demonstrates how to use the toolkit to elucidate multiple molecules
from their NMR spectra through iterative LLM prompting.
"""

import os
import sys
import time
from pathlib import Path
from typing import List

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import ConfigManager
from src.data_utils import MolecularDataLoader
from src.llm_interface import LLMInterface
from src.elucidation_engine import ElucidationEngine, ElucidationConfig
from src.visualization import ElucidationVisualizer


def select_molecules_for_batch(data_loader: MolecularDataLoader, 
                             max_molecules: int = 5) -> List[str]:
    """
    Select molecules for batch elucidation.
    
    Args:
        data_loader: Molecular data loader instance
        max_molecules: Maximum number of molecules to select
        
    Returns:
        List of molecule IDs selected for elucidation
    """
    print(f"\nSelecting up to {max_molecules} molecules for batch elucidation...")
    
    # Get dataset info
    dataset_info = data_loader.get_dataset_info()
    total_molecules = dataset_info['total_molecules']
    
    if total_molecules == 0:
        print("ERROR: No molecules available in dataset!")
        return []
    
    # Select molecules - you can customize this selection strategy
    selected_ids = []
    
    # Strategy 1: Select first N molecules
    available_ids = data_loader.data['molecule_id'].head(max_molecules).tolist()
    selected_ids.extend(available_ids)
    
    # Strategy 2: Select molecules with different complexity (optional)
    # You could add logic here to select diverse molecules based on:
    # - Number of functional groups
    # - Degree of unsaturation
    # - Molecular weight
    # - etc.
    
    print(f"✓ Selected {len(selected_ids)} molecules:")
    for i, molecule_id in enumerate(selected_ids, 1):
        molecule = data_loader.get_molecule_by_id(molecule_id)
        if molecule:
            smiles = molecule.get('SMILES', 'Unknown')
            functional_groups = [k for k, v in molecule.items() 
                               if v and k not in ['SMILES', 'H_NMR', 'C_NMR', 'molecule_id']]
            print(f"   {i}. {molecule_id}: {smiles[:30]}... (Groups: {len(functional_groups)})")
    
    return selected_ids


def main():
    """Main function for batch molecule elucidation."""
    
    print("=" * 60)
    print("SPECTRUM ELUCIDATOR TOOLKIT - BATCH ELUCIDATION EXAMPLE")
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
    
    # Select molecules for batch processing
    print("\n6. Selecting molecules for batch elucidation...")
    max_molecules = 3  # Limit for demonstration - adjust as needed
    selected_molecule_ids = select_molecules_for_batch(data_loader, max_molecules)
    
    if not selected_molecule_ids:
        print("ERROR: No molecules selected for elucidation!")
        return
    
    # Perform batch elucidation
    print(f"\n7. Starting batch elucidation for {len(selected_molecule_ids)} molecules...")
    print(f"   Max iterations per molecule: {elucidation_config.max_iterations}")
    print(f"   Similarity threshold: {elucidation_config.similarity_threshold}")
    print(f"   Temperature: {elucidation_config.temperature}")
    
    start_time = time.time()
    
    try:
        results = engine.batch_elucidation(selected_molecule_ids)
        
        total_time = time.time() - start_time
        
        print(f"\n8. Batch elucidation completed in {total_time:.2f} seconds!")
        
        # Analyze results
        print("\n9. Analyzing batch results...")
        
        successful_elucidations = [r for r in results if r.success]
        failed_elucidations = [r for r in results if not r.success]
        
        print(f"   Successful elucidations: {len(successful_elucidations)}")
        print(f"   Failed elucidations: {len(failed_elucidations)}")
        print(f"   Success rate: {len(successful_elucidations)/len(results)*100:.1f}%")
        
        if successful_elucidations:
            avg_similarity = sum(r.final_similarity for r in successful_elucidations) / len(successful_elucidations)
            avg_iterations = sum(r.total_iterations for r in successful_elucidations) / len(successful_elucidations)
            print(f"   Average final similarity: {avg_similarity:.3f}")
            print(f"   Average iterations needed: {avg_iterations:.1f}")
        
        # Create batch visualizations
        print("\n10. Creating batch visualizations...")
        
        # Batch results overview
        fig1 = visualizer.plot_batch_results(
            results,
            save_path="results/batch_results_overview.png",
            show_plot=viz_config.show_plots
        )
        
        # Learning curves comparison
        fig2 = visualizer.plot_learning_curves(
            results,
            save_path="results/batch_learning_curves.png",
            show_plot=viz_config.show_plots
        )
        
        print("✓ Batch visualizations created")
        
        # Generate individual reports for each molecule
        print("\n11. Generating individual reports...")
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        for i, result in enumerate(results):
            if result.steps:  # Only generate report if there are steps
                # Individual similarity progression
                fig = visualizer.plot_similarity_progression(
                    result,
                    save_path=f"results/{result.target_molecule_id}_progression.png",
                    show_plot=False  # Don't show individual plots in batch mode
                )
                plt.close(fig)  # Close to free memory
                
                # Individual summary report
                report = visualizer.create_summary_report(
                    result,
                    save_path=f"results/{result.target_molecule_id}_summary.txt"
                )
        
        print("✓ Individual reports generated")
        
        # Save batch results summary
        print("\n12. Saving batch results summary...")
        
        batch_summary = {
            'timestamp': time.time(),
            'total_molecules': len(results),
            'successful_elucidations': len(successful_elucidations),
            'failed_elucidations': len(failed_elucidations),
            'success_rate': len(successful_elucidations) / len(results),
            'total_execution_time': total_time,
            'average_execution_time': total_time / len(results),
            'molecule_results': []
        }
        
        for result in results:
            molecule_summary = {
                'molecule_id': result.target_molecule_id,
                'success': result.success,
                'final_similarity': result.final_similarity,
                'total_iterations': result.total_iterations,
                'execution_time': result.execution_time,
                'final_smiles': result.final_smiles
            }
            batch_summary['molecule_results'].append(molecule_summary)
        
        # Save batch summary
        import json
        batch_summary_file = results_dir / "batch_summary.json"
        with open(batch_summary_file, 'w') as f:
            json.dump(batch_summary, f, indent=2, default=str)
        
        print(f"✓ Batch summary saved to: {batch_summary_file}")
        
        print("\n" + "=" * 60)
        print("BATCH ELUCIDATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: Batch elucidation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print final batch summary
    print("\nBATCH ELUCIDATION SUMMARY:")
    print(f"Total Molecules Processed: {len(results)}")
    print(f"Successful Elucidations: {len(successful_elucidations)}")
    print(f"Failed Elucidations: {len(failed_elucidations)}")
    print(f"Overall Success Rate: {len(successful_elucidations)/len(results)*100:.1f}%")
    print(f"Total Execution Time: {total_time:.2f} seconds")
    print(f"Average Time per Molecule: {total_time/len(results):.2f} seconds")
    
    if successful_elucidations:
        print(f"Average Final Similarity: {avg_similarity:.3f}")
        print(f"Average Iterations Needed: {avg_iterations:.1f}")
    
    print("\nResults saved to 'results/' directory:")
    print("  - batch_results_overview.png")
    print("  - batch_learning_curves.png")
    print("  - batch_summary.json")
    print("  - Individual molecule reports and plots")
    
    if len(successful_elucidations) == len(results):
        print("\n🎉 All molecules elucidated successfully!")
    elif len(successful_elucidations) > 0:
        print(f"\n✅ {len(successful_elucidations)} out of {len(results)} molecules elucidated successfully.")
    else:
        print("\n⚠️  No molecules were elucidated successfully. Consider:")
        print("   - Increasing max iterations")
        print("   - Adjusting temperature parameter")
        print("   - Checking API key and rate limits")


if __name__ == "__main__":
    main()
