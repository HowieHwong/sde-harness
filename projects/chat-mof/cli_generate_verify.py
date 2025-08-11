"""
Generate specific MOF names and verify them against the database
python -u cli_generate_verify.py --model openai/o3-2025-04-16 --target-surface-area 3000
"""

import argparse
import sys
import re
from pathlib import Path
from typing import List, Dict, Any
import random

import weave

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import MOFGeneration
from src.mof_name_generator import MOFNameGeneratorPrompt
from src.oracle import MOFOracle


def extract_mof_names_from_response(response_text: str) -> List[str]:
    """
    Extract MOF names from LLM response, using the delimiter to skip thinking process.
    
    Args:
        response_text: Raw LLM response
        
    Returns:
        List of MOF names
    """
    # Look for the delimiter that marks the start of MOF names
    delimiter = "BELOW ARE GENERATED MOFS:"
    
    if delimiter in response_text:
        # Only parse text after the delimiter
        mof_section = response_text.split(delimiter, 1)[1]
    else:
        # Fallback to original parsing if delimiter not found
        mof_section = response_text
    
    lines = mof_section.strip().split('\n')
    mof_names = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Remove numbering, bullets, dashes
        line = re.sub(r'^\d+\.\s*', '', line)
        line = re.sub(r'^[-•*]\s*', '', line)
        
        # Remove any trailing explanations or descriptions
        # Keep only the MOF name part
        if ':' in line:
            line = line.split(':')[0].strip()
        
        # Skip obvious non-MOF lines
        skip_words = ['explanation', 'based on', 'analysis', 'note:', 'here are', 'mof names', 'following']
        if any(skip_word in line.lower() for skip_word in skip_words):
            continue
        
        # Skip markdown code blocks and other formatting
        if line.startswith('```') or line == '```':
            continue
        
        # Basic validation - reasonable MOF name length
        if line and 3 <= len(line) <= 50:
            mof_names.append(line)
    
    return mof_names


def main():
    """Main function for MOF name generation and verification."""
    parser = argparse.ArgumentParser(description="Generate specific MOF names and verify them in database")
    
    # Core parameters
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--target-surface-area", type=float, default=2000.0, 
                       help="Target surface area threshold (m²/g)")
    
    # Generation parameters
    parser.add_argument("--population-size", type=int, default=10,
                       help="Number of MOF names to generate per iteration")
    parser.add_argument("--n-generations", type=int, default=5,
                       help="Number of generations/iterations")
    
    # Learning parameters
    parser.add_argument("--use-iterative-learning", action="store_true", default=True,
                       help="Use iterative learning from successful/failed attempts")
    parser.add_argument("--database-examples", type=int, default=20,
                       help="Number of real MOF examples to show in prompt")
    
    # Display options
    parser.add_argument("--show-all-attempts", action="store_true",
                       help="Show all generated names, not just successful ones")
    
    # Logging
    parser.add_argument("--project-name", default="ChatMOF-Generate-Verify",
                       help="Weave project name for logging")
    
    args = parser.parse_args()
    
    # Initialize weave logging
    weave.init(args.project_name)
    
    print("🔮 ChatMOF: Generate Specific MOF Names & Verify")
    print(f"Model: {args.model}")
    print(f"Target surface area: {args.target_surface_area} m²/g")
    print(f"Generating {args.population_size} MOF names per iteration")
    print(f"Running {args.n_generations} generations")
    
    try:
        # Initialize components
        print("\n📝 Setting up components...")
        
        generator = MOFGeneration(model_name=args.model)
        prompt = MOFNameGeneratorPrompt()
        oracle = MOFOracle()
        
        print("✓ All components initialized")
        
        # Get database examples for prompting
        print(f"\n📚 Loading database examples...")
        high_sa_examples = oracle.get_high_surface_area_mofs(
            threshold=args.target_surface_area * 0.7,  # Slightly lower for examples
            top_n=args.database_examples
        )
        
        # Sample some random MOFs for naming pattern diversity
        random_sample = oracle.df.sample(min(10, len(oracle.df))).to_dict('records')
        all_examples = high_sa_examples + random_sample
        
        print(f"✓ Loaded {len(all_examples)} database examples")
        
        # Track results
        all_tested_mofs = []
        successful_mofs = []
        generation_stats = []
        
        # Run generations
        for generation in range(args.n_generations):
            print(f"\n--- Generation {generation + 1}/{args.n_generations} ---")
            
            # Build prompt
            if generation == 0 or not args.use_iterative_learning:
                # Initial generation - use database examples
                examples_text = prompt.format_real_mof_examples(all_examples)
                
                prompt_text = prompt.build_mof_name_prompt(
                    target_surface_area=args.target_surface_area,
                    num_samples=args.population_size,
                    real_mof_examples=examples_text
                )
            else:
                # Iterative generation - learn from previous attempts
                # Show ALL successful MOFs (no limit)
                successful_text = prompt.format_successful_mofs(successful_mofs, max_show=None)
                
                # Get failed MOFs (not successful) and show ALL (no limit)
                failed_mofs = [mof for mof in all_tested_mofs if not mof.get('above_threshold', False)]
                failed_text = prompt.format_failed_mofs(failed_mofs, max_show=None)
                
                examples_text = prompt.format_real_mof_examples(all_examples, max_examples=10)
                
                prompt_text = prompt.build_iterative_mof_names_prompt(
                    current_iteration=generation + 1,
                    max_iterations=args.n_generations,
                    target_surface_area=args.target_surface_area,
                    num_samples=args.population_size,
                    successful_mofs=successful_text,
                    failed_mofs=failed_text,
                    database_examples=examples_text
                )
            
            # Generate MOF names
            print(f"🧠 Generating {args.population_size} MOF names...")
            response = generator.generate_mof_names(
                prompt=prompt_text,
                model_name=args.model,
                temperature=0.8,  # Higher temperature for more creativity
                max_tokens=10000
            )
            
            # Extract MOF names
            candidate_names = extract_mof_names_from_response(response['text'])
            print(f"✓ Extracted {len(candidate_names)} MOF names")
            
            if args.show_all_attempts:
                print("Generated names:", candidate_names[:8], "..." if len(candidate_names) > 8 else "")
            
            # Verify against database
            print(f"🔍 Verifying names against database...")
            evaluation_results = oracle.evaluate_mof_candidates(
                candidate_names,
                threshold=args.target_surface_area
            )
            
            # Update tracking
            all_tested_mofs.extend(evaluation_results)
            
            # Find successful MOFs
            new_successful = [mof for mof in evaluation_results if mof['above_threshold']]
            found_any = [mof for mof in evaluation_results if mof['found']]
            
            successful_mofs.extend(new_successful)
            
            # Calculate statistics
            found_rate = len(found_any) / len(evaluation_results) if evaluation_results else 0
            success_rate = len(new_successful) / len(evaluation_results) if evaluation_results else 0
            
            surface_areas = [mof.get('surface_area') for mof in evaluation_results if mof.get('surface_area') is not None]
            best_sa_this_gen = max(surface_areas) if surface_areas else 0
            
            stats = {
                'generation': generation + 1,
                'names_generated': len(candidate_names),
                'names_found_in_db': len(found_any),
                'names_above_threshold': len(new_successful),
                'found_rate': found_rate,
                'success_rate': success_rate,
                'best_surface_area': best_sa_this_gen
            }
            generation_stats.append(stats)
            
            # Print results
            print(f"📊 Results: {len(found_any)}/{len(candidate_names)} found in DB ({found_rate:.1%})")
            print(f"🎯 Success: {len(new_successful)} above {args.target_surface_area} m²/g threshold ({success_rate:.1%})")
            
            if new_successful:
                print("🏆 NEW SUCCESSFUL MOFs:")
                for mof in new_successful:
                    name = mof['mof_name']
                    sa = mof['surface_area']
                    print(f"   ✓ {name}: {sa:.0f} m²/g")
            
            if found_any and not new_successful:
                print("📋 Found in DB but below threshold:")
                for mof in found_any[:3]:  # Show first 3
                    if not mof['above_threshold']:
                        name = mof['mof_name']
                        sa = mof.get('surface_area', 'N/A')
                        print(f"   • {name}: {sa} m²/g")
            
            # Early stopping if we're finding good MOFs consistently
            if generation >= 2:
                recent_success_rates = [s['success_rate'] for s in generation_stats[-3:]]
                avg_recent_success = sum(recent_success_rates) / len(recent_success_rates)
                
                if avg_recent_success > 0.2:  # 20% success rate
                    print(f"🎯 High success rate achieved ({avg_recent_success:.1%}), could stop here!")
        
        # Final summary
        total_generated = sum(s['names_generated'] for s in generation_stats)
        total_found = len([m for m in all_tested_mofs if m['found']])
        total_successful = len(successful_mofs)
        
        overall_found_rate = total_found / total_generated if total_generated > 0 else 0
        overall_success_rate = total_successful / total_generated if total_generated > 0 else 0
        
        print(f"\n🏁 FINAL RESULTS")
        print("=" * 50)
        print(f"📊 Generated {total_generated} MOF names over {args.n_generations} generations")
        print(f"🔍 Found {total_found} names in database ({overall_found_rate:.1%} found rate)")
        print(f"🎯 Discovered {total_successful} MOFs above {args.target_surface_area} m²/g ({overall_success_rate:.1%} success rate)")
        
        if successful_mofs:
            print(f"\n🏆 ALL SUCCESSFUL MOFs DISCOVERED:")
            # Sort by surface area
            successful_mofs.sort(key=lambda x: x.get('surface_area', 0), reverse=True)
            
            for i, mof in enumerate(successful_mofs, 1):
                name = mof['mof_name']
                sa = mof['surface_area']
                properties = mof.get('properties', {})
                metal = properties.get('Metal type', 'Unknown') if properties else 'Unknown'
                void_frac = properties.get('void fraction', 'N/A') if properties else 'N/A'
                
                print(f"{i:2d}. {name}")
                print(f"     Surface Area: {sa:.0f} m²/g")
                print(f"     Metal: {metal}, Void Fraction: {void_frac}")
                
                if i >= 10:  # Limit display
                    remaining = len(successful_mofs) - 10
                    if remaining > 0:
                        print(f"     ... and {remaining} more")
                    break
        
        # Save results
        import json
        model_safe = args.model.replace('/', '_')
        results_file = f"mof_generation_verification_{model_safe}_{args.target_surface_area}sa.json"
        
        results_data = {
            'parameters': {
                'model': args.model,
                'target_surface_area': args.target_surface_area,
                'population_size': args.population_size,
                'n_generations': args.n_generations,
                'use_iterative_learning': args.use_iterative_learning
            },
            'summary': {
                'total_generated': total_generated,
                'total_found': total_found,
                'total_successful': total_successful,
                'overall_found_rate': overall_found_rate,
                'overall_success_rate': overall_success_rate
            },
            'successful_mofs': successful_mofs,
            'generation_stats': generation_stats,
            'all_tested_mofs': all_tested_mofs[-50:]  # Keep last 50 for file size
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n💾 Results saved to {results_file}")
        
        if successful_mofs:
            best_mof = successful_mofs[0]
            print(f"\n🥇 BEST MOF DISCOVERED: {best_mof['mof_name']}")
            print(f"🔺 Surface Area: {best_mof['surface_area']:.0f} m²/g")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        generator.close()


if __name__ == "__main__":
    sys.exit(main())