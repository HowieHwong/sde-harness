"""
Visualization utilities for elucidation results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import json
from pathlib import Path

from .elucidation_engine import ElucidationResult


class ElucidationVisualizer:
    """Visualize and analyze elucidation results."""
    
    def __init__(self, style: str = 'default'):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        self.style = style
        plt.style.use(style)
        
        # Set default figure size and DPI
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['figure.dpi'] = 100
        
        # Set color palette
        self.colors = sns.color_palette("husl", 10)
    
    def plot_similarity_progression(self, 
                                  result: ElucidationResult,
                                  save_path: Optional[str] = None,
                                  show_plot: bool = True) -> plt.Figure:
        """
        Plot the progression of NMR similarity across iterations.
        
        Args:
            result: ElucidationResult to visualize
            save_path: Path to save the plot (optional)
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract data
        iterations = [step.iteration for step in result.steps]
        similarities = [step.nmr_similarity for step in result.steps]
        smiles_list = [step.generated_smiles for step in result.steps]
        
        # Plot 1: Similarity progression
        ax1.plot(iterations, similarities, 'o-', linewidth=2, markersize=8, 
                color=self.colors[0], label='NMR Similarity')
        
        # Add threshold line
        threshold = 0.8  # Default threshold, could be made configurable
        ax1.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
                   label=f'Threshold ({threshold})')
        
        # Highlight best result
        best_idx = np.argmax(similarities)
        ax1.plot(iterations[best_idx], similarities[best_idx], 'o', 
                markersize=12, color='green', label='Best Result')
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('NMR Similarity Score')
        ax1.set_title(f'Similarity Progression - {result.target_molecule_id}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # Plot 2: Bar chart of similarities
        bars = ax2.bar(iterations, similarities, color=self.colors[1], alpha=0.7)
        
        # Color bars based on similarity
        for i, (bar, sim) in enumerate(zip(bars, similarities)):
            if sim >= threshold:
                bar.set_color('green')
            elif sim > 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('NMR Similarity Score')
        ax2.set_title('Similarity by Iteration')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for i, (bar, sim) in enumerate(zip(bars, similarities)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{sim:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            # Ensure directory exists
            try:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_smiles_evolution(self, 
                             result: ElucidationResult,
                             save_path: Optional[str] = None,
                             show_plot: bool = True) -> plt.Figure:
        """
        Plot the evolution of SMILES structures across iterations.
        
        Args:
            result: ElucidationResult to visualize
            save_path: Path to save the plot (optional)
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Extract data
        iterations = [step.iteration for step in result.steps]
        similarities = [step.nmr_similarity for step in result.steps]
        smiles_list = [step.generated_smiles for step in result.steps]
        
        # Create a scatter plot with size based on similarity
        scatter = ax.scatter(iterations, similarities, 
                           s=[max(100, sim * 500) for sim in similarities],
                           c=similarities, cmap='viridis', alpha=0.7)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('NMR Similarity')
        
        # Add SMILES labels
        for i, (iter_num, sim, smiles) in enumerate(zip(iterations, similarities, smiles_list)):
            if smiles:
                # Truncate long SMILES for display
                display_smiles = smiles[:30] + "..." if len(smiles) > 30 else smiles
                ax.annotate(display_smiles, (iter_num, sim), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('NMR Similarity Score')
        ax.set_title(f'SMILES Evolution - {result.target_molecule_id}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save_path:
            # Ensure directory exists
            try:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_batch_results(self, 
                          results: List[ElucidationResult],
                          save_path: Optional[str] = None,
                          show_plot: bool = True) -> plt.Figure:
        """
        Plot batch elucidation results.
        
        Args:
            results: List of ElucidationResult objects
            save_path: Path to save the plot (optional)
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract data
        molecule_ids = [r.target_molecule_id for r in results]
        final_similarities = [r.final_similarity for r in results]
        total_iterations = [r.total_iterations for r in results]
        execution_times = [r.execution_time for r in results]
        success_rates = [r.success for r in results]
        
        # Plot 1: Final similarities
        bars1 = ax1.bar(range(len(molecule_ids)), final_similarities, 
                        color=[self.colors[0] if sim >= 0.8 else self.colors[2] for sim in final_similarities])
        ax1.set_xlabel('Molecule ID')
        ax1.set_ylabel('Final NMR Similarity')
        ax1.set_title('Final Similarity Scores')
        ax1.set_xticks(range(len(molecule_ids)))
        ax1.set_xticklabels([mid[:10] + "..." if len(mid) > 10 else mid for mid in molecule_ids], 
                           rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # Plot 2: Iterations needed
        ax2.bar(range(len(molecule_ids)), total_iterations, color=self.colors[1])
        ax2.set_xlabel('Molecule ID')
        ax2.set_ylabel('Total Iterations')
        ax2.set_title('Iterations Required')
        ax2.set_xticks(range(len(molecule_ids)))
        ax2.set_xticklabels([mid[:10] + "..." if len(mid) > 10 else mid for mid in molecule_ids], 
                           rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Execution times
        ax3.bar(range(len(molecule_ids)), execution_times, color=self.colors[3])
        ax3.set_xlabel('Molecule ID')
        ax3.set_ylabel('Execution Time (seconds)')
        ax3.set_title('Execution Times')
        ax3.set_xticks(range(len(molecule_ids)))
        ax3.set_xticklabels([mid[:10] + "..." if len(mid) > 10 else mid for mid in molecule_ids], 
                           rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Success rate
        success_count = sum(success_rates)
        total_count = len(success_rates)
        ax4.pie([success_count, total_count - success_count], 
                labels=[f'Success ({success_count})', f'Failed ({total_count - success_count})'],
                colors=[self.colors[0], self.colors[2]], autopct='%1.1f%%')
        ax4.set_title('Overall Success Rate')
        
        plt.tight_layout()
        
        if save_path:
            # Ensure directory exists
            try:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        
        return fig
    
    def create_summary_report(self, 
                             result: ElucidationResult,
                             save_path: Optional[str] = None) -> str:
        """
        Create a text summary report of the elucidation process.
        
        Args:
            result: ElucidationResult to summarize
            save_path: Path to save the report (optional)
            
        Returns:
            Formatted summary report string
        """
        report = f"""
ELUCIDATION SUMMARY REPORT
==========================

Target Molecule: {result.target_molecule_id}
Status: {'SUCCESS' if result.success else 'FAILED'}
Final NMR Similarity: {result.final_similarity:.3f}
Total Iterations: {result.total_iterations}
Execution Time: {result.execution_time:.2f} seconds

ITERATION DETAILS:
"""
        
        for i, step in enumerate(result.steps):
            report += f"""
Iteration {step.iteration}:
- Generated SMILES: {step.generated_smiles or 'None'}
- NMR Similarity: {step.nmr_similarity:.3f}
- Response Length: {len(step.response)} characters
- Timestamp: {pd.Timestamp(step.timestamp, unit='s')}
"""
        
        report += f"""

ANALYSIS:
- Best Similarity: {max([step.nmr_similarity for step in result.steps]):.3f}
- Average Similarity: {np.mean([step.nmr_similarity for step in result.steps]):.3f}
- Improvement Rate: {np.mean([result.steps[i].nmr_similarity - result.steps[i-1].nmr_similarity for i in range(1, len(result.steps))]):.3f}
- Final Structure: {result.final_smiles or 'None'}

RECOMMENDATIONS:
"""
        
        if result.success:
            report += "- Elucidation successful! Target similarity threshold reached.\n"
        else:
            report += "- Elucidation incomplete. Consider:\n"
            report += "  * Increasing max iterations\n"
            report += "  * Adjusting temperature parameter\n"
            report += "  * Providing additional molecular context\n"
        
        if save_path:
            # Ensure directory exists
            try:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
    
    def plot_learning_curves(self, 
                            results: List[ElucidationResult],
                            save_path: Optional[str] = None,
                            show_plot: bool = True) -> plt.Figure:
        """
        Plot learning curves showing how different molecules improve over iterations.
        
        Args:
            results: List of ElucidationResult objects
            save_path: Path to save the plot (optional)
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot learning curve for each molecule
        for i, result in enumerate(results):
            if result.steps:  # Only plot if there are steps
                iterations = [step.iteration for step in result.steps]
                similarities = [step.nmr_similarity for step in result.steps]
                
                # Use different colors and line styles
                color = self.colors[i % len(self.colors)]
                linestyle = ['-', '--', '-.', ':'][i % 4]
                
                ax.plot(iterations, similarities, 
                       color=color, linestyle=linestyle, linewidth=2,
                       marker='o', markersize=6,
                       label=f"{result.target_molecule_id[:10]}...")
        
        # Add threshold line
        threshold = 0.8
        ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
                   label=f'Success Threshold ({threshold})')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('NMR Similarity Score')
        ax.set_title('Learning Curves - Multiple Molecules')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save_path:
            # Ensure directory exists
            try:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        
        return fig
