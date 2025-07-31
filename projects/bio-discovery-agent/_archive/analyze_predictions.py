import json
import csv
import numpy as np

# Load the results
with open('gpt4/openai/gpt-4o-2024-08-06_IFNG/test/final_results.json', 'r') as f:
    results = json.load(f)

# Load ground truth data
ground_truth = {}
with open('datasets/ground_truth_IFNG.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        ground_truth[row['Gene']] = float(row['Score'])

# Get top scoring genes from ground truth (likely true positives)
sorted_genes = sorted(ground_truth.items(), key=lambda x: x[1], reverse=True)
top_200_genes = set([gene for gene, score in sorted_genes[:200]])
top_500_genes = set([gene for gene, score in sorted_genes[:500]])
top_1000_genes = set([gene for gene, score in sorted_genes[:1000]])

print("Analysis of Bio-Discovery Agent Results for IFNG Task")
print("=" * 60)
print()

# Analyze each round
for round_result in results['round_results']:
    round_num = round_result['round']
    predicted_genes = round_result['predicted_genes']
    
    print(f"Round {round_num} Analysis:")
    print(f"  Total predicted genes: {len(predicted_genes)}")
    
    # Find true positives in different thresholds
    tp_200 = [g for g in predicted_genes if g in top_200_genes]
    tp_500 = [g for g in predicted_genes if g in top_500_genes]
    tp_1000 = [g for g in predicted_genes if g in top_1000_genes]
    
    print(f"  True positives in top 200: {len(tp_200)} ({len(tp_200)/len(predicted_genes)*100:.1f}%)")
    print(f"  True positives in top 500: {len(tp_500)} ({len(tp_500)/len(predicted_genes)*100:.1f}%)")
    print(f"  True positives in top 1000: {len(tp_1000)} ({len(tp_1000)/len(predicted_genes)*100:.1f}%)")
    
    if tp_200:
        print(f"  Top 200 hits: {', '.join(tp_200[:10])}")
    
    # Check for specific genes
    if 'TNFRSF11A' in predicted_genes:
        score = ground_truth.get('TNFRSF11A', 0)
        rank = sorted_genes.index(('TNFRSF11A', score)) + 1
        print(f"  TNFRSF11A found - Score: {score}, Rank: {rank}")
    
    print()

# Overall statistics
print("Overall Performance:")
print(f"  Mean hit rate: {results['aggregate']['mean_hit_rate']:.4f}")
print(f"  Mean precision: {results['aggregate']['mean_precision']:.4f}")
print(f"  Mean recall: {results['aggregate']['mean_recall']:.4f}")
print(f"  Mean F1 score: {results['aggregate']['mean_f1_score']:.4f}")
print(f"  Total unique genes predicted: {results['aggregate']['total_unique_genes']}")

# Performance trend
hit_rates = results['aggregate']['hit_rates_by_round']
print(f"\nPerformance trend across rounds:")
for i, rate in enumerate(hit_rates, 1):
    print(f"  Round {i}: {rate:.4f}")

# Pattern analysis
print("\nPrediction Patterns:")
all_predicted_genes = []
for round_result in results['round_results']:
    all_predicted_genes.extend(round_result['predicted_genes'])

# Count gene family occurrences
gene_families = {}
for gene in all_predicted_genes:
    family = gene.split('_')[0] if '_' in gene else gene[:4]
    gene_families[family] = gene_families.get(family, 0) + 1

top_families = sorted(gene_families.items(), key=lambda x: x[1], reverse=True)[:10]
print("  Most frequently predicted gene families:")
for family, count in top_families:
    print(f"    {family}: {count} occurrences")