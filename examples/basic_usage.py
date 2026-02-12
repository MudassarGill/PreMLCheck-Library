"""
Basic usage example for PreMLCheck
"""
import pandas as pd
import numpy as np
from premlcheck import PreMLCheck

# Create a sample classification dataset
np.random.seed(42)
n_samples = 500

data = {
    'feature_1': np.random.randn(n_samples),
    'feature_2': np.random.randn(n_samples),
    'feature_3': np.random.randn(n_samples),
    'feature_4': np.random.randn(n_samples),
    'feature_5': np.random.randn(n_samples),
    'target': np.random.choice(['A', 'B', 'C'], size=n_samples, p=[0.5, 0.3, 0.2])
}

df = pd.DataFrame(data)

# Add some missing values
df.loc[df.sample(frac=0.1).index, 'feature_1'] = np.nan
df.loc[df.sample(frac=0.05).index, 'feature_2'] = np.nan

print("=" * 60)
print("PreMLCheck - Basic Usage Example")
print("=" * 60)

# Initialize analyzer
analyzer = PreMLCheck()

# Run analysis
print("\nRunning analysis...")
results = analyzer.analyze(df, target_column='target')

# Print summary
print("\n" + results.summary())

# Print detailed results
print("\n" + "=" * 60)
print("Detailed Results")
print("=" * 60)

print(f"\n📊 Task Type: {results.task_type}")
print(f"   Confidence: {results.task_confidence:.1%}")

print(f"\n💯 Dataset Health Score: {results.quality_score:.1f}/100")

print(f"\n⚠️ Overfitting Risk: {results.overfitting_risk}")
if results.overfitting_factors:
    print("   Risk Factors:")
    for factor in results.overfitting_factors:
        print(f"   - {factor['factor']}: {factor['description']}")

print(f"\n🎯 Top Model Recommendations:")
for i, rec in enumerate(results.model_recommendations[:3], 1):
    print(f"   {i}. {rec.name} (Score: {rec.score:.1f}/100)")
    print(f"      {rec.reason}")

print(f"\n📈 Performance Estimate:")
if results.performance_estimate:
    print(f"   {results.performance_estimate.get('description', 'N/A')}")

print(f"\n🔧 Preprocessing Suggestions:")
for sugg in results.preprocessing_suggestions[:3]:
    print(f"   - [{sugg.priority}] {sugg.action}")
    print(f"     {sugg.description}")

# Generate reports
print("\n" + "=" * 60)
print("Generating Reports...")
print("=" * 60)

analyzer.generate_report(results, 'analysis_report.md', format='markdown')
print("✓ Markdown report saved to: analysis_report.md")

analyzer.generate_report(results, 'analysis_report.html', format='html')
print("✓ HTML report saved to: analysis_report.html")

analyzer.generate_report(results, 'analysis_report.json', format='json')
print("✓ JSON report saved to: analysis_report.json")

print("\n" + "=" * 60)
print("Analysis Complete!")
print("=" * 60)
