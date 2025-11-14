# UCI Wholesale Customers Dataset - Clustering Analysis

A comprehensive clustering analysis comparing **K-means** and **DBSCAN** algorithms on the UCI Wholesale Customers Dataset. This project performs customer segmentation to identify distinct purchasing patterns and provide actionable business insights.

## Authors

- **Students**: Ainedembe Denis, Musinguzi Benson
- **Lecturer**: Harriet Sibitenda (PhD)

## Dataset Overview
**Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers)
**Description**: The dataset contains annual spending (in monetary units) of wholesale distributor clients across different product categories.

### Dataset Characteristics

- **Type**: Multivariate
- **Subject Area**: Business
- **Associated Tasks**: Classification, Clustering
- **Feature Type**: Integer
- **Instances**: 440
- **Features**: 7
- **Missing Values**: No

### Variables

| Variable Name | Role | Type | Description |
|--------------|------|------|-------------|
| Channel | Feature | Categorical | Customer channel (1=Horeca, 2=Retail) |
| Region | Target | Categorical | Customer region (1=Lisbon, 2=Oporto, 3=Other) |
| Fresh | Feature | Integer | Annual spending on fresh products (m.u.) |
| Milk | Feature | Integer | Annual spending on milk products (m.u.) |
| Grocery | Feature | Integer | Annual spending on grocery products (m.u.) |
| Frozen | Feature | Integer | Annual spending on frozen products (m.u.) |
| Detergents_Paper | Feature | Integer | Annual spending on detergents and paper products (m.u.) |
| Delicassen | Feature | Integer | Annual spending on delicatessen products (m.u.) |

### Descriptive Statistics

| Category | Min | Max | Mean | Std. Deviation |
|----------|-----|-----|------|----------------|
| Fresh | 3 | 112,151 | 12,000.30 | 12,647.33 |
| Milk | 55 | 73,498 | 5,796.27 | 7,380.38 |
| Grocery | 3 | 92,780 | 7,951.28 | 9,503.16 |
| Frozen | 25 | 60,869 | 3,071.93 | 4,854.67 |
| Detergents_Paper | 3 | 40,827 | 2,881.49 | 4,767.85 |
| Delicassen | 3 | 47,943 | 1,524.87 | 2,820.11 |

### Distribution

**Region Distribution:**
- Lisbon: 77 customers
- Oporto: 47 customers
- Other Region: 316 customers

**Channel Distribution:**
- Horeca (Hotel/Restaurant/Café): 298 customers
- Retail: 142 customers

## Project Structure

```
UCI-Wholesale-Customers-Dataset/
│
├── dataset/
│   └── Wholesale-customers-data.csv     # Raw dataset
│
├── clustering_analysis.ipynb            # Main analysis notebook
│
└── README.md                            # This file
```

## Launch Jupyter Notebook:
```bash
jupyter notebook
```
Navigate to `clustering_analysis.ipynb` in the browser

## Clearing outputs via commandline
```bash
jupyter nbconvert --clear-output --inplace clustering_analysis.ipynb
```

## Analysis Overview

The analysis is structured into six main parts:

### Part A: Data Loading & Preprocessing
- Load and inspect dataset
- Handle missing values (if present)
- Remove exact duplicates
- Scale numeric features using RobustScaler (robust to outliers)
- Encode categorical variables for interpretation

### Part B: First Exploratory Data Analysis
- Descriptive statistics and boxplots for spending categories
- Outlier detection and analysis
- Log transformation of highly skewed variables
- Correlation heatmap to identify co-purchasing patterns

### Part C: Feature Engineering & Aggregation
- **TotalSpend**: Sum of all spending categories
- **ProportionFresh**: Fresh spending as proportion of total
- **LogTotalSpend**: Log-transformed total spending
- **GroceryMilkRatio**: Ratio of Grocery to Milk spending
- **NonFreshProportion**: Proportion of non-fresh products

### Part D: Clustering Modelling & Parameter Selection

#### K-means Clustering
- Test k values from 2 to 8
- Evaluate using Silhouette Score and Davies-Bouldin Index
- Select optimal k based on evaluation metrics
- Visualize metrics (Silhouette, DB Index, Elbow Method)

#### DBSCAN Clustering
- Compute k-distance plot to select optimal eps parameter
- Test multiple eps and min_samples combinations
- Visualize parameter sensitivity using heatmaps
- Select optimal parameters based on cluster quality and noise percentage

### Part E: Second EDA & Statistical Inference
- Tabulate cluster centroids/medoids in original units
- Test significant differences in TotalSpend across clusters (ANOVA/Kruskal-Wallis)
- Test associations between clusters and Region/Channel (Chi-square tests)

### Part F: Presentation & Reflection
- Visualize clusters using PCA (2D projection)
- Discuss DBSCAN sensitivity to parameters
- Analyze K-means assumptions (spherical clusters)
- Provide domain-specific business recommendations

##  Getting Started

### Prerequisites

Install the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

Or using conda:

```bash
conda install pandas numpy matplotlib seaborn scikit-learn scipy
```

### Running the Analysis

1. **Clone or download this repository**

2. **Open the Jupyter notebook**:
   ```bash
   jupyter notebook clustering_analysis.ipynb
   ```

3. **Run all cells** sequentially to perform the complete analysis

4. **Review the results**:
   - Visualizations are displayed inline
   - Statistical test results are printed
   - Cluster characteristics are summarized

## Key Features

- **Comprehensive preprocessing**: Handles outliers, missing values, and feature scaling
- **Dual clustering approach**: Compares K-means and DBSCAN algorithms
- **Statistical validation**: Uses ANOVA, Kruskal-Wallis, and Chi-square tests
- **Visual analytics**: Multiple visualizations including PCA projections, heatmaps, and boxplots
- **Business insights**: Actionable recommendations based on cluster analysis

## Key Insights

The analysis reveals:

1. **Customer Segmentation**: Distinct customer groups based on spending patterns
2. **Product Preferences**: Different clusters show varying preferences for fresh vs. processed products
3. **Channel Associations**: Clusters may associate with specific channels (Horeca vs. Retail)
4. **Regional Patterns**: Potential regional differences in purchasing behavior
5. **Outlier Detection**: DBSCAN identifies unique customers that don't fit standard patterns

## Clustering Algorithms

### K-means
- **Advantages**: Fast, interpretable, works well with spherical clusters
- **Limitations**: Assumes spherical clusters, requires pre-specified number of clusters
- **Best for**: Well-separated, similarly-sized clusters

### DBSCAN
- **Advantages**: Can find irregularly shaped clusters, identifies outliers, no need to specify number of clusters
- **Limitations**: Sensitive to parameters (eps, min_samples), may struggle with varying densities
- **Best for**: Datasets with noise and irregular cluster shapes

## Business Applications

The clustering results can be used for:

1. **Customer Segmentation**: Identify distinct customer groups for targeted marketing
2. **Product Mix Optimization**: Adjust inventory based on cluster preferences
3. **Channel Strategy**: Develop channel-specific marketing campaigns
4. **Regional Targeting**: Customize offerings by region
5. **Outlier Management**: Investigate unique customers for special handling
6. **Promotional Strategies**: Design promotions tailored to each segment
7. **Inventory Management**: Optimize stock levels based on cluster demand patterns

## Notes

- The dataset uses monetary units (m.u.) for all spending values
- Channel and Region are categorical variables used for interpretation but not included in clustering distance calculations
- RobustScaler is used instead of StandardScaler due to the presence of outliers
- All statistical tests use α = 0.05 significance level

## References

- **Dataset**: [UCI Machine Learning Repository - Wholesale Customers](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers)
- **Donated**: March 30, 2014

## Contributing

Feel free to fork this repository and submit pull requests for improvements.

## License
This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
This allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given.

## Creators
Margarida Cardoso

## DOI
10.24432/C5030X


