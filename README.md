# E-Commerce Data Engineering Pipeline

A PySpark-based data engineering project for processing and analyzing e-commerce datasets. The project focuses on data profiling, revenue analysis, delivery analysis, customer-product insights, Spark SQL operations, and data validation.

## Overview

E-commerce platforms generate data across customers, orders, products, payments, sellers, and reviews. Analyzing this data requires structured processing and reliable data quality checks.

This project organizes these datasets into a Python-based PySpark pipeline to support data analysis and validation.

## Key Features

- **Data Loading:** Load e-commerce CSV datasets using PySpark.
- **Data Profiling:** Explore datasets and understand their structure and quality.
- **Revenue Analysis:** Analyze order and item data for revenue-related insights.
- **Delivery Analysis:** Examine order delivery information.
- **Customer and Product Analysis:** Explore customer and product-related data.
- **Spark SQL:** Perform SQL-based analysis using Spark.
- **Data Validation:** Check required tables, missing values in key fields, duplicate identifiers, invalid numeric values, and selected data relationships.
- **Output Generation:** Export analysis and validation results for further inspection.

## Tech Stack

- **Language:** Python
- **Data Processing:** Apache Spark / PySpark
- **Data Analysis:** Pandas
- **Storage Formats:** CSV and Parquet
- **Development Environment:** Visual Studio Code
- **Version Control:** Git and GitHub

## Project Structure

```text
ecommerce-gauntlet/
├── src/
│   ├── common.py
│   ├── profiling.py
│   ├── revenue.py
│   ├── delivery.py
│   ├── customer_product.py
│   ├── spark_sql.py
│   ├── validation.py
│   └── pipeline.py
├── data/                 # Local datasets (not committed)
├── output/               # Generated results (not committed)
├── .gitignore
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/muskan8006/ecommerce-data-engineering.git
cd ecommerce-data-engineering
```

Replace `muskan8006` with your GitHub username.

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add the datasets

Place the required e-commerce CSV files in the local `data/` directory.

The dataset files should use the filenames expected by `src/common.py`. The data directory is excluded from Git, so users must obtain and add the datasets separately.

### 5. Run the pipeline

From the project root directory:

```powershell
python src/pipeline.py
```

Follow any setup instructions or error messages related to your local Python and Spark environment.

## Data Quality and Validation

The validation module includes checks for:

- Required dataset availability
- Empty datasets
- Null and duplicate key identifiers
- Required order columns
- Invalid or negative order-item numeric values
- Review scores outside the expected 1–5 range or values that cannot be parsed
- Selected order-customer and order-item-order relationships

Validation results help identify potential data quality issues before using the datasets for analysis.

## Learning Objectives

This project provides practical experience with:

- Building a multi-module Python project
- Processing structured datasets with PySpark
- Using Spark DataFrames and Spark SQL
- Performing data profiling and analytical transformations
- Implementing data quality checks
- Managing project dependencies and source code with Git

## Future Improvements

- Add automated unit tests for transformation and validation functions.
- Add logging and structured error handling.
- Automate tests with GitHub Actions.
- Add visualizations and a project architecture diagram.
- Explore cloud storage and cloud-based data processing.
- Document sample outputs using non-sensitive example data.

## Author

**Muskan**

GitHub:(https://github.com/muskan8006)

---

*This project is developed for learning and demonstrating practical data engineering concepts. Analysis results depend on the datasets provided.