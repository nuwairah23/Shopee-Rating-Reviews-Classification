# Classification and Visualization of E-Commerce Product Reviews Comparison Using Support Vector Machine

## Overview
This project is a **web-based system** designed to classify and visualize e-commerce product reviews to address the growing challenges of information overload and unhelpful reviews on online shopping platforms like Shopee. By utilizing the **Support Vector Machine (SVM)** algorithm, the system helps users make informed purchasing decisions efficiently by distinguishing between "Useful" and "Not Useful" reviews.

## Features
- **Review Classification**
  - Classifies reviews into "Useful" or "Not Useful" categories.
  - Classification based on:
    - Review text
    - Star rating
    - Duplicated spam detection
    - Sentiment analysis
- **Product Comparison**
  - Users can input up to **six product links**.
  - Recommends the **best shop** to purchase from based on aggregated results.
  - Provides **visual comparisons** of product reviews across different shops.
- **Visualization**
  - Dynamic and interactive visual representation of star ratings and sentiment analysis.
- **Enhanced Decision-Making**
  - Reduces time spent evaluating reviews manually.
  - Helps users prioritize valuable insights.

## System Accuracy
- SVM classifier achieved an accuracy of **96.8%** during testing.
- Evaluations using the Mann-Whitney U Test confirmed the system's reliability, showing a **significant reduction in decision-making time** compared to manual review analysis.

## Technical Specifications
- **Backend**: Python with SVM implementation for classification.
- **Frontend**: Web-based interface for ease of use.
- **Visualization**: Utilizes Python’s Plotly for dynamic visualizations.
- **System Testing**: Passed all test cases with expected functionality and usability.

## Usage
1. Enter up to six Shopee product links into the system.
2. The system processes the reviews and classifies them.
3. View visualized outputs and product comparisons.
4. Receive a recommendation for the best shop based on the analysis.

## Results
- Successfully classified reviews with **96.8% accuracy**.
- Proven to significantly improve user efficiency in analyzing product reviews.
- Reliable and functional system as validated by robust statistical testing.

## Conclusion
This system provides a practical solution to the challenges faced by online shoppers, helping them save time and make well-informed purchasing decisions. Its high accuracy, reliability, and ease of use make it a valuable tool for e-commerce platforms and their users.