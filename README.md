# Facebook Misinformation Detection Using Intelligent Systems 

This project applies intelligent forensic techniques to analyze and detect misinformation patterns on Facebook. It simulates user-generated post data, applies Natural Language Processing (NLP) and anomaly detection models, and visualizes the spread of misleading information based on engagement and sentiment trends.

---

## 🎯 Objectives
- Generate a simulated dataset of Facebook posts with engagement metrics.  
- Preprocess and clean unstructured text data for analysis.  
- Apply **Named Entity Recognition (NER)** and **Sentiment Analysis** using SpaCy.  
- Use **Isolation Forest** for anomaly detection of viral or suspicious posts.  
- Create visualizations to show misinformation trends and patterns.

---

## 📂 Project Files
| File | Description |
|------|--------------|
| `generate_data.py` | Generates simulated Facebook posts with engagement and sentiment scores |
| `preprocess_data.py` | Cleans, normalizes, and engineers features from raw post data |
| `analyze_data.py` | Performs NLP, anomaly detection, and visualization |
| `final_project_raw_data.csv` | Simulated Facebook post dataset |
| `final_project_cleaned_data.csv` | Preprocessed dataset |
| `final_project_anomalies.csv` | Analysis results with flagged misinformation posts |
| `final_project_chart.png` | Visualization showing misinformation trends |
| `final_project_report.md` | Comprehensive forensic report of the investigation |

---

## 📊 Features and Techniques
- **Text Cleaning & Preprocessing**  
  Tokenization, stopword removal, normalization, and sentiment scoring.

- **Intelligent Systems**  
  - *NLP (SpaCy)* for extracting entities and detecting keywords.  
  - *Isolation Forest (Scikit-learn)* for anomaly detection based on engagement patterns.  

- **Visualization**  
  - Bar chart for misinformation count.  
  - Pie chart for sentiment distribution.  
  - Line graph showing misinformation frequency over time.  
  - Word cloud highlighting most frequent suspicious terms.

---

## 🧩 Technologies Used
- Python 3  
- Pandas  
- Scikit-learn  
- SpaCy  
- Matplotlib  
- NumPy  

---

## 📌 Summary
This project highlights how intelligent systems can support **digital forensic investigations** by identifying and analyzing misinformation on social media. Through anomaly detection and natural language analysis, it demonstrates how data-driven approaches can help detect deceptive content and strengthen online information integrity.
