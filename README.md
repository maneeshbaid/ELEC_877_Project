# ELEC 877 Project - Fake News Detection Based on News Content and Social Contexts Using Deep Learning Methods

Overview
--------
This project implements a comprehensive pipeline for fake news detection:
 - Text Preprocessing: Cleans and prepares text data using SpaCy (en_core_web_md)
 - Word Embeddings: Creates supervised FastText embeddings
 - Data Balancing: Uses undersampling for handling class imbalance
 - Hybrid Model: Combines TCN, Attention, and LSTM for classification
 - Explainability: Provides interpretability through LIME

Files Provided:
----------------
1. tcn_attention_lstm_architecture.py
   - Uses a TCN-Attention-LSTM architecture. This is the only model defined in the paper.
2. cnn_bilstm_architecture.py
   - Uses a CNN-BiLSTM architecture. I also tried this model to check performance.
3. cnn_bilstm_multiheadattention_architecture.py
   - Uses a CNN-BiLSTM-MultiHeadAttention architecture. I also tried this model to check performance.

Installation
------------
Prerequisites:
 • A recent version of Python (3.8 or higher)
 • Other Python packages: pandas, numpy, seaborn, matplotlib, spacy (with en_core_web_md), tensorflow, scikit-learn, lime, fasttext, tqdm

Install the required packages:
```python
pip install pandas numpy matplotlib seaborn tensorflow scikit-learn tqdm
pip install spacy
python -m spacy download en_core_web_md
python -m spacy download en_core_web_sm
pip install fasttext lime
```

Dataset Structure
-----------------
FakeNewsNet Dataset used:

 - gossipcop_fake.csv
 - gossipcop_real.csv
 - politifact_fake.csv
 - politifact_real.csv

Place these datasets in the dataset directory before running.

Execution
---------

To run the training pipeline, execute the desired Python script from the command line.

For example, to run the TCN-Attention-LSTM:
   ```python
   $ python tcn_attention_lstm_architecture.py
   ```

Alternatively, you may run CNN-BiLSTM and CNN-BiLSTM-MultiHeadAttention:
   ```python
   $ python cnn_bilstm_architecture.py
   ```
   ```python
   $ python cnn_bilstm_multiheadattention_architecture.py
   ```

The script will:
 - Create necessary directories (dataset, models, logs)
 - Load and preprocess data
 - Train the hybrid model
 - Evaluate performance
 - Save metrics and visualizations
 - Generate LIME explanations

Directory Structure
--------------------
```
|- dataset/            # Data files
|- models/             # Saved models and metrics
|- logs/               # Training logs and visualizations
|- tcn_attention_lstm_architecture.py  # Main script as defined in the paper
|- cnn_bilstm_architecture.py
|- cnn_bilstm_multiheadattention_architecture.py
```

## References


## License
This project is provided for educational and research purposes.