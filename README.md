# **BERT-Based Sentiment Analysis**

This repository implements a sentiment analysis pipeline using BERT (Bidirectional Encoder Representations from Transformers). The model is trained to classify sentences into positive or negative sentiment using a custom dataset.

The project includes data preprocessing, tokenization, and model training with PyTorch. It also provides utilities for saving/loading preprocessed data, tokens, and model weights.

---

## **Features**
- **Preprocessing**: Clean and standardize text data with custom functions.
- **Tokenization**: Utilize BERT tokenizer for text encoding, generating input IDs and attention masks.
- **Model**: Fine-tune BERT for sentiment classification with a custom classifier head.
- **Custom Dataset**: Implement a PyTorch-compatible dataset class for efficient data loading.
- **Training and Validation**: Monitor loss and accuracy using `torchmetrics` during training and evaluation.
- **Utilities**: Save/load data, tokens, and model weights with pickle for reproducibility.

---

## **Project Structure**
- **`data.pkl`**: Stores preprocessed data (sentences and labels).
- **`saved_data/labels.pkl`**: Pickled labels for sentiment classification.
- **`saved_data/tokens.pkl`**: Tokenized inputs (input IDs and attention masks).
- **`Article.txt`**: Text data used for testing predictions.
- **`bert_pre_trained.pth`**: Pre-trained weights of the fine-tuned BERT model.

---

## **Getting Started**

### **Requirements**
Before running the code, ensure the following dependencies are installed:

- Python 3.8+
- PyTorch 1.11.0+ (with CUDA support for GPU acceleration, if available)
- Transformers
- TorchMetrics
- Pandas
- Numpy
- Scikit-learn
- Tqdm
- Kaggle (optional, for downloading datasets)
- Pickle (standard Python library)

You can install all required dependencies with the following command:

```bash
pip install torch torchvision torchaudio transformers torchmetrics pandas numpy scikit-learn tqdm kaggle
```

---

### **Usage**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your_username/bert-sentiment-analysis.git
   cd bert-sentiment-analysis
   ```

2. **Download Dataset**:
   Use `kagglehub` to fetch the dataset:
   ```python
   path = kagglehub.dataset_download("endofnight17j03/bert-sentiment-analysis")
   print("Path to dataset files:", path)
   ```

3. **Run Preprocessing**:
   Clean and preprocess the dataset to prepare for tokenization:
   ```python
   data = pd.read_csv('/path/to/your/dataset.csv')
   ```

4. **Tokenization**:
   Tokenize the data using the BERT tokenizer and save tokens:
   ```python
   handle_pickle(data=tokens, filepath="saved_data/tokens.pkl", mode="save")
   ```

5. **Model Training**:
   Train the BERT-based classifier on the tokenized data:
   ```python
   python train.py
   ```

6. **Inference**:
   Test predictions on new sentences or evaluate model performance.

---

## **Results**
The model achieves high accuracy on the test set, showing robust performance for binary sentiment classification tasks.

---

### **Contributions**
Feel free to submit issues, suggest improvements, or contribute by making pull requests. All contributions are welcome! 😊
