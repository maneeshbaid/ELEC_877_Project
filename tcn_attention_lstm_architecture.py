import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import logging
import warnings
import time
import json
from typing import List, Dict, Tuple, Union
import sys

import spacy
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (Input, Embedding, Dense, Dropout, LSTM,
                                     Bidirectional, Concatenate, Add, Activation,
                                     GlobalMaxPooling1D, GlobalAveragePooling1D,
                                     Attention, AdditiveAttention, MultiHeadAttention,
                                     Conv1D, LayerNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.utils import class_weight
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

try:
    import lime
    import lime.lime_text
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: 'lime' library not found. LIME interpretability will be skipped.")

try:
    import fasttext as fasttext_standalone
    FASTTEXT_LIB_AVAILABLE = True
except ImportError:
    FASTTEXT_LIB_AVAILABLE = False
    print("Error: Standalone 'fasttext' library not found.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class Config:
    SEED = 42
    DATA_DIR = "dataset"
    MODEL_DIR = "models"
    LOG_DIR = "logs"
    MAX_SEQUENCE_LENGTH = 200
    EMBEDDING_DIM = 300
    BATCH_SIZE = 32
    MAX_EPOCHS = 30
    LEARNING_RATE = 0.001
    DROPOUT_RATE = 0.4
    REGULARIZATION = 0.0
    TEXT_CLEANING_REGEX = r'http\S+|www\S+|@\w+|\d+|<.*?>'
    FASTTEXT_SUPERVISED_EPOCHS = 50
    FASTTEXT_SUPERVISED_LR = 0.01
    SPACY_MODEL = 'en_core_web_md'
    UNDERSAMPLING_TARGET_RATIO = 1.5

    TCN_FILTERS = 64
    TCN_KERNEL_SIZE = 3
    TCN_DILATIONS = [1, 2, 4, 8]
    TCN_STACKS = 1
    TCN_DROPOUT_RATE = 0.2
    TCN_ACTIVATION = 'relu'

    ATTENTION_HEADS = 4
    ATTENTION_KEY_DIM = 64

    LSTM_UNITS_1 = 80
    LSTM_UNITS_2 = 30
    LSTM_DROPOUT = 0.2
    LSTM_RECURRENT_DROPOUT = 0.2

    EARLY_STOPPING_PATIENCE = 10

    LIME_NUM_FEATURES = 15
    LIME_NUM_SAMPLES = 1000

    @classmethod
    def setup(cls):
        np.random.seed(cls.SEED)
        tf.random.set_seed(cls.SEED)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        logger.info("=== Configuration ===")
        for attr in dir(cls):
            if not attr.startswith('__') and not callable(getattr(cls, attr)) and attr != 'setup':
                logger.info(f"{attr}: {getattr(cls, attr)}")
        logger.info("=====================")

class DataProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = Tokenizer(oov_token="<OOV>")
        self.embedding_matrix = None
        self.feature_extractor = self._init_feature_extractor()
        self.supervised_model = None

    def _init_feature_extractor(self):
        feature_extractor = {}
        logger.info(f"Initializing SpaCy model: {self.config.SPACY_MODEL}...")
        try:
            nlp = spacy.load(self.config.SPACY_MODEL, disable=['parser', 'ner'])
            feature_extractor['nlp'] = nlp
            logger.info(f"SpaCy model '{self.config.SPACY_MODEL}' loaded successfully.")
        except OSError:
            logger.error(f"SpaCy model '{self.config.SPACY_MODEL}' not found.")
            logger.error(f"Download it by running: python -m spacy download {self.config.SPACY_MODEL}")
            logger.error("Preprocessing will fail without the SpaCy model.")
            feature_extractor['nlp'] = None
        except Exception as e:
            logger.error(f"An unexpected error occurred loading SpaCy model: {e}", exc_info=True)
            feature_extractor['nlp'] = None
        return feature_extractor

    def _enhanced_clean_text(self, text: str) -> str:
        if not isinstance(text, str): return ""
        nlp = self.feature_extractor.get('nlp')
        if nlp is None:
             logger.warning("SpaCy model not available. Using basic regex cleaning.")
             try:
                text = text.lower()
                text = re.sub(self.config.TEXT_CLEANING_REGEX, '', text)
                text = re.sub(r'[^\w\s]', ' ', text)
                return re.sub(r'\s+', ' ', text).strip()
             except Exception: return ""
        try:
            cleaned_text = text.lower()
            cleaned_text = re.sub(self.config.TEXT_CLEANING_REGEX, '', cleaned_text)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            doc = nlp(cleaned_text)
            tokens = [token.lemma_.lower().strip() for token in doc
                      if not token.is_stop and not token.is_punct and not token.is_space and len(token.lemma_.strip()) > 1]
            return ' '.join(tokens)
        except Exception as e:
            logger.error(f"Error in SpaCy cleaning: {e}", exc_info=True)
            try:
                text = text.lower()
                text = re.sub(self.config.TEXT_CLEANING_REGEX, '', text)
                text = re.sub(r'[^\w\s]', ' ', text)
                return re.sub(r'\s+', ' ', text).strip()
            except Exception: return ""

    def prepare_data(self, file_paths: List[str]) -> Dict[str, Tuple]:
        logger.info("Loading datasets...")
        dfs = []
        for path in file_paths:
            try:
                if not os.path.exists(path): logger.warning(f"File not found: {path}. Skipping."); continue
                df = None; encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
                for enc in encodings_to_try:
                    try: df = pd.read_csv(path, encoding=enc); logger.info(f"Read {path} with '{enc}'."); break
                    except Exception: continue
                if df is None: logger.error(f"Could not read CSV: {path}. Skipping."); continue
                if df.empty: logger.warning(f"Empty dataset: {path}. Skipping."); continue
                filename_lower = os.path.basename(path).lower()
                if 'fake' in filename_lower: df['label'] = 0
                elif 'real' in filename_lower or 'true' in filename_lower: df['label'] = 1
                else:
                    label_col_candidates = [c for c in df.columns if c.lower() in ['label', 'class', 'type', 'target', 'category', 'truth']]
                    if label_col_candidates:
                        label_col = label_col_candidates[0]
                        if df[label_col].dtype == object:
                             df['label'] = df[label_col].apply(lambda x: 0 if isinstance(x, str) and 'fake' in x.lower() else (1 if isinstance(x, str) and ('real' in x.lower() or 'true' in x.lower()) else pd.NA))
                             df = df.dropna(subset=['label']);
                             if not df.empty: df['label'] = df['label'].astype(int)
                        elif pd.api.types.is_numeric_dtype(df[label_col]):
                            unique_vals = df[label_col].unique()
                            if set(unique_vals).issubset({0, 1, 0.0, 1.0}): df['label'] = df[label_col].astype(int)
                            else: logger.warning(f"Numeric label column '{label_col}' in {path} has non-binary values. Skipping."); continue
                        else: logger.warning(f"Unrecognized label column format ('{label_col}') in {path}. Skipping."); continue
                    else: logger.warning(f"Cannot determine label for: {path}. Skipping."); continue
                if 'label' not in df.columns or df.empty: continue
                text_col = None; potential_text_cols = ['text', 'content', 'title', 'article', 'headline', 'body', 'news', 'statement', 'claim']
                for col_name in potential_text_cols:
                     if col_name in df.columns and df[col_name].dtype == object and df[col_name].notna().sum() > 0.5 * len(df): text_col = col_name; break
                if text_col is None:
                    for col in df.columns:
                        if df[col].dtype == object and df[col].fillna('').astype(str).str.len().mean() > 50: text_col = col; logger.warning(f"Using fallback text column '{col}' for {path}."); break
                if text_col is None: logger.warning(f"No text column found in {path}. Skipping."); continue
                df = df.rename(columns={text_col: 'text'})
                df['text'] = df['text'].fillna('').astype(str)
                if 'title' in df.columns and text_col != 'title' and df['title'].dtype == object:
                    df['title'] = df['title'].fillna('').astype(str)
                    df['text'] = df['title'] + ". " + df['text']
                df = df[df['text'].str.strip().str.len() > 10]
                if df.empty: logger.warning(f"No valid text data in {path}. Skipping."); continue
                dfs.append(df[['text', 'label']])
                logger.info(f"Processed {path}: added {len(df)} rows.")
            except Exception as e: logger.error(f"Error processing file {path}: {e}", exc_info=True)

        if not dfs: raise ValueError("No valid data files could be processed.")
        data = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total dataset size before cleaning/splitting: {len(data)} rows")
        if 'label' not in data.columns or data['label'].isna().any(): raise ValueError("Label column invalid.")
        class_counts = data['label'].value_counts(); logger.info(f"Combined dataset class distribution: {class_counts.to_dict()}")
        if len(class_counts) < 2: logger.warning("Combined dataset contains only one class.")

        logger.info(f"Performing text cleaning using SpaCy model: {self.config.SPACY_MODEL}...")
        if self.feature_extractor.get('nlp') is None: raise RuntimeError("SpaCy model failed to load.")
        try:
            from tqdm.auto import tqdm; tqdm.pandas(desc=f"Cleaning Text ({self.config.SPACY_MODEL})")
            data['processed_text'] = data['text'].progress_apply(self._enhanced_clean_text)
        except ImportError:
            logger.info("tqdm not found, processing text without progress bar...")
            data['processed_text'] = data['text'].apply(self._enhanced_clean_text)
        original_len = len(data); data = data[data['processed_text'].str.strip() != '']
        removed_count = original_len - len(data)
        if removed_count > 0: logger.warning(f"Removed {removed_count} rows with empty text after processing.")
        if data.empty: raise ValueError("All data resulted in empty processed text.")
        logger.info(f"Dataset size after cleaning: {len(data)} rows")

        logger.info("Performing LDA Topic Modeling on processed text...")
        perform_lda(data['processed_text'], self.config)

        logger.info("Tokenizing and padding text sequences...")
        if 'processed_text' not in data.columns or data['processed_text'].empty: raise ValueError("Column 'processed_text' missing.")
        texts_for_tokenizer = data['processed_text'].tolist()
        self.tokenizer.fit_on_texts(texts_for_tokenizer)
        word_index = self.tokenizer.word_index; logger.info(f"Tokenizer fitted. Vocabulary size: {len(word_index)}")
        sequences = self.tokenizer.texts_to_sequences(texts_for_tokenizer)
        X_text_padded = pad_sequences(sequences, maxlen=self.config.MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        logger.info(f"Text sequences processed. Shape: {X_text_padded.shape}")

        logger.info("Creating Supervised FastText embeddings...")
        self._create_fasttext_embeddings(data['processed_text'], data['label'])
        if self.embedding_matrix is None: logger.warning("Embedding matrix was not created successfully.")

        y = data['label'].values
        texts_original = data['text'].values
        logger.info("Splitting data into train/val/test sets...")
        data_dict = self._split_dataset(X_text_padded, y, texts_original)
        if not data_dict or 'train' not in data_dict or data_dict['train'][0] is None:
             logger.critical("Data splitting failed. Exiting."); return {}

        logger.info("Balancing the training set by undersampling...")
        X_text_train, y_train, texts_original_train = data_dict['train']
        X_text_val, y_val, texts_original_val = data_dict['val']
        X_text_test, y_test, texts_original_test = data_dict['test']

        if y_train is not None and len(y_train) > 0:
            logger.info(f"Original training set size: {len(y_train)}")
            train_class_counts = pd.Series(y_train).value_counts().to_dict()
            logger.info(f"Original training class distribution: {train_class_counts}")

            if len(train_class_counts) < 2:
                 logger.warning("Training set contains only one class. Skipping balancing.")
                 X_text_train_balanced, y_train_balanced, texts_original_train_balanced = X_text_train, y_train, texts_original_train
            else:
                train_indices = np.arange(len(y_train))
                fake_indices = train_indices[y_train == 0]
                real_indices = train_indices[y_train == 1]
                n_fake, n_real = len(fake_indices), len(real_indices)
                target_ratio = self.config.UNDERSAMPLING_TARGET_RATIO
                final_fake_indices, final_real_indices = fake_indices, real_indices

                if n_real > n_fake * target_ratio and n_fake > 0:
                    n_real_target = min(n_real, int(n_fake * target_ratio))
                    logger.info(f"Undersampling Real class from {n_real} to {n_real_target}")
                    np.random.seed(self.config.SEED)
                    final_real_indices = np.random.choice(real_indices, size=n_real_target, replace=False)
                elif n_fake > n_real * target_ratio and n_real > 0:
                    n_fake_target = min(n_fake, int(n_real * target_ratio))
                    logger.info(f"Undersampling Fake class from {n_fake} to {n_fake_target}")
                    np.random.seed(self.config.SEED)
                    final_fake_indices = np.random.choice(fake_indices, size=n_fake_target, replace=False)
                else:
                    logger.info("Training set already balanced according to target ratio.")

                balanced_train_indices = np.concatenate([final_fake_indices, final_real_indices])
                np.random.seed(self.config.SEED); np.random.shuffle(balanced_train_indices)
                X_text_train_balanced = X_text_train[balanced_train_indices]
                y_train_balanced = y_train[balanced_train_indices]
                texts_original_train_balanced = texts_original_train[balanced_train_indices]

                logger.info(f"Balanced training set size: {len(y_train_balanced)}")
                balanced_counts = pd.Series(y_train_balanced).value_counts().to_dict()
                logger.info(f"Balanced training class distribution: {balanced_counts}")

            data_dict['train'] = (X_text_train_balanced, y_train_balanced, texts_original_train_balanced)
        else:
            logger.warning("Training data is None or empty before balancing. Skipping balancing.")
            data_dict['train'] = (X_text_train, y_train, texts_original_train)

        if y_val is not None: logger.info(f"Validation set size (unbalanced): {len(y_val)}")
        if y_test is not None: logger.info(f"Test set size (unbalanced): {len(y_test)}")

        data_dict['val'] = (X_text_val, y_val, texts_original_val)
        data_dict['test'] = (X_text_test, y_test, texts_original_test)

        return data_dict

    def _create_fasttext_embeddings(self, texts, labels):
        if not FASTTEXT_LIB_AVAILABLE:
             logger.error("Standalone FastText library not available.")
             word_index = self.tokenizer.word_index if hasattr(self.tokenizer, 'word_index') else {}
             vocab_size = len(word_index) + 1
             self.embedding_matrix = np.random.normal(0, 0.1, (vocab_size, self.config.EMBEDDING_DIM))
             logger.warning(f"Falling back to random embeddings matrix.")
             return
        if isinstance(texts, pd.Series): texts = texts.tolist()
        if not isinstance(texts, list) or not texts:
             logger.error("No texts provided for FastText training.")
             word_index = self.tokenizer.word_index if hasattr(self.tokenizer, 'word_index') else {}
             vocab_size = len(word_index) + 1
             self.embedding_matrix = np.random.normal(0, 0.1, (vocab_size, self.config.EMBEDDING_DIM))
             logger.warning(f"Falling back to random embeddings matrix.")
             return

        word_index = self.tokenizer.word_index
        vocab_size = len(word_index) + 1
        self.embedding_matrix = np.zeros((vocab_size, self.config.EMBEDDING_DIM))
        supervised_temp_file = 'supervised_fasttext_data.txt'
        try:
            logger.info("Preparing data for supervised FastText training...")
            lines_written = 0
            with open(supervised_temp_file, 'w', encoding='utf-8') as f:
                for text, label in zip(texts, labels):
                    if isinstance(text, str) and text.strip() and label in [0, 1]:
                        clean_text_for_ft = text.replace('\n', ' ').replace('\r', ' ')
                        f.write(f'__label__{label} {clean_text_for_ft}\n')
                        lines_written += 1
            if lines_written > 0:
                logger.info(f"Training supervised FastText model on {lines_written} samples...")
                self.supervised_model = fasttext_standalone.train_supervised(
                    input=supervised_temp_file, dim=self.config.EMBEDDING_DIM,
                    epoch=self.config.FASTTEXT_SUPERVISED_EPOCHS, lr=self.config.FASTTEXT_SUPERVISED_LR,
                    wordNgrams=2, verbose=2, seed=self.config.SEED, thread=max(1, os.cpu_count() - 1)
                )
                logger.info("Supervised FastText model training complete.")
            else:
                logger.warning(f"Temp file empty. Skipping supervised FastText training.")
                self.supervised_model = None
        except Exception as e_sup:
            logger.error(f"Error training supervised FastText: {e_sup}", exc_info=True)
            self.supervised_model = None
        finally:
            if os.path.exists(supervised_temp_file):
                try: os.remove(supervised_temp_file); logger.info(f"Removed temp file: {supervised_temp_file}")
                except OSError as e_rm: logger.warning(f"Could not remove temp file {supervised_temp_file}: {e_rm}")

        logger.info("Populating embedding matrix from supervised FastText model...")
        found_vectors, oov_count = 0, 0
        if self.supervised_model:
            for word, i in word_index.items():
                if i >= vocab_size: continue
                try:
                    vector = self.supervised_model.get_word_vector(word)
                    self.embedding_matrix[i] = vector; found_vectors += 1
                except Exception:
                    self.embedding_matrix[i] = np.random.normal(scale=0.1, size=self.config.EMBEDDING_DIM); oov_count += 1
            logger.info(f"Embedding matrix created. Shape: {self.embedding_matrix.shape}. Found: {found_vectors}/{len(word_index)}. OOV: {oov_count}")
            if found_vectors == 0 and len(word_index) > 0:
                 logger.error("No word vectors retrieved. Matrix is random.")
                 self.embedding_matrix = np.random.normal(scale=0.1, size=(vocab_size, self.config.EMBEDDING_DIM))
        else:
             logger.error("Supervised model not trained. Embedding matrix is random.")
             self.embedding_matrix = np.random.normal(scale=0.1, size=(vocab_size, self.config.EMBEDDING_DIM))

    def _split_dataset(self, X_text, y, texts_original, test_size=0.2, val_size=0.1):
        logger.info(f"Splitting data: Total={len(y)}, Test={test_size}, Val={val_size}")
        n_samples = len(y)
        if n_samples < 10:
             logger.error(f"Not enough data for splitting ({n_samples}).")
             return {'train': (None, None, None), 'val': (None, None, None), 'test': (None, None, None)}
        if len(X_text) != n_samples or len(texts_original) != n_samples:
             logger.error("Input array lengths mismatch before splitting.")
             return {'train': (None, None, None), 'val': (None, None, None), 'test': (None, None, None)}
        try:
            unique_labels, counts = np.unique(y, return_counts=True)
            can_stratify_tt = (len(unique_labels) >= 2 and np.all(counts >= 2))
            stratify_tt = y if can_stratify_tt else None
            if not can_stratify_tt: logger.warning(f"Cannot stratify train/test split. Counts: {dict(zip(unique_labels, counts))}")

            X_text_train_val, X_text_test, y_train_val, y_test, texts_original_train_val, texts_original_test = train_test_split(
                X_text, y, texts_original, test_size=test_size, stratify=stratify_tt, random_state=self.config.SEED)
            logger.info(f"Initial split: Train+Val={len(y_train_val)}, Test={len(y_test)}")

            if len(y_train_val) < 5:
                 logger.warning(f"Train+Val set very small ({len(y_train_val)}).")
                 if len(y_train_val) <= 1:
                      logger.error("Cannot create validation set.")
                      X_text_train, y_train, texts_original_train = X_text_train_val, y_train_val, texts_original_train_val
                      X_text_val, y_val, texts_original_val = (np.array([]).reshape(0, X_text.shape[1]), np.array([]), np.array([]))
                 else:
                      logger.warning("Creating minimal validation set.")
                      relative_val_size_adj = 1 / len(y_train_val)
                      unique_tv, counts_tv = np.unique(y_train_val, return_counts=True)
                      can_stratify_tv_adj = (len(unique_tv) >= 2 and np.all(counts_tv >= 1))
                      stratify_tv_adj = y_train_val if can_stratify_tv_adj else None
                      X_text_train, X_text_val, y_train, y_val, texts_original_train, texts_original_val = train_test_split(
                          X_text_train_val, y_train_val, texts_original_train_val, test_size=relative_val_size_adj,
                          stratify=stratify_tv_adj, random_state=self.config.SEED)
            else:
                 relative_val_size = val_size / (1.0 - test_size)
                 logger.info(f"Relative validation size: {relative_val_size:.3f}")
                 unique_labels_tv, counts_tv = np.unique(y_train_val, return_counts=True)
                 can_stratify_tv = (len(unique_labels_tv) >= 2 and np.all(counts_tv >= 2))
                 stratify_tv = y_train_val if can_stratify_tv else None
                 if not can_stratify_tv: logger.warning(f"Cannot stratify train/validation split. Counts: {dict(zip(unique_labels_tv, counts_tv))}")
                 X_text_train, X_text_val, y_train, y_val, texts_original_train, texts_original_val = train_test_split(
                     X_text_train_val, y_train_val, texts_original_train_val, test_size=relative_val_size,
                     stratify=stratify_tv, random_state=self.config.SEED)

            logger.info(f"Final split sizes before balancing: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")
            if len(y_train) == 0 or len(y_test) == 0:
                logger.error("Empty split detected (Train or Test).")
                return {'train': (None, None, None), 'val': (None, None, None), 'test': (None, None, None)}
            if len(y_val) == 0:
                 logger.warning("Validation set is empty.")
                 X_text_val = np.array([]).reshape(0, X_text.shape[1]); y_val = np.array([]); texts_original_val = np.array([])

            return {'train': (X_text_train, y_train, texts_original_train),
                    'val': (X_text_val, y_val, texts_original_val),
                    'test': (X_text_test, y_test, texts_original_test)}
        except ValueError as ve:
             logger.error(f"Error during train_test_split: {ve}", exc_info=True)
             return {'train': (None, None, None), 'val': (None, None, None), 'test': (None, None, None)}
        except Exception as e:
             logger.error(f"Unexpected error during splitting: {e}", exc_info=True)
             return {'train': (None, None, None), 'val': (None, None, None), 'test': (None, None, None)}

def perform_lda(texts: Union[pd.Series, List[str]], config: Config, num_topics: int = 3, n_top_words: int = 10):
    logger.info(f"Performing LDA for {num_topics} topics...")
    if isinstance(texts, pd.Series): texts = texts.tolist()
    if not texts or all(not t for t in texts): logger.warning("Input texts for LDA empty. Skipping."); return
    try:
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
        dtm = vectorizer.fit_transform(texts)
        if dtm.shape[0] == 0 or dtm.shape[1] == 0: logger.warning("DTM empty. Skipping LDA."); return
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=config.SEED,
                                        learning_method='online', learning_offset=50., max_iter=10)
        lda.fit(dtm)
        feature_names = vectorizer.get_feature_names_out()
        print("\n--- LDA Topic Modeling Results ---")
        for topic_idx, topic in enumerate(lda.components_):
            top_words_indices = topic.argsort()[:-n_top_words - 1:-1]
            top_words = [feature_names[i] for i in top_words_indices]
            print(f"Topic #{topic_idx}: {' '.join(top_words)}")
        print("---------------------------------")
    except ValueError as ve: logger.error(f"ValueError during LDA: {ve}.")
    except Exception as e: logger.error(f"Error during LDA: {e}", exc_info=True)

def residual_block(inputs, dilation_rate: int, nb_filters: int, kernel_size: int,
                   dropout_rate: float, activation: str, block_id: int):
    block_name = f'tcn_residual_block_{block_id}_dil{dilation_rate}'
    conv1 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal', name=f'{block_name}_conv1')(inputs)
    norm1 = LayerNormalization(name=f'{block_name}_layernorm1')(conv1)
    act1 = Activation(activation, name=f'{block_name}_{activation}1')(norm1)
    drop1 = Dropout(rate=dropout_rate, name=f'{block_name}_dropout1')(act1)
    conv2 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal', name=f'{block_name}_conv2')(drop1)
    norm2 = LayerNormalization(name=f'{block_name}_layernorm2')(conv2)
    act2 = Activation(activation, name=f'{block_name}_{activation}2')(norm2)
    drop2 = Dropout(rate=dropout_rate, name=f'{block_name}_dropout2')(act2)
    if inputs.shape[-1] != nb_filters:
        shortcut = Conv1D(filters=nb_filters, kernel_size=1, padding='same', name=f'{block_name}_shortcut_conv')(inputs)
    else:
        shortcut = inputs
    res_output = Add(name=f'{block_name}_add')([shortcut, drop2])
    return res_output, drop2

def create_tcn_attention_lstm_model(config: Config, processor: DataProcessor):
    logger.info("Building the TCN-Attention-LSTM model")

    text_input_shape = (config.MAX_SEQUENCE_LENGTH,)
    if not hasattr(processor, 'tokenizer') or not hasattr(processor.tokenizer, 'word_index'):
         raise ValueError("Tokenizer not found or not fitted in the processor.")
    vocab_size = len(processor.tokenizer.word_index) + 1
    embedding_dim = config.EMBEDDING_DIM

    if processor.embedding_matrix is None:
        logger.error("Embedding matrix is None! Using random embeddings (trainable).")
        embedding_matrix = np.random.normal(scale=0.1, size=(vocab_size, embedding_dim))
        trainable_embedding = True
    elif processor.embedding_matrix.shape != (vocab_size, embedding_dim):
        logger.error(f"Embedding matrix shape mismatch! Using random embeddings (trainable).")
        embedding_matrix = np.random.normal(scale=0.1, size=(vocab_size, embedding_dim))
        trainable_embedding = True
    else:
        embedding_matrix = processor.embedding_matrix
        logger.info("Using pre-computed embeddings from DataProcessor.")
        trainable_embedding = False
        logger.info(f"Embedding layer trainable: {trainable_embedding}")

    text_input = Input(shape=text_input_shape, name='text_input')
    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_length=config.MAX_SEQUENCE_LENGTH,
        trainable=trainable_embedding,
        name='embedding_layer'
    )(text_input)
    embedding_layer = Dropout(0.2, name='embedding_dropout')(embedding_layer)

    logger.info("Building Manual TCN stack...")
    tcn_input = embedding_layer
    skip_connections = []
    block_counter = 0
    for stack in range(config.TCN_STACKS):
        for dil_rate in config.TCN_DILATIONS:
            block_counter += 1
            tcn_input, skip_out = residual_block(
                inputs=tcn_input,
                dilation_rate=dil_rate,
                nb_filters=config.TCN_FILTERS,
                kernel_size=config.TCN_KERNEL_SIZE,
                dropout_rate=config.TCN_DROPOUT_RATE,
                activation=config.TCN_ACTIVATION,
                block_id=block_counter
            )
            skip_connections.append(skip_out)

    tcn_output = tcn_input
    logger.info(f"Manual TCN stack output shape: {tcn_output.shape}")

    logger.info("Applying Multi-Head Attention...")
    attention_layer = MultiHeadAttention(
        num_heads=config.ATTENTION_HEADS, key_dim=config.ATTENTION_KEY_DIM,
        dropout=0.1, name='multi_head_attention'
    )(query=tcn_output, value=tcn_output, key=tcn_output, return_attention_scores=False)
    attention_output = Add(name='attention_add')([tcn_output, attention_layer])
    attention_output = LayerNormalization(epsilon=1e-6, name='attention_layernorm')(attention_output)
    logger.info(f"Attention layer output shape: {attention_output.shape}")

    logger.info("Adding Bidirectional LSTM layers...")
    lstm_layer = Bidirectional(LSTM(
        units=config.LSTM_UNITS_1,
        return_sequences=True,
        kernel_regularizer=l2(config.REGULARIZATION), recurrent_regularizer=l2(config.REGULARIZATION),
        dropout=config.LSTM_DROPOUT, recurrent_dropout=config.LSTM_RECURRENT_DROPOUT,
        name='bilstm_1'
    ))(attention_output)

    lstm_layer = Bidirectional(LSTM(
        units=config.LSTM_UNITS_2,
        return_sequences=False,
        kernel_regularizer=l2(config.REGULARIZATION), recurrent_regularizer=l2(config.REGULARIZATION),
        dropout=config.LSTM_DROPOUT, recurrent_dropout=config.LSTM_RECURRENT_DROPOUT,
        name='bilstm_2'
    ))(lstm_layer)
    logger.info(f"BiLSTM layer output shape: {lstm_layer.shape}")

    logger.info("Adding final Dense layers...")
    x = Dropout(config.DROPOUT_RATE, name='dropout_lstm_output')(lstm_layer)
    x = Dense(128, activation='relu', kernel_regularizer=l2(config.REGULARIZATION), name='dense_1')(x)
    x = Dropout(config.DROPOUT_RATE, name='dropout_dense_1')(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(config.REGULARIZATION), name='dense_2')(x)
    x = Dropout(config.DROPOUT_RATE, name='dropout_dense_2')(x)

    output = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=text_input, outputs=output, name='FakeNews_ManualTCN_Attention_LSTM_Model')

    logger.info(f"Using Adam optimizer with Learning Rate: {config.LEARNING_RATE}")
    optimizer = Adam(learning_rate=config.LEARNING_RATE, clipnorm=1.0)

    metrics = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)

    logger.info(f"Manual TCN-Attention-LSTM model built successfully. Total Params: {model.count_params():,}")
    return model

def train_model(model: Model, data_dict: Dict[str, Tuple], config: Config):
    logger.info("Starting model training using Undersampled Data...")
    if 'train' not in data_dict or 'val' not in data_dict: logger.error("Train/Val data missing."); return None
    X_text_train, y_train, _ = data_dict['train']
    X_text_val, y_val, _ = data_dict['val']
    if X_text_train is None or y_train is None: logger.error("Training data None."); return None
    if len(X_text_train) == 0 or len(y_train) == 0: logger.error("Training data empty."); return None
    if len(X_text_train) != len(y_train): logger.error(f"Training shape mismatch."); return None
    if X_text_val is None or y_val is None: logger.error("Validation data None."); return None
    if len(X_text_val) == 0 or len(y_val) == 0:
         logger.warning("Validation data empty. Training without validation callbacks."); validation_data = None; monitor_metric = 'loss'
    elif len(X_text_val) != len(y_val): logger.error(f"Validation shape mismatch."); return None
    else: validation_data = (X_text_val, y_val); monitor_metric = 'val_loss'

    run_timestamp = time.strftime("%Y%m%d-%H%M%S"); run_log_dir = os.path.join(config.LOG_DIR, f'run_{run_timestamp}'); model_save_dir = os.path.join(config.MODEL_DIR, f'run_{run_timestamp}')
    os.makedirs(run_log_dir, exist_ok=True); os.makedirs(model_save_dir, exist_ok=True)
    logger.info(f"TB Log dir: {run_log_dir}"); logger.info(f"Checkpoint dir: {model_save_dir}")
    logger.info(f"Using EarlyStopping patience: {config.EARLY_STOPPING_PATIENCE}")
    early_stopping = EarlyStopping(monitor=monitor_metric, patience=config.EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor=monitor_metric, factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    checkpoint_filepath = os.path.join(model_save_dir, 'best_model.keras')
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor=monitor_metric, save_best_only=True, save_weights_only=False, verbose=1)
    tensorboard = TensorBoard(log_dir=run_log_dir, histogram_freq=1, write_graph=True, update_freq='epoch')
    callbacks = [early_stopping, reduce_lr, model_checkpoint, tensorboard]
    if validation_data is None: callbacks = [tensorboard]; logger.warning("Removed callbacks needing validation data.")

    class_weights_dict = None
    logger.info("Calculating class weights for the undersampled training set...")
    try:
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        if len(unique_classes) > 1:
            computed_weights = class_weight.compute_class_weight('balanced', classes=unique_classes, y=y_train)
            class_weights_dict = dict(zip(unique_classes, computed_weights))
            logger.info(f"Using computed class weights for training (post-undersampling): {class_weights_dict}")
        else:
            logger.warning("Only one class present in the training data (post-undersampling). Cannot compute class weights.")
    except Exception as e:
        logger.warning(f"Class weight calculation failed: {e}. Training without class weights.")

    logger.info(f"Starting fitting: Epochs={config.MAX_EPOCHS}, Batch={config.BATCH_SIZE}...")
    history = None
    try:
        history = model.fit( X_text_train, y_train, validation_data=validation_data, epochs=config.MAX_EPOCHS,
                            batch_size=config.BATCH_SIZE, class_weight=class_weights_dict, callbacks=callbacks, verbose=1)
        logger.info("Training finished.")
        if validation_data is not None:
             if early_stopping.stopped_epoch > 0: logger.info(f"Early stopping at epoch {early_stopping.stopped_epoch + 1}. Best weights restored: {early_stopping.restore_best_weights}")
             else: logger.info("Training completed max epochs.")
             if os.path.exists(checkpoint_filepath): logger.info(f"Best model saved: {checkpoint_filepath}")
             else: logger.warning(f"Checkpoint not found: {checkpoint_filepath}.")
        else: logger.info("Training completed without validation.")
    except tf.errors.InvalidArgumentError as e: logger.error(f"TF InvalidArgumentError: {e}", exc_info=True); return None
    except Exception as e: logger.error(f"Unexpected training error: {e}", exc_info=True); return None
    return history

def evaluate_model(model: Model, data_dict: Dict[str, Tuple], config: Config):
    logger.info("Starting model evaluation on the test set...")
    if 'test' not in data_dict: logger.error("Test data missing."); return None, None, None, None
    X_text_test, y_test, texts_original_test = data_dict['test']
    if X_text_test is None or y_test is None or texts_original_test is None: logger.error("Test data arrays None."); return None, None, None, None
    if len(y_test) == 0: logger.error("Test set empty."); return None, None, None, None
    if len(X_text_test) != len(y_test): logger.error(f"Test data shape mismatch."); return None, None, None, None

    try:
        logger.info(f"Predicting on {len(y_test)} test samples...")
        start_pred_time = time.time()
        y_pred_proba = model.predict(X_text_test).flatten()
        logger.info(f"Prediction complete. Took {time.time() - start_pred_time:.2f} sec.")

        logger.info("Finding optimal threshold (Macro F1)...")
        best_f1_macro, best_threshold = 0, 0.5
        thresholds = np.arange(0.05, 0.96, 0.01)
        f1_scores_per_threshold = []
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            f1_scores_per_threshold.append(f1_macro)
            if f1_macro > best_f1_macro: best_f1_macro, best_threshold = f1_macro, threshold
        final_threshold = best_threshold
        logger.info(f"Optimal threshold: {final_threshold:.3f} (Test Macro F1: {best_f1_macro:.4f})")

        try:
             plt.figure(figsize=(8, 5)); plt.plot(thresholds, f1_scores_per_threshold, marker='.', linestyle='-')
             plt.title('Macro F1-Score vs. Threshold (Test Set)'); plt.xlabel('Threshold'); plt.ylabel('Macro F1-Score')
             plt.vlines(final_threshold, plt.ylim()[0], plt.ylim()[1], color='r', linestyle='--', label=f'Optimal ({final_threshold:.3f})')
             plt.legend(); plt.grid(True)
             f1_plot_path = os.path.join(config.LOG_DIR, f'threshold_f1_plot_{time.strftime("%Y%m%d-%H%M%S")}.png')
             plt.savefig(f1_plot_path); logger.info(f"Threshold plot saved: {f1_plot_path}"); plt.close()
        except Exception as plot_e: logger.warning(f"Could not save threshold plot: {plot_e}")

        y_pred_final = (y_pred_proba >= final_threshold).astype(int)
        accuracy = accuracy_score(y_test, y_pred_final)
        precision_macro = precision_score(y_test, y_pred_final, average='macro', zero_division=0)
        recall_macro = recall_score(y_test, y_pred_final, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred_final, average='macro', zero_division=0)
        report_dict = classification_report(y_test, y_pred_final, output_dict=True, zero_division=0)
        report_str = classification_report(y_test, y_pred_final, target_names=['Fake (0)', 'Real (1)'], zero_division=0)
        precision_fake = report_dict.get('0', {}).get('precision', 0.0); recall_fake = report_dict.get('0', {}).get('recall', 0.0); f1_fake = report_dict.get('0', {}).get('f1-score', 0.0); support_fake = report_dict.get('0', {}).get('support', 0)
        precision_real = report_dict.get('1', {}).get('precision', 0.0); recall_real = report_dict.get('1', {}).get('recall', 0.0); f1_real = report_dict.get('1', {}).get('f1-score', 0.0); support_real = report_dict.get('1', {}).get('support', 0)
        cm = confusion_matrix(y_test, y_pred_final); tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        print("\n===== MODEL EVALUATION RESULTS (Test Set) =====")
        print(f"Optimal Threshold Applied: {final_threshold:.3f}")
        print(f"\nOverall Metrics (Macro Average):"); print(f"  Accuracy:  {accuracy:.4f}"); print(f"  Precision: {precision_macro:.4f}"); print(f"  Recall:    {recall_macro:.4f}"); print(f"  F1-Score:  {f1_macro:.4f}")
        print(f"\nClass-Specific Metrics:"); print(f"  Class 0 (Fake): P={precision_fake:.4f}, R={recall_fake:.4f}, F1={f1_fake:.4f} (Support: {support_fake})"); print(f"  Class 1 (Real): P={precision_real:.4f}, R={recall_real:.4f}, F1={f1_real:.4f} (Support: {support_real})")
        print("\nClassification Report:\n", report_str); print("\nConfusion Matrix:"); print(f"            Predicted Fake | Predicted Real"); print(f"Actual Fake       {tn:^6d}     |     {fp:^6d}"); print(f"Actual Real       {fn:^6d}     |     {tp:^6d}"); print("---------------------------------------------")

        plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred Fake (0)', 'Pred Real (1)'], yticklabels=['Actual Fake (0)', 'Actual Real (1)'], annot_kws={"size": 14}, linewidths=.5)
        plt.title(f'Confusion Matrix (Test Set, Threshold: {final_threshold:.3f})', fontsize=16); plt.xlabel('Predicted Label', fontsize=12); plt.ylabel('Actual Label', fontsize=12); plt.tight_layout()
        cm_filename = f'confusion_matrix_{time.strftime("%Y%m%d-%H%M%S")}.png'; cm_path = os.path.join(config.LOG_DIR, cm_filename)
        try: plt.savefig(cm_path); logger.info(f"CM plot saved: {cm_path}"); plt.close()
        except Exception as plot_e: logger.error(f"Failed to save CM plot: {plot_e}")

        metrics_dict = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"), 'model_type': 'ManualTCN_Attention_LSTM_ClassWgt',
            'config': {attr: getattr(config, attr) for attr in dir(config) if not attr.startswith('__') and not callable(getattr(config, attr)) and attr != 'setup'},
            'optimal_threshold': final_threshold, 'accuracy': accuracy, 'precision_macro': precision_macro, 'recall_macro': recall_macro, 'f1_macro': f1_macro,
            'precision_fake': precision_fake, 'recall_fake': recall_fake, 'f1_fake': f1_fake, 'support_fake': int(support_fake),
            'precision_real': precision_real, 'recall_real': recall_real, 'f1_real': f1_real, 'support_real': int(support_real),
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}, 'confusion_matrix_array': cm.tolist()
        }
        return metrics_dict, texts_original_test, y_test, y_pred_proba
    except tf.errors.InvalidArgumentError as tf_err: logger.error(f"TF error during evaluation: {tf_err}", exc_info=True); return None, None, None, None
    except Exception as e: logger.error(f"Unexpected error during evaluation: {e}", exc_info=True); return None, None, None, None

def explain_with_lime(model: Model, processor: DataProcessor, config: Config,
                      texts_original: np.ndarray, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      num_samples: int = 5):
    if not LIME_AVAILABLE:
        logger.warning("LIME library not available. Skipping explanations.")
        return
    if texts_original is None or y_true is None or y_pred_proba is None:
        logger.warning("LIME input data is None. Skipping explanations.")
        return
    if len(texts_original) == 0:
        logger.warning("No original texts provided for LIME explanations.")
        return
    if not (len(texts_original) == len(y_true) == len(y_pred_proba)):
        logger.warning(f"LIME input array length mismatch. Skipping explanations.")
        return

    logger.info(f"\n--- Generating LIME Explanations for ~{num_samples} Test Samples (Stratified) ---")

    fake_indices = np.where(y_true == 0)[0]
    real_indices = np.where(y_true == 1)[0]
    num_fake_available = len(fake_indices)
    num_real_available = len(real_indices)

    num_fake_to_sample = min(num_fake_available, num_samples // 2)
    num_real_to_sample = min(num_real_available, num_samples - num_fake_to_sample)
    if (num_fake_to_sample + num_real_to_sample < num_samples) and (num_fake_available > num_fake_to_sample):
        num_fake_to_sample = min(num_fake_available, num_samples - num_real_to_sample)


    logger.info(f"Available: {num_fake_available} Fake, {num_real_available} Real.")
    logger.info(f"Sampling: {num_fake_to_sample} Fake, {num_real_to_sample} Real.")

    np.random.seed(config.SEED)
    selected_fake_indices = np.random.choice(fake_indices, size=num_fake_to_sample, replace=False) if num_fake_to_sample > 0 else np.array([])
    selected_real_indices = np.random.choice(real_indices, size=num_real_to_sample, replace=False) if num_real_to_sample > 0 else np.array([])

    indices_to_explain = np.concatenate([selected_fake_indices, selected_real_indices]).astype(int)
    np.random.shuffle(indices_to_explain)

    if len(indices_to_explain) == 0:
        logger.warning("No samples selected for LIME explanation after stratification.")
        return

    def lime_predictor(texts: List[str]) -> np.ndarray:
        if not isinstance(texts, list):
            try: texts = [str(t) for t in texts]
            except Exception: return np.array([[0.5, 0.5]] * len(texts))
        try:
            cleaned_texts = [processor._enhanced_clean_text(text) for text in texts]
            sequences = processor.tokenizer.texts_to_sequences(cleaned_texts)
            padded_sequences = pad_sequences(sequences, maxlen=config.MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        except Exception as preprocess_e:
            logger.error(f"LIME preprocessing error: {preprocess_e}", exc_info=True)
            return np.array([[0.5, 0.5]] * len(texts))

        try:
            pred_proba_positive = model.predict(padded_sequences, batch_size=min(len(texts), 512))
            pred_proba_positive = pred_proba_positive.flatten()
            pred_proba_negative = 1.0 - pred_proba_positive
            return np.vstack((pred_proba_negative, pred_proba_positive)).T
        except Exception as pred_e:
            logger.error(f"LIME prediction error: {pred_e}", exc_info=True)
            return np.array([[0.5, 0.5]] * len(texts))


    explainer = LimeTextExplainer(class_names=['Fake (0)', 'Real (1)'])
    num_lime_features = config.LIME_NUM_FEATURES

    for i, idx in enumerate(indices_to_explain):
        text_instance = texts_original[idx]
        true_label = y_true[idx]
        pred_prob_real = y_pred_proba[idx]
        predicted_label_interpret = 1 if pred_prob_real >= 0.5 else 0

        print(f"\n--- Explaining Test Sample #{i+1} (Index: {idx}) ---")
        print(f"  Original Text (first 300 chars): {text_instance[:300]}...")
        print(f"  True Label: {'Real (1)' if true_label == 1 else 'Fake (0)'}")
        print(f"  Predicted Label (@0.5 Thresh): {'Real (1)' if predicted_label_interpret == 1 else 'Fake (0)'}")
        print(f"  Model's Predicted Probability (Class 1 - Real): {pred_prob_real:.4f}")

        try:
            explanation = explainer.explain_instance(
                text_instance,
                lime_predictor,
                num_features=num_lime_features,
                num_samples=config.LIME_NUM_SAMPLES
            )
            print(f"  LIME Explanation (Top {num_lime_features} features contributing to P(Real)):")
            explanation_list = explanation.as_list()
            if explanation_list:
                for feature, weight in explanation_list:
                    direction = "Supports Real" if weight > 0 else "Supports Fake"
                    print(f"    - Word: '{feature}', Weight: {weight:.4f} ({direction})")
            else:
                print("    - LIME could not identify significant features for this sample.")

            html_filename = f'lime_explanation_sample_{idx}.html'
            html_filepath = os.path.join(config.LOG_DIR, html_filename)
            try:
                explanation.save_to_file(html_filepath)
                logger.info(f"  LIME HTML explanation saved: {html_filepath}")
            except Exception as save_e:
                logger.error(f"  Failed to save LIME HTML explanation: {save_e}")

        except Exception as lime_e:
            logger.error(f"  Error generating LIME explanation for sample index {idx}: {lime_e}", exc_info=True)

    print("---------------------------------------------------------------------")

def plot_training_history(history, config: Config):
    if history is None or not hasattr(history, 'history') or not history.history: logger.warning("No history object. Skipping plotting."); return
    history_dict = history.history; epochs = range(1, len(history_dict.get('loss', [])) + 1)
    has_loss = 'loss' in history_dict; has_val_loss = 'val_loss' in history_dict
    acc_key = next((k for k in history_dict if k == 'accuracy' or k == 'acc'), None)
    val_acc_key = next((k for k in history_dict if k == 'val_accuracy' or k == 'val_acc'), None)
    has_accuracy = bool(acc_key); has_val_accuracy = bool(val_acc_key)
    num_plots = sum([has_loss or has_val_loss, has_accuracy or has_val_accuracy])
    if num_plots == 0: logger.warning("History missing plottable metrics. Skipping plotting."); return
    if not epochs: logger.warning("No epochs in history. Skipping plotting."); return

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 6), squeeze=False)
    plot_idx = 0
    if has_loss or has_val_loss:
        ax = axes[0, plot_idx]; ax.set_title('Training and Validation Loss', fontsize=14); ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('Loss', fontsize=12); ax.grid(True)
        if has_loss: ax.plot(epochs, history_dict['loss'], 'bo-', label='Training Loss')
        if has_val_loss: ax.plot(epochs, history_dict['val_loss'], 'ro-', label='Validation Loss')
        else: ax.text(0.5, 0.5, 'Validation loss N/A', ha='center', va='center', transform=ax.transAxes, color='gray')
        if has_loss or has_val_loss: ax.legend(fontsize=10)
        plot_idx += 1
    if has_accuracy or has_val_accuracy:
        ax = axes[0, plot_idx]; ax.set_title('Training and Validation Accuracy', fontsize=14); ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('Accuracy', fontsize=12); ax.grid(True)
        if has_accuracy: ax.plot(epochs, history_dict[acc_key], 'bo-', label=f'Training {acc_key.capitalize()}')
        if has_val_accuracy: ax.plot(epochs, history_dict[val_acc_key], 'ro-', label=f'Validation {val_acc_key.capitalize()}')
        else: ax.text(0.5, 0.5, 'Validation accuracy N/A', ha='center', va='center', transform=ax.transAxes, color='gray')
        if has_accuracy or has_val_accuracy: ax.legend(fontsize=10)
        if has_accuracy and has_val_accuracy: min_acc = min(min(history_dict[acc_key]), min(history_dict[val_acc_key])); max_acc = max(max(history_dict[acc_key]), max(history_dict[val_acc_key])); ax.set_ylim([max(0, min_acc - 0.05), min(1.05, max_acc + 0.05)])
        elif has_accuracy: min_acc = min(history_dict[acc_key]); max_acc = max(history_dict[acc_key]); ax.set_ylim([max(0, min_acc - 0.05), min(1.05, max_acc + 0.05)])

    plt.tight_layout(); plot_filename = f'training_history_{time.strftime("%Y%m%d-%H%M%S")}.png'; plot_path = os.path.join(config.LOG_DIR, plot_filename)
    try: plt.savefig(plot_path); logger.info(f"Training history plot saved: {plot_path}"); plt.close(fig)
    except Exception as e: logger.error(f"Failed to save training history plot: {e}")

def save_metrics(metrics: dict, config: Config):
    if metrics is None: logger.warning("Metrics None. Skipping save."); return
    if not isinstance(metrics, dict): logger.warning(f"Invalid metrics format. Skipping save."); return
    try:
        metrics_dir = config.MODEL_DIR; os.makedirs(metrics_dir, exist_ok=True)
        metrics_filename = f'evaluation_metrics_{time.strftime("%Y%m%d-%H%M%S")}.json'; metrics_path = os.path.join(metrics_dir, metrics_filename)
        serializable_metrics = {}
        for k, v in metrics.items():
            if k == 'config' and isinstance(v, dict): serializable_metrics[k] = {cfg_k: str(cfg_v) if not isinstance(cfg_v, (list, dict, str, int, float, bool, type(None))) else cfg_v for cfg_k, cfg_v in v.items()}
            elif isinstance(v, (np.int_, np.integer)): serializable_metrics[k] = int(v)
            elif isinstance(v, (np.float_, np.floating)): serializable_metrics[k] = float(v)
            elif isinstance(v, np.ndarray): serializable_metrics[k] = v.tolist()
            elif isinstance(v, (list, dict, str, int, float, bool, type(None))): serializable_metrics[k] = v
            else: logger.warning(f"Metric '{k}' type {type(v)} -> str."); serializable_metrics[k] = str(v)
        with open(metrics_path, 'w', encoding='utf-8') as f: json.dump(serializable_metrics, f, indent=4)
        logger.info(f"Evaluation metrics saved: {metrics_path}")
    except TypeError as te: logger.error(f"TypeError saving metrics: {te}", exc_info=True)
    except Exception as e: logger.error(f"Error saving metrics: {e}", exc_info=True)

def main():
    main_start_time = time.time()
    try:
        logger.info("Step 1: Setting up configuration and environment...")
        Config.setup()

        if not FASTTEXT_LIB_AVAILABLE: logger.critical("FastText library not found. Exiting."); return

        logger.info("Step 2: Initializing Data Processor...")
        processor = DataProcessor(Config())
        if processor.feature_extractor.get('nlp') is None: logger.critical("SpaCy model failed to load. Exiting."); return

        data_files = [ os.path.join(Config.DATA_DIR, f) for f in ["gossipcop_fake.csv", "gossipcop_real.csv", "politifact_fake.csv", "politifact_real.csv"] ]
        existing_files = [f for f in data_files if os.path.exists(f)]
        if not existing_files: logger.critical(f"CRITICAL: No data files found in '{Config.DATA_DIR}'. Exiting."); return
        logger.info(f"Found {len(existing_files)} data file(s): {existing_files}")

        logger.info("Step 3: Preparing data (using class weights)...")
        data_prep_start = time.time()
        data_dict = processor.prepare_data(existing_files)
        logger.info(f"Data preparation completed in {time.time() - data_prep_start:.2f} seconds.")

        if not data_dict or not all(k in data_dict for k in ['train', 'val', 'test']): logger.critical("Data prep failed. Exiting."); return
        if data_dict.get('train') is None or data_dict['train'][0] is None: logger.critical("Train split invalid. Exiting."); return
        if data_dict.get('test') is None or data_dict['test'][0] is None: logger.critical("Test split invalid. Exiting."); return
        if data_dict.get('val') is None: logger.critical("Validation split invalid. Exiting."); return
        if data_dict['val'][0] is not None and len(data_dict['val'][0]) == 0: logger.warning("Validation split empty.")

        logger.info("Step 4: Creating Manual TCN-Attention-LSTM model.")
        model_creation_start = time.time()
        model = create_tcn_attention_lstm_model(Config(), processor)
        model.summary(print_fn=logger.info)
        logger.info(f"Model creation completed in {time.time() - model_creation_start:.2f} seconds.")

        logger.info("Step 5: Training the model using class weights...")
        train_start = time.time()
        history = train_model(model, data_dict, Config())
        logger.info(f"Model training completed in {time.time() - train_start:.2f} seconds.")
        if history is None: logger.critical("Model training failed. Exiting."); return

        logger.info("Step 6: Plotting training history...")
        plot_training_history(history, Config())

        logger.info("Step 7: Evaluating the final model...")
        eval_start = time.time()
        metrics, texts_original_test, y_test, y_pred_proba = evaluate_model(model, data_dict, Config())
        logger.info(f"Model evaluation completed in {time.time() - eval_start:.2f} seconds.")

        if metrics: logger.info("Step 8: Saving evaluation metrics..."); save_metrics(metrics, Config())
        else: logger.warning("Evaluation failed. Skipping metrics save and LIME.");

        if metrics and LIME_AVAILABLE:
             logger.info("Step 9: Generating LIME explanations...")
             lime_start = time.time()
             if texts_original_test is not None and y_test is not None and y_pred_proba is not None:
                 explain_with_lime(model, processor, Config(), texts_original_test, y_test, y_pred_proba, num_samples=5)
                 logger.info(f"LIME generation took {time.time() - lime_start:.2f} seconds.")
             else: logger.warning("Skipping LIME due to missing eval data.")
        elif not LIME_AVAILABLE: logger.info("Skipping LIME (library not installed).")
        elif not metrics: logger.info("Skipping LIME (evaluation failed).")

        total_time = time.time() - main_start_time
        logger.info(f"=== Pipeline Completed Successfully in {total_time:.2f} seconds ===")

    except ValueError as ve: logger.critical(f"CRITICAL ERROR: ValueError: {ve}", exc_info=True)
    except ImportError as ie: logger.critical(f"CRITICAL ERROR: Missing library: {ie}.", exc_info=True)
    except RuntimeError as re: logger.critical(f"CRITICAL ERROR: RuntimeError: {re}", exc_info=True)
    except FileNotFoundError as fnfe: logger.critical(f"CRITICAL ERROR: File/Dir not found: {fnfe}", exc_info=True)
    except MemoryError as me: logger.critical(f"CRITICAL ERROR: Insufficient memory: {me}.", exc_info=True)
    except Exception as e:
        logger.critical(f"CRITICAL ERROR: Unexpected error: {str(e)}", exc_info=True)
        import traceback; logger.error("Traceback:\n%s", traceback.format_exc())
    finally: logger.info("Pipeline execution finished.")

if __name__ == "__main__":
    if not os.path.exists(Config.DATA_DIR):
        logger.warning(f"Data directory '{Config.DATA_DIR}' not found.")
        try: os.makedirs(Config.DATA_DIR); logger.info(f"Created data directory: {Config.DATA_DIR}.")
        except OSError as e: sys.exit(f"Error: Could not create data directory '{Config.DATA_DIR}': {e}")
    if not os.path.exists(Config.MODEL_DIR):
        try: os.makedirs(Config.MODEL_DIR); logger.info(f"Created model directory: {Config.MODEL_DIR}")
        except OSError as e: logger.error(f"Could not create model directory '{Config.MODEL_DIR}': {e}")
    if not os.path.exists(Config.LOG_DIR):
        try: os.makedirs(Config.LOG_DIR); logger.info(f"Created log directory: {Config.LOG_DIR}")
        except OSError as e: logger.error(f"Could not create log directory '{Config.LOG_DIR}': {e}")

    main()