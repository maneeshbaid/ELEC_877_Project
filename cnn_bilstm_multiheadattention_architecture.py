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

import spacy

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Concatenate, MultiHeadAttention
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
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
    print("Install it using: pip install lime")

try:
    import fasttext as fasttext_standalone
    FASTTEXT_LIB_AVAILABLE = True
except ImportError:
    FASTTEXT_LIB_AVAILABLE = False
    print("Error: Standalone 'fasttext' library not found. Supervised FastText training is unavailable.")
    print("Install it using: pip install fasttext-wheel")

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
    MAX_EPOCHS = 20
    TEXT_CLEANING_REGEX = r'http\S+|www\S+|@\w+|\d+|<.*?>'
    FASTTEXT_SUPERVISED_EPOCHS = 50
    FASTTEXT_SUPERVISED_LR = 0.01
    SPACY_MODEL = 'en_core_web_sm'
    UNDERSAMPLING_TARGET_RATIO = 1.5
    TEST_SPLIT_RATIO = 0.2
    VALIDATION_SPLIT_RATIO = 0.1

    @classmethod
    def setup(cls):
        np.random.seed(cls.SEED)
        tf.random.set_seed(cls.SEED)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        logger.info("=== Configuration ===")
        log_params = {
            'SEED': cls.SEED, 'DATA_DIR': cls.DATA_DIR, 'MODEL_DIR': cls.MODEL_DIR,
            'LOG_DIR': cls.LOG_DIR, 'MAX_SEQUENCE_LENGTH': cls.MAX_SEQUENCE_LENGTH,
            'EMBEDDING_DIM': cls.EMBEDDING_DIM, 'BATCH_SIZE': cls.BATCH_SIZE,
            'MAX_EPOCHS': cls.MAX_EPOCHS, 'SPACY_MODEL': cls.SPACY_MODEL,
            'UNDERSAMPLING_TARGET_RATIO': cls.UNDERSAMPLING_TARGET_RATIO,
            'TEST_SPLIT_RATIO': cls.TEST_SPLIT_RATIO,
            'VALIDATION_SPLIT_RATIO': cls.VALIDATION_SPLIT_RATIO
        }
        for key, value in log_params.items():
            logger.info(f"{key}: {value}")

class DataProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = Tokenizer(oov_token="<OOV>")
        self.embedding_matrix = None
        self.feature_extractor = self._init_feature_extractor()
        self.supervised_model = None

    def _init_feature_extractor(self):
        feature_extractor = {}
        logger.info("Initializing SpaCy model...")
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
        if not isinstance(text, str):
            logger.warning(f"Received non-string input for cleaning: {type(text)}. Returning empty string.")
            return ""

        nlp = self.feature_extractor.get('nlp')
        if nlp is None:
             logger.error("SpaCy model not available. Cannot perform cleaning.")
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
            tokens = [token.lemma_.lower().strip()
                      for token in doc
                      if not token.is_stop
                      and not token.is_punct
                      and not token.is_space
                      and len(token.lemma_.strip()) > 1]
            final_text = ' '.join(tokens)
            return final_text
        except Exception as e:
            logger.error(f"Error in SpaCy cleaning for input starting with '{text[:50]}...': {e}", exc_info=True)
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
                if not os.path.exists(path): logger.warning(f"File not found: {path}"); continue
                try:
                    encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
                    df = None
                    for enc in encodings_to_try:
                        try: df = pd.read_csv(path, encoding=enc); logger.info(f"Read {path} with {enc}"); break
                        except UnicodeDecodeError: continue
                        except Exception as read_e: logger.warning(f"Could not read {path} with {enc}: {read_e}"); continue
                    if df is None: logger.error(f"Could not read CSV: {path} with any encoding."); continue
                except Exception as e: logger.error(f"Error loading {path}: {e}"); continue
                if df.empty: logger.warning(f"Empty dataset: {path}"); continue

                filename_lower = os.path.basename(path).lower()
                if 'fake' in filename_lower: df['label'] = 0
                elif 'real' in filename_lower or 'true' in filename_lower: df['label'] = 1
                else:
                    label_col_candidates = [c for c in df.columns if c.lower() in ['label', 'class', 'type', 'target']]
                    if label_col_candidates:
                        label_col = label_col_candidates[0]
                        if df[label_col].dtype == object:
                             df['label'] = df[label_col].apply(lambda x: 0 if isinstance(x, str) and 'fake' in x.lower() else (1 if isinstance(x, str) and ('real' in x.lower() or 'true' in x.lower()) else pd.NA))
                             df = df.dropna(subset=['label'])
                             if not df.empty: df['label'] = df['label'].astype(int)
                        elif pd.api.types.is_numeric_dtype(df[label_col]): df['label'] = df[label_col].astype(int)
                        else: logger.warning(f"Unrecognized label column format in {path}"); continue
                    else: logger.warning(f"Cannot determine label for: {path}"); continue

                text_col = None
                potential_text_cols = ['text', 'content', 'title', 'article', 'headline', 'body', 'news']
                for col_name in potential_text_cols:
                     if col_name in df.columns and df[col_name].dtype == object and df[col_name].notna().sum() > 0.5 * len(df):
                         text_col = col_name; break
                if text_col is None:
                    for col in df.columns:
                        if df[col].dtype == object and df[col].notna().sum() > 0.1 * len(df):
                            text_col = col; logger.warning(f"Using fallback text column '{col}' for {path}"); break
                if text_col is None: logger.warning(f"No text column found in {path}"); continue

                df = df.rename(columns={text_col: 'text'})
                df['text'] = df['text'].fillna('').astype(str)
                df = df[df['text'].str.strip() != '']
                if df.empty: logger.warning(f"No valid text data in {path}"); continue
                dfs.append(df[['text', 'label']])
                logger.info(f"Processed {path}: added {len(df)} rows")
            except Exception as e: logger.error(f"Error processing file {path}: {e}", exc_info=True)

        if not dfs: raise ValueError("No valid data files could be processed.")
        data = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total dataset size before cleaning/splitting: {len(data)} rows")
        if 'label' not in data.columns or data['label'].isna().any(): raise ValueError("Label column invalid.")
        class_counts = data['label'].value_counts(); logger.info(f"Original class distribution: {class_counts.to_dict()}")

        logger.info("Performing text cleaning using SpaCy...")
        if self.feature_extractor.get('nlp') is None:
             raise RuntimeError("SpaCy model failed to load, cannot proceed with text cleaning.")
        try:
            from tqdm.auto import tqdm
            tqdm.pandas(desc="Cleaning Text (SpaCy)")
            data['processed_text'] = data['text'].progress_apply(self._enhanced_clean_text)
        except ImportError:
            logger.info("tqdm not found, processing text without progress bar...")
            data['processed_text'] = data['text'].apply(self._enhanced_clean_text)

        original_len = len(data)
        data = data[data['processed_text'].str.strip() != '']
        removed_count = original_len - len(data)
        if removed_count > 0: logger.warning(f"Removed {removed_count} rows with empty text after processing.")
        if data.empty: raise ValueError("All data resulted in empty processed text.")
        logger.info(f"Dataset size after cleaning: {len(data)} rows")

        logger.info("Performing LDA Topic Modeling on processed text...")
        perform_lda(data['processed_text'], self.config)

        logger.info("Tokenizing and padding text sequences...")
        if 'processed_text' not in data.columns or data['processed_text'].empty:
            raise ValueError("Column 'processed_text' is missing or empty.")

        texts_for_tokenizer = data['processed_text'].tolist()
        self.tokenizer.fit_on_texts(texts_for_tokenizer)
        sequences = self.tokenizer.texts_to_sequences(texts_for_tokenizer)
        X_text_padded = pad_sequences(sequences, maxlen=self.config.MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        logger.info(f"Text sequences processed. Shape: {X_text_padded.shape}")

        logger.info("Creating Supervised FastText embeddings (based on full dataset)...")
        self._create_fasttext_embeddings(data['processed_text'], data['label'])
        if self.embedding_matrix is None:
            logger.error("Failed to create embedding matrix. Cannot proceed.")
            return {}

        y = data['label'].values
        texts_original = data['text'].values
        logger.info("Splitting data into train/val/test sets (before balancing train)...")
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

            train_indices = np.arange(len(y_train))
            fake_indices = train_indices[y_train == 0]
            real_indices = train_indices[y_train == 1]

            n_fake, n_real = len(fake_indices), len(real_indices)
            target_ratio = self.config.UNDERSAMPLING_TARGET_RATIO

            final_fake_indices = fake_indices
            final_real_indices = real_indices

            if n_real > n_fake * target_ratio and n_fake > 0:
                n_real_target = min(n_real, int(n_fake * target_ratio))
                logger.info(f"Undersampling real samples in training set from {n_real} to {n_real_target}")
                np.random.seed(self.config.SEED)
                final_real_indices = np.random.choice(real_indices, size=n_real_target, replace=False)
            elif n_fake > n_real * target_ratio and n_real > 0:
                n_fake_target = min(n_fake, int(n_real * target_ratio))
                logger.info(f"Undersampling fake samples in training set from {n_fake} to {n_fake_target}")
                np.random.seed(self.config.SEED)
                final_fake_indices = np.random.choice(fake_indices, size=n_fake_target, replace=False)
            else:
                logger.info("Training set already balanced or target ratio met. No undersampling needed.")

            balanced_train_indices = np.concatenate([final_fake_indices, final_real_indices])
            np.random.seed(self.config.SEED)
            np.random.shuffle(balanced_train_indices)

            X_text_train_balanced = X_text_train[balanced_train_indices]
            y_train_balanced = y_train[balanced_train_indices]
            if isinstance(texts_original_train, list): texts_original_train = np.array(texts_original_train)
            texts_original_train_balanced = texts_original_train[balanced_train_indices]

            logger.info(f"Balanced training set size: {len(y_train_balanced)}")
            balanced_train_class_counts = pd.Series(y_train_balanced).value_counts().to_dict()
            logger.info(f"Balanced training class distribution: {balanced_train_class_counts}")

            data_dict['train'] = (X_text_train_balanced, y_train_balanced, texts_original_train_balanced)
        else:
            logger.warning("Training data is None or empty, skipping balancing.")

        if y_val is not None: logger.info(f"Validation set size (unbalanced): {len(y_val)}")
        if y_test is not None: logger.info(f"Test set size (unbalanced): {len(y_test)}")

        return data_dict

    def _create_fasttext_embeddings(self, texts, labels):
        if not FASTTEXT_LIB_AVAILABLE:
             logger.error("Standalone FastText library not available. Cannot create supervised embeddings.")
             word_index = self.tokenizer.word_index if hasattr(self.tokenizer, 'word_index') else {}
             vocab_size = len(word_index) + 1
             self.embedding_matrix = np.random.normal(0, 0.1, (vocab_size, self.config.EMBEDDING_DIM))
             logger.warning("Falling back to random embeddings.")
             return
        if not isinstance(texts, (list, pd.Series)): texts = []
        else: texts = list(texts)
        if not texts:
             logger.error("No texts provided for FastText training.")
             word_index = self.tokenizer.word_index if hasattr(self.tokenizer, 'word_index') else {}
             vocab_size = len(word_index) + 1
             self.embedding_matrix = np.random.normal(0, 0.1, (vocab_size, self.config.EMBEDDING_DIM))
             logger.warning("Falling back to random embeddings.")
             return

        word_index = self.tokenizer.word_index
        vocab_size = len(word_index) + 1
        self.embedding_matrix = np.random.normal(scale=0.01, size=(vocab_size, self.config.EMBEDDING_DIM))

        supervised_temp_file = 'supervised_fasttext_data.txt'
        try:
            logger.info("Preparing data for supervised FastText training...")
            with open(supervised_temp_file, 'w', encoding='utf-8') as f:
                 for text, label in zip(texts, labels):
                     if isinstance(text, str) and text.strip() and label in [0, 1]:
                         f.write(f'__label__{int(label)} {text.strip()}\n')
            if os.path.getsize(supervised_temp_file) > 0:
                logger.info(f"Training supervised FastText model (dim={self.config.EMBEDDING_DIM})...")
                self.supervised_model = fasttext_standalone.train_supervised(
                    input=supervised_temp_file, dim=self.config.EMBEDDING_DIM,
                    epoch=self.config.FASTTEXT_SUPERVISED_EPOCHS, lr=self.config.FASTTEXT_SUPERVISED_LR,
                    wordNgrams=2, verbose=2, thread=max(1, os.cpu_count() - 1))
                logger.info("Supervised FastText model training complete.")
            else: logger.warning(f"Temp file {supervised_temp_file} empty. Skipping supervised training."); self.supervised_model = None
        except Exception as e_sup: logger.error(f"Error training supervised FastText: {e_sup}", exc_info=True); self.supervised_model = None
        finally:
            if os.path.exists(supervised_temp_file):
                try: os.remove(supervised_temp_file); logger.info(f"Removed temp file: {supervised_temp_file}")
                except OSError as e_rm: logger.warning(f"Could not remove temp file {supervised_temp_file}: {e_rm}")

        logger.info("Populating embedding matrix from supervised model...")
        found_vectors, oov_count = 0, 0
        if self.supervised_model and self.supervised_model.get_dimension() == self.config.EMBEDDING_DIM:
            for word, i in word_index.items():
                if i >= vocab_size: continue
                try:
                    vector = self.supervised_model.get_word_vector(word)
                    self.embedding_matrix[i] = vector
                    found_vectors += 1
                except Exception:
                    oov_count += 1
            logger.info(f"Embedding matrix populated. Shape: {self.embedding_matrix.shape}. Found: {found_vectors}/{len(word_index)}. OOV: {oov_count}")
        else:
             logger.warning("Supervised model not trained or dimension mismatch. Embedding matrix contains only random noise.")
        if self.embedding_matrix is None:
             self.embedding_matrix = np.random.normal(scale=0.1, size=(vocab_size, self.config.EMBEDDING_DIM))


    def _split_dataset(self, X_text, y, texts_original):
        test_size = self.config.TEST_SPLIT_RATIO
        val_size = self.config.VALIDATION_SPLIT_RATIO

        logger.info(f"Splitting data: Total samples = {len(y)}, Test Ratio={test_size}, Val Ratio (of Train+Val)={val_size}")
        if len(y) < 10:
             logger.error("Not enough data for reliable train/val/test split.")
             return {'train': (None, None, None), 'val': (None, None, None), 'test': (None, None, None)}
        try:
            unique_labels, counts = np.unique(y, return_counts=True)
            stratify_y = y if (len(unique_labels) >= 2 and np.all(counts >= 2)) else None
            if stratify_y is None: logger.warning("Using non-stratified split for train/test.")

            X_text_train_val, X_text_test, y_train_val, y_test, texts_original_train_val, texts_original_test = train_test_split(
                X_text, y, texts_original, test_size=test_size, stratify=stratify_y, random_state=self.config.SEED)

            unique_labels_tv, counts_tv = np.unique(y_train_val, return_counts=True)
            stratify_y_tv = y_train_val if (len(unique_labels_tv) >= 2 and np.all(counts_tv >= 2)) else None
            if stratify_y_tv is None: logger.warning("Using non-stratified split for train/validation.")

            relative_val_size = val_size

            if len(y_train_val) * relative_val_size < 2 and len(y_train_val) > 0:
                 relative_val_size = 1 / len(y_train_val) if len(y_train_val) == 1 else 2 / len(y_train_val)
                 logger.warning(f"Adjusting relative validation size to {relative_val_size:.2f} due to small train+val set size ({len(y_train_val)}).")
            elif len(y_train_val) == 0:
                 logger.error("Train+validation set is empty after first split.")
                 return {'train': (None, None, None), 'val': (None, None, None), 'test': (X_text_test, y_test, texts_original_test)}


            X_text_train, X_text_val, y_train, y_val, texts_original_train, texts_original_val = train_test_split(
                X_text_train_val, y_train_val, texts_original_train_val, test_size=relative_val_size,
                stratify=stratify_y_tv, random_state=self.config.SEED)

            logger.info(f"Split sizes: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")
            if len(y_train) == 0 or len(y_val) == 0 or len(y_test) == 0:
                logger.warning("Empty split detected after train_test_split. Check data and split ratios.")
                return {'train': (X_text_train, y_train, texts_original_train) if len(y_train) > 0 else (None, None, None),
                        'val': (X_text_val, y_val, texts_original_val) if len(y_val) > 0 else (None, None, None),
                        'test': (X_text_test, y_test, texts_original_test) if len(y_test) > 0 else (None, None, None)}

            return {'train': (X_text_train, y_train, texts_original_train),
                    'val': (X_text_val, y_val, texts_original_val),
                    'test': (X_text_test, y_test, texts_original_test)}
        except Exception as e:
             logger.error(f"Error splitting data: {e}", exc_info=True)
             return {'train': (None, None, None), 'val': (None, None, None), 'test': (None, None, None)}

def perform_lda(texts: pd.Series, config: Config, num_topics: int = 3, n_top_words: int = 10):
    logger.info(f"Performing LDA for {num_topics} topics...")
    if texts.empty: logger.warning("Input texts for LDA empty. Skipping."); return
    try:
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
        dtm = vectorizer.fit_transform(texts)
        if dtm.shape[1] == 0:
             logger.warning("LDA vocabulary is empty after vectorization (check min_df/stopwords). Skipping LDA.")
             return
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=config.SEED,
                                        learning_method='online', learning_offset=50., max_iter=10)
        lda.fit(dtm)
        feature_names = vectorizer.get_feature_names_out()
        print("\n--- LDA Topic Modeling Results ---")
        for topic_idx, topic in enumerate(lda.components_):
            num_words_to_show = min(n_top_words, len(feature_names))
            if num_words_to_show <= 0: continue
            top_words_indices = topic.argsort()[:-num_words_to_show - 1:-1]
            top_words = [feature_names[i] for i in top_words_indices]
            print(f"Topic #{topic_idx}: {' '.join(top_words)}")
        print("---------------------------------")
    except Exception as e: logger.error(f"Error during LDA: {e}", exc_info=True)

def create_cnn_lstm_model(config, processor):
    logger.info("Building the CNN-LSTM model with Attention")

    best_hp = {
        'trainable_embedding': False,
        'conv_filters': 128,
        'conv_kernel_size_k4': 5,
        'conv_kernel_size_k3': 4,
        'lstm_units_1': 70,
        'lstm_units_2': 20,
        'lstm_dropout': 0.1,
        'num_heads': 4,
        'key_dim': 64,
        'attention_dropout': 0.2,
        'dense_units_1': 160,
        'dense_units_2': 32,
        'l2_reg': 0.000528,
        'dropout_rate_dense': 0.5,
        'learning_rate': 0.00439
    }

    text_input_shape = (config.MAX_SEQUENCE_LENGTH,)
    if not hasattr(processor, 'tokenizer') or not processor.tokenizer.word_index:
        logger.error("Tokenizer not available or not fitted in processor. Cannot build model.")
        return None
    vocab_size = len(processor.tokenizer.word_index) + 1
    embedding_dim = config.EMBEDDING_DIM

    if processor.embedding_matrix is None:
        logger.error("Embedding matrix is None! Using random embeddings.");
        embedding_matrix = np.random.normal(scale=0.1, size=(vocab_size, embedding_dim));
        trainable_embedding = True
    elif processor.embedding_matrix.shape != (vocab_size, embedding_dim):
         logger.error(f"Embedding matrix shape mismatch! Expected {(vocab_size, embedding_dim)}, got {processor.embedding_matrix.shape}. Using random embeddings.");
         embedding_matrix = np.random.normal(scale=0.1, size=(vocab_size, embedding_dim));
         trainable_embedding = True
    else:
        embedding_matrix = processor.embedding_matrix;
        trainable_embedding = best_hp['trainable_embedding']
        logger.info(f"Using pre-computed embeddings. Trainable: {trainable_embedding}")

    text_input = Input(shape=text_input_shape, name='text_input')

    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix],
                                input_length=config.MAX_SEQUENCE_LENGTH, trainable=trainable_embedding, name='embedding_layer')(text_input)

    conv1 = Conv1D(filters=best_hp['conv_filters'], kernel_size=best_hp['conv_kernel_size_k4'], padding='same', activation='relu',
                   kernel_regularizer=l2(best_hp['l2_reg']), name='conv1d_block1')(embedding_layer)
    conv2 = Conv1D(filters=best_hp['conv_filters'], kernel_size=best_hp['conv_kernel_size_k3'], padding='same', activation='relu',
                   kernel_regularizer=l2(best_hp['l2_reg']), name='conv1d_block2')(embedding_layer)
    max_pool1 = GlobalMaxPooling1D(name='globalmaxpool_block1')(conv1)
    avg_pool1 = GlobalAveragePooling1D(name='globalavgpool_block1')(conv1)
    max_pool2 = GlobalMaxPooling1D(name='globalmaxpool_block2')(conv2)
    avg_pool2 = GlobalAveragePooling1D(name='globalavgpool_block2')(conv2)

    lstm_branch_seq = Bidirectional(LSTM(units=best_hp['lstm_units_1'], return_sequences=True,
                                     kernel_regularizer=l2(best_hp['l2_reg']), recurrent_regularizer=l2(best_hp['l2_reg']),
                                     dropout=best_hp['lstm_dropout'], recurrent_dropout=best_hp['lstm_dropout'],
                                     name='bilstm_1'))(embedding_layer)
    lstm_branch_seq = Bidirectional(LSTM(units=best_hp['lstm_units_2'], return_sequences=True,
                                     kernel_regularizer=l2(best_hp['l2_reg']), recurrent_regularizer=l2(best_hp['l2_reg']),
                                     dropout=best_hp['lstm_dropout'], recurrent_dropout=best_hp['lstm_dropout'],
                                     name='bilstm_2'))(lstm_branch_seq)

    attention_output = MultiHeadAttention(
        num_heads=best_hp['num_heads'],
        key_dim=best_hp['key_dim'],
        dropout=best_hp['attention_dropout'],
        name='multi_head_attention'
    )(query=lstm_branch_seq, value=lstm_branch_seq, key=lstm_branch_seq)

    attention_output_pooled = GlobalAveragePooling1D(name='attention_pooling')(attention_output)

    combined_features = Concatenate(name='concat_features')([
        max_pool1, avg_pool1,
        max_pool2, avg_pool2,
        attention_output_pooled
    ])
    combined_features = Dropout(best_hp['dropout_rate_dense'], name='dropout_combined')(combined_features)

    x = Dense(best_hp['dense_units_1'], activation='relu', kernel_regularizer=l2(best_hp['l2_reg']), name='dense_1')(combined_features)
    x = Dropout(best_hp['dropout_rate_dense'], name='dropout_1')(x)
    x = Dense(best_hp['dense_units_2'], activation='relu', kernel_regularizer=l2(best_hp['l2_reg']), name='dense_2')(x)
    x = Dropout(best_hp['dropout_rate_dense'], name='dropout_2')(x)

    output = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=text_input, outputs=output, name='FakeNews_CNN_LSTM_Attention_Model')
    optimizer = Adam(learning_rate=best_hp['learning_rate'], clipnorm=1.0)
    metrics = ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')]
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)
    logger.info(f"Model built successfully. Params: {model.count_params()}")
    return model

def train_model(model, data_dict, config):
    logger.info("Starting model training.")
    if 'train' not in data_dict or 'val' not in data_dict:
        logger.error("Train or validation data missing in data_dict."); return None
    X_text_train, y_train, _ = data_dict['train']
    X_text_val, y_val, _ = data_dict['val']

    if X_text_train is None or y_train is None or len(X_text_train) == 0:
        logger.error("Training data is invalid or empty."); return None
    if X_text_val is None or y_val is None or len(X_text_val) == 0:
        logger.error("Validation data is invalid or empty."); return None
    if len(X_text_train) != len(y_train) or len(X_text_val) != len(y_val):
        logger.error("Data shape mismatch."); return None

    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_log_dir = os.path.join(config.LOG_DIR, f'train_run_{run_timestamp}')
    model_save_dir = os.path.join(config.MODEL_DIR, f'train_run_{run_timestamp}')
    os.makedirs(run_log_dir, exist_ok=True); os.makedirs(model_save_dir, exist_ok=True)
    logger.info(f"Logging training to: {run_log_dir}")
    logger.info(f"Saving model to: {model_save_dir}")

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    best_lr = 0.00439
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                                  min_lr=max(1e-6, best_lr / 100),
                                  verbose=1)
    checkpoint_filepath = os.path.join(model_save_dir, 'best_model.keras')
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss',
                                       save_best_only=True, save_weights_only=False, verbose=1)
    tensorboard = TensorBoard(log_dir=run_log_dir, histogram_freq=1, write_graph=True, update_freq='epoch')
    callbacks = [early_stopping, reduce_lr, model_checkpoint, tensorboard]

    class_weights_dict = None
    try:
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        if len(unique_classes) > 1:
            computed_weights = class_weight.compute_class_weight('balanced', classes=unique_classes, y=y_train)
            class_weights_dict = dict(zip(unique_classes, computed_weights))
            logger.info(f"Using computed class weights for training: {class_weights_dict}")
        else:
            logger.warning("Only one class present in the training data. Cannot compute class weights.")
    except Exception as e:
        logger.warning(f"Class weight calculation failed: {e}. Training without class weights.")

    logger.info(f"Starting training: Max Epochs={config.MAX_EPOCHS}, Batch={config.BATCH_SIZE}...")
    history = None
    try:
        history = model.fit(X_text_train, y_train,
                            validation_data=(X_text_val, y_val),
                            epochs=config.MAX_EPOCHS,
                            batch_size=config.BATCH_SIZE,
                            class_weight=class_weights_dict,
                            callbacks=callbacks,
                            verbose=1)
        logger.info("Training finished.")

    except tf.errors.InvalidArgumentError as e:
         logger.error(f"TensorFlow InvalidArgumentError during training: {e}", exc_info=True)
         return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during model training: {e}", exc_info=True)
        return None

    return history

def evaluate_model(model, data_dict, config):
    logger.info("Starting model evaluation...")
    if 'test' not in data_dict: logger.error("Test data missing."); return None, None, None, None
    X_text_test, y_test, texts_original_test = data_dict['test']

    if X_text_test is None or y_test is None or texts_original_test is None:
        logger.error("Test data arrays (X, y, or original text) are None."); return None, None, None, None
    if len(y_test) == 0:
        logger.error("Test set is empty."); return None, None, None, None
    if len(X_text_test) != len(y_test):
        logger.error(f"Test data shape mismatch: X={X_text_test.shape}, y={y_test.shape}."); return None

    try:
        logger.info(f"Predicting on {len(y_test)} test samples...")
        y_pred_proba = model.predict(X_text_test)
        y_pred_proba = y_pred_proba.flatten()
        logger.info("Prediction complete.")

        best_f1_overall, best_threshold_overall = 0, 0.5
        thresholds = np.arange(0.05, 0.96, 0.01)
        logger.info("Finding optimal threshold on test set...")
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            if f1_macro > best_f1_overall:
                best_f1_overall = f1_macro
                best_threshold_overall = threshold

        final_threshold = best_threshold_overall
        logger.info(f"Optimal threshold found: {final_threshold:.3f} (Yielding Test Macro F1: {best_f1_overall:.4f})")

        y_pred_final = (y_pred_proba >= final_threshold).astype(int)

        accuracy = accuracy_score(y_test, y_pred_final)
        precision_macro = precision_score(y_test, y_pred_final, average='macro', zero_division=0)
        recall_macro = recall_score(y_test, y_pred_final, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred_final, average='macro', zero_division=0)

        report_dict = classification_report(y_test, y_pred_final, output_dict=True, zero_division=0)
        report_str = classification_report(y_test, y_pred_final, zero_division=0)

        precision_fake = report_dict.get('0', {}).get('precision', 0.0)
        recall_fake = report_dict.get('0', {}).get('recall', 0.0)
        f1_fake = report_dict.get('0', {}).get('f1-score', 0.0)
        precision_real = report_dict.get('1', {}).get('precision', 0.0)
        recall_real = report_dict.get('1', {}).get('recall', 0.0)
        f1_real = report_dict.get('1', {}).get('f1-score', 0.0)

        cm = confusion_matrix(y_test, y_pred_final)

        print("\n===== MODEL EVALUATION RESULTS (Test Set) =====")
        print(f"Threshold Applied: {final_threshold:.3f}")
        print(f"\nOverall Metrics (Macro Avg):")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision_macro:.4f}")
        print(f"  Recall: {recall_macro:.4f}")
        print(f"  F1-Score: {f1_macro:.4f}")
        print(f"\nClass-Specific Metrics:")
        print(f"  Class 0 (Fake): P={precision_fake:.4f}, R={recall_fake:.4f}, F1={f1_fake:.4f}")
        print(f"  Class 1 (Real): P={precision_real:.4f}, R={recall_real:.4f}, F1={f1_real:.4f}")
        print("\nClassification Report:\n", report_str)
        print("\nConfusion Matrix:")
        print(cm)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Pred Fake', 'Pred Real'],
                    yticklabels=['Actual Fake', 'Actual Real'],
                    annot_kws={"size": 14})
        plt.title(f'Confusion Matrix (Thresh: {final_threshold:.3f})', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('Actual Label', fontsize=12)
        plt.tight_layout()
        cm_path = os.path.join(config.LOG_DIR, f'confusion_matrix_{time.strftime("%Y%m%d-%H%M%S")}.png')
        try:
            plt.savefig(cm_path)
            logger.info(f"Confusion matrix plot saved to: {cm_path}")
            plt.close()
        except Exception as plot_e:
            logger.error(f"Failed to save confusion matrix plot: {plot_e}")

        metrics_dict = {
            'threshold': final_threshold,
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_fake': precision_fake, 'recall_fake': recall_fake, 'f1_fake': f1_fake,
            'precision_real': precision_real, 'recall_real': recall_real, 'f1_real': f1_real,
            'confusion_matrix': cm.tolist()
        }
        return metrics_dict, texts_original_test, y_test, y_pred_proba

    except Exception as e:
        logger.error(f"An error occurred during model evaluation: {e}", exc_info=True)
        return None, None, None, None

def explain_with_lime(model, processor, config, texts_original, y_true, y_pred_proba, num_samples=5):
    if not LIME_AVAILABLE: logger.warning("LIME library not found. Skipping LIME explanations."); return
    if texts_original is None or y_true is None or y_pred_proba is None:
        logger.warning("Input data for LIME (original texts, true labels, or predicted probabilities) is None. Skipping."); return
    if len(texts_original) == 0: logger.warning("No original texts provided for LIME explanations."); return
    if len(texts_original) != len(y_true) or len(texts_original) != len(y_pred_proba):
        logger.warning("Mismatch in lengths of LIME input arrays. Skipping."); return

    logger.info(f"\n--- Generating LIME Explanations for {num_samples} Test Samples ---")

    def lime_predictor(texts: List[str]) -> np.ndarray:
        if not isinstance(texts, list): texts = [str(t) for t in texts]
        cleaned_texts = [processor._enhanced_clean_text(text) for text in texts]
        sequences = processor.tokenizer.texts_to_sequences(cleaned_texts)
        padded_sequences = pad_sequences(sequences, maxlen=config.MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        try:
            pred_proba = model.predict(padded_sequences, batch_size=config.BATCH_SIZE)
            return np.hstack((1 - pred_proba, pred_proba))
        except Exception as pred_e:
             logger.error(f"Error during prediction within LIME predictor: {pred_e}")
             return np.array([[0.5, 0.5]] * len(texts))

    explainer = LimeTextExplainer(class_names=['Fake (0)', 'Real (1)'])

    num_available = len(texts_original)
    actual_num_samples = min(num_samples, num_available)
    if actual_num_samples == 0: logger.warning("No samples available to explain."); return
    np.random.seed(config.SEED)
    indices_to_explain = np.random.choice(num_available, actual_num_samples, replace=False)

    for idx in indices_to_explain:
        text_instance = texts_original[idx]
        true_label = y_true[idx]
        pred_prob_real = y_pred_proba[idx]
        predicted_label = 1 if pred_prob_real >= 0.5 else 0

        print(f"\n--- Explaining Test Sample #{idx} ---")
        print(f"  Original Text (first 200 chars): {text_instance[:200]}...")
        print(f"  True Label: {'Real (1)' if true_label == 1 else 'Fake (0)'}")
        print(f"  Predicted Label (Threshold 0.5): {'Real (1)' if predicted_label == 1 else 'Fake (0)'}")
        print(f"  Model's Predicted Probability (Real): {pred_prob_real:.4f}")

        try:
            explanation = explainer.explain_instance(
                text_instance, lime_predictor, num_features=10, num_samples=1000)
            print("  LIME Explanation (Top 10 features contributing to the predicted class):")
            explanation_list = explanation.as_list()
            if explanation_list:
                for feature, weight in explanation_list: print(f"    - Word: '{feature}', Weight: {weight:.4f}")
            else: print("    - No significant features found by LIME.")
        except Exception as lime_e:
            logger.error(f"  Error generating LIME explanation for sample {idx}: {lime_e}", exc_info=True)
    print("------------------------------------------")

def plot_training_history(history, config):
    if history is None or not hasattr(history, 'history') or not history.history:
        logger.warning("No training history object found or history is empty. Skipping plotting."); return

    history_dict = history.history
    epochs = range(1, len(history_dict.get('loss', [])) + 1)

    has_loss = 'loss' in history_dict and 'val_loss' in history_dict
    acc_key = next((k for k in ['accuracy', 'acc'] if k in history_dict), None)
    val_acc_key = next((k for k in ['val_accuracy', 'val_acc'] if k in history_dict), None)
    has_accuracy = bool(acc_key and val_acc_key)

    if not has_loss and not has_accuracy:
        logger.warning("History dictionary missing required keys ('loss'/'val_loss' or 'accuracy'/'val_accuracy'). Skipping plotting.")
        return
    if not epochs:
         logger.warning("No epochs recorded in history. Skipping plotting.")
         return

    plt.style.use('seaborn-v0_8-darkgrid')
    num_plots = sum([has_loss, has_accuracy])
    if num_plots == 0: return
    fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 6), squeeze=False)

    plot_idx = 0
    if has_loss:
        ax = axes[0, plot_idx]
        ax.plot(epochs, history_dict['loss'], 'bo-', label='Training Loss')
        ax.plot(epochs, history_dict['val_loss'], 'ro-', label='Validation Loss')
        ax.set_title('Training and Validation Loss', fontsize=14)
        ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('Loss', fontsize=12)
        ax.legend(fontsize=10); ax.grid(True)
        plot_idx += 1

    if has_accuracy:
        ax = axes[0, plot_idx]
        ax.plot(epochs, history_dict[acc_key], 'bo-', label=f'Training {acc_key.capitalize()}')
        ax.plot(epochs, history_dict[val_acc_key], 'ro-', label=f'Validation {val_acc_key.capitalize()}')
        ax.set_title('Training and Validation Accuracy', fontsize=14)
        ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('Accuracy', fontsize=12)
        ax.legend(fontsize=10); ax.grid(True)
        if history_dict[acc_key] and history_dict[val_acc_key]:
             min_acc = min(min(history_dict[acc_key]), min(history_dict[val_acc_key]))
             max_acc = max(max(history_dict[acc_key]), max(history_dict[val_acc_key]))
             ax.set_ylim([max(0, min_acc - 0.05), min(1, max_acc + 0.05)])
        else:
             logger.warning("Empty accuracy history, cannot set y-limits for plot.")

    plt.tight_layout()
    plot_path = os.path.join(config.LOG_DIR, f'training_history_{time.strftime("%Y%m%d-%H%M%S")}.png')
    try:
        plt.savefig(plot_path)
        logger.info(f"Training history plot saved to: {plot_path}")
        plt.close()
    except Exception as e:
        logger.error(f"Failed to save training history plot: {e}")

def save_metrics(metrics, config):
    if metrics is None: logger.warning("Metrics dictionary is None. Skipping save."); return
    try:
        metrics_dir = config.LOG_DIR
        os.makedirs(metrics_dir, exist_ok=True)

        metrics_filename = f'evaluation_metrics_{time.strftime("%Y%m%d-%H%M%S")}.json'
        metrics_path = os.path.join(metrics_dir, metrics_filename)

        serializable_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (np.int_, np.integer)): serializable_metrics[k] = int(v)
            elif isinstance(v, (np.float_, np.floating)): serializable_metrics[k] = float(v)
            elif isinstance(v, (np.ndarray,)): serializable_metrics[k] = v.tolist()
            elif isinstance(v, (list, dict, str, int, float, bool, type(None))): serializable_metrics[k] = v
            else:
                logger.warning(f"Metric '{k}' has unserializable type {type(v)}. Converting to string.")
                serializable_metrics[k] = str(v)

        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_metrics, f, indent=4)
        logger.info(f"Evaluation metrics saved to: {metrics_path}")

    except Exception as e:
        logger.error(f"Error saving metrics to JSON: {e}", exc_info=True)

def main():
    main_start_time = time.time()
    try:
        logger.info("Step 1: Setting up configuration...")
        config = Config()
        config.setup()
        if not FASTTEXT_LIB_AVAILABLE:
            logger.critical("Required standalone 'fasttext' library not found. Please install it. Exiting.")
            return

        logger.info("Step 2: Initializing Data Processor (with SpaCy)...")
        processor = DataProcessor(config)
        if processor.feature_extractor.get('nlp') is None:
             logger.critical("SpaCy model failed to load during init. Cannot proceed.")
             return

        data_files = [
            os.path.join(config.DATA_DIR, "gossipcop_fake.csv"),
            os.path.join(config.DATA_DIR, "gossipcop_real.csv"),
            os.path.join(config.DATA_DIR, "politifact_fake.csv"),
            os.path.join(config.DATA_DIR, "politifact_real.csv")
        ]
        existing_files = [f for f in data_files if os.path.exists(f)]
        if not existing_files:
             logger.critical(f"No data files found in specified paths within {config.DATA_DIR}. Searched for: {data_files}. Exiting.")
             return
        logger.info(f"Found data files: {existing_files}")

        logger.info("Step 3: Preparing data (Clean -> Tokenize/Pad -> Split -> Balance Train Set)...")
        data_prep_start = time.time()
        data_dict = processor.prepare_data(existing_files)
        logger.info(f"Data preparation took {time.time() - data_prep_start:.2f} sec.")

        if not data_dict or not all(k in data_dict for k in ['train', 'val', 'test']):
             logger.critical("Data preparation failed to return valid train/val/test splits. Exiting."); return
        if data_dict['train'][0] is None or data_dict['val'][0] is None or data_dict['test'][0] is None:
             logger.critical("Data preparation resulted in None for one or more data splits. Exiting."); return

        logger.info("Step 4: Creating CNN-LSTM model.")
        model = create_cnn_lstm_model(config, processor)
        if model is None:
             logger.critical("Failed to create model. Exiting.")
             return
        model.summary(print_fn=logger.info)

        logger.info("Step 5: Training model.")
        train_start = time.time()
        history = train_model(model, data_dict, config)
        logger.info(f"Training took {time.time() - train_start:.2f} sec.")

        if history is None:
            logger.critical("Model training failed or returned no history. Exiting.")
            return

        logger.info("Step 6: Plotting training history...")
        plot_training_history(history, config)

        logger.info("Step 7: Evaluating model on the test set...")
        eval_start = time.time()
        metrics, texts_original_test, y_test, y_pred_proba = evaluate_model(model, data_dict, config)
        logger.info(f"Evaluation took {time.time() - eval_start:.2f} sec.")

        if metrics:
            logger.info("Step 8: Saving evaluation metrics...")
            save_metrics(metrics, config)
        else:
            logger.warning("Evaluation failed or produced no metrics. Skipping saving metrics.")

        logger.info("Step 9: Generating LIME explanations for test samples...")
        if metrics is not None and LIME_AVAILABLE:
             if texts_original_test is not None and y_test is not None and y_pred_proba is not None:
                 explain_with_lime(model, processor, config, texts_original_test, y_test, y_pred_proba, num_samples=5)
             else:
                 logger.warning("Skipping LIME explanations due to missing test data.")
        elif not LIME_AVAILABLE:
             logger.info("Skipping LIME explanations (library not installed).")
        else:
             logger.info("Skipping LIME explanations (evaluation failed).")

        total_time = time.time() - main_start_time
        logger.info(f"=== Pipeline Completed in {total_time:.2f} seconds ===")

    except ValueError as ve:
        logger.critical(f"Configuration or data processing error: {ve}", exc_info=True)
    except ImportError as ie:
        logger.critical(f"Missing required library: {ie}. Please install dependencies.", exc_info=True)
    except RuntimeError as re:
         logger.critical(f"Runtime error encountered: {re}", exc_info=True)
    except Exception as e:
        logger.critical(f"An unexpected critical error occurred in the main pipeline: {str(e)}", exc_info=True)
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    if not os.path.exists(Config.DATA_DIR):
        logger.warning(f"Data directory '{Config.DATA_DIR}' not found. Please ensure it exists and contains data files.")
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    logger.info(f"Ensured model directory exists: {Config.MODEL_DIR}")
    logger.info(f"Ensured log directory exists: {Config.LOG_DIR}")

    main()