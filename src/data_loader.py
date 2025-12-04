"""
data_loader.py
Functions to download and load Stanford Politeness Corpus
"""

import pandas as pd
import json
import urllib.request
import zipfile
import os
from src import config


def download_official_corpus(save_path=None):
    """
    Download Stanford Politeness Corpus from Cornell servers
    
    Args:
        save_path (str): Directory to save raw data. 
                        Defaults to config.RAW_DATA_DIR
    
    Returns:
        tuple: (wiki_data, se_data) as pandas DataFrames
    """
    if save_path is None:
        save_path = config.RAW_DATA_DIR
    
    # Ensure directory exists
    os.makedirs(save_path, exist_ok=True)
    
    datasets = {
        'wikipedia': config.WIKI_CORPUS_ID,
        'stack-exchange': config.SE_CORPUS_ID
    }
    
    all_data = {}
    
    for name, dataset_id in datasets.items():
        print(f"\n📦 Downloading {name.title()} Politeness Corpus...")
        
        try:
            # Download the zip file
            zip_url = f"{config.CONVOKIT_BASE_URL}{dataset_id}/{dataset_id}.zip"
            zip_path = os.path.join(save_path, f"{dataset_id}.zip")
            extract_path = os.path.join(save_path, dataset_id)
            
            print(f"   Downloading from: {zip_url}")
            urllib.request.urlretrieve(zip_url, zip_path)
            print(f"   ✅ Downloaded zip file")
            
            # Extract it
            print(f"   Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print(f"   ✅ Extracted files")
            
            # Find and read the utterances file
            utterances_file = os.path.join(extract_path, dataset_id, "utterances.jsonl")
            
            if os.path.exists(utterances_file):
                print(f"   Reading utterances...")
                data_list = []
                
                with open(utterances_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            item = json.loads(line.strip())
                            text = item.get('text', '')
                            meta = item.get('meta', {})
                            score = meta.get('Normalized Score', 0.5)
                            
                            if text:
                                data_list.append({
                                    'Request': text,
                                    'Normalized Score': score
                                })
                        except json.JSONDecodeError:
                            continue
                
                df = pd.DataFrame(data_list)
                all_data[name] = df
                print(f"   ✅ Loaded {len(df)} samples from {name}")
            else:
                print(f"   ⚠️ Utterances file not found")
                all_data[name] = pd.DataFrame()
            
        except Exception as e:
            print(f"   ❌ Error downloading {name}: {e}")
            all_data[name] = pd.DataFrame()
    
    wiki_data = all_data.get('wikipedia', pd.DataFrame())
    se_data = all_data.get('stack-exchange', pd.DataFrame())
    
    return wiki_data, se_data

def download_new_politeness_data():
    """
    Downloads the new politeness dataset from the Tag-and-Generate GitHub repository.
    
    Returns:
        pd.DataFrame: A DataFrame containing 'Utterance' (raw text) and 
                      'Normalized Score' (politeness score 0-1) from the new corpus, 
                      or an empty DataFrame on failure.
    """
    
    # URL for the raw JSON file from the GitHub repository
    URL = "https://raw.githubusercontent.com/tag-and-generate/politeness-dataset/master/TagAndGenerateData/politeness_data.json"
    print("\n📦 Downloading New Politeness Dataset (Tag and Generate Corpus)...")
    
    try:
        # Fetch the JSON content
        with urllib.request.urlopen(URL) as url:
            # Note: This loads the entire JSON file into memory
            data = json.load(url)
        
        print(f"   ✅ Downloaded {len(data)} samples.")
        
        # Transform into a DataFrame, mapping 'text' to 'Utterance' 
        # and 'politeness_score' to 'Normalized Score'
        new_data_df = pd.DataFrame([
            {
                'Utterance': item['text'],
                'Normalized Score': item['politeness_score']
            }
            for item in data
        ])
        
        print("   ✅ New data loaded into DataFrame.")
        return new_data_df
    
    except Exception as e:
        print(f"   ❌ Error downloading or parsing new data: {e}")
        return pd.DataFrame()

def load_processed_data(file_path=None):
    """
    Load preprocessed politeness data
    
    Args:
        file_path (str): Path to processed CSV. 
                        Defaults to config.PROCESSED_DATA_FILE
    
    Returns:
        pd.DataFrame: Processed data with all features
    """
    if file_path is None:
        file_path = config.PROCESSED_DATA_FILE
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Processed data not found at {file_path}. "
            f"Please run preprocessing first."
        )
    
    return pd.read_csv(file_path)


def save_processed_data(df, file_path=None):
    """
    Save processed data to CSV
    
    Args:
        df (pd.DataFrame): Processed dataframe to save
        file_path (str): Path to save CSV. 
                        Defaults to config.PROCESSED_DATA_FILE
    """
    if file_path is None:
        file_path = config.PROCESSED_DATA_FILE
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    df.to_csv(file_path, index=False)
    print(f"💾 Saved processed data to: {file_path}")

if __name__ == "__main__":
    print("Running data_loader.py directly for testing...")
    
    try:
        # Attempt to download the official corpus
        wiki_data, se_data = download_official_corpus()
        
        # Attempt to download the new corpus
        new_data = download_new_politeness_data()
        
        # Total
        total_samples = len(wiki_data) + len(se_data) + len(new_data)
        print(f"\n✅ All corpora downloaded successfully, totaling {total_samples} raw samples.")
        
    except Exception as e:
        print(f"\n❌ A critical error occurred during download: {e}")