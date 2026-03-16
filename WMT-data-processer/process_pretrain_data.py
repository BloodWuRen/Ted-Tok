import datasets
import extraction
import os
import argparse
from copy import deepcopy

def create_huggingface_dataset_from_wmt(archive_files: list, deduplicated_keys_file: str, output_raw_path: str = None):
    print("deduplicating WMT docs...")
    wmt_docs = extraction.get_deduplicated_wmt_docs(
        wmt_archive_files=archive_files,
        deduplicated_sorting_keys_file=deduplicated_keys_file,
    )
    
    print("split docs into sentence chunks...")
    wmt_passages = extraction.get_wmt_passages_from_docs(
        wmt_docs=wmt_docs,
        prepend_date=False,
    )
    
    data_list = []
    text_list = []
    for passage in wmt_passages:
        text = passage.text.decode('utf-8')
        data_list.append({
            "id": passage.id,
            "text": text,
        })
        text_list.append(text)
    
    if output_raw_path:
        print("Printing original text to file...")
        with open(output_raw_path, 'w') as f:
            for text in text_list:
                f.write(text + '\n')
        print(f"Original text saved to {output_raw_path}")
        
    print(f"Processed data length: {len(data_list)}")

    print("Creating Hugging Face Dataset...")
    features = datasets.Features({
        "id": datasets.Value("string"),
        "text": datasets.Value("string"),
    })
    dataset = datasets.Dataset.from_list(data_list, features=features)
    
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process WMT pretraining data.")
    parser.add_argument("--start_year", type=int, default=2007,
                        help="Start year for document extraction (inclusive)")
    parser.add_argument("--end_year", type=int, default=2007,
                        help="End year for document extraction (inclusive)")
    parser.add_argument("--meta_archive_path", type=str, default='/data/huangjiameng/data/streamingQA/news-docs.{year}.en.filtered.gz',
                        help="Pattern for the input archive files, e.g. '/path/to/news-docs.2007.en.filtered.gz'")
    parser.add_argument("--deduplicated_keys_path", type=str, default='/data/huangjiameng/data/streamingQA/wmt_sorting_key_ids.txt.gz',
                        help="Path to wmt_sorting_key_ids.txt.gz")

if __name__ == "__main__":
    args = parser.parse_args()

    filenames = []
    for year in range(args.start_year, args.end_year+1):
        filename = args.meta_archive_path.format(year=year)
        filenames.append(filename)


    out_dir = f"/data/zhangzhi/data/local_datasets"
    os.makedirs(out_dir, exist_ok=True)
    output_raw_path = os.path.join(out_dir, f'wmt_{args.start_year}-{args.end_year}.raw')

    wmt_dataset = create_huggingface_dataset_from_wmt(filenames, args.deduplicated_keys_path, output_raw_path)
    
    print(f"Dataset size: {len(wmt_dataset)}")
    print("Dataset overview:")
    print(wmt_dataset)

    dataset_dir = os.path.join(out_dir, f'wmt_{args.start_year}-{args.end_year}')
    print(f"\n--- save dataset to local dir:{dataset_dir} ---")
    wmt_dataset.save_to_disk(dataset_dir)
    
    print("\n--- laod for double check ---")
    reloaded_dataset = datasets.load_from_disk(dataset_dir)
    
    print(reloaded_dataset)
    print("\nFist 2 :")
    for i in range(min(2, len(reloaded_dataset))):
        example = reloaded_dataset[i]
        print(f"id: {example['id']}")
        print(f"text: {example['text']}")
        print("-" * 20)
