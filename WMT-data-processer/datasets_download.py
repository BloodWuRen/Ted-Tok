import os
import subprocess
import concurrent.futures
import extraction
import argparse

def download_with_wget(url, output_path):
  os.makedirs(output_path, exist_ok=True)
  filename = os.path.basename(url)
  filepath = os.path.join(output_path, filename)
  done_marker = os.path.join(output_path, f".{filename}.done")

  if os.path.exists(done_marker):
    print(f"✅ File already exists: {filepath}")
    return filepath

  try:
    print(f"⬇️ Downloading {filepath} from {url} ...")
    subprocess.run(
      ["wget", "-c", url, "-O", filepath],
      check=True
    )
    print(f"✅ Download completed: {filepath}")
    with open(done_marker, 'w') as f:
      f.write("done")
  except subprocess.CalledProcessError as e:
    print(f"❌ Download failed: {e}")
    raise

  return filepath

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Download WMT dataset.")
  parser.add_argument("--data_path", default="data/WMT",
                      help="Path to save the downloaded data.")
  args = parser.parse_args()

  data_path = args.data_path

  download_with_wget("https://storage.googleapis.com/dm-streamingqa/wmt_sorting_key_ids.txt.gz", data_path)

  _archive_file_names = [
    'news-docs.2007.en.filtered.gz',
    'news-docs.2008.en.filtered.gz',
    'news-docs.2009.en.filtered.gz',
    'news-docs.2010.en.filtered.gz',
    'news-docs.2011.en.filtered.gz',
    'news-docs.2012.en.filtered.gz',
    'news-docs.2013.en.filtered.gz',
    'news-docs.2014.en.filtered.gz',
    'news-docs.2015.en.filtered.gz',
    'news-docs.2016.en.filtered.gz',
    'news-docs.2017.en.filtered.gz',
    'news-docs.2018.en.filtered.gz',
    'news-docs.2019.en.filtered.gz',
    'news-docs.2020.en.filtered.gz',
    'news-docs.2021.en.filtered.gz',
  ]
  def download_one(file_name):
    url = f"https://data.statmt.org/news-crawl/doc/en/{file_name}"
    return download_with_wget(url, data_path)

  with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
    list(executor.map(download_one, _archive_file_names))
