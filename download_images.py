from bing_image_downloader import downloader

downloader.download("cat", limit=200, output_dir="data", adult_filter_off=True)
downloader.download("dog", limit=200, output_dir="data", adult_filter_off=True)

print("Download complete!")