from pathlib import Path

def get_file_list(folder, suffix):
    p = Path(folder).rglob(suffix)
    return [str(item) for item in p]

if __name__ == "__main__":
    print(get_file_list("/data/vott-csv-export","*.png"))