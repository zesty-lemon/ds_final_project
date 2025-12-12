# Install requirement: pip install deep-translator tqdm openpyxl
import time
import os
import pickle
from deep_translator import GoogleTranslator
from tqdm.auto import tqdm
import pandas as pd
import glob

def translate_column_deep_translator(df, col, out_col=None, cache_path=None, pause=0.2, debug=False):
    """
    Translate df[col] -> English using deep_translator.GoogleTranslator (no API key).
    Uses a simple on-disk cache to avoid repeated requests.
    - pause: seconds to sleep between requests to reduce rate-limit risk
    """
    if out_col is None:
        out_col = f"{col}_en"

    texts = df[col].fillna("").astype(str).tolist()
    unique_texts = list(dict.fromkeys([t for t in texts if t.strip()]))

    cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
        except Exception:
            cache = {}

    to_translate = [t for t in unique_texts if t not in cache]
    if to_translate:
        translator = GoogleTranslator(source="ja", target="en")#"auto", target="en")
        for t in tqdm(to_translate, desc="translating (deep_translator)"):
            try:
                translated = translator.translate(t)
                cache[t] = translated
            except Exception as e:
                if debug:
                    print("translate error:", e, "for text:", t[:80])
                # fallback: keep original text (or set to empty string "")
                cache[t] = ""
                time.sleep(1.0)
            time.sleep(pause)

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(cache, f)

    # map back preserving empties
    out = [cache.get(t, "") if t.strip() else "" for t in texts]
    df2 = df.copy()
    df2[out_col] = out
    return df2

def translate_all_files_in_folder(src_folder: str,
                                  out_folder: str = None,
                                  file_patterns: list = None,
                                  artist_col: str = "ARTIST",
                                  song_col: str = "SONG",
                                  keep_unique_only: bool = True,
                                  remove_date: bool = True):
    """
    Translate ARTIST and SONG columns for every .xlsx/.csv in src_folder using deep_translator helper.
    Overwrites original columns with English translations.
    Optionally keeps only unique song-artist pairs and removes DATE column.
    Saves translated files to out_folder (defaults to src_folder/translated).
    """
    if file_patterns is None:
        file_patterns = ["*.xlsx", "*.csv"]
    src_folder = os.path.abspath(src_folder)
    if out_folder is None:
        out_folder = os.path.join(src_folder, "translated")
    os.makedirs(out_folder, exist_ok=True)
    cache_dir = os.path.join("/Users/albinmeli/CS5870/ds_final_project/cache")
    os.makedirs(cache_dir, exist_ok=True)

    files = []
    for pat in file_patterns:
        files.extend(sorted(glob.glob(os.path.join(src_folder, pat))))
    if not files:
        print("No files found in", src_folder)
        return

    for fpath in files:
        fname = os.path.basename(fpath)
        print("Processing:", fname)
        try:
            if fname.lower().endswith(".xlsx") or fname.lower().endswith(".xls"):
                df = pd.read_excel(fpath, engine="openpyxl")
                ext = "xlsx"
            else:
                df = pd.read_csv(fpath, low_memory=False)
                ext = "csv"
        except Exception as e:
            print("  Failed to read file:", fpath, "error:", e)
            continue

        # check for artist and song columns
        has_artist = artist_col in df.columns
        has_song = song_col in df.columns
        if not has_artist and not has_song:
            print(f"  Columns '{artist_col}' and '{song_col}' not found — skipping.")
            continue

        original_row_count = len(df)
        translated = df.copy()
        
        # translate artist column and overwrite original
        if has_artist:
            cache_path_artist = os.path.join(cache_dir, f"{os.path.splitext(fname)[0]}_artist_cache.pkl")
            try:
                temp = translate_column_deep_translator(translated,
                                                        col=artist_col,
                                                        out_col=f"{artist_col}_en",
                                                        cache_path=cache_path_artist,
                                                        pause=0.2,
                                                        debug=False)
                # overwrite original artist column with translated
                translated[artist_col] = temp[f"{artist_col}_en"]
                # drop temporary _en column
                translated = translated.drop(columns=[f"{artist_col}_en"], errors='ignore')
            except Exception as e:
                print("  Artist translation failed for", fname, "error:", e)
        
        # translate song column and overwrite original
        if has_song:
            cache_path_song = os.path.join(cache_dir, f"{os.path.splitext(fname)[0]}_song_cache.pkl")
            try:
                temp = translate_column_deep_translator(translated,
                                                        col=song_col,
                                                        out_col=f"{song_col}_en",
                                                        cache_path=cache_path_song,
                                                        pause=0.2,
                                                        debug=False)
                # overwrite original song column with translated
                translated[song_col] = temp[f"{song_col}_en"]
                # drop temporary _en column
                translated = translated.drop(columns=[f"{song_col}_en"], errors='ignore')
            except Exception as e:
                print("  Song translation failed for", fname, "error:", e)

        # Remove DATE column if present and requested
        if remove_date:
            date_cols = [col for col in translated.columns if col.upper() == 'DATE']
            if date_cols:
                translated = translated.drop(columns=date_cols)
                print(f"  Removed DATE column(s): {date_cols}")

        # Keep only unique song-artist pairs if requested
        if keep_unique_only and has_song and has_artist:
            # Drop duplicates based on song and artist, keeping first occurrence
            translated = translated.drop_duplicates(subset=[song_col, artist_col], keep='first')
            unique_count = len(translated)
            print(f"  Kept {unique_count} unique song-artist pairs (removed {original_row_count - unique_count} duplicates)")

        out_name = fname.replace(f".{ext}", f"_translated.{ext}")
        out_path = os.path.join(out_folder, out_name)
        try:
            if ext == "xlsx":
                translated.to_excel(out_path, index=False, engine="openpyxl")
            else:
                translated.to_csv(out_path, index=False)
            print("  Saved:", out_path)
        except Exception as e:
            print("  Failed to save translated file:", e)

if __name__ == "__main__":
    SRC = "/Users/albinmeli/CS5870/ds_final_project/selected_cities_apple"
    translate_all_files_in_folder(SRC, keep_unique_only=True, remove_date=True)