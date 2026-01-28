import pandas as pd

def _aggregate_csv_max(data_df, rows_per_segment):
    """
    Aggregate dataframe by taking MAX over numeric columns
    every `rows_per_segment` rows.

    This logic is extracted verbatim from Notebook 1.
    """

    if len(data_df) % rows_per_segment != 0:
        new_len = (len(data_df) // rows_per_segment) * rows_per_segment
        data_df = data_df.iloc[:new_len]

    rows = []
    duration_seconds = int(len(data_df) / 10)

    for i in range(0, len(data_df), rows_per_segment):
        block = data_df.iloc[i:i + rows_per_segment]
        numeric_max = block.max(numeric_only=True)

        new_row = {
            "patient_id": block.iloc[0]["patient_id"]
        }
        new_row.update(numeric_max)
        rows.append(new_row)

    df_new = pd.DataFrame(rows).reset_index(drop=True)
    return df_new, duration_seconds


def segment_dataset(raw_path, processed_path, time_ms, devices):
    """
    Segmentation pipeline entry point.

    NOTE:
    Only CSV aggregation is implemented at this stage.
    Audio segmentation will be added later.
    """
    raise NotImplementedError(
        "CSV aggregation implemented, audio segmentation pending."
    )

import numpy as np
import librosa

def _segment_audio_waveform(
    audio_path,
    rows_per_segment,
    duration_seconds
):
    """
    Segment an audio file into equal-length chunks aligned
    with aggregated CSV rows.

    Extracted verbatim from Notebook 1 logic.
    No file writing is performed here.
    """

    # Load audio at native sampling rate
    wav, sr = librosa.load(audio_path, sr=None)

    # Trim audio to match CSV duration
    wav = wav[: int(duration_seconds * sr)]

    # Compute segmentation boundaries (Notebook 1 logic)
    cut_points = np.linspace(
        0,
        len(wav),
        int(len(wav) / (rows_per_segment * (sr / 10))) + 1
    )

    segments = []
    for i in range(len(cut_points) - 1):
        start = int(cut_points[i])
        end = int(cut_points[i + 1])
        segments.append(wav[start:end])

    return segments, sr
