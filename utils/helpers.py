import numpy as np

NOTE_NAMES_FULL = [f"{n}{o}" for o in range(1, 8) for n in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]]


def freq_to_note_name(freq):
    if freq is None or freq <= 0:
        return "..."
    n = int(round(12 * np.log2(freq / 440.0)))
    note = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][(n + 9) % 12]
    octave = 4 + ((n + 9) // 12)
    return f"{note}{octave}"


def freq_to_note_index(freq):
    if freq is None or freq <= 0:
        return None
    n = int(round(12 * np.log2(freq / 440.0)))
    name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][(n + 9) % 12]
    octave = 4 + ((n + 9) // 12)
    full_note = f"{name}{octave}"
    return NOTE_NAMES_FULL.index(full_note) if full_note in NOTE_NAMES_FULL else None


def note_index_to_freq(index):
    """Returns frequency in Hz for a given note index in NOTE_NAMES_FULL"""
    if 0 <= index < len(NOTE_NAMES_FULL):
        return 440.0 * 2 ** ((index - NOTE_NAMES_FULL.index("A4")) / 12)
    return None




