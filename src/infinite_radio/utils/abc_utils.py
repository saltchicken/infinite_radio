import re

def force_instrumental_abc(abc_text, instrument_name="Lead Synth", midi_program=81):
    """Strips lyrics and forces a specific instrument/MIDI program in the ABC text."""
    lines = abc_text.splitlines()
    out_lines = []

    for line in lines:
        stripped = line.lstrip()
        # 1. Drop all lyric lines
        if stripped.startswith('w:'):
            continue

        # 2. Modify Voice lines to force instrument names
        if stripped.startswith('V:'):
            # Overwrite full name
            if 'name=' in line:
                line = re.sub(r'name="[^"]*"', f'name="{instrument_name}"', line)
            else:
                line = line.rstrip() + f' name="{instrument_name}"'

            # Overwrite short name (snm) to remove lingering "Vocal" triggers
            if 'snm=' in line:
                line = re.sub(r'snm="[^"]*"', f'snm="{instrument_name}"', line)

            out_lines.append(line)
            # Inject standard MIDI directive immediately after the voice declaration
            out_lines.append(f'%%MIDI program {midi_program}')
            continue

        out_lines.append(line)

    return '\n'.join(out_lines)
