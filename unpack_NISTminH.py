import numpy as np

# Choose your file (Temporal or Spatial)
input_file = r"data/raw/3hours_nopeople/temporal_3hraw_bitstream.bin"
output_file = r"temporal_unpacked_for_nist.bin"

# Load the packed bytes
packed_data = np.fromfile(input_file, dtype=np.uint8)

# Unpack them back into a bitstream (array of 0s and 1s)
bitstream = np.unpackbits(packed_data)

# Save as a raw binary file where each byte is exactly 0 or 1
bitstream.tofile(output_file)

print(f"-> Success! NIST-ready file created: {output_file}")