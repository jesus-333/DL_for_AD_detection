#!/bin/sh

# Script used in interactive mode to decompress the downloaded data files.


#  
folder_with_files="${SCRATCH}/ADNI_MRI_NORMALIZED/"
filename="Brain_MRI_Preprocess__GradWarp-B1-N3__-_SCALED_MRI_"
n_files=10

folder_output="${SCRATCH}/ADNI_MRI_NORMALIZED_DECOMPRESSED/"

rm -rf folder_output

# Decompress the files. Note that indices start at 1, not 0.
for i in $(seq 1 $n_files); do
	tmp_file="${folder_with_files}${filename}${i}.zip"

	if [ -f "$tmp_file" ]; then
		# unzip -o "$tmp_file" -d "$folder_output"
		7z x "$tmp_file" -o"$folder_output" -aoa -mmt=on
	else
		echo "File $tmp_file does not exist."
	fi
done


