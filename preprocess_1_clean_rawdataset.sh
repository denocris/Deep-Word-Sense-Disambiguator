#!/bin/sh

text=$1

# Remove non-ASCII characters...
echo "Remove non-ASCII characters except ! ' ?"
sed -e "s/[^a-zA-Z0-9àèéìòù@_\!'?\* ]//g" $1 > corpus_tmp_v0

# Select only sentences with more than 2 words
echo "Select only sentences with more than 2 words..."
awk 'NF>=2' corpus_tmp_v0 > corpus_tmp_v1

echo "Select only sentences with less than 2 words..."
awk 'NF<=17' corpus_tmp_v1 > corpus_tmp_v2

# Lowering characters
echo "Lowering character..."
cat corpus_tmp_v2 | tr '[:upper:]' '[:lower:]' > corpus_tmp_v3

# Remove whitespace at the beginning and at the end of the line...
echo "Remove whitespace at the beginning and at the end of the line and strip multiple whitespaces..."
sed -r 's/[[:blank:]]*$//g' corpus_tmp_v3 | sed -r 's/^[[:blank:]]*//g' | sed 's/  */ /g ' > $1_cleaned

rm -rf corpus_tmp_v*
