#!/bin/sh

text=$1

# Grep è as a verb and remove 'è, e, e'
echo "Grepping è..."
cat $1 | grep "\bè\b" | grep -v "\b'è\b" | grep -v "\be\b" | grep -v "e'" | grep -Pv "__" > è_verb_tmp0
num_verb=$(cat è_verb_tmp0 | wc -l )
echo "Number of sentences: " ${num_verb}
num_cong=$(grep "\be\b" è_verb_tmp0 | wc -l )
echo "Check zero entries for e: " ${num_cong}



echo "Grepping e..."
cat $1 | grep "\be\b" | grep -v "\bè\b" | grep -v "\be'\b" | grep -v "e'" | grep -Pv "__" > e_cong_tmp0
num_cong=$(cat e_cong_tmp0 | wc -l )
echo "Number of sentences: " ${num_cong}
num_verb=$(grep "\bè\b" e_cong_tmp0 | wc -l )
echo "Check zero entries for e: " ${num_verb}


# Some refinement
cat è_verb_tmp0 | grep -Pv "(\bè\b).*(\bè\b).*" > è_verb_tmp1
cat e_cong_tmp0 | grep -Pv "(\bè\b).*(\bè\b).*" > e_cong_tmp1

# Unique and then sort
sort -u è_verb_tmp1 | sort -R > è_verb
sort -u e_cong_tmp1 | sort -R > e_cong

rm -rf è_verb_tmp* e_cong_tmp*
rm -rf è_verb_temp* e_cong_temp*
