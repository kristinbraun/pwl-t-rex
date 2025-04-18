#!/bin/bash

FILE=$1

echo "read $FILE.nl write problem $FILE.gms quit" | scip

sed 's/\r//g' $FILE.gms > ${FILE}_2.gms

mv ${FILE}_2.gms ${FILE}.gms

echo "osil ${FILE}.osil" > convert.opt
echo "${FILE}.gms" | python add_optline.py

gams ${FILE}_add.gms minlp=convert

rm convert.opt
rm $FILE.gms ${FILE}_2.gms
