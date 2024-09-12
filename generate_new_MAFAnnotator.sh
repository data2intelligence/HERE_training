#!/bin/bash

PROJECT=${1}
VERSION="v2"
CODE_ROOT=${2}

if [ -e ${PROJECT}/mutation_effects_sortedv2.csv ]; then
  exit;
fi

if [ ! -e ${PROJECT}/mafs${VERSION} ]; then
mkdir -p ${PROJECT}/mafs${VERSION};
cd ${PROJECT}/mafs${VERSION};
for f in `find ${PROJECT}/all_maf/ -name "*.maf"`; do
filename=$(basename $f)
sed -n '8,$p' < $f > $filename
done
fi

cd $CODE_ROOT

## 
ONKOKB_TOKEN=XXXXXXXXX   # register in https://faq.oncokb.org/technical
current_dir=$(pwd)
cd oncokb-annotator;
for f in `find ${PROJECT}/mafs${VERSION}/ -name "*.maf"`; do
python MafAnnotator.py -i $f -o $f.oncokb${VERSION}.txt -q Genomic_Change -b ${ONKOKB_TOKEN};
echo $f
done
cd $current_dir

### 3. finally the *.oncokb.txt files will have the "MUTATION_EFFECT" column
python convert_tcga_maf_to_valid_input_v2.py ${PROJECT} ${VERSION}









