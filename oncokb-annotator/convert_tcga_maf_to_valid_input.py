import sys,os,glob,shutil
import pandas as pd
import json
import requests
import numpy as np

"""
PROJECT=TCGA-BRCA
1. download all the *_ensemble_masked.maf from GDC Portal into the following directory:
/data/Jiang_Lab/Data/Zisha_Zhong/BigData/${PROJECT}/ensemble_masked_mafs/

2. run the following commands one by one
for f in `find /data/Jiang_Lab/Data/Zisha_Zhong/BigData/${PROJECT}/ensemble_masked_mafs/ -name "*.gz"`; 
do
 echo $f; gunzip $f; 
done

cd /data/zhongz2/BigData/${PROJECT}
mkdir mafs
cd mafs

for f in `find /data/Jiang_Lab/Data/Zisha_Zhong/BigData/${PROJECT}/ensemble_masked_mafs/ -name "*.maf"`; do
filename=$(basename $f)
sed -n '8,$p' < $f > $filename
done

cd /data/zhongz2/HistoVAE/oncokb-annotator;
for f in `find /data/zhongz2/BigData/${PROJECT}/mafs/ -name "*.maf"`; do
python MafAnnotator.py -i $f -o $f.oncokb.txt -b 30a43cc4-caf0-4651-af3d-e96053b3f686;
echo $f
done

3. finally the *.oncokb.txt files will have the "MUTATION_EFFECT" column
"""

## -------------- JSON Filters constructor :
class Filter(object):

    def __init__(self):
        self.filter = {"op": "and", "content": []}

    def add_filter(self, Field, Value, Operator):
        self.filter['content'].append({"op": Operator, "content": {"field": Field, "value": Value}})

    def create_filter(self):
        self.final_filter = json.dumps(self.filter, separators=(',', ':'))
        return self.final_filter
    
    
## -------------- Function for reading manifest file :
def read_manifest(manifest_loc):
    uuid_list = []
    with open(manifest_loc, 'r') as myfile:
        if myfile.readline()[0:2] != 'id':
            raise ValueError('Bad Manifest File')
        else:
            for x in myfile:
                uuid = x.split('\t')[0]
                uuid_list.append(uuid)
    return uuid_list

    
project = 'BRCA'
maf_root = '/data/zhongz2/BigData/TCGA-{}/mafs'.format(project)
filepaths = glob.glob(os.path.join(maf_root, '*.oncokb.txt'))

Manifest_Loc = '/data/zhongz2/HistoVAE/tcga_paad/gdc_manifest_gene_masked_{}.2022-11-21.txt'.format(project.lower())
UUIDs = read_manifest(Manifest_Loc)

# 2. Get info about files in manifest
# -------------------------------------------------------
File_Filter = Filter()
File_Filter.add_filter("files.file_id", UUIDs, "in")
File_Filter.create_filter()

EndPoint = 'files'
Fields = 'cases.samples.portions.analytes.aliquots.submitter_id,file_name,cases.samples.sample_type,file_id,md5sum,experimental_strategy,analysis.workflow_type,data_type'
Size = '10000'

Payload = {'filters': File_Filter.create_filter(),
           'format': 'json',
           'fields': Fields,
           'size': Size}
r = requests.post('https://api.gdc.cancer.gov/files', json=Payload)
data = json.loads(r.text)
file_list = data['data']['hits']

Dictionary = {}
TCGA_Barcode_Dict = {}
for file in file_list:
    UUID = file['file_id']
    Barcode = file['cases'][0]['samples'][0]['portions'][0]['analytes'][0]['aliquots'][0]['submitter_id']
    File_Name = file['file_name']

    Dictionary[UUID] = {'File Name': File_Name,
                        'TCGA Barcode': Barcode,
                        'MD5': file['md5sum'],
                        'Sample Type': file['cases'][0]['samples'][0]['sample_type'],
                        'Experimental Strategy': file['experimental_strategy'],
                        'Workflow Type': file['analysis']['workflow_type'],
                        'Data Type': file['data_type']}

    TCGA_Barcode_Dict[File_Name] = Barcode
TCGA_Barcode_Dict_Reverse = {v:k for k, v in TCGA_Barcode_Dict.items()}
"""
TCGA_Barcode_Dict will be like this:
{
 'b3512896-94fe-489d-bd88-9d0858857bd0.wxs.aliquot_ensemble_masked.maf.gz': 'TCGA-A8-A0A7-01A-11W-A019-09',
 ...
 }
"""

# remove duplicates
# barcodes = list(TCGA_Barcode_Dict.values())
# barcodes, counts = np.unique(barcodes, return_counts=True)

alldata = {}
for filepath in filepaths:
    # filepath = os.path.join(maf_root, TCGA_Barcode_Dict_Reverse[barcode].replace('.gz', '.oncokb.txt'))
    barcode = TCGA_Barcode_Dict[os.path.basename(filepath).replace('.oncokb.txt', '.gz')]
    df = pd.read_csv(filepath, sep='\t', low_memory=False)
    for ii, gene_symbol in enumerate(df['Hugo_Symbol'].values):
        if gene_symbol not in alldata:
            alldata[gene_symbol] = {}
        alldata[gene_symbol][barcode] = df.iloc[ii]['MUTATION_EFFECT']

alldf = pd.DataFrame(alldata).T
alldf.to_csv('/data/zhongz2/temp/{}_mutation_effects.csv'.format(project))

all_mutation_effects = alldf.values.astype(np.str_)
mutations, counts = np.unique(all_mutation_effects, return_counts=True)
print('statistics on {}'.format(project))
for mut, c in zip(mutations, counts):
    print('\t{}: {}'.format(mut, c))
"""
BRCA:
Gain-of-function: 75
Likely Gain-of-function: 17
Likely Loss-of-function: 1052
Unknown: 82447
nan: 16175643
"""

project = 'BRCA'
alldf = pd.read_csv('/data/zhongz2/temp/{}_mutation_effects.csv'.format(project), low_memory=True)
alldf = alldf.rename(columns={'Unnamed: 0': 'Hugo_Symbol'})
gene_symbols = alldf[alldf.columns[0]].tolist()
valid_gene_inds = []
unknown_nan_counts = []
for i, gene_symbol in enumerate(gene_symbols):
    rowdf = alldf.iloc[i]
    values = rowdf.values[1:].astype(np.str_)
    labels, counts = np.unique(values, return_counts=True)
    dd = {'Unknown': 0, 'nan': 0}
    dd.update({l: c for l, c in zip(labels, counts)})
    if len(set(labels) - {'Unknown', 'nan'}) > 0:
        valid_gene_inds.append(i)
        unknown_nan_counts.append(dd['Unknown'] + dd['nan'])
valid_gene_inds = np.array(valid_gene_inds)
unknown_nan_counts = np.array(unknown_nan_counts)
inds = np.argsort(unknown_nan_counts)
unknown_nan_counts = unknown_nan_counts[inds]
valid_gene_inds = valid_gene_inds[inds]
valid_counts = len(alldf.columns) - np.array(unknown_nan_counts)
valid_df = alldf.iloc[valid_gene_inds].copy()
valid_df.insert(1, 'valid_mut_counts', valid_counts)
valid_df = valid_df.rename(columns={'Unnamed: 0': 'Hugo_Symbol'})
valid_df.to_csv('/data/zhongz2/temp/{}_mutation_effects_sorted.csv'.format(project), index=False)







