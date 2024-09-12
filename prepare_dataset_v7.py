import sys, os, glob, shutil
import numpy as np
import pandas as pd
import json
import CytoSig
import pdb
import re

"""
this script based on the prepare_tcga_luad_lusc_dataset.py
2022-04-16 modified following Prof. Jiang's suggestions using cbioportal data
https://www.cbioportal.org/

2023-01-12 using Receptor_Cytokine for prediction

2023-01-25 using MsigDB Hallmark 50 gene sets for regression prediction

"""
PREFIX = sys.argv[1]
MUTATION_VERSION = sys.argv[2]
PATCH_SIZE = int(float(sys.argv[3]))

DATA_ROOT = f'./tcga/TCGA-{PREFIX}_{PATCH_SIZE}/'
DATA_ROOT2 = './tcga/tcga_{}/'.format(PREFIX.lower())

SAVE_ROOT = os.path.join(DATA_ROOT, 'generated7')  # updated gene classification, using mean for MsigDB hallmark
if not os.path.exists(SAVE_ROOT):
    os.makedirs(SAVE_ROOT, exist_ok=True)


def step_1_check_brca():
    save_filename = os.path.join(SAVE_ROOT, 'all.csv')
    if os.path.exists(save_filename):
        return

    print('begin step_1_check_brca')
    filename = glob.glob(os.path.join(DATA_ROOT, 'clinical.*.json'))[0]
    with open(filename, 'r') as fp:
        clinical = json.load(fp)

    clinical = [clinical[i] for i in range(len(clinical)) if 'exposures' in clinical[i]]
    case_uuids = [clinical[i]['case_id'] for i in range(len(clinical))]

    groups = {
        'exposures': [
            'cigarettes_per_day', 'alcohol_history', 'years_smoked', 'alcohol_intensity'
        ],
        'diagnoses':
            ['synchronous_malignancy', 'ajcc_pathologic_stage', 'days_to_diagnosis', 'last_known_disease_status',
             'tissue_or_organ_of_origin',
             'days_to_last_follow_up', 'primary_diagnosis', 'prior_malignancy', 'days_to_last_known_disease_status',
             'ajcc_staging_system_edition', 'ajcc_pathologic_t', 'morphology',
             'ajcc_pathologic_n', 'ajcc_pathologic_m', 'icd_10_code',
             'site_of_resection_or_biopsy', 'tumor_grade', 'progression_or_recurrence',
             'submitter_id', 'diagnosis_id', 'classification_of_tumor'
             ],
        'demographic':
            ['vital_status', 'gender', 'race', 'ethnicity', 'age_at_index', 'year_of_death']
    }

    items = {}
    for group, keys in groups.items():
        if group == 'demographic':
            for key in keys:
                values = []
                for i in range(len(clinical)):
                    if key in clinical[i][group]:
                        values.append(clinical[i][group][key])
                    else:
                        values.append(np.nan)
                items[key] = values
        else:
            for key in keys:
                values = []
                for i in range(len(clinical)):
                    if key in clinical[i][group][0]:
                        values.append(clinical[i][group][0][key])
                    else:
                        values.append(np.nan)
                items[key] = values

    items['case_uuids'] = case_uuids
    clinical = pd.DataFrame(items)

    diagnostic_svs_root = os.path.join(DATA_ROOT, 'allsvs')

    if PREFIX == 'BRCA':
        excel_filename = os.path.join(DATA_ROOT, 'DataS1.xlsx')
        if os.path.exists(excel_filename):
            df = pd.read_excel(excel_filename)
            dfs = {}
            for index, name in enumerate(df.CLID.values):
                for ind2, name2 in enumerate(clinical.submitter_id.values):
                    if name[:12] == name2[:12]:
                        dfs[name] = clinical.iloc[ind2]
                        break
            df1 = pd.DataFrame(dfs).T
            df = df.set_index('CLID')
            df = df.join(df1)
    else:
        df = clinical
        df['CLID'] = clinical.submitter_id.values.tolist()
        df = df.set_index('CLID')

    all_svs_filenames = glob.glob(os.path.join(diagnostic_svs_root, '*.svs'))

    filenames = []
    svs_counts = []
    dx_svs_counts = []
    ts_svs_counts = []
    for index, name in enumerate(df.index.values):
        filenames_for_this_case = []
        for svs_filename in all_svs_filenames:
            if name[:12] in svs_filename:
                # print(name, svs_filename)
                filenames_for_this_case.append(os.path.basename(svs_filename))
        # print(name, len(filenames_for_this_case))

        filenames.append(filenames_for_this_case)
        svs_counts.append(len(filenames_for_this_case))
        dx_svs_count = len([name for name in filenames_for_this_case if '-DX' in name])
        ts_svs_count = len(filenames_for_this_case) - dx_svs_count
        dx_svs_counts.append(dx_svs_count)
        ts_svs_counts.append(ts_svs_count)

    df['all_svs_filenames'] = filenames
    df['number_of_DX_slides'] = dx_svs_counts
    df['number_of_TS_slides'] = ts_svs_counts

    df.to_csv(save_filename)

    print('done')


def step_2_get_fpkm():
    save_filename = os.path.join(SAVE_ROOT, 'all_with_fpkm.csv')
    if os.path.exists(save_filename):
        return
    print('begin get_fpkm')

    df = pd.read_csv(os.path.join(SAVE_ROOT, 'all.csv'), delimiter=',')
    # fpkm = pd.read_json('fpkm.2022-01-19.json')

    # filename = '/data/zhongz2/BigData/TCGA-LUAD-LUSC/clinical.project-TCGA-LUAD-LUSC.2022-04-08.json'
    filename = glob.glob(os.path.join(DATA_ROOT, 'clinical*.json'))[0]
    cases = pd.read_json(filename)

    case_uuids = cases.case_id
    diagnoses = cases.diagnoses
    submitter_ids = []
    uuids = []
    for i, (uuid, diagnosis) in enumerate(zip(case_uuids, diagnoses)):
        if len(uuid) != 36:
            continue
        try:
            submitter_id = diagnosis[0]['submitter_id']
            uuids.append(uuid)
            submitter_ids.append(submitter_id[:12])

        except:
            continue

    case_uuids = []
    for index, name in enumerate(df.CLID.values):
        case_uuid_this_case = []
        for j, (uuid, submitter_id) in enumerate(zip(uuids, submitter_ids)):
            if name[:12] == submitter_id:
                case_uuid_this_case.append(uuid)

        if len(case_uuid_this_case) != 1:
            print(name)

        case_uuids.append(case_uuid_this_case)

    uuid_counts = [len(item) for item in case_uuids]

    df['case_uuid'] = case_uuids
    df['num_uuid'] = uuid_counts
    df1 = df[df.num_uuid <= 0]
    if len(df1) > 0:
        print('there are some cases without UUID, check them')

    df = df[df.num_uuid > 0].reset_index(drop=True)

    removed_columns = [col for col in df.columns if 'Unnamed' in col]
    if len(removed_columns) > 0:
        df = df.drop(columns=removed_columns, axis=1)

    df = df.set_index('CLID')
    df.to_csv(save_filename)


def Print(str):
    print('\033[2;31;43m %s \033[0;0m' % (str))


def step_3_prepare_TIDE_scores_v3():
    hist_save_dir = os.path.join(SAVE_ROOT, 'histograms')
    os.makedirs(hist_save_dir, exist_ok=True)
    save_filename = os.path.join(hist_save_dir, '{}_CTL_hist.png'.format(PREFIX))
    if os.path.exists(save_filename):
        return

    print('begin prepare_TIDE_scores_v3')

    if PREFIX == 'BRCA':
        df = pd.read_csv(os.path.join(SAVE_ROOT, 'all_with_fpkm.csv'), delimiter=',')
        normal_df = df[df['2016 Histology Annotations'].isin(['True Normal'])]
        normal_CLID_values = normal_df.CLID.values.tolist()
        normal_CLID_values = [name[:12] for name in normal_CLID_values]
        normal_CLID_values_set = set(normal_CLID_values)
    else:
        filename = os.path.join(SAVE_ROOT, 'all_with_fpkm.csv')
        df = pd.read_csv(filename)
        case_uuids = []
        for case_uuid in df.case_uuid.values:
            case_uuids.append(eval(case_uuid)[0])
        df['case_uuids'] = case_uuids

        filename = glob.glob(os.path.join(DATA_ROOT, 'biospecimen.*.json'))[0]
        with open(filename, 'r') as fp:
            clinical = json.load(fp)

        sample_counts_per_case = [len(x['samples']) for x in clinical]
        uuids = [x['case_id'] for x in clinical]
        sample_types_per_case = [[sample['sample_type'] for sample in x['samples']] for x in clinical]
        normal_uuids = []
        normal_CLID_values = []
        invalid_uuids = []
        for uuid, sample_types in zip(uuids, sample_types_per_case):
            print(uuid, sample_types)
            sample_types = '_'.join(sample_types).lower()
            if 'normal' in sample_types and 'solid' in sample_types:
                normal_uuids.append(uuid)
                tmpdf = df[df.case_uuids == uuid]
                if len(tmpdf) > 0:
                    normal_CLID_values.append(tmpdf['CLID'].values[0])
                else:
                    invalid_uuids.append(uuid)

        Print('number of normal cases: {}'.format(len(normal_uuids)))
        Print('number of total cases: {}'.format(len(uuids)))
        Print('number of invalid cases: {}'.format(len(invalid_uuids)))
        normal_CLID_values = [name[:12] for name in normal_CLID_values]
        normal_CLID_values_set = set(normal_CLID_values)

    data = pd.read_csv(os.path.join(DATA_ROOT, 'Merged_FPKM.tsv'), delimiter='\t', index_col='gene_name',
                       low_memory=False)

    columns = data.columns
    removed_columns = [col for col in columns if 'TCGA' not in col]  # must be TCGA dataset
    if len(removed_columns) > 0:
        data = data.drop(removed_columns, axis=1)

    data = np.log2(data + 1)

    if len(normal_CLID_values_set) > 0:
        normal_column_names = []
        for name in data.columns.values:
            if name[:12] in normal_CLID_values_set:
                normal_column_names.append(name)
        data_normal = data.filter(normal_column_names, axis=1)
        data_tumor = data.drop(normal_column_names, axis=1)
    else:
        data_normal = data
        data_tumor = data

    data = data_tumor.subtract(data_normal.mean(axis=1), axis=0)  # Y   # data_tumor

    data = data.groupby(data.index).median()

    # 43 Zscores for regression
    signature = '/data/zhongz2/HistoVAE/CytoSig/CytoSig/signature.centroid'  # load cytokine response signature installed in your python system path
    signature = pd.read_csv(signature, sep='\t', index_col=0)

    beta, std, zscore, pvalue = CytoSig.ridge_significance_test(signature, data, alpha=1E4, alternative="two-sided",
                                                                nrand=1000, cnt_thres=10, flag_normalize=True,
                                                                verbose=True)
    zscore = zscore.T
    new_columns = {col: 'CytoSig_{}'.format(col.replace(' ', '_')) for col in zscore.columns}
    zscore = zscore.rename(columns=new_columns)
    lines = ['\"{}\",\n'.format(v) for v in zscore.columns.tolist()]
    with open(os.path.join(SAVE_ROOT, 'CytoSig_zscore_list.txt'), 'w') as fp:
        fp.writelines(lines)

    CTL = data.loc[['CD8A', 'CD8B', 'GZMA', 'GZMB', 'PRF1']].median()

    signature = pd.read_csv('/data/zhongz2/HistoVAE/TIDE_Results/Exclusion_scores/exclusion.signature', delimiter='\t')
    metrics = signature.apply(lambda v: data.corrwith(v))

    signature_dysfunction = pd.read_csv('/data/zhongz2/HistoVAE/run.summary.full.trans.gz', delimiter='\t',
                                        index_col=0).dropna().median(
        axis=1)
    signature_dysfunction = signature_dysfunction.to_frame()
    signature_dysfunction.index.name = 'gene_name'
    signature_dysfunction.columns = ['Dys']

    print(signature_dysfunction)

    dysfunction_metrics = signature_dysfunction.apply(lambda v: data.corrwith(v))

    print(dysfunction_metrics)

    indices = \
        np.where(metrics.isnull())[0].tolist() + \
        np.where(CTL.isnull())[0].tolist() + \
        np.where(dysfunction_metrics.isnull())[0].tolist()
    if len(indices) > 0:
        indices = np.unique(indices)
        print('have invalid values: ', indices)
        metrics = metrics.drop(metrics.index[indices])
        CTL = CTL.drop(CTL.index[indices])
        dysfunction_metrics = dysfunction_metrics.drop(dysfunction_metrics.index[indices])

    try:
        from scipy.stats import pearsonr
        import matplotlib.pyplot as plt

        for col in metrics.columns:
            corr, pvalue = pearsonr(CTL.values, metrics[col].values)
            print('Corr ({}, {}) is {}(p={}) '.format('CTL', col, corr, pvalue))
            _ = plt.hist(metrics[col].values, bins='auto')
            plt.savefig(os.path.join(hist_save_dir, '{}_{}_hist.png'.format(PREFIX, col)))
            plt.close()

        for col in zscore.columns:
            _ = plt.hist(zscore[col].values, bins='auto')
            plt.savefig(os.path.join(hist_save_dir, '{}_{}_hist.png'.format(PREFIX, col)))
            plt.close()

        _ = plt.hist(CTL.values, bins='auto')
        plt.savefig(save_filename)
        plt.close()
    except:
        print('ERROR: TIDE histogram error, check it!')

    TIDE_scores = pd.concat([metrics.drop(['Mean'], axis=1), dysfunction_metrics], axis=1)

    new_columns = {col: 'TIDE_{}'.format(col) for col in TIDE_scores.columns}
    TIDE_scores = TIDE_scores.rename(columns=new_columns)
    # combine the CytoSig values to TIDE_scores for convenience

    # from Receptor_Cytokine.xlsx
    xls = pd.ExcelFile('/data/zhongz2/HistoVAE/Receptor_Cytokine.xlsx')
    newdata1 = {}
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name)
        for i, rowname in enumerate(df['ID'].values):

            gene_list = [v.strip() for v in re.split(',|\+', df.loc[i, 'Gene'])] if isinstance(df.loc[i, 'Gene'], str) else []
            receptor_list = [v.strip() for v in re.split(',|\+', df.loc[i, 'Receptor'])] if isinstance(df.loc[i, 'Receptor'], str) else []
            print(i, rowname, gene_list, receptor_list)
            if len(gene_list) > 0:
                newdata1['Cytokine_Receptor_{}_{}_GeneSum'.format(sheet_name, rowname)] = data.loc[gene_list].sum()
            if len(receptor_list) > 0:
                newdata1['Cytokine_Receptor_{}_{}_ReceptorSum'.format(sheet_name, rowname)] = data.loc[receptor_list].sum()
    newdata1 = pd.DataFrame(newdata1)
    lines = ['\"{}\",\n'.format(v) for v in newdata1.columns.tolist()]
    with open(os.path.join(SAVE_ROOT, 'Cytokine_Receptor_list.txt'), 'w') as fp:
        fp.writelines(lines)

    try:
        from scipy.stats import pearsonr
        import matplotlib.pyplot as plt

        for col in newdata1.columns:
            corr, pvalue = pearsonr(CTL.values.flatten(), newdata1[col].values)
            print('Corr ({}, {}) is {}(p={}) '.format('CTL', col, corr, pvalue))
            _ = plt.hist(newdata1[col].values, bins='auto')
            plt.savefig(os.path.join(hist_save_dir, '{}_{}_hist.png'.format(PREFIX, col)))
            plt.close()
    except:
        print('ERROR: Receptor_Cytokine histogram error, check it!')

    # MsigDB Hallmark 50 gene sets
    with open('/data/zhongz2/HistoVAE/h.all.v2022.1.Hs.json', 'r') as fp:
        hallmark_dict = json.load(fp)
    newdata2 = {}
    for hall_key, hall_item_dict in hallmark_dict.items():
        gene_list = [v for v in hall_item_dict['geneSymbols'] if v in data.index.values]
        if len(gene_list) > 0:
            newdata2['{}_sum'.format(hall_key)] = data.loc[gene_list].mean()   # the difference between v5 and v6
    newdata2 = pd.DataFrame(newdata2)
    lines = ['\"{}\",\n'.format(v) for v in newdata2.columns.tolist()]
    with open(os.path.join(SAVE_ROOT, 'MsigDB_Hallmark_list.txt'), 'w') as fp:
        fp.writelines(lines)

    try:
        from scipy.stats import pearsonr
        import matplotlib.pyplot as plt

        for col in newdata2.columns:
            corr, pvalue = pearsonr(CTL.values.flatten(), newdata2[col].values)
            print('Corr ({}, {}) is {}(p={}) '.format('CTL', col, corr, pvalue))
            _ = plt.hist(newdata2[col].values, bins='auto')
            plt.savefig(os.path.join(hist_save_dir, '{}_{}_hist.png'.format(PREFIX, col)))
            plt.close()
    except:
        print('ERROR: MsigDB hallmark histogram error, check it!')

    CTL = pd.DataFrame(CTL)
    CTL.columns = ['Cytotoxic_T_Lymphocyte']
    CTL.index.name = 'BARCODE'
    TIDE_scores1 = pd.concat([TIDE_scores, CTL, zscore, newdata1, newdata2], axis=1)

    TIDE_scores1.to_csv(os.path.join(SAVE_ROOT, 'TIDE_scores_v2.csv'))

    print('done')


def step_6_add_CTL_and_TIDE_v2():
    save_filename = os.path.join(SAVE_ROOT, 'all_with_fpkm_withTIDECytoSig.csv')
    if os.path.exists(save_filename):
        return
    print('begin add_CTL_and_TIDE')

    filename = os.path.join(SAVE_ROOT, 'all_with_fpkm.csv')
    alldata = pd.read_csv(filename, delimiter=',', index_col=0)

    TIDE = pd.read_csv(os.path.join(SAVE_ROOT, 'TIDE_scores_v2.csv'), delimiter=',', index_col=0)

    dict12, dict15, dict16 = {}, {}, {}
    TIDE_new = {}
    for clid in alldata.index:
        inds12, inds15, inds16 = [], [], []
        for index, barcode in enumerate(TIDE.index):
            if clid[:12] == barcode[:12]:
                inds12.append(index)
            if PREFIX == 'BRCA':
                if clid[:15] == barcode[:15]:
                    inds15.append(index)
            else:
                if clid[:12] + '-01' == barcode[:15]:
                    inds15.append(index)
            if clid[:16] == barcode[:16]:
                inds16.append(index)
            dict12[clid] = inds12
            dict15[clid] = inds15
            dict16[clid] = inds16

        if PREFIX == 'BRCA':
            if len(inds15) > 0:
                TIDE_new[clid] = TIDE.iloc[inds15].mean()
        else:
            if len(inds15) > 0:
                TIDE_new[clid] = TIDE.iloc[inds15].mean()

    TIDE_new = pd.DataFrame(TIDE_new).T
    TIDE_new.index.name = 'CLID'
    alldata1 = alldata.copy().join(TIDE_new, how='inner')

    alldata1.to_csv(save_filename)

    print('done')


def step_7_check_brca_svs_files():
    save_filename = os.path.join(SAVE_ROOT, 'all_with_fpkm_withTIDECytoSig_withMPP.csv')
    if os.path.exists(save_filename):
        return

    print('begin check_brca_svs_files')

    import pyvips
    svs_root = os.path.join(DATA_ROOT, 'svs')  # '/media/ubuntu/BigData/TCGA-BRCA/svs'

    df = pd.read_csv(os.path.join(SAVE_ROOT, 'all_with_fpkm_withTIDECytoSig.csv'), delimiter=',',
                     index_col=0, low_memory=False)

    # df = df[df.days_to_last_follow_up > 0]
    new_columns = {}
    for column in df.columns:
        new_columns[column] = column.replace(' ', '_')
    df = df.rename(columns=new_columns)

    print(df.columns)

    df = df[df.number_of_DX_slides > 0]

    filepaths = {}
    for clid, svs_filenames in zip(df.index, df.all_svs_filenames):
        # svs_filenames = [name for name in eval(svs_filenames) if '-DX1.' in name]
        svs_filenames1 = []
        for ii in range(10):
            for name in eval(svs_filenames):
                if '-DX{}.'.format(ii) in name:  # prefer DX1, then DX2
                    svs_filenames1.append(name)
                    break
        filepaths[clid] = []
        for svs_filename in svs_filenames1:
            svs_filepath = os.path.join(svs_root, svs_filename)
            if not os.path.exists(svs_filepath):
                print(clid, svs_filename)
            else:
                filepaths[clid].append(svs_filepath)

    svs_filenames = []
    filesizes = {}
    for clid, svs_filename in filepaths.items():
        print(clid, len(svs_filename))
        if len(svs_filename) == 0:
            continue
        svs_filename = svs_filename[0]
        filesizes[clid] = os.path.getsize(os.path.realpath(svs_filename))
        svs_filenames.append(svs_filename)
    print(filesizes)
    total_size = sum([v for k, v in filesizes.items()])
    print(total_size // 1024 // 1024, 'M')
    print(total_size // 1024 // 1024 // 1024, 'G')  # 865G

    # VAE:
    # (image) encoder --> z_dim (mu,sigma) --> decoder (image)
    # (image) encoder --> feature --> cls()
    #                     feature --> MDSC, CAF, ..

    widths = []
    heights = []
    appmags = []
    mpps = []
    clids = []
    valid_svsfilenames = []
    clid_prefixes = []
    for index, (clid, size) in enumerate(filesizes.items()):
        svs_filename = svs_filenames[index]
        prefix = os.path.basename(svs_filename).replace('.svs', '')
        print(clid, svs_filename)
        image = pyvips.Image.new_from_file(svs_filename)

        try:
            appmag = image.get('aperio.AppMag')
            mpp = image.get('aperio.MPP')
        except:
            appmag = '20'
            mpp = '0.5000'

        appmags.append(appmag)
        mpps.append(mpp)
        widths.append(image.get('width'))
        heights.append(image.get('height'))
        clids.append(clid)
        valid_svsfilenames.append(svs_filename)
        clid_prefixes.append(clid[:12])


    df1 = pd.DataFrame({'CLID': clids,
                        'CLID_PREFIX': clid_prefixes,
                        'width': widths,
                        'height': heights,
                        'AppMag': appmags,
                        'MPP': mpps,
                        'DX_filename': valid_svsfilenames})
    df1 = df1.drop_duplicates(subset='CLID_PREFIX')
    df1 = df1.set_index('CLID')
    df2 = df.join(df1, how='inner')

    df2.to_csv(save_filename)

    print('done')


def step_add_gene_mutation_classification():
    save_filename = os.path.join(SAVE_ROOT, 'all_with_fpkm_withTIDECytoSig_withMPP_withGene.csv')
    if os.path.exists(save_filename):
        return

    print('begin add gene mutations for classification')
    one_case_df = pd.read_csv(os.path.join(SAVE_ROOT, 'all_with_fpkm_withTIDECytoSig_withMPP.csv'), index_col=0,
                              low_memory=False)

    VALID_GENES_BAK = {
        'GBM': ['TP53', 'PTEN'],
        'COAD': ['APC', 'TP53', 'KRAS'],
        'HNSC': ['TP53'],
        'LGG': ['IDH1', 'TP53', 'ATRX'],
        'LIHC': ['TP53'],
        'LUAD': ['TP53', 'KRAS'],
        'LUSC': ['TP53'],
        'READ': ['APC', 'TP53'],
        'SKCM': ['NRAS'],
        'STAD': ['TP53'],
        'BLCA': ['TP53'],
        'KIRC': ['VHL', 'PBRM1'],
        'OV': ['TP53'],
        'THCA': ['NRAS'],
        'PRAD': ['TP53', 'SPOP'],
        'CESC': ['PIK3CA'],
        'SARC': ['TP53'],
        'ESCA': ['TP53'],
        'PCPG': ['HRAS'],
        'TGCT': ['KIT'],
        'THYM': ['HRAS'],
        'KICH': ['TP53'],
        'ACC': ['TP53'],
        'MESO': ['BAP1'],
        'UCEC': ['PTEN', 'PIK3CA', 'ARID1A', 'TP53', 'PIK3R1'],
        'UVM': ['GNAQ'],
        'UCS': ['TP53'],
        'PAAD': ['KRAS', 'TP53'],
        'BRCA': ['TP53', 'PIK3CA', 'CDH1', 'GATA3']
    }
    # CDH1, GATA3, PIK3CA, TP53, KRAS, ARID1A, PIK3R1, PTEN, APC, ATRX, IDH1
    VALID_GENES_BAK2 = {
        'GBM': ['TP53', 'PTEN'],
        'COAD': ['APC', 'TP53', 'KRAS'],
        'HNSC': ['TP53'],
        'LGG': ['IDH1', 'TP53', 'ATRX'],
        'LIHC': ['TP53'],
        'LUAD': ['TP53', 'KRAS'],
        'LUSC': ['TP53'],
        'READ': ['APC', 'TP53'],
        'STAD': ['TP53'],
        'BLCA': ['TP53'],
        'OV': ['TP53'],
        'PRAD': ['TP53'],
        'CESC': ['PIK3CA'],
        'SARC': ['TP53'],
        'ESCA': ['TP53'],
        'KICH': ['TP53'],
        'ACC': ['TP53'],
        'UCEC': ['PTEN', 'PIK3CA', 'ARID1A', 'TP53', 'PIK3R1'],
        'UCS': ['TP53'],
        'PAAD': ['KRAS', 'TP53'],
        'BRCA': ['TP53', 'PIK3CA', 'CDH1', 'GATA3']
    }
    VALID_GENES = {
        'GBM': ['TP53', 'PTEN'],
        'COAD': ['APC', 'TP53', 'KRAS', 'PIK3CA'],
        'HNSC': ['TP53'],
        'KIRP': [],
        'LGG': ['IDH1', 'TP53', 'ATRX'],
        'LIHC': ['TP53'],
        'LUAD': ['TP53', 'KRAS'],
        'LUSC': ['TP53'],
        'READ': ['APC', 'TP53'],
        'SKCM': ['BRAF'],
        'STAD': ['TP53'],
        'BLCA': ['TP53'],
        'KIRC': [],
        'OV': ['TP53'],
        'THCA': ['BRAF'],
        'PRAD': ['TP53'],
        'CESC': ['PIK3CA'],
        'SARC': ['TP53'],
        'ESCA': ['TP53'],
        'PCPG': [],
        'TGCT': [],
        'THYM': [],
        'KICH': ['TP53'],
        'ACC': ['TP53'],
        'MESO': [],
        'UCEC': ['PTEN', 'PIK3CA', 'ARID1A', 'TP53'],
        'UVM': [],
        'DLBC': [],
        'UCS': ['TP53'],
        'CHOL': [],
        'PAAD': ['KRAS', 'TP53'],
        'BRCA': ['TP53', 'PIK3CA', 'CDH1', 'GATA3']
    }

    gene_symbols = {
        'TP53': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'PIK3CA': {'Loss_Or_Unknown_Or_NaN': 0, 'gain': 1, 'Other': 2},
        'PTEN': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'KRAS': {'Loss_Or_Unknown_Or_NaN': 0, 'gain': 1, 'Other': 2},
        'ARID1A': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'BRAF': {'Loss_Or_Unknown_Or_NaN': 0, 'gain': 1, 'Other': 2},
        'APC': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'IDH1': {'Gain_Or_Loss_Or_Unknown_Or_NaN': 0, 'switch': 1, 'Other': 2},
        'KMT2D': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'FBXW7': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'CDKN2A': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'NF1': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'RB1': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'KMT2C': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'ATRX': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'CTNNB1': {'Loss_Or_Unknown_Or_NaN': 0, 'gain': 1, 'Other': 2},
        'NRAS': {'Loss_Or_Unknown_Or_NaN': 0, 'gain': 1, 'Other': 2},
        'FAT1': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'PBRM1': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'PIK3R1': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'ATM': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'RNF43': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'EGFR': {'Loss_Or_Unknown_Or_NaN': 0, 'gain': 1, 'Other': 2},
        'KDM6A': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'ARID2': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'CDH1': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'SETD2': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'CTCF': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'EP300': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'NFE2L2': {'Loss_Or_Unknown_Or_NaN': 0, 'gain': 1, 'Other': 2},
        'CIC': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'SMAD4': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'VHL': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'KMT2B': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'GATA3': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'CREBBP': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'ZFHX3': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'ERBB2': {'Loss_Or_Unknown_Or_NaN': 0, 'gain': 1, 'Other': 2},
        'MAP3K1': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'JAK1': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'NSD1': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'STAG2': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'KMT2A': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'BRCA2': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'MGA': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'BAP1': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'MSH6': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'HRAS': {'Loss_Or_Unknown_Or_NaN': 0, 'gain': 1, 'Other': 2},
        'SPOP': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'B2M': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'NOTCH1': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'BCORL1': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2},
        'CASP8': {'Gain_Or_Unknown_Or_NaN': 0, 'loss': 1, 'Other': 2}
    }

    gene_df = pd.read_csv(os.path.join(DATA_ROOT2, 'mutation_effects_sorted{}.csv'.format(MUTATION_VERSION)), low_memory=False)
    gene_data = {}
    for gene_symbol in gene_symbols.keys():
        df1 = gene_df[gene_df['Hugo_Symbol'] == gene_symbol]
        if len(df1) > 0:
            columns = [col for col in df1.columns if 'TCGA' in col]
            dicts = {col[:12]: df1[col].values.astype(np.str_)[0].lower() for col in columns}
            gene_data[gene_symbol] = dicts

    df = one_case_df.copy()

    for gene_symbol, labels_dict in gene_symbols.items():
        if gene_symbol not in gene_data:
            continue
        if PREFIX not in VALID_GENES or len(VALID_GENES[PREFIX]) == 0 or \
                gene_symbol.lower() not in ''.join(VALID_GENES[PREFIX]).lower():
            continue
        values = []
        labels_dict_reverse = {v: k.lower() for k, v in labels_dict.items()}
        for case_id in df.index.values:
            case_id = case_id[:12]
            if case_id in gene_data[gene_symbol]:
                label = gene_data[gene_symbol][case_id]
                if 'gain' in label: label = 'gain'
                if 'loss' in label: label = 'loss'
                if 'switch' in label: label = 'switch'
                if 'nan' in label: label = 'nan'
                if 'unknown' in label or 'inconclusive' in label: label = 'unknown'
                found = False
                for k, v in labels_dict_reverse.items():
                    if label in v:
                        values.append(k)
                        found = True
                        break
                if not found:
                    print(label)
                    values.append(0)
            else:
                values.append(2)

        if len(values) > 0 and len(np.unique(values)) > 1:
            df['{}_cls'.format(gene_symbol)] = values

    columns = [col for col in df.columns if '_cls' in col]
    orig_stdout = sys.stdout
    log_fp = open(os.path.join(SAVE_ROOT, 'GENE_CLS_STATISTICS.txt'), 'w')
    sys.stdout = log_fp
    for col in columns:
        print('=' * 20)
        print(df[col].value_counts())
    log_fp.close()
    sys.stdout = orig_stdout
    df.to_csv(save_filename)

    print('done')


def step_add_cbioportal_data():
    save_filename = os.path.join(SAVE_ROOT, 'all_with_fpkm_withTIDECytoSig_withMPP_withGene_withCBIO.csv')
    if os.path.exists(save_filename):
        return

    print('begin add gene mutations for classification')
    df = pd.read_csv(os.path.join(SAVE_ROOT, 'all_with_fpkm_withTIDECytoSig_withMPP_withGene.csv'), index_col=0,
                     low_memory=False)

    cbio_dict = {
        "UCEC": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_data/ucec_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
        "THCA": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_data/thca_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
        "LUAD": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_data/luad_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
        "KIRP": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_data/kirp_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
        "LGG": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_data/lgg_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
        "GBM": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_data/gbm_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
        "LIHC": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_data/lihc_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
        "BLCA": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_data/blca_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
        "PRAD": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_data/prad_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
        "SARC": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_data/sarc_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
        "STAD": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_data/stad_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
        "PAAD": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_data/paad_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
        "MESO": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_data/meso_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
        "LUSC": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_data/lusc_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
        "KIRC": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_data/kirc_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
        "OV": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_data/ov_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
        "HNSC": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_data/hnsc_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
        "UCS": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_data/ucs_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
        "UVM": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_data/uvm_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
        "ACC": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_data/acc_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
        "CHOL": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_data/chol_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
        "TGCT": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_data/tgct_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
        "THYM": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_data/thym_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
        "KICH": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_data/kich_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
        "SKCM": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_data/skcm_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
        "ESCA": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_data/esca_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
        "BRCA": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_data/brca_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
        "DLBC": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_data/dlbc_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
        "PCPG": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_data/pcpg_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
        "COAD": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_data/coadread_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
        "READ": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_data/coadread_tcga_pan_can_atlas_2018/data_clinical_patient.txt",
        "CESC": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_data/cesc_tcga_pan_can_atlas_2018/data_clinical_patient.txt"
    }

    cbio_firehose_dict = {
        "DLBC": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_firehose_data/dlbc_tcga/data_clinical_patient.txt",
        "LUSC": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_firehose_data/lusc_tcga/data_clinical_patient.txt",
        "CHOL": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_firehose_data/chol_tcga/data_clinical_patient.txt",
        "COAD": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_firehose_data/coadread_tcga/data_clinical_patient.txt",
        "READ": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_firehose_data/coadread_tcga/data_clinical_patient.txt",
        "CESC": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_firehose_data/cesc_tcga/data_clinical_patient.txt",
        "THYM": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_firehose_data/thym_tcga/data_clinical_patient.txt",
        "LAML": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_firehose_data/laml_tcga/data_clinical_patient.txt",
        "PRAD": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_firehose_data/prad_tcga/data_clinical_patient.txt",
        "MESO": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_firehose_data/meso_tcga/data_clinical_patient.txt",
        "BRCA": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_firehose_data/brca_tcga/data_clinical_patient.txt",
        "LGG": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_firehose_data/lgg_tcga/data_clinical_patient.txt",
        "ESCA": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_firehose_data/esca_tcga/data_clinical_patient.txt",
        "KIRP": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_firehose_data/kirp_tcga/data_clinical_patient.txt",
        "SKCM": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_firehose_data/skcm_tcga/data_clinical_patient.txt",
        "OV": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_firehose_data/ov_tcga/data_clinical_patient.txt",
        "KICH": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_firehose_data/kich_tcga/data_clinical_patient.txt",
        "PAAD": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_firehose_data/paad_tcga/data_clinical_patient.txt",
        "GBM": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_firehose_data/gbm_tcga/data_clinical_patient.txt",
        "STAD": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_firehose_data/stad_tcga/data_clinical_patient.txt",
        "LIHC": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_firehose_data/lihc_tcga/data_clinical_patient.txt",
        "BLCA": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_firehose_data/blca_tcga/data_clinical_patient.txt",
        "SARC": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_firehose_data/sarc_tcga/data_clinical_patient.txt",
        "UCEC": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_firehose_data/ucec_tcga/data_clinical_patient.txt",
        "KIRC": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_firehose_data/kirc_tcga/data_clinical_patient.txt",
        "ACC": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_firehose_data/acc_tcga/data_clinical_patient.txt",
        "HNSC": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_firehose_data/hnsc_tcga/data_bcr_clinical_data_patient.txt",
        "LUAD": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_firehose_data/luad_tcga/data_bcr_clinical_data_patient.txt",
        "PCPG": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_firehose_data/pcpg_tcga/data_bcr_clinical_data_patient.txt",
        "TGCT": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_firehose_data/tgct_tcga/data_bcr_clinical_data_patient.txt",
        "THCA": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_firehose_data/thca_tcga/data_bcr_clinical_data_patient.txt",
        "UCS": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_firehose_data/ucs_tcga/data_bcr_clinical_data_patient.txt",
        "UVM": "/data/Jiang_Lab/Data/Zisha_Zhong/BigData/cbioportal_firehose_data/uvm_tcga/data_bcr_clinical_data_patient.txt"
    }

    if PREFIX in cbio_dict:
        # using the survival_months from cbioportal
        print('using the survival_months from cbioportal')
        cbioportal_txt_filename = cbio_dict[PREFIX]
        if os.path.exists(cbioportal_txt_filename):
            cbio_df = pd.read_csv(cbioportal_txt_filename, sep='\t', skiprows=4)

            dfs = {}
            for index, name in enumerate(df.index.values):
                for ind2, name2 in enumerate(cbio_df.PATIENT_ID.values):
                    if name[:12] == name2[:12]:
                        dfs[name] = cbio_df.iloc[ind2]
                        break
            df1 = pd.DataFrame(dfs).T
            df = df.join(df1)

    if PREFIX in cbio_firehose_dict:
        print('using the survival_months from cbioportal')
        cbioportal_txt_filename = cbio_firehose_dict[PREFIX]
        if os.path.exists(cbioportal_txt_filename):
            cbio_df = pd.read_csv(cbioportal_txt_filename, sep='\t', skiprows=4)

            dfs = {}
            for index, name in enumerate(df.index.values):
                for ind2, name2 in enumerate(cbio_df.PATIENT_ID.values):
                    if name[:12] == name2[:12]:
                        dfs[name] = cbio_df.iloc[ind2]
                        break
            df1 = pd.DataFrame(dfs).T
            print('merge cbio firehose data')
            df = df.combine_first(df1)

    df.to_csv(save_filename)


def add_CLAM_information():
    save_filename = os.path.join(SAVE_ROOT, 'all_with_fpkm_withTIDECytoSig_withMPP_withGene_withCBIO_withCLAM.csv')
    if os.path.exists(save_filename):
        return
    print('begin CLAM for {}'.format(PREFIX))
    df = pd.read_csv(os.path.join(SAVE_ROOT, 'all_with_fpkm_withTIDECytoSig_withMPP_withGene_withCBIO.csv'),
                     low_memory=False)

    if 'slide_id' not in df.columns:
        slide_ids = [os.path.basename(f).replace('.svs', '') for f in df['DX_filename'].values]
        df['slide_id'] = slide_ids
    if 'case_id' not in df.columns:
        case_ids = [os.path.basename(f)[:12] for f in df['DX_filename'].values]
        df['case_id'] = case_ids

    invalid_CLIDs = set([
        'TCGA-5P-A9KA',
        'TCGA-5P-A9KC',
        'TCGA-HT-7483',
        'TCGA-UZ-A9PQ'
    ])

    invalid_inds = []
    for ind, DX_filename in enumerate(df['DX_filename'].values):
        fileprefix = os.path.basename(DX_filename).replace('.svs', '')
        if fileprefix[:12] in invalid_CLIDs:
            invalid_inds.append(ind)
    try:
        if len(invalid_inds) > 0:
            df = df.drop(invalid_inds).reset_index(drop=True)
    except:
        pdb.set_trace()

    df.to_csv(save_filename)

    print('add CLAM done')


def process_BRCA_data():
    if 'BRCA' not in PREFIX:
        return

    save_filename = os.path.join(SAVE_ROOT, 'all_with_fpkm_withTIDECytoSig_withMPP_withGene_withCBIO_withCLAM.csv')
    if os.path.exists(save_filename):
        shutil.copyfile(save_filename, save_filename.replace('.csv', '_bak.csv'))

    print('begin BRCA processing for {}'.format(PREFIX))
    df = pd.read_csv(os.path.join(SAVE_ROOT, 'all_with_fpkm_withTIDECytoSig_withMPP_withGene_withCBIO_withCLAM.csv'),
                     low_memory=False)

    SUBTYPES_DICT = {
        'LumA': 0,
        'LumB': 1,
        'Basal': 2,
        'HER2E': 3,
        'normal-like': 4,
        'normal': 4,
        'CLOW': 4
    }
    df = df[df['PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype'].isin(SUBTYPES_DICT.keys())]
    df['CLS_Molecular_Subtype'] = \
        df['PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype'].map(SUBTYPES_DICT)

    IHC_HER2_dict = {k: 2 for k in df['HER2.newly.derived'].value_counts().index}
    IHC_HER2_dict['Positive'] = 1
    IHC_HER2_dict['Negative'] = 0
    df['CLS_IHC_HER2'] = df['HER2.newly.derived'].map(IHC_HER2_dict)

    HistoAnno_dict = {k: 2 for k in df['2016_Histology_Annotations'].value_counts().index}
    HistoAnno_dict['Invasive ductal carcinoma'] = 1
    HistoAnno_dict['Invasive lobular carcinoma'] = 0
    df['CLS_HistoAnno'] = df['2016_Histology_Annotations'].map(HistoAnno_dict)

    df.to_csv(save_filename)
    print('add CLAM done')


if __name__ == '__main__':
    if True:
        step_1_check_brca()
        step_2_get_fpkm()
        step_3_prepare_TIDE_scores_v3()

        step_6_add_CTL_and_TIDE_v2()

        step_7_check_brca_svs_files()

        step_add_gene_mutation_classification()
        step_add_cbioportal_data()

        add_CLAM_information()
