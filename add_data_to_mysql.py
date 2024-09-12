
import sys,os,shutil,io,time,json,pickle,gc,glob
import numpy as np
import h5py
import torch
import idr_torch
from common import BACKBONE_DICT
import openslide
from utils import _assertLevelDownsamplesV2
import faiss
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 12660162500
from PIL import Image, ImageFile, ImageDraw
ImageFile.LOAD_TRUNCATED_IMAGES = True




def merge_background_samples_for_deployment_v2():
    import os
    import pickle
    import numpy as np
    data = {}
    for method in ['HERE_ProvGigaPath', 'HERE_CONCH', 'HERE_PLIP', 'HERE_UNI']:
        if method not in data:
            data[method] = {}
        version = 'V6'
        for project_name in ['KenData_20240814', 'ST', 'TCGA-COMBINED']:
            filename = f'randomly_background_samples_for_train_{project_name}_{method}{version}.pkl'
            if project_name in data[method] or not os.path.exists(filename):
                print('wrong')
                import pdb
                pdb.set_trace()
                continue
            with open(filename, 'rb') as fp:
                data1 = pickle.load(fp)
            if project_name == 'TCGA-COMBINED' or project_name == 'KenData_20240814':
                embeddings = data1[method][project_name]['embeddings']
                data[method][project_name] = embeddings[np.random.randint(0, len(embeddings), 10000), :]
            else:
                data[method][project_name] = data1[method][project_name]['embeddings']
        data[method]['ALL'] = np.concatenate([
            vv for kk, vv in data[method].items() 
        ])
    with open('randomly_1000_data_with_PLIP_ProvGigaPath_CONCH_20240814.pkl', 'wb') as fp:
        pickle.dump(data, fp)




def get_all_scales_20240813():

    project_names = ['TCGA-COMBINED', 'KenData_20240814', 'ST'] # get_project_names()
    project_names = ['TCGA-COMBINED', 'KenData_20240814', 'ST_20240903'] # get_project_names()
    version = '20240814'
    version = '20240908'

    all_results = {}
    image_ext = '.svs'
    for project_name in project_names:

        assert project_name in project_names, 'check project_name'

        # allfeats_dir, image_ext, svs_dir, patches_dir = get_allfeats_dir(
        #     project_name)
        svs_dir = f'/data/zhongz2/{project_name}_256/svs'
        patches_dir = f'/data/zhongz2/{project_name}_256/patches'

        h5filenames = sorted(glob.glob(patches_dir + '/*.h5'))
        for file_index, h5filename in enumerate(h5filenames):

            svs_prefix = os.path.basename(h5filename).replace('.h5', '')
            svs_filename = os.path.join(svs_dir, svs_prefix + image_ext)
            slide = openslide.open_slide(svs_filename)

            h5file = h5py.File(h5filename, 'r')
            patch_level = h5file['coords'].attrs['patch_level']
            patch_size = h5file['coords'].attrs['patch_size']
            h5file.close()

            if len(slide.level_dimensions) == 1:
                vis_level = 0
            else:
                dimension = slide.level_dimensions[1] if len(
                    slide.level_dimensions) > 1 else slide.level_dimensions[0]
                if dimension[0] > 100000 or dimension[1] > 100000:
                    vis_level = 2
                else:
                    vis_level = 1
                if len(slide.level_dimensions) == 1:
                    vis_level = 0

            level_downsamples = _assertLevelDownsamplesV2(
                slide.level_dimensions, slide.level_downsamples)
            downsample0 = level_downsamples[patch_level]
            downsample = level_downsamples[vis_level]
            scale = [downsample0[0] / downsample[0], downsample0[1] /
                     downsample[1]]  # Scaling from 0 to desired level
            # scale = [1 / downsample[0], 1 / downsample[1]]
            patch_size_vis_level = np.ceil(patch_size * scale[0]).astype(int)

            all_results['{}_{}'.format(project_name, svs_prefix)] = {
                'patch_level': patch_level,
                'patch_size': patch_size,
                'vis_level': vis_level,
                'level_dimensions': slide.level_dimensions,
                'level_downsamples': slide.level_downsamples,
                'scale': scale,
                'patch_size_vis_level': patch_size_vis_level
            }

            slide.close()

    save_dir = f'/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240801/'
    os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_dir}/all_scales_{version}_newKenData.pkl', 'wb') as fp:  # TCGA-COMBINED
        pickle.dump(all_results, fp)


def gen_faiss_infos_to_mysqldb_v2(): # combined to reduce memory
    # tcga_names = ["TCGA-ACC", "TCGA-BLCA", "TCGA-BRCA", "TCGA-CESC", "TCGA-CHOL",
    #               "TCGA-COAD", "TCGA-DLBC", "TCGA-ESCA", "TCGA-GBM", "TCGA-HNSC",
    #               "TCGA-KICH", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LGG", "TCGA-LIHC",
    #               "TCGA-LUAD", "TCGA-LUSC", "TCGA-MESO", "TCGA-OV", "TCGA-PAAD",
    #               "TCGA-PCPG", "TCGA-PRAD", "TCGA-READ", "TCGA-SARC", "TCGA-SKCM",
    #               "TCGA-STAD", "TCGA-TGCT", "TCGA-THCA", "TCGA-THYM", "TCGA-UCEC",
    #               "TCGA-UCS", "TCGA-UVM"]
    # project_names = ['Adoptive_TIL_Breast', 'TransNEO', 'WiemDataCheck', 'METABRIC',
    #                  'ST', 'ShebaV3', 'BintrafuspAlfa', 'Mouse'] + tcga_names + ['KenData']

    import os,h5py,glob,time,pickle
    import pymysql
    import numpy as np
    import json
    import pandas as pd

    project_names = ['TCGA-COMBINED', 'KenData_20240814', 'ST'] # get_project_names()
    project_names = ['TCGA-COMBINED', 'KenData_20240814', 'ST_20240903'] # get_project_names()

    data_root = 'data_HiDARE_PLIP_20240208'
    version = '20240812' # for TCGA-COMBINED
    version = '20240814' # for KenData_20240814
    version = '20240903' # for ST_20240903

    DB_USER = os.environ['HERE_DB_USER']
    DB_PASSWORD = os.environ['HERE_DB_PASSWORD']
    DB_HOST = os.environ['HERE_DB_HOST']
    DB_DATABASE = os.environ['HERE_DB_DATABASE']

    conn = pymysql.connect(user=DB_USER, password=DB_PASSWORD, host=DB_HOST, database=DB_DATABASE)
    conn.autocommit = False
    cur = conn.cursor()

    sql_command = f'CREATE TABLE IF NOT EXISTS image_table_{version} '\
        '(rowid BIGINT NOT NULL PRIMARY KEY AUTO_INCREMENT, '\
            'project_id INT NOT NULL, '\
                'svs_prefix_id INT NOT NULL, svs_prefix VARCHAR(1024) NOT NULL, '\
                    'scale FLOAT NOT NULL, patch_size_vis_level SMALLINT NOT NULL, external_link VARCHAR(2048), note TEXT);'
    cur.execute(sql_command)
    sql_command = f'CREATE TABLE IF NOT EXISTS project_table_{version} '\
        '(rowid BIGINT NOT NULL PRIMARY KEY AUTO_INCREMENT, '\
            'project_id INT NOT NULL, project_name VARCHAR(256) NOT NULL);'
    cur.execute(sql_command)

    with open(f'{data_root}/assets/all_scales_{version}_newKenData.pkl', 'rb') as fp:
        all_scales = pickle.load(fp) # proj_name+svs_prefix
    with open(f'{data_root}/assets/all_notes_all.pkl', 'rb') as fp:
        all_notes = pickle.load(fp) # svs_prefix
    case_uuids = {}
    with open(f'{data_root}/assets/metadata.repository.2024-08-13.json', 'r') as fp:
        case_uuids = {item['file_name'].replace('.svs', ''): item['associated_entities'][0]['case_id'] for item in json.load(fp)}
    ST_df = pd.read_excel(f'{data_root}/assets/ST_list_cancer.xlsx')
    ST_df = pd.read_excel(f'{data_root}/assets/ST_{version}.xlsx')

    all_items = []
    project_items = []
    for proj_id, project_name in enumerate(project_names):
        project_items.append((proj_id, project_name))
        print(f'begin {project_name}')
        patches_dir = os.path.join(
            data_root, 'assets', 'all_patches', project_name, 'patches')

        h5filenames = sorted(glob.glob(patches_dir + '/*.h5'))

        for svs_prefix_id, h5filename in enumerate(h5filenames):
            
            svs_prefix = os.path.basename(h5filename).replace('.h5', '')

            key = '{}_{}'.format(project_name, svs_prefix)
            if key in all_scales:
                scale = all_scales[key]['scale'][0]
                patch_size_vis_level = all_scales[key]['patch_size_vis_level']
            else:
                scale = 1.0
                patch_size_vis_level = 256
            note = all_notes[svs_prefix] if svs_prefix in all_notes else svs_prefix
            # if 'TCGA' == svs_prefix[:4] and svs_prefix in case_uuids:
            #     note = f'Link: <a href=\"https://portal.gdc.cancer.gov/cases/{case_uuids[svs_prefix]}\" target=\"_blank\">{svs_prefix}</a>\n\n' + note
            external_link = ''
            if project_name == 'ST':
                external_link = ST_df.loc[ST_df['ID']==svs_prefix, 'Source'].values[0]
            elif project_name == 'TCGA-COMBINED':
                external_link = f'https://portal.gdc.cancer.gov/cases/{case_uuids[svs_prefix]}' if svs_prefix in case_uuids else ''
            all_items.append((proj_id, svs_prefix_id, svs_prefix, scale, patch_size_vis_level, external_link, note))
    cur.executemany(f"INSERT INTO project_table_{version} (project_id, project_name) VALUES (%s, %s)",
                    project_items)
    conn.commit()
    cur.executemany(f"INSERT INTO image_table_{version} (project_id, svs_prefix_id, svs_prefix, scale, patch_size_vis_level, external_link, note) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    all_items)
    conn.commit()



    #################
    sql_command = f'CREATE TABLE IF NOT EXISTS faiss_table_{version} '\
        '(rowid BIGINT NOT NULL PRIMARY KEY AUTO_INCREMENT, '\
            'x INT NOT NULL, '\
                'y INT NOT NULL, svs_prefix_id INT NOT NULL, project_id INT NOT NULL);'
    cur.execute(sql_command)
    
    total_count = 0 
    for proj_id, project_name in enumerate(project_names):
        print(f'begin {project_name}')
        patches_dir = os.path.join(
            data_root, 'assets', 'all_patches', project_name, 'patches')

        h5filenames = sorted(glob.glob(patches_dir + '/*.h5'))

        for svs_prefix_id, h5filename in enumerate(h5filenames):

            svs_prefix = os.path.basename(h5filename).replace('.h5', '')

            with h5py.File(h5filename, 'r') as file:
                coords = file['coords'][()].astype(np.int32)

            total_count += len(coords)
            svs_prefix_ids = svs_prefix_id * \
                np.ones((len(coords), 1), dtype=np.int32)
            project_ids = proj_id * np.ones((len(coords), 1), dtype=np.int32) 

            cur.executemany(f"INSERT INTO faiss_table_{version} (x, y, svs_prefix_id, project_id) VALUES (%s, %s, %s, %s)",
                            np.concatenate([coords, svs_prefix_ids, project_ids], axis=1).tolist())
            conn.commit()

            if svs_prefix_id % 1000 == 0:
                print(svs_prefix_id, svs_prefix)
        print(f'end {project_name}')


    conn.close()
    time.sleep(1)









def prepare_hidare_mysqldb():
    import pymysql 

    DB_USER = os.environ['HERE_DB_USER']
    DB_PASSWORD = os.environ['HERE_DB_PASSWORD']
    DB_HOST = os.environ['HERE_DB_HOST']
    DB_DATABASE = os.environ['HERE_DB_DATABASE']

    conn = pymysql.connect(user=DB_USER, password=DB_PASSWORD, host=DB_HOST, database=DB_DATABASE)
    conn.autocommit = False
    cur = conn.cursor()
    
    version = '_20240814'
    sql_commands=[
    f'DROP TABLE IF EXISTS gene_table{version};',
    f'DROP TABLE IF EXISTS st_table{version};',
    f'DROP TABLE IF EXISTS cluster_setting_table{version};',    
    f'DROP TABLE IF EXISTS cluster_table{version};',
    f'DROP TABLE IF EXISTS cluster_result_table{version};',
    f'''CREATE TABLE IF NOT EXISTS gene_table{version} (
    id BIGINT NOT NULL PRIMARY KEY AUTO_INCREMENT, 
    symbol VARCHAR(128) NOT NULL, 
    alias VARCHAR(128) NOT NULL);''',
    f'''CREATE TABLE IF NOT EXISTS st_table{version} (
    id BIGINT NOT NULL PRIMARY KEY AUTO_INCREMENT, 
    prefix VARCHAR(1024) NOT NULL);''',
    f'''CREATE TABLE IF NOT EXISTS cluster_setting_table{version} (
    id BIGINT NOT NULL PRIMARY KEY AUTO_INCREMENT, 
    cluster_setting VARCHAR(1024) NOT NULL);''',
    f'''CREATE TABLE IF NOT EXISTS cluster_table{version} (
    id BIGINT NOT NULL PRIMARY KEY AUTO_INCREMENT, 
    st_id BIGINT REFERENCES st_table{version}(id),
    cs_id BIGINT REFERENCES cluster_setting_table{version}(id),
    cluster_label INT NOT NULL,
    cluster_info MEDIUMTEXT NOT NULL);''',
    f'''CREATE TABLE IF NOT EXISTS cluster_result_table{version} (
    id BIGINT NOT NULL PRIMARY KEY AUTO_INCREMENT, 
    c_id BIGINT REFERENCES cluster_table{version}(id),
    gene_id BIGINT REFERENCES gene_table{version}(id),
    cohensd FLOAT NOT NULL,
    pvalue FLOAT NOT NULL,
    pvalue_corrected FLOAT NOT NULL,
    zscore FLOAT NOT NULL);
    '''
    ]
    for sql_command in sql_commands:
        print('begin')
        cur.execute(sql_command)
        print('end')
    conn.close()


def add_ST_data_to_mysqldb():

    import pymysql
    import pandas as pd
    import os
    import glob
    import pickle
    import numpy as np


    # val = input("Did you create tables in HiDARE database? [Yes/No]")
    # if 'yes' not in val:
    #     print('check the function prepare_hidare_mariadb()')
    #     return 

    version = ''
    version = '_20240814'

    root = '/mnt/hidare-efs/data/differential_results/ST/20231030v2_ST/PanCancer2GPUsFP/shared_attention_imagenetmobilenetv3/split3_e1_h224_density_vis/feat_before_attention_feat/test/'
    root = '/mnt/hidare-efs/data_20240208/differential_analysis/20240202v4_ST/PanCancer2GPUsFP/shared_attention_imagenetPLIP/split1_e95_h224_density_vis/feat_before_attention_feat/test'
    root = '/mnt/hidare-efs/data_20240208/ST_kmeans_clustering/'
    root = '/mnt/hidare-efs/data/HERE_assets/assets/ST_kmeans_clustering/'
    root = ''

    all_prefixes = [os.path.basename(f).replace(
        '.tsv', '') for f in sorted(glob.glob(f'{root}/vst_dir/*.tsv'))]
    
    df = pd.read_excel(f'{root}/../ST_list_cancer.xlsx')
    all_prefixes = df['ID'].values.tolist()

    DB_USER = os.environ['HERE_DB_USER']
    DB_PASSWORD = os.environ['HERE_DB_PASSWORD']
    DB_HOST = os.environ['HERE_DB_HOST']
    DB_DATABASE = os.environ['HERE_DB_DATABASE']

    conn = pymysql.connect(user=DB_USER, password=DB_PASSWORD, host=DB_HOST, database=DB_DATABASE)
    conn.autocommit = False
    cur = conn.cursor()

    # add ST table
    for prefix in all_prefixes:
        cur.execute(f"INSERT INTO st_table{version} (prefix) VALUES (%s)", (prefix,))
        conn.commit()

    num_clusters = [8, 12, 16, 20]
    keep_thresholds = [1, 10, 20, 30, 50]
    dimension_reduction_methods = ['umap3d', 'pca3d']
    clustering_methods = ['kmeans', 'hierarchical']
    num_clusters = [8]
    keep_thresholds = [1]
    dimension_reduction_methods = ['umap3d']
    clustering_methods = ['hierarchical']

    num_clusters = [8]
    keep_thresholds = [1]
    dimension_reduction_methods = ['none']
    clustering_methods = ['hierarchical']
    feature_normalization_type = 'meanstd'
    clustering_distance_metric = 'euclidean'

    # 20240814
    num_clusters = [8]
    keep_thresholds = [1]
    dimension_reduction_methods = ['none']
    clustering_methods = ['kmeans']
    feature_normalization_type = 'meanstd'
    clustering_distance_metric = 'euclidean'

    #
    all_items = []
    gene_symbol_dict = {}
    st_dict = {}
    cs_dict = {}
    for prefix in all_prefixes:
        print(prefix)
        if prefix not in st_dict:
            cur.execute(
                f'select id from st_table{version} where prefix = %s', (prefix, ))
            result = cur.fetchone()
            if result is None:
                cur.execute(
                    f'insert into st_table{version} (prefix) value (%s)', (prefix, ))
                conn.commit()
                cur.execute(
                    f'select id from st_table{version} where prefix = %s', (prefix, ))
                result = cur.fetchone()
            st_dict[prefix] = result[0]
        st_id = st_dict[prefix]

        gene_data_filename = f'{root}/gene_data/{prefix}_gene_data.pkl'
        if not os.path.exists(gene_data_filename):
            continue

        with open(gene_data_filename, 'rb') as fp:
            gene_data_dict = pickle.load(fp)

        # gene_data_dict['gene_data_vst']
        vst_filename = f'{root}/vst_dir/{prefix}.tsv'
        # coord_filename = gene_data_dict['coord_df']
        # counts_filename = gene_data_dict['counts_df']
        barcode_col_name = gene_data_dict['barcode_col_name']
        Y_col_name = gene_data_dict['Y_col_name']
        X_col_name = gene_data_dict['X_col_name']
        mpp = gene_data_dict['mpp']
        coord_df = gene_data_dict['coord_df']
        counts_df = gene_data_dict['counts_df']

        if not os.path.exists(vst_filename):
            continue
        vst = pd.read_csv(vst_filename, sep='\t', index_col=0)
        vst = vst.subtract(vst.mean(axis=1), axis=0)

        # only use the spots in the tissue
        # coord_df = coord_df[coord_df[barcode_col_name].isin(vst.columns)]
        # coord_df = coord_df.reset_index(drop=True)

        barcodes = coord_df[barcode_col_name].values.tolist()
        stY = coord_df[Y_col_name].values.tolist()
        stX = coord_df[X_col_name].values.tolist()

        st_patch_size = int(
            pow(2, np.ceil(np.log(64 / mpp) / np.log(2))))
        st_all_coords = np.array([stX, stY]).T
        st_all_coords[:, 0] -= st_patch_size // 2
        st_all_coords[:, 1] -= st_patch_size // 2
        st_all_coords = st_all_coords.astype(np.int32)

        vst = vst.T
        vst.index.name = 'barcode'
        valid_barcodes = set(vst.index.values.tolist())
        print(len(valid_barcodes))

        for num_cluster in num_clusters:
            for keep_threshold in keep_thresholds:
                for dimension_reduction_method in dimension_reduction_methods:
                    for clustering_method in clustering_methods:
                        cluster_setting = '{}_{}_{}_{}_{}_{}_clustering'.format(
                            feature_normalization_type, dimension_reduction_method, clustering_method, clustering_distance_metric, num_cluster, keep_threshold)

                        if cluster_setting not in cs_dict:
                            cur.execute(
                                f'select id from cluster_setting_table{version} where cluster_setting = %s', (cluster_setting, ))
                            result = cur.fetchone()
                            if result is None:
                                cur.execute(
                                    f'insert into cluster_setting_table{version} (cluster_setting) value (%s)', (cluster_setting, ))
                                conn.commit()
                                cur.execute(
                                    f'select id from cluster_setting_table{version} where cluster_setting = %s', (cluster_setting, ))
                                result = cur.fetchone()
                            cs_dict[cluster_setting] = result[0]
                        cs_id = cs_dict[cluster_setting]

                        cluster_data_filename = f'{root}/analysis/one_patient_top_128/{cluster_setting}/{prefix}/{prefix}_cluster_data.pkl'
                        result_data_filename = f'{root}/analysis/one_patient_top_128/{cluster_setting}/{prefix}/{prefix}_cluster_tests.pkl'

                        if not os.path.exists(cluster_data_filename):
                            continue
                        if not os.path.exists(result_data_filename):
                            continue

                        with open(cluster_data_filename, 'rb') as fp:
                            cluster_data = pickle.load(fp)

                        cluster_coords = cluster_data['coords_in_original']
                        cluster_labels = cluster_data['cluster_labels']

                        with open(result_data_filename, 'rb') as fp:
                            # gene_names, labels, 0, 1, 2, ...
                            result_data = pickle.load(fp)

                        gene_names = result_data['gene_names']
                        gene_ids = []
                        for gene_name in gene_names:
                            if gene_name not in gene_symbol_dict:
                                cur.execute(
                                    f'select id from gene_table{version} where symbol = %s', (gene_name, ))
                                result = cur.fetchone()
                                if result is None:
                                    cur.execute(
                                        f'insert into gene_table{version} (symbol) value (%s)', (gene_name, ))
                                    conn.commit()
                                    cur.execute(
                                        f'select id from gene_table{version} where symbol = %s', (gene_name, ))
                                    result = cur.fetchone()
                                gene_symbol_dict[gene_name] = result[0]
                            gene_id = gene_symbol_dict[gene_name]
                            gene_ids.append(gene_id)

                        cluster_barcodes = []
                        innnvalid = 0
                        iinds = []
                        for iiii, (x, y) in enumerate(cluster_coords):
                            ind = np.where((st_all_coords[:, 0] == x) & (
                                st_all_coords[:, 1] == y))[0]
                            if len(ind) > 0:
                                barcoode = barcodes[ind[0]]
                                if barcoode in valid_barcodes:
                                    cluster_barcodes.append(barcoode)
                                    iinds.append(iiii)
                            else:
                                innnvalid += 1
                        cluster_labels = cluster_labels[iinds]
                        cluster_coords = cluster_coords[iinds]
                        vst1 = vst.loc[cluster_barcodes]
                        counts_df1 = counts_df.T
                        coord_df1 = coord_df.set_index(barcode_col_name)
                        coord_df1.index.name = 'barcode'

                        for label in result_data['labels'].tolist():
                            inds = np.where(cluster_labels == label)[0]
                            if len(inds) == 0:
                                continue
                            coords_this_label = cluster_coords[inds]  # nx2
                            barcodes_this_label = [
                                cluster_barcodes[iiii] for iiii in inds]
                            cluster_info = []
                            for barcode, coord in zip(barcodes_this_label, coords_this_label):
                                cluster_info.append('{},{},{}'.format(
                                    barcode, coord[0], coord[1]))
                            cur.execute(
                                f'insert into cluster_table{version} (st_id, cs_id, cluster_label, cluster_info) values (%s, %s, %s, %s)', (st_id, cs_id, label, '\n'.join(cluster_info)))
                            conn.commit()
                            cur.execute(
                                f'select id from cluster_table{version} where st_id = %s and cs_id = %s and cluster_label = %s', (st_id, cs_id, label))
                            result = cur.fetchone()
                            c_id = result[0]

                            dff = result_data[label].astype(float)
                            dff['c_id'] = [c_id for _ in range(len(dff))]
                            dff['gene_id'] = gene_ids
                            # dff = dff.fillna(100)
                            if not dff.isnull().values.any():

                                cur.executemany(
                                    f'INSERT INTO cluster_result_table{version} (zscore, pvalue, pvalue_corrected, cohensd, c_id, gene_id) VALUES (%s, %s, %s, %s, %s, %s)',
                                    list(dff.itertuples(index=False, name=None)))
                                conn.commit()



def generate_ST_vst_for_visualization():

    import os,h5py,glob,time,pickle
    import numpy as np
    import openslide
    import base64
    import pandas as pd
    import json
    import matplotlib.pyplot as plt

    # from https://github.com/mahmoodlab/CLAM
    def _assertLevelDownsamples(slide):
        level_downsamples = []
        dim_0 = slide.level_dimensions[0]

        for downsample, dim in zip(slide.level_downsamples, slide.level_dimensions):
            estimated_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
            level_downsamples.append(estimated_downsample) if estimated_downsample != (
                downsample, downsample) else level_downsamples.append((downsample, downsample))

        return level_downsamples

    ST_DIR = '/data/Jiang_Lab/Data/Zisha_Zhong/temp_20240202/differential_results/20240202v4_ST/PanCancer2GPUsFP/shared_attention_imagenetPLIP/split1_e95_h224_density_vis/feat_before_attention_feat/test'
    svs_dir = os.path.join(ST_DIR, 'svs')
    vst_dir = os.path.join(ST_DIR, 'vst_dir')
    vst_dir_db = os.path.join(ST_DIR, 'vst_dir_db')
    os.makedirs(vst_dir_db, exist_ok=True)
    spatial_dir = os.path.join(ST_DIR, 'spatial')
    gene_data_dir = os.path.join(ST_DIR, 'gene_data')

    project_name = 'ST'

    allfeats_dir, image_ext, svs_dir, patches_dir = get_allfeats_dir(project_name)

    h5filenames = [f for f in sorted(glob.glob(patches_dir + '/*.h5')) if not os.path.exists(os.path.join(vst_dir_db, os.path.basename(f).replace('.h5', '_original_VST.db')))]

    if 'idr_torch' in globals():
        index_splits = np.array_split(
            np.arange(len(h5filenames)), indices_or_sections=idr_torch.world_size)

        h5filenames_sub = [h5filenames[i] for i in index_splits[idr_torch.rank]]
    else:
        h5filenames_sub = h5filenames

    for _, h5filename in enumerate(h5filenames_sub):
        svs_prefix = os.path.basename(h5filename).replace('.h5', '')
        slide_name = svs_prefix
        vst_filename_db = f'{vst_dir_db}/{slide_name}.db'
        vst_filename_db_VST = f'{vst_dir_db}/{slide_name}_original_VST.db'
        if os.path.exists(vst_filename_db_VST) and os.path.exists(vst_filename_db):
            continue
        
        gene_data_filename = f'{gene_data_dir}/{slide_name}_gene_data.pkl'
        vst_filename = f'{vst_dir}/{slide_name}.tsv'
        svs_filename = f'{svs_dir}/{slide_name}.svs'
        if not os.path.exists(svs_filename):
            prefix1 = slide_name.replace('10x_', '')
            svs_filename = f'{svs_dir}/{prefix1}.svs'

        if not os.path.exists(gene_data_filename):
            continue
        if not os.path.exists(vst_filename):
            continue
        if not os.path.exists(svs_filename):
            continue

        with open(gene_data_filename, 'rb') as fp:
            gene_data_dict = pickle.load(fp)

        coord_df = gene_data_dict['coord_df']
        counts_df = gene_data_dict['counts_df']
        barcode_col_name = gene_data_dict['barcode_col_name']
        Y_col_name = gene_data_dict['Y_col_name']
        X_col_name = gene_data_dict['X_col_name']
        mpp = gene_data_dict['mpp']

        vst = pd.read_csv(vst_filename, sep='\t', index_col=0)
        vst = vst.subtract(vst.mean(axis=1), axis=0)
        vst = vst.T
        vst.index.name = '__barcode'
        vst.columns = [n.upper() for n in vst.columns]

        coord_df1 = coord_df.rename(columns={barcode_col_name: '__barcode', X_col_name: 'X', Y_col_name: 'Y'}).set_index('__barcode')
        coord_df1 = coord_df1.loc[vst.index.values]
        st_XY = coord_df1[['X', 'Y']].values 

        st_patch_size = int(pow(2, np.ceil(np.log(64 / mpp) / np.log(2))))

        if '10x_' in slide_name:
            with open('{}/10x/{}/spatial/scalefactors_json.json'.format(spatial_dir, slide_name.replace('10x_', '')), 'r') as fp:
                spot_size = float(json.load(fp)['spot_diameter_fullres'])
        elif 'TNBC_' in slide_name:
            with open('{}/TNBC/{}/spatial/scalefactors_json.json'.format(spatial_dir, slide_name), 'r') as fp:
                spot_size = float(json.load(fp)['spot_diameter_fullres'])
        else:
            spot_size = st_patch_size

        slide = openslide.open_slide(svs_filename)
        if len(slide.level_dimensions) == 1:
            vis_level = 0
        else:
            dimension = slide.level_dimensions[1] if len(slide.level_dimensions) > 1 else slide.level_dimensions[0]
            if dimension[0] > 100000 or dimension[1] > 100000:
                vis_level = 2
            else:
                vis_level = 1

        level_downsamples = _assertLevelDownsamples(slide)
        slide.close()

        downsample = level_downsamples[vis_level]
        scale = 1 / downsample[0]
        circle_radius = int(spot_size * scale * 0.5)
        st_XY_for_shown = (st_XY * scale).astype(np.int32)

        if os.path.exists(vst_filename_db_VST):
            continue

        vst1 = vst.copy()
        vst1 = vst1.astype('float32')
        vst1[['__spot_X', '__spot_Y']] = st_XY.astype('uint16')
        st_XY_upperleft = st_XY - st_patch_size//2
        vst1[['__upperleft_X', '__upperleft_Y']] = st_XY_upperleft.astype('uint16')
        vst1.to_parquet(vst_filename_db_VST, engine='fastparquet')
        del vst1

        if os.path.exists(vst_filename_db):
            continue

        low_percentile = vst.quantile(0.01)
        high_percentile = vst.quantile(0.99)

        vst = vst.apply(lambda col: col.clip(lower=low_percentile[col.name], upper=high_percentile[col.name]))
        vst = 255 * (vst - low_percentile) / (high_percentile - low_percentile)
        vst = vst.astype(np.uint8)

        vst[['__coordX', '__coordY']] = st_XY_for_shown
        vst['__circle_radius'] = circle_radius
        vst['__st_patch_size'] = st_patch_size
        vst['__circle_radius'] = vst['__circle_radius'].astype('uint16')
        vst['__st_patch_size'] = vst['__st_patch_size'].astype('uint16')
        vst.to_parquet(vst_filename_db, engine='fastparquet')


if __name__ == '__main__':
    merge_background_samples_for_deployment_v2()
    get_all_scales_20240813()
    prepare_hidare_mysqldb()
    add_ST_data_to_mysqldb()
    generate_ST_vst_for_visualization()
    


