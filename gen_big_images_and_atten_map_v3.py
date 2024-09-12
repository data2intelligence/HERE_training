import sys,os
import openslide
import idr_torch
import time
import pyvips
import pickle
import h5py
import pandas as pd
from utils import get_svs_prefix, visHeatmap, visWSI
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 12660162500
from PIL import Image, ImageFile, ImageDraw
ImageFile.LOAD_TRUNCATED_IMAGES = True


def main():

    df = pd.read_excel(sys.argv[1])
    for _, row in df.iterrows():
        svs_filename = row['svs_filename']
        cache_filename = row['cache_filename']
        save_root3 = row['save_root3']  # for heatmap image
        save_root3_1 = row['save_root3_1'] # for origin slide image
        image_ext = row['image_ext']
        h5filename = row['h5filename']
        j = int(float(row['j']))

        if 'CLUSTER_NAME' in os.environ:
            local_temp_dir = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], str(idr_torch.rank),
                                      str(idr_torch.local_rank))
        else: 
            local_temp_dir = './tmp'
        os.makedirs(local_temp_dir, exist_ok=True)

        svs_prefix = get_svs_prefix(svs_filename)
        svs_filename1 = os.path.realpath(svs_filename)
        local_svs_filename = os.path.join(local_temp_dir, os.path.basename(svs_filename1))
        if not os.path.exists(local_svs_filename):
            os.system(f'cp -RL "{svs_filename1}" "{local_svs_filename}"')

        with h5py.File(h5filename, 'r') as h5file:  # the mask_root is the CLAM patches dir
            coords = h5file['coords'][()]
            patch_level = h5file['coords'].attrs['patch_level']
            patch_size = h5file['coords'].attrs['patch_size']

        with open(cache_filename, 'rb') as fff:
            tmp = pickle.load(fff)
            A = tmp['A']
            jj = tmp['jj']
            cc = tmp['cc']
            del tmp

        slide = openslide.open_slide(local_svs_filename)
        print("do heatmap big")
        dimension = slide.level_dimensions[1] if len(slide.level_dimensions) > 1 else slide.level_dimensions[0]
        if dimension[0] > 100000 or dimension[1] > 100000:
            vis_level = 2
        else:
            vis_level = 1
        if len(slide.level_dimensions) == 1:
            vis_level = 0
        print('dimension', dimension)


        save_filename = '{}/{}_big_orig.zip'.format(save_root3_1, svs_prefix)
        if not os.path.exists(save_filename):
            img = visWSI(slide, vis_level=vis_level)
            print(type(img), img.size)
            img_vips = pyvips.Image.new_from_array(img)
            img_vips.dzsave(save_filename, tile_size=1024)
            time.sleep(1)
            del img, img_vips

        save_filename = '{}/{}_big_attention_map.zip'.format(save_root3, svs_prefix)
        if not os.path.exists(save_filename):
            img = visHeatmap(slide, scores=A[cc[j]:(cc[j] + len(coords))], coords=coords,
                             vis_level=vis_level, patch_size=(patch_size, patch_size),
                             convert_to_percentiles=False)
            print(type(img), img.size)
            img_vips = pyvips.Image.new_from_array(img)
            img_vips.dzsave(save_filename, tile_size=1024)
            time.sleep(1)
            del img, img_vips

        if os.path.exists(local_svs_filename):
            os.system(f'rm -rf "{local_svs_filename}"')


if __name__ == '__main__':
    main()



















