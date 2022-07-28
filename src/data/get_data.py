# -*- coding: utf-8 -*-
import gdown
from zipfile import ZipFile
import os
import logging
from pathlib import Path
import shutil
from dotenv import find_dotenv, load_dotenv

def main():
    """"
"""
    logger = logging.getLogger(__name__)
    url = 'https://drive.google.com/file/d/17fyteuN2MdGdbP5_Xq_sySN_yH91vTup/view?usp=sharing'
    output = os.fspath(raw_data_directory)+'/credito-imoves.zip'
    gdown.download(url=url, output=output, fuzzy=True)
    
    logger.info('.zip file download into /data/raw')
    
    with ZipFile(output, mode='r') as zip_ref:
        zip_ref.extractall(raw_data_directory)
    
    logger.info('.zip file extracted into /data/raw')

    os.remove(output)
    # Checking for __MACOXS folder for removal
    macosx_directory = raw_data_directory / '__MACOSX'
    if macosx_directory.exists() and macosx_directory.is_dir():
        shutil.rmtree(macosx_directory)    

    logger.info('.zip file deleted from /data/raw')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    project_directory = Path(__file__).resolve().parents[2]
    raw_data_directory = project_directory.joinpath('data', 'raw')

    main()
