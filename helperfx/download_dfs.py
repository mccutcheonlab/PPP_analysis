# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 11:39:45 2021

@author: admin
"""
def download_dfs():
    import os
    if not os.path.exists('data.zip'):
        import urllib.request
        import sys
        import time

        url = 'http://www.mccutcheonlab.com/s/data.zip'
        print('downloading data...')

        def reporthook(count, block_size, total_size):
            global start_time
            if count == 0:
                start_time = time.time()
                return
            duration = time.time() - start_time
            progress_size = int(count * block_size)
            if duration > 0:
                speed = int(progress_size / (1024 * duration))
                percent = min(int(count * block_size * 100 / total_size), 100)
                sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds elapsed" %
                                (percent, progress_size / (1024 * 1024), speed, duration))
                sys.stdout.flush()

        urllib.request.urlretrieve(url, 'data.zip', reporthook)
        print()

    try:
        print('unzipping demo data...')
        import zipfile
        zip_ref = zipfile.ZipFile('data.zip', 'r')
        zip_ref.extractall('..\\data')
        zip_ref.close()
        os.remove('data.zip')
    except:
        print('problem with zip, downloading again')
        os.remove('data.zip')
        return download_demo_data()

    print('demo data ready')