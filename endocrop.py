import click
import os, functools
from pathlib import Path
import concurrent.futures
import tqdm
import cv2

class Mask:
    def __init__(self, mask_file, shape):
        self._mask = cv2.imread(str(mask_file))
        if self._mask is None:
            raise ValueError(f"Failed to load mask file '{str(mask_file)}'")
        self._shape = self._mask.shape
        if shape is not None and self._shape != shape:
            raise ValueError(f"shape of '{str(mask_file)}' is expected to be {shape}, but {self._shape}")

    def is_same_shape(self, image):
        return image.shape == self._shape

    def mask(self, image):
        return cv2.bitwise_and(image, self._mask)

_mask_fuji = Mask(Path(__file__).parent/'mask_fuji.png', (1024, 1280, 3))
_mask_cce = Mask(Path(__file__).parent/'mask_cce.png', (512, 512, 3))

def conv_endocrop(image):
    i = image
    
    # grayscale
    ip = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    # gaussian blur to reduce noise
    ip = cv2.GaussianBlur(ip, (5, 5), 0)
    # filter
    _, ip = cv2.threshold(ip, 10, 255, cv2.THRESH_BINARY)
    
    ct,_ = cv2.findContours(ip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnt = max(ct, key=lambda cnt: cv2.contourArea(cnt))
    x,y,w,h = cv2.boundingRect(cnt)

    return i[y:y+h, x:x+w]

def convert(src, out, flag_endocrop, flag_maskfuji, flag_maskcce):
    i = cv2.imread(str(src))

    if flag_maskfuji and _mask_fuji.is_same_shape(i):
        i = _mask_fuji.mask(i)

    if flag_maskcce and _mask_cce.is_same_shape(i):
        i = _mask_cce.mask(i)

    if flag_endocrop:
        i = conv_endocrop(i)

    cv2.imwrite(str(out), i)
    
def walk_and_convert(src_dir, dst_dir, rename_map, function, overwrite=False, tqdm_unit='unit'):
    if not os.path.isdir(src_dir):
        raise

    if os.path.isdir(dst_dir):
        pass
    elif os.path.isfile(dst_dir):
        raise
    else:
        os.makedirs(dst_dir)

    args = []
    
    for parent, dirs, files in os.walk(src_dir):
        for d in dirs:
            p = Path(os.path.join(parent, d))
            o = Path(dst_dir)/p.relative_to(src_dir)
            os.makedirs(o, exist_ok=True)

            
        for f in files:
            p = Path(os.path.join(parent, f))
            dst_name = rename_map(p.relative_to(src_dir))
            
            if not dst_name:
                continue
            
            o = Path(dst_dir)/dst_name

            if not overwrite and o.exists():
                continue

            args.append((p,o))


    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(function, s, o) for s,o in args]
        for f in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), unit='image'):
            pass
        dfs = [f.result() for f in futures]

@click.command()
@click.argument('src_dir', type=click.Path(exists=True))
@click.argument('dst_dir')
@click.option('--overwrite', is_flag=True)
@click.option('--endocrop/--no-endocrop', default=True)
@click.option('--maskfuji/--no-maskfuji', default=True)
@click.option('--maskcce', is_flag=True)
def command(src_dir, dst_dir, overwrite, endocrop, maskfuji, maskcce):
    def rename_map(path):
        if path.suffix.lower() not in ['.jpg','.png']:
            return None
        return path.with_suffix('.jpg')

    function = functools.partial(convert, flag_maskfuji=maskfuji, flag_maskcce=maskcce, flag_endocrop=endocrop)

    walk_and_convert(src_dir, dst_dir, rename_map, function, overwrite, 'image')

if __name__ == '__main__':
    command()
