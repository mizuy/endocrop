import click
import os
from pathlib import Path
from PIL import Image
import PIL
import concurrent.futures
from PIL import ImageFilter,ImageChops,ImageMath
import tqdm
import numpy as np
import cv2

def cv2pil(image_cv):
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_cv)
    image_pil = image_pil.convert('RGB')

    return image_pil

@click.command()
@click.argument('src', type=click.Path(exists=True))
@click.argument('out')
@click.option('--overwrite', default=False)
def cmd(src, out, overwrite):
    if not os.path.isdir(src):
        raise

    if os.path.isdir(out):
        pass
    elif os.path.isfile(out):
        raise
    else:
        os.makedirs(out)

    args = []
    
    for parent, dirs, files in os.walk(src):
        for d in dirs:
            p = Path(os.path.join(parent, d))
            o = Path('out')/p.relative_to(src)
            os.makedirs(o, exist_ok=True)

            
        for f in files:
            p = Path(os.path.join(parent, f))
            o = Path('out')/p.relative_to(src).with_suffix('.png')

            if not overwrite and o.exists():
                continue

            if p.suffix.lower() not in ['.jpg','.png']:
                continue


            args.append((p,o))


    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(endo_image_crop, s,o) for s,o in args]
        for f in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), unit='image'):
            pass
        dfs = [f.result() for f in futures]

        

def endo_image_crop(src,out):
    i = cv2.imread(str(src))
    ip = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    ip = cv2.GaussianBlur(ip, (5, 5), 0)
    _, ip = cv2.threshold(ip, 10, 255, cv2.THRESH_BINARY)
    
    ct,_ = cv2.findContours(ip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ii = np.copy(i)

    cnt = max(ct, key=lambda cnt: cv2.contourArea(cnt))
    x,y,w,h = cv2.boundingRect(cnt)

    r = i[y:y+h, x:x+w]
    #cv2pil(r).show()
    cv2.imwrite(str(out), r)

    
    '''    
    origin = Image.open(src).convert('RGB') #

    h,s,v = origin.convert('HSV').split()
    s.show()

    r,g,b = origin.split()
    m = origin.convert('L')

    i = ImageMath.eval('(((r-m)**2+(g-m)**2+(b-m)**2)>100)*255', r=r, g=g, b=b, m=m)
    i = i.filter(ImageFilter.MinFilter(3))
    i.show()
    bbox = i.getbbox()
    origin.crop(bbox).show()

    v.show()
    vm = Image.new('L',i.size)
    
    w,h = i.size
    for y in range(h):
        for x in range(w):
            r,g,b = i.getpixel((x,y))
            m = (r+g+b)/3
            v = (r-m)**2+(g-m)**2+(b-m)**2
            c = 255 if v > 200 else 0
            vm.putpixel((x,y),c)
            
    vm = vm.filter(ImageFilter.MinFilter(5))
    bbox = vm.getbbox()
    i.crop(bbox).save(out)
'''    

    
def main():
    cmd()


if __name__ == '__main__':
    main()
