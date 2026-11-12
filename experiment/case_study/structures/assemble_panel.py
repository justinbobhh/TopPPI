import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from common import read_complex_manifest

HERE = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description='TopPPI: assemble panel.')
    p.add_argument('--images', type=Path, default=HERE.parents[2]/'outputs/case_study/structures/figures')
    p.add_argument('--manifest', type=Path, default=HERE/'complex_manifest.csv')
    p.add_argument('--output', type=Path, default=HERE.parents[2]/'outputs/case_study/structures/figure6.png')
    p.add_argument('--font', type=Path, default=None)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    rows = read_complex_manifest(args.manifest)
    if len(rows) != 12:
        raise ValueError('The full panel requires exactly 12 complexes.')
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    missing = [str(args.images/f'{r["pdb_id"]}_{kind}.png') for r in rows for kind in ('cartoon','surface')
               if not (args.images/f'{r["pdb_id"]}_{kind}.png').is_file()]
    if missing:
        raise FileNotFoundError('Missing rendered images: '+', '.join(missing))
    font = None
    for path in ([args.font] if args.font else [Path('C:/Windows/Fonts/arial.ttf'),Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')]):
        if path and path.is_file():
            font = ImageFont.truetype(str(path), 34)
            break
    if font is None:
        font = ImageFont.load_default()
    cell_w, cell_h, image_w, image_h = 1000, 448, 490, 368
    canvas = Image.new('RGB',(cell_w*3,cell_h*4),'white')
    draw = ImageDraw.Draw(canvas)
    for i,row in enumerate(rows):
        x,y = (i%3)*cell_w, (i//3)*cell_h
        for j,kind in enumerate(('cartoon','surface')):
            with Image.open(args.images/f'{row["pdb_id"]}_{kind}.png') as src:
                img = src.convert('RGBA')
                background = Image.new('RGBA',img.size,'white')
                img = Image.alpha_composite(background,img).convert('RGB')
                img.thumbnail((image_w,image_h),Image.Resampling.LANCZOS)
                canvas.paste(img,(x+j*500+(image_w-img.width)//2,y+(image_h-img.height)//2))
        label = row['pdb_id']
        bbox = draw.textbbox((0,0),label,font=font)
        draw.text((x+(cell_w-(bbox[2]-bbox[0]))//2,y+374),label,fill='black',font=font)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(args.output,dpi=(300,300))
    print(f'Wrote {args.output.resolve()}')

if __name__ == '__main__':
    main()
