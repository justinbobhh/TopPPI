import argparse
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
HEADER=['protein_a_id','protein_b_id','seq_a','seq_b','label']

def main():
    p=argparse.ArgumentParser(description='TopPPI: data process.')
    p.add_argument('--task',required=True,choices=('yeast','multispecies','multiclass'))
    p.add_argument('--dictionary',type=Path,help='Two-column protein ID / sequence TSV; defaults to bundled snapshot.')
    p.add_argument('--pairs',type=Path,nargs='+',help='Custom three-column ID / ID / label TSV files.')
    p.add_argument('--output-dir',type=Path,help='Defaults to outputs/preprocessed/TASK; bundled data are preserved.')
    args=p.parse_args()
    raw=ROOT/'data/raw'/args.task
    dictionary=args.dictionary or raw/'sequences.tsv'
    names=['train.tsv','test.tsv'] if args.task!='multispecies' else [f'{split}_{identity}.tsv' for split in ('train','test') for identity in ('01','10','25','40','any')]
    pairs=args.pairs or [raw/n for n in names]
    output=(args.output_dir or ROOT/'outputs/preprocessed'/args.task).resolve()
    for protected in (ROOT/'data',ROOT/'checkpoints',ROOT/'pretrained'):
        if output==protected or output.is_relative_to(protected):
            p.error('Choose an output folder outside bundled data and model folders.')
    sequences={}
    with dictionary.open(encoding='utf-8-sig',newline='') as handle:
        for line in handle:
            fields=line.rstrip('\r\n').split('\t')
            if len(fields)<2:continue
            pid,seq=fields[:2]
            if pid in sequences and sequences[pid]!=seq:raise ValueError(f'Conflicting sequence for {pid}')
            sequences[pid]=seq
    output.mkdir(parents=True,exist_ok=True)
    records=[]; emitted={}
    for source in pairs:
        target=output/(source.stem+'.csv')
        if target in emitted.values():raise ValueError('Input filenames must have different stems.')
        kept=missing=0
        with source.open(encoding='utf-8-sig',newline='') as handle,target.open('w',encoding='utf8',newline='') as out:
            writer=csv.writer(out,lineterminator='\n');writer.writerow(HEADER)
            for number,line in enumerate(handle,1):
                if not line.strip():continue
                fields=line.strip().split()
                if len(fields)!=3:raise ValueError(f'{source}:{number}: expected three columns.')
                a,b,label=fields
                if a not in sequences or b not in sequences:missing+=1;continue
                label=int(label)
                if label not in (range(7) if args.task=='multiclass' else (0,1)):raise ValueError(f'Unexpected label at {source}:{number}')
                writer.writerow([a,b,sequences[a],sequences[b],label]);kept+=1
        emitted[source.stem]=target
        records.append({'input':str(source),'output':str(target),'kept':kept,'missing_sequence':missing})
    if not args.pairs:
        merge_names=['train','test'] if args.task!='multispecies' else ['train_40','test_40']
        merged=output/{'yeast':'data_all.csv','multispecies':'dataset40_train_test_merged.csv','multiclass':'dataset_merged_all.csv'}[args.task]
        with merged.open('w',encoding='utf8',newline='') as out:
            writer=csv.writer(out,lineterminator='\n');writer.writerow(HEADER)
            for name in merge_names:
                with emitted[name].open(encoding='utf8',newline='') as handle:
                    reader=csv.reader(handle);next(reader);writer.writerows(reader)
    report={'task':args.task,'dictionary':str(dictionary),'dictionary_ids':len(sequences),'conversion':records,
            'note':'Offline conversion; missing sequences reported explicitly. Verify rows against bundled snapshots for paper reproduction.'}
    (output/'conversion_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
