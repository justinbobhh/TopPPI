import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from shapely.geometry import LineString, Point
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path(__file__).resolve().parent
CORE = {'CTNNB', 'AXIN1', 'DVL1', 'WNT9A'}

def route_edges(graph, pos):

    edges=sorted(tuple(sorted(e)) for e in graph.edges())
    nodes={n:Point(pos[n]) for n in graph.nodes()}

    radii=[0.0, .06, -.06, .12, -.12, .20, -.20, .30, -.30, .42, -.42]
    samples=np.linspace(0,1,35)[:,None]
    candidates={}
    for a,b in edges:
        start=np.asarray(pos[a],dtype=float); end=np.asarray(pos[b],dtype=float)
        v=end-start
        choices=[]
        for rad in radii:
            control=(start+end)/2+rad*np.array([v[1],-v[0]])
            points=(1-samples)**2*start+2*(1-samples)*samples*control+samples**2*end
            line=LineString(points)
            penalty=abs(rad)*8
            for n,point in nodes.items():
                if n in (a,b): continue
                safe=34 if n in CORE else 25
                dist=line.distance(point)
                if dist < safe: penalty+=65+(safe-dist)*3
            choices.append((rad,line,penalty))
        candidates[(a,b)]=choices
    picked={edge:min(range(len(radii)),key=lambda i:candidates[edge][i][2]) for edge in edges}
    for _ in range(3):
        for a,b in edges:
            edge=(a,b)
            def cost(index):
                _,line,penalty=candidates[edge][index]
                for other in edges:
                    if other==edge or a in other or b in other: continue
                    otherline=candidates[other][picked[other]][1]
                    if line.crosses(otherline): penalty+=5
                    elif line.distance(otherline)<5: penalty+=3
                return penalty
            picked[edge]=min(range(len(radii)),key=cost)
    return {edge:candidates[edge][picked[edge]][0] for edge in edges}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-npz', type=Path, help='Output from export_representations.py')
    p.add_argument('--threshold', type=float, help='Fixed threshold for own model')
    p.add_argument('--outdir', type=Path, default=BASE.parents[2]/'outputs/case_study/wnt')
    p.add_argument('--model-only', action='store_true', help='Draw only the user model')
    p.add_argument('--no-annotation', action='store_true', help='Omit panel title and legend')
    args = p.parse_args()
    if args.model_only and args.model_npz is None:
        p.error('--model-only requires --model-npz')
    if args.model_npz is not None and (args.threshold is None or not 0 <= args.threshold <= 1):
        p.error('An explicit --threshold in [0, 1] is required for own predictions')
    cases = pd.read_csv(BASE/'inputs'/'case_pairs.csv')
    ref = pd.read_csv(BASE/'inputs'/'ssppi_predictions.csv')
    if len(cases) != len(ref) or not ((cases.protein_a_name == ref.proteinA) & (cases.protein_b_name == ref.proteinB)).all():
        raise ValueError('Case input and reference rows differ; refusing to align by row')
    if not (cases.label == 1).all() or not (ref.label == 1).all():
        raise ValueError('Expected a positive-only reference case')
    cases['ssppi_prediction'] = ref['predicted label'].astype(int)
    if not cases.ssppi_prediction.isin([0,1]).all():
        raise ValueError('Invalid SSPPI label')
    if args.model_npz:
        with np.load(args.model_npz, allow_pickle=False) as dat:
            idx = np.asarray(dat['row_index']); prob = np.asarray(dat['probability']).reshape(-1)
            labels = np.asarray(dat['labels']).reshape(-1)
            metadata = json.loads(str(dat['metadata']))
        source = BASE/'inputs'/'case_pairs.csv'
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if metadata.get('source_csv_sha256') != [digest] or metadata.get('fold') is not None:
            raise ValueError('Prediction provenance does not match the full case input')
        if len(idx) != len(cases) or sorted(idx.tolist()) != list(range(len(cases))) or len(prob) != len(cases):
            raise ValueError('Inference must cover each case row exactly once')
        if not np.isfinite(prob).all() or (prob < 0).any() or (prob > 1).any() or not (labels == 1).all():
            raise ValueError('Invalid probabilities or labels')
        values = np.empty(len(cases)); values[idx.astype(int)] = prob
        cases['model_score'] = values
        cases['model_prediction'] = (values > args.threshold).astype(int)
        predcols = [('TopPPI', 'model_prediction')]
        if not args.model_only:
            predcols.append(('SSPPI (published)', 'ssppi_prediction'))
    else:
        predcols = [('SSPPI (published; layout check)', 'ssppi_prediction')]
    args.outdir.mkdir(parents=True, exist_ok=True)
    cases.drop(columns=['seq_a','seq_b']).to_csv(args.outdir/'network_results.csv',index=False,encoding='utf-8-sig')

    G=nx.Graph()
    for row in cases.itertuples():
        G.add_edge(row.protein_a_name,row.protein_b_name)
    for _, col in predcols:
        seen = {}
        for row in cases.itertuples(index=False):
            key = tuple(sorted((row.protein_a_name,row.protein_b_name)))
            value = getattr(row,col)
            if key in seen and seen[key] != value:
                raise ValueError(f'Inconsistent predictions for duplicate pair: {key}')
            seen[key] = value

    layout=pd.read_csv(BASE/'reference_layout.csv')
    if layout.protein.duplicated().any() or set(layout.protein) != set(G):
        raise ValueError('Layout and case protein names do not match')
    pos={r.protein: (r.x, -r.y) for r in layout.itertuples(index=False)}
    routes=route_edges(G,pos)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'], 'svg.fonttype':'none','pdf.fonttype':42})
    fig, axes=plt.subplots(1,len(predcols),figsize=(9.24*len(predcols),8.65),squeeze=False)
    node_order=list(G.nodes())
    node_sizes=[1850 if node in CORE else 1050 for node in node_order]
    for ax,(title,col) in zip(axes[0],predcols):
        correct=set(); missed=set()
        for row in cases.itertuples(index=False):
            (correct if getattr(row,col)==1 else missed).add(tuple(sorted((row.protein_a_name,row.protein_b_name))))
        for edge in sorted(correct | missed):
            is_missed=edge in missed
            nx.draw_networkx_edges(G,pos,edgelist=[edge],nodelist=node_order,node_size=node_sizes,ax=ax,width=2.7 if is_missed else .9,edge_color='#E00000' if is_missed else '#292929',alpha=1 if is_missed else .9,arrows=True,arrowstyle='-',connectionstyle=f'arc3,rad={routes[edge]}')
        for nodes,color,size in [(sorted(set(G)-CORE),'#7996E6',1050),(sorted(set(G)&CORE),'#F08080',1850)]:
            nx.draw_networkx_nodes(G,pos,nodelist=nodes,node_color=color,node_size=size,linewidths=0,alpha=0.72,ax=ax)
        nx.draw_networkx_labels(G,pos,labels={n:n for n in G if n not in CORE},font_size=5.2,font_weight='bold',ax=ax)
        nx.draw_networkx_labels(G,pos,labels={n:n for n in G if n in CORE},font_size=9.5,font_weight='bold',ax=ax)
        if not args.no_annotation:
            ax.set_title(f'{title}  |  identified {len(correct)}/{G.number_of_edges()} unique pairs',fontsize=12)
        ax.set_axis_off(); ax.set_aspect('equal'); ax.set_xlim(30,870); ax.set_ylim(-820,0)
        if not args.no_annotation:
            ax.legend(handles=[Line2D([0],[0],color='#444444',lw=1,label='Identified positive'),Line2D([0],[0],color='#D73832',lw=2,label='Missed positive')],loc='lower center',bbox_to_anchor=(.5,-.05),frameon=False,ncol=2,fontsize=9)
    fig.tight_layout()
    stem=args.outdir/('model_only' if args.model_only else 'model_vs_ssppi' if args.model_npz else 'ssppi_layout_check')
    for ext in ('png','pdf','svg'):
        fig.savefig(stem.with_suffix('.'+ext),dpi=250,facecolor='white',bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {stem}.[png/pdf/svg] and network_results.csv')

if __name__=='__main__':main()
