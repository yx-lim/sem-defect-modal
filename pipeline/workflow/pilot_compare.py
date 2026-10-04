import json, collections, csv, sys
P='/home/ubuntu/sem_b3/pilot/Batch_3/'; R='/home/ubuntu/sem_b3/v001ref/'
out={}
for d in ('BSE','Inlens'):
    ID=f'Batch_3__img_ptg8lmto_{d}'; o={}
    a1=json.load(open(R+ID+'/pass1/annotation.json')); b1=json.load(open(P+ID+'/v001/pass1/annotation.json'))
    ka={i['id']:tuple(i['bbox_xyxy']) for i in a1['instances']}; kb={i['id']:tuple(i['bbox_xyxy']) for i in b1['instances']}
    da={x['id']:x['class'] for x in a1['dark_regions']}; db={x['id']:x['class'] for x in b1['dark_regions']}
    o['pass1_identical_instances']=ka==kb; o['pass1_identical_dark']=da==db; o['pass1_n']=(len(ka),len(da))
    for st in ('review_pass2','review_final'):
        ra=json.load(open(R+ID+f'/{st}.json')); rb=json.load(open(P+ID+f'/v001/{st}.json'))
        s={}
        for k in ('remove_instances','relabel_instances','reclassify_dark'):
            A={x['id']:x.get('class') for x in ra.get(k,[])}; B={x['id']:x.get('class') for x in rb.get(k,[])}
            s[k]={'v001':len(A),'pilot':len(B),'both':len(set(A)&set(B)),'same_class':sum(1 for x in set(A)&set(B) if A[x]==B[x]),
                  'pilot_only':len(set(B)-set(A)),'v001_only':len(set(A)-set(B))}
        for k in ('add_seeds','artifact_regions','ignore_regions','verified_exterior','inspected_regions'):
            s[k]={'v001':len(ra.get(k,[])),'pilot':len(rb.get(k,[]))}
        s['feature_overrides']={'v001':sorted((ra.get('feature_overrides') or {}).keys()),'pilot':sorted((rb.get('feature_overrides') or {}).keys())}
        o[st]=s
    # among v001 actions, how many were on objects the pilot queue actually showed
    q=json.load(open(P+ID+'/v001/.verify/pass1/queue.json')); shown={i for c in q['crops'] for i in c['object_ids']}
    ra=json.load(open(R+ID+'/review_pass2.json'))
    v1ids={x['id'] for k in ('remove_instances','relabel_instances','reclassify_dark') for x in ra.get(k,[])}
    o['v001_pass2_actions_on_objects_pilot_showed']=f"{len(v1ids&shown)}/{len(v1ids)}"
    # on shown objects that v001 changed, did pilot agree?
    rb=json.load(open(P+ID+'/v001/review_pass2.json'))
    pa={x['id']:(k,x.get('class')) for k in ('remove_instances','relabel_instances','reclassify_dark') for x in rb.get(k,[])}
    va={x['id']:(k,x.get('class')) for k in ('remove_instances','relabel_instances','reclassify_dark') for x in ra.get(k,[])}
    sh=[i for i in va if i in shown]
    o['on_shown_v001_changed']={'n':len(sh),'pilot_same_action':sum(1 for i in sh if pa.get(i)==va[i]),'pilot_other_action':sum(1 for i in sh if i in pa and pa[i]!=va[i]),'pilot_no_action':sum(1 for i in sh if i not in pa)}
    shk=[i for i in shown if i not in va]
    o['on_shown_v001_unchanged']={'n':len(shk),'pilot_changed':sum(1 for i in shk if i in pa)}
    fa=json.load(open(R+ID+'/final/annotation.json')); fb=json.load(open(P+ID+'/v001/final/annotation.json'))
    o['final_instances']={'v001':len(fa['instances']),'pilot':len(fb['instances'])}
    o['final_inst_classes']={'v001':dict(collections.Counter(i['class'] for i in fa['instances'])),'pilot':dict(collections.Counter(i['class'] for i in fb['instances']))}
    o['final_dark_classes_nonvoid']={'v001':dict(collections.Counter(x['class'] for x in fa['dark_regions'] if x['class']!='void_like_region')),'pilot':dict(collections.Counter(x['class'] for x in fb['dark_regions'] if x['class']!='void_like_region'))}
    sa={f['id']:f['state'] for f in fa['features']}; sb={f['id']:f['state'] for f in fb['features']}
    o['feature_states']={'v001':dict(collections.Counter(sa.values())),'pilot':dict(collections.Counter(sb.values())),
                         'differ':{k:(sa[k],sb[k]) for k in sa if sa[k]!=sb.get(k)}}
    o['coverage']={'v001':fa.get('coverage'),'pilot':fb.get('coverage')}
    out[d]=o
json.dump(out,open('/home/ubuntu/sem_b3/pilot/compare.json','w'),indent=1); print(json.dumps(out,indent=1)[:9000])
