from pathlib import Path
import json, zipfile, re, base64, argparse
from lxml import html
parser=argparse.ArgumentParser()
parser.add_argument("report",type=Path)
a=parser.parse_args()
OUT=Path(__file__).resolve().parents[1]/"data"
with zipfile.ZipFile(a.report) as archive:
    source = archive.read(archive.namelist()[0]).decode('utf8')
reports = json.loads(re.search(r'<script[^>]*id=["\']reports-data["\'][^>]*>(.*?)</script>', source, re.S)[1])
data = {}
for model, report in zip(['1.7B', '4B'], reports[1:3]):
    document = html.fromstring(base64.b64decode(report['html']).decode())
    sections = {}
    for key in ['s6', 's8', 's9']:
        rows = [[cell.text_content().strip() for cell in row] for row in document.get_element_by_id(key).xpath('.//tr')]
        sections[key] = [dict(zip(rows[0], row)) for row in rows[1:] if row[0] == 'FULL']
    motion = {int(row['总更新步']): row for row in sections['s6']}
    full = {int(row['总更新步']): row for row in sections['s8'] if row['范围'] == 'full'}
    projected = {int(row['总更新步']): row for row in sections['s8'] if row['范围'] == 'u'}
    spectra = {int(row['总更新步']): json.loads(row['所有已保存r与覆盖比例']) for row in sections['s9'] if row['范围'] == 'full'}
    coverage = {}
    for step, row in full.items():
        basis = json.loads(row['U参考定义'])
        if basis['frozen'] and basis['cumulative_to'] == 10:
            coverage[step] = float(projected[step]['energy_per_token']) / float(row['energy_per_token'])
    assert len(motion) == 22 and len(full) == 23 and 106 in coverage
    data[model] = dict(motion=motion, full=full, spectra=spectra, coverage=coverage)

(OUT/"rq2.json").write_text(json.dumps(data,ensure_ascii=False),encoding="utf8")
with zipfile.ZipFile(a.report) as z:
    source = z.read(z.namelist()[0]).decode('utf8')
reports = json.loads(re.search(r'<script[^>]*id=["\']reports-data["\'][^>]*>(.*?)</script>', source, re.S)[1])
data = {}
for model, report in zip(['1.7B', '4B'], reports[1:3]):
    doc = html.fromstring(base64.b64decode(report['html']).decode())
    sections = {}
    for key in ['s2','s14','s15','s16','s17','s18','s21']:
        rows = [[c.text_content().strip() for c in tr] for tr in doc.get_element_by_id(key).xpath('.//tr')]
        sections[key] = {'headers': rows[0], 'rows': [r for r in rows[1:] if (r[0] == 'FULL' if key != 's21' else 'full' in r[0].lower())]}
    vals = {int(r[1]): [float(r[2]), float(r[3])] for r in sections['s2']['rows']}
    steps = {int(r[1]): r for r in sections['s17']['rows']}
    tv = {int(r[1]): float(r[7]) for r in sections['s18']['rows'] if r[2]=='all'}
    batch = {int(r[1]): float(r[3]) for r in sections['s16']['rows'] if r[2]=='all'}
    hidden = {int(r[1]): float(r[3]) for r in sections['s14']['rows'] if r[2]=='full'}
    overlap = {int(r[1]): float(r[4]) for r in sections['s15']['rows'] if r[2]=='full' and r[3]=='centered_k4'}
    assert len(vals)==12 and len(steps)==11 and len(tv)==11 and len(batch)==len(hidden)==len(overlap)==4
    for r in steps.values():
        assert abs(float(r[2])-float(r[3])-float(r[4])) < 2e-8
        assert abs(sum(float(r[j]) for j in [10,11,12])-1)<1e-7
    data[model] = dict(sections=sections, vals=vals, steps=steps, tv=tv, batch=batch, hidden=hidden, overlap=overlap)

for values in data.values(): values.pop('sections')
(OUT/"rq3.json").write_text(json.dumps(data,ensure_ascii=False),encoding="utf8")
