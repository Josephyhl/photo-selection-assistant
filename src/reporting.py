"""Export reviewable ranking results without modifying original photos."""

import csv
import html
import json
from pathlib import Path


def export_reports(groups, errors, output, settings):
    output.mkdir(parents=True, exist_ok=True)
    public_groups = []
    for group in groups:
        public_groups.append({**group, 'photos': [
            {k: v for k, v in p.items() if k != 'descriptor'} for p in group['photos']
        ]})
    payload = {'settings': settings, 'groups': public_groups, 'errors': errors}
    (output / 'results.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    fields = ['group_id', 'rank', 'recommended', 'path', 'score', 'score_region',
              'sharpness', 'exposure', 'composition', 'review_needed']
    with (output / 'rankings.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for group in public_groups:
            for photo in group['photos']:
                writer.writerow({
                    **{k: photo[k] for k in fields if k in photo},
                    **photo['components'], 'review_needed': group['review_needed'],
                })
    cards = []
    labels = {'sharpness': '清晰度', 'exposure': '曝光', 'composition': '构图'}
    regions = {'face': '人脸', 'person': '人物', 'scene': '全图'}
    for group in public_groups:
        status = '需要人工复核：前两名接近或缺少一致的人脸测量' if group['review_needed'] else '建议供人工确认'
        if len(group['photos']) == 1:
            status = '单张照片，暂无相似候选可比较'
        cards.append(f'<section><h2>组 {group["id"]} · {len(group["photos"])} 张</h2><p>{status}</p><div class="photos">')
        for photo in group['photos']:
            caption = '推荐' if photo['recommended'] else f'第 {photo["rank"]} 名'
            components = ' · '.join(f'{labels[k]}: {v:.1f}' for k, v in photo['components'].items())
            cards.append(
                '<article>'
                f'<a href="{html.escape(Path(photo["path"]).as_uri(), quote=True)}">'
                f'<img loading="lazy" src="{html.escape(photo["thumbnail"], quote=True)}" alt=""></a>'
                f'<strong>{caption} · {photo["score"]:.1f}</strong>'
                f'<p>{html.escape(Path(photo["path"]).name)}</p>'
                f'<p>{html.escape(components)}</p><p>测量区域：{regions[photo["score_region"]]}</p>'
                + ''.join(f'<p>{html.escape(note)}</p>' for note in photo['notes'])
                + '</article>'
            )
        cards.append('</div></section>')
    if errors:
        cards.append('<h2>未能分析的文件</h2><ul>')
        cards.extend(f'<li>{html.escape(e["path"])}: {html.escape(e["error"])}</li>' for e in errors)
        cards.append('</ul>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>Photo Culling</title><style>
body{font:16px system-ui;margin:32px;background:#f6f6f6;color:#222}
.photos{display:flex;gap:16px;flex-wrap:wrap}article{background:white;padding:12px;width:300px}
img{width:300px;height:220px;object-fit:contain}article p{font-size:13px;overflow-wrap:anywhere}
section{margin:32px 0}strong{display:block;margin-top:8px}
</style><h1>照片分组与推荐</h1>
<p>分数仅用于组内比较。推荐表示组内优先，不表示质量合格。当前使用启发式规则，闭眼预测未参与；原图保持原样。</p>'''
    mode = '全图模式（跳过人物、人脸检测）' if settings['no_detection'] else '主体检测模式；检测不可用时使用全图'
    (output / 'report.html').write_text(page + f'<p>{mode}</p>' + ''.join(cards) + '</html>', encoding='utf-8')
