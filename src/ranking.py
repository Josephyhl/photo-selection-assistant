"""Rank only within a similar-photo group, exposing the component scores."""

from .config import RANK_WEIGHTS, REVIEW_SCORE_MARGIN


def rank_group(group, group_id):
    # One measurement region for the whole group; never mix face and scene detail.
    region = 'scene'
    for candidate in ('face', 'person'):
        if all(p.get(f'{candidate}_quality') is not None for p in group):
            if candidate == 'face':
                scales = [p.get('face_scale') for p in group]
                if all(s is not None for s in scales) and max(scales) / min(scales) > 1.5:
                    continue
            region = candidate
            break
    measurements = [p[f'{region}_quality'] for p in group]
    maximum = max(q['sharpness'] for q in measurements)
    composition_available = all(p['composition'] is not None for p in group)
    weights = dict(RANK_WEIGHTS)
    if not composition_available:
        weights.pop('composition')
    for photo, quality in zip(group, measurements):
        detail = quality['sharpness'] / maximum * 100 if maximum > 0 else 0.0
        components = {'sharpness': detail, 'exposure': quality['exposure']['score']}
        if composition_available:
            components['composition'] = photo['composition']['composition_score']
        photo.update({
            'group_id': group_id,
            'score_region': region,
            'components': components,
            'score': sum(components[k] * w for k, w in weights.items()) / sum(weights.values()),
        })
    ranked = sorted(group, key=lambda p: (-p['score'], p['path']))
    margin = ranked[0]['score'] - ranked[1]['score'] if len(ranked) > 1 else None
    for rank, photo in enumerate(ranked, 1):
        photo.update({'rank': rank, 'recommended': rank == 1})
    return {
        'id': group_id,
        'photos': ranked,
        'winner': ranked[0]['path'],
        'score_margin': margin,
        'review_needed': margin is not None and (margin < REVIEW_SCORE_MARGIN or region != 'face'),
    }
