"""Batch analysis, visual grouping, ranking, and report orchestration."""

import argparse
from contextlib import nullcontext
from pathlib import Path

from PIL import Image, ImageOps

from .analysis.composition import analyze_subject_composition
from .analysis.quality import crop_box, exposure_quality, normalized_sharpness
from .config import (
    DETECTION_LONG_SIDE, FACE_MODEL_PATH, GROUP_HASH_DISTANCE,
    MIN_RANK_FACE_SIZE, PERSON_MODEL_PATH, PHOTO_FOLDER, PROJECT_FOLDER,
    QUALITY_LONG_SIDE, RANK_WEIGHTS,
)
from .grouping import describe_image, group_photos
from .detection_session import DetectionSession
from .ranking import rank_group
from .reporting import export_reports

PHOTO_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp'}


def collect_photos(folder, output):
    """Search nested folders, excluding our generated output and directory entries."""
    return sorted(
        p for p in folder.rglob('*')
        if p.is_file() and p.suffix.lower() in PHOTO_EXTENSIONS
        and not p.resolve().is_relative_to(output.resolve())
    )


def select_subject(image):
    """Choose a person once, then use only that person's matched face."""
    from .analysis.detection import (
        detect_face_with_person_guidance, detect_people, fraction_inside,
        select_primary_person,
    )
    people = detect_people(image) if PERSON_MODEL_PATH.exists() else []
    person = select_primary_person(image, people, [], show_details=False)
    faces = detect_face_with_person_guidance(
        image, people=people, primary_person=person,
    ) if FACE_MODEL_PATH.exists() else []
    if person is not None:
        faces = [f for f in faces if fraction_inside(f, person) >= 0.6
                 and f['y'] + f['height'] / 2 <= person['y'] + person['height'] * 0.45]
    face = max(faces, key=lambda f: f['width'] * f['height'], default=None)
    return person, face


def measure_quality(image):
    return {'sharpness': normalized_sharpness(image), 'exposure': exposure_quality(image)}


def analyze_photo(path, thumbnail_path, no_detection=False, detector=None):
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert('RGB')
    original_size = image.size
    # The same cap applies to every image, and all detection coordinates use it.
    image.thumbnail((DETECTION_LONG_SIDE, DETECTION_LONG_SIDE), Image.Resampling.LANCZOS)
    notes = []
    person = face = None
    if not no_detection:
        if not PERSON_MODEL_PATH.exists():
            notes.append('人物模型缺失，主体选择仅依据人脸。')
        if not FACE_MODEL_PATH.exists():
            notes.append('人脸模型缺失，使用全图质量。')
        try:
            person, face = (detector or select_subject)(image)
        except Exception as error:
            notes.append(f'主体检测失败，使用全图质量：{type(error).__name__}: {error}')
    face_quality = None
    person_quality = None
    if person is not None:
        person_crop = crop_box(image, person)
        if person_crop is not None and min(person_crop.size) >= MIN_RANK_FACE_SIZE:
            person_quality = measure_quality(person_crop)
    if face is not None:
        crop = crop_box(image, face)
        if crop is not None and min(crop.size) >= MIN_RANK_FACE_SIZE:
            face_quality = measure_quality(crop)
        else:
            notes.append('人脸尺寸不足，无法可靠比较人脸清晰度。')
    composition = analyze_subject_composition(image, person)
    scene_quality = measure_quality(image)
    descriptor = describe_image(image)
    thumbnail = image.copy()
    thumbnail.thumbnail((400, 300))
    thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
    thumbnail.save(thumbnail_path, quality=85)
    return {
        'path': str(path.resolve()), 'original_size': list(original_size),
        'analysis_size': list(image.size), 'scene_quality': scene_quality,
        'face_quality': face_quality, 'person_quality': person_quality,
        'face_scale': (face['width'] / person['width'] if person else face['width'] / image.width)
        if face is not None else None,
        'composition': composition,
        'person': person, 'face': face, 'notes': notes, 'descriptor': descriptor,
        'thumbnail': f'thumbnails/{thumbnail_path.name}',
    }


def run_batch(folder, output, no_detection=False, hash_distance=GROUP_HASH_DISTANCE):
    folder, output = Path(folder).resolve(), Path(output).resolve()
    if not folder.is_dir():
        raise ValueError(f'照片文件夹不存在：{folder}')
    if folder.is_relative_to(output):
        raise ValueError('输出文件夹不能与输入相同，也不能包含输入文件夹。')
    paths = collect_photos(folder, output)
    if not paths:
        raise ValueError(f'文件夹内没有支持的图片：{folder}')
    photos, errors = [], []
    session = nullcontext(None) if no_detection else DetectionSession()
    with session as detector:
        for index, path in enumerate(paths, 1):
            print(f'[{index}/{len(paths)}] {path.relative_to(folder)}', flush=True)
            try:
                photos.append(analyze_photo(
                    path, output / 'thumbnails' / f'{index:06d}.jpg',
                    no_detection, detector=detector,
                ))
            except Exception as error:
                errors.append({'path': str(path), 'error': f'{type(error).__name__}: {error}'})
    groups = [rank_group(group, i) for i, group in enumerate(group_photos(photos, hash_distance), 1)]
    export_reports(groups, errors, output, {
        'input': str(folder), 'no_detection': no_detection,
        'hash_distance': hash_distance, 'quality_long_side': QUALITY_LONG_SIDE,
        'rank_weights': RANK_WEIGHTS, 'closed_eye_used': False,
        'score_scope': 'within_group',
    })
    return groups, errors


def main(argv=None):
    parser = argparse.ArgumentParser(description='批量分析相似照片，排序并导出推荐报告。')
    parser.add_argument('folder', nargs='?', type=Path, default=PHOTO_FOLDER)
    parser.add_argument('--output', type=Path, default=PROJECT_FOLDER / 'results' / 'culling')
    parser.add_argument('--no-detection', action='store_true', help='跳过人物与人脸模型，统一用全图评分')
    parser.add_argument('--hash-distance', type=int, default=GROUP_HASH_DISTANCE, help='分组相似度距离，0 最严格，最大 63')
    args = parser.parse_args(argv)
    if not 0 <= args.hash_distance <= 63:
        parser.error('--hash-distance 必须在 0 到 63 之间')
    try:
        groups, errors = run_batch(args.folder, args.output, args.no_detection, args.hash_distance)
    except ValueError as error:
        parser.error(str(error))
    print(f'完成：{sum(len(g["photos"]) for g in groups)} 张照片，{len(groups)} 组，{len(errors)} 个文件失败。')
    print(f'报告：{args.output.resolve() / "report.html"}')
    if not groups:
        raise SystemExit(1)
