"""Behavior checks for grouping, ranking, input handling, and report export."""

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw, ImageFilter

from src.analysis.composition import analyze_subject_composition
from src.analysis.quality import exposure_quality, normalized_sharpness
from src.grouping import describe_image, group_photos
from src.detection_session import DetectionFailure, DetectionSession
from src.ranking import rank_group
from src.workflow import analyze_photo, run_batch, select_subject


def crashing_worker(connection):
    connection.close()
    os._exit(17)


def scene():
    image = Image.new('RGB', (512, 384), '#526684')
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 60, 180, 290), fill='#e2b475')
    draw.ellipse((280, 35, 400, 155), fill='#5a8b51')
    for x in range(55, 170, 12):
        draw.line((x, 70, x, 270), fill='#815f40', width=2)
    draw.polygon([(290, 220), (450, 200), (410, 350)], fill='#bb6f65')
    return image


def record(path, descriptor=None, detail=1, exposure=90, face=None, composition=None):
    return {'path': path, 'descriptor': descriptor,
            'scene_quality': {'sharpness': detail, 'exposure': {'score': exposure}},
            'face_quality': face, 'composition': composition}


class QualityTests(unittest.TestCase):
    def test_blur_reduces_detail(self):
        image = scene()
        self.assertGreater(normalized_sharpness(image), normalized_sharpness(image.filter(ImageFilter.GaussianBlur(3))))

    def test_resolution_normalization(self):
        image = scene()
        enlarged = image.resize((1024, 768), Image.Resampling.LANCZOS)
        self.assertAlmostEqual(normalized_sharpness(image), normalized_sharpness(enlarged), delta=0.03)

    def test_clipped_exposures_score_below_midtones(self):
        middle = exposure_quality(Image.new('RGB', (100, 100), (120, 120, 120)))['score']
        for color in [(0, 0, 0), (255, 255, 255), (255, 120, 120)]:
            self.assertLess(exposure_quality(Image.new('RGB', (100, 100), color))['score'], middle)

    def test_thirds_bonus_affects_final_score(self):
        image = Image.new('RGB', (900, 600))
        box = {'x': 400, 'y': 50, 'width': 100, 'height': 400}
        center = analyze_subject_composition(image, box)['composition_score']
        third = analyze_subject_composition(image, {**box, 'x': 250})['composition_score']
        self.assertGreater(third, center)
        self.assertLessEqual(third, 100)


class GroupingTests(unittest.TestCase):
    def test_blurred_variant_joins_original(self):
        image = scene()
        records = [record('a', describe_image(image)), record('b', describe_image(image.filter(ImageFilter.GaussianBlur(2))))]
        self.assertEqual(len(group_photos(records)), 1)

    def test_different_colors_do_not_merge_flat_images(self):
        records = [record('a', describe_image(Image.new('RGB', (300, 200), 'red'))),
                   record('b', describe_image(Image.new('RGB', (300, 200), 'blue')))]
        self.assertEqual(len(group_photos(records)), 2)

    def test_no_transitive_chaining(self):
        records = [record(name, {'hash': signature, 'colors': [1.0], 'aspect': 1.0})
                   for name, signature in [('a', 0), ('b', 1), ('c', 3)]]
        groups = group_photos(records, hash_distance=1)
        self.assertEqual([len(g) for g in groups], [2, 1])


class RankingTests(unittest.TestCase):
    def test_sharp_photo_wins_when_other_components_match(self):
        result = rank_group([record('blurred', detail=0.1), record('sharp', detail=1)], 1)
        self.assertEqual(result['winner'], 'sharp')
        self.assertEqual(sum(p['recommended'] for p in result['photos']), 1)

    def test_missing_face_makes_whole_group_use_scene(self):
        face = {'sharpness': 100, 'exposure': {'score': 100}}
        result = rank_group([record('a', detail=0.1, face=face), record('b', detail=1)], 1)
        self.assertEqual(result['winner'], 'b')
        self.assertTrue(all(p['score_region'] == 'scene' for p in result['photos']))

    def test_ties_are_deterministic_and_marked_for_review(self):
        result = rank_group([record('b'), record('a')], 1)
        self.assertEqual(result['winner'], 'a')
        self.assertTrue(result['review_needed'])

    def test_person_region_is_used_when_faces_are_unavailable(self):
        photos = [record('a'), record('b')]
        photos[0]['person_quality'] = {'sharpness': 0.1, 'exposure': {'score': 90}}
        photos[1]['person_quality'] = {'sharpness': 1, 'exposure': {'score': 90}}
        result = rank_group(photos, 1)
        self.assertEqual(result['winner'], 'b')
        self.assertTrue(all(p['score_region'] == 'person' for p in result['photos']))

    def test_inconsistent_face_framing_uses_person_measurements(self):
        photos = [record('a'), record('b')]
        for photo, scale in zip(photos, [0.3, 0.7]):
            photo['face_quality'] = {'sharpness': 1, 'exposure': {'score': 90}}
            photo['face_scale'] = scale
            photo['person_quality'] = {'sharpness': 1, 'exposure': {'score': 90}}
        result = rank_group(photos, 1)
        self.assertTrue(all(p['score_region'] == 'person' for p in result['photos']))
        self.assertTrue(result['review_needed'])


class BatchTests(unittest.TestCase):
    def test_worker_crash_is_contained_and_not_restarted(self):
        with patch('src.detection_session._worker', crashing_worker):
            with DetectionSession(timeout=5) as detector:
                with self.assertRaises(DetectionFailure):
                    detector(scene())
                process = detector.process
                with self.assertRaises(DetectionFailure):
                    detector(scene())
                self.assertIs(detector.process, process)
                self.assertFalse(process.is_alive())

    def test_subject_face_is_matched_to_selected_person(self):
        module = types.ModuleType('src.analysis.detection')
        person = {'x': 10, 'y': 10, 'width': 100, 'height': 200}
        inside = {'x': 30, 'y': 20, 'width': 30, 'height': 30}
        outside = {'x': 200, 'y': 20, 'width': 100, 'height': 100}
        clothing = {'x': 30, 'y': 150, 'width': 60, 'height': 50}
        module.detect_people = Mock(return_value=[person])
        module.select_primary_person = Mock(return_value=person)
        module.detect_face_with_person_guidance = Mock(return_value=[outside, clothing, inside])
        module.fraction_inside = lambda face, subject: 0.0 if face is outside else 1.0
        with patch.dict(sys.modules, {'src.analysis.detection': module}):
            with patch('src.workflow.PERSON_MODEL_PATH') as people_path, patch('src.workflow.FACE_MODEL_PATH') as face_path:
                people_path.exists.return_value = face_path.exists.return_value = True
                image = scene()
                subject, face = select_subject(image)
        self.assertIs(subject, person)
        self.assertIs(face, inside)
        module.detect_people.assert_called_once()
        module.select_primary_person.assert_called_once()
        module.detect_face_with_person_guidance.assert_called_once_with(image, people=[person], primary_person=person)

    def test_detection_error_keeps_photo_with_fallback_note(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = folder / 'scene.png'
            scene().save(path)
            failed_detector = Mock(side_effect=DetectionFailure('model failed'))
            photo = analyze_photo(path, folder / 'thumb.jpg', detector=failed_detector)
            self.assertIsNone(photo['face_quality'])
            self.assertTrue(any('model failed' in note for note in photo['notes']))
            self.assertGreater(photo['scene_quality']['sharpness'], 0)

    def test_nested_batch_bad_file_exports_and_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            nested = folder / 'burst'
            nested.mkdir()
            image = scene()
            image.save(nested / 'sharp.png')
            image.filter(ImageFilter.GaussianBlur(2)).save(nested / 'blurred.png')
            (folder / 'broken.jpg').write_bytes(b'not an image')
            output = folder / 'reports'
            before = (nested / 'sharp.png').read_bytes()
            groups, errors = run_batch(folder, output, no_detection=True)
            self.assertEqual(len(groups), 1)
            self.assertEqual(Path(groups[0]['winner']).name, 'sharp.png')
            self.assertEqual(len(errors), 1)
            self.assertEqual((nested / 'sharp.png').read_bytes(), before)
            for name in ['report.html', 'rankings.csv', 'results.json']:
                self.assertTrue((output / name).is_file())
            payload = json.loads((output / 'results.json').read_text())
            self.assertFalse(payload['settings']['closed_eye_used'])
            groups, errors = run_batch(folder, output, no_detection=True)
            self.assertEqual(sum(len(g['photos']) for g in groups), 2)

    def test_exif_orientation(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            image = Image.new('RGB', (100, 200), 'gray')
            exif = Image.Exif()
            exif[274] = 6
            path = folder / 'rotated.jpg'
            image.save(path, exif=exif)
            photo = analyze_photo(path, folder / 'thumb.jpg', no_detection=True)
            self.assertEqual(photo['original_size'], [200, 100])

    def test_output_cannot_overwrite_input_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_batch(directory, directory, no_detection=True)


if __name__ == '__main__':
    unittest.main()
