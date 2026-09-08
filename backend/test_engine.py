"""
Unit tests for the matching decision, with synthetic embeddings.

Deliberately no camera, no InsightFace, no cv2 and **no database**:
`_decide_face` was split out of `_recognize` precisely so the decision that
governs whether someone is marked present can be tested with hand-built
vectors. numpy is the only dependency.

`_mark` is stubbed here so these tests stay about the *decision*. What happens
to a mark once it reaches SQLite — day rollover, the unique constraint,
check-in/check-out — is tested in test_persistence.py, against a real database.

Run from the repo root:

    py backend/test_engine.py
    py -m unittest discover -s backend -p "test_*.py"
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import config  # noqa: E402
from config import LIVENESS_UNAVAILABLE  # noqa: E402
from engine import (  # noqa: E402
    MATCH_PRESENT,
    MATCH_UNCERTAIN,
    MATCH_UNKNOWN,
    AttendanceEngine,
)


def unit(vec):
    """L2-normalize, so a dot product is a cosine similarity."""
    arr = np.asarray(vec, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def embedding_at(similarity, dim=512):
    """A unit vector whose dot product with `basis(dim)` is `similarity`.

    Built from two orthogonal axes, so the resulting similarity is exact rather
    than approximate — the tests can then sit deliberately just above and just
    below a threshold.
    """
    vec = np.zeros(dim, dtype=np.float32)
    vec[0] = similarity
    vec[1] = float(np.sqrt(max(0.0, 1.0 - similarity ** 2)))
    return unit(vec)


def basis(dim=512):
    """The enrolled reference embedding every similarity is measured against."""
    vec = np.zeros(dim, dtype=np.float32)
    vec[0] = 1.0
    return vec


def axis(index, dim=512):
    """A unit vector along one axis — orthogonal to every other axis()."""
    vec = np.zeros(dim, dtype=np.float32)
    vec[index] = 1.0
    return vec


def stub_marks(eng):
    """Replace `_mark` with a recorder, so no database is touched."""
    eng.marked = []

    def record(person_id, status, confidence=None, multi_template=False,
               liveness=LIVENESS_UNAVAILABLE):
        eng.marked.append(
            {
                "person_id": person_id,
                "status": status,
                "confidence": confidence,
                "multi_template": multi_template,
                # 'unavailable' is what every mark carries with no liveness
                # provider configured, which is the state these tests run in —
                # they pass an embedding and no frame, so no check can happen.
                "liveness": liveness,
            }
        )

    eng._mark = record
    return eng


def make_engine(*names):
    """An engine with a synthetic gallery, no database and no camera.

    `_gallery_loaded = True` stops `load_gallery()` from opening SQLite. Each
    name gets exactly one template, which is the configuration STRONG_MATCH was
    calibrated at; template id and person id coincide only for that reason.
    """
    eng = stub_marks(AttendanceEngine())
    eng._gallery_loaded = True
    for i, name in enumerate(names, start=1):
        eng._append_known_locked(basis(), template_id=i, person_id=i, name=name)
        eng._person_template_counts[i] = 1
    return eng


def marked_names(eng):
    """Names the engine tried to record, in order."""
    return [eng._person_names[m["person_id"]] for m in eng.marked]


class MatchingDecisionTests(unittest.TestCase):
    def test_confident_match_is_recorded_as_present(self):
        eng = make_engine("Ada")
        decision, name, sim = eng._decide_face(embedding_at(0.90))

        self.assertEqual(decision, MATCH_PRESENT)
        self.assertEqual(name, "Ada")
        self.assertAlmostEqual(sim, 0.90, places=5)

        self.assertEqual(len(eng.marked), 1)
        self.assertEqual(marked_names(eng), ["Ada"])
        self.assertEqual(eng.marked[0]["status"], "present")
        self.assertAlmostEqual(eng.marked[0]["confidence"], 0.90, places=5)

    def test_uncertain_match_is_displayed_but_not_recorded(self):
        """The central guarantee: the uncertain band never becomes attendance.

        Measured open-set FAR at WEAK_MATCH is 1.335% — roughly one stranger
        frame in 75. A false accept here would be proxy attendance.
        """
        eng = make_engine("Ada")
        midband = (config.WEAK_MATCH + config.STRONG_MATCH) / 2.0
        decision, name, sim = eng._decide_face(embedding_at(midband))

        self.assertEqual(decision, MATCH_UNCERTAIN)
        self.assertEqual(name, "Ada")          # still identified for display
        self.assertGreaterEqual(sim, config.WEAK_MATCH)
        self.assertLess(sim, config.STRONG_MATCH)
        self.assertEqual(eng.marked, [])       # and recorded nowhere

    def test_unknown_face_is_neither_named_nor_recorded(self):
        eng = make_engine("Ada")
        decision, name, sim = eng._decide_face(embedding_at(0.10))

        self.assertEqual(decision, MATCH_UNKNOWN)
        self.assertIsNone(name)
        self.assertLess(sim, config.WEAK_MATCH)
        self.assertEqual(eng.marked, [])

    def test_uncertain_then_confident_records_only_the_confident_frame(self):
        """The first-write-wins bug: an early weak frame used to pin the session.

        Old behaviour: the uncertain frame wrote status "uncertain" and the later
        confident frame was ignored. New behaviour: the uncertain frame writes
        nothing, and the confident frame is the only one that records.
        """
        eng = make_engine("Ada")

        eng._decide_face(embedding_at((config.WEAK_MATCH + config.STRONG_MATCH) / 2.0))
        self.assertEqual(eng.marked, [], "uncertain must not record")

        eng._decide_face(embedding_at(0.95))
        self.assertEqual(len(eng.marked), 1)
        self.assertEqual(eng.marked[0]["status"], "present")

    def test_each_band_is_entered_at_its_threshold(self):
        """Bands switch at the thresholds, checked either side of a float32 margin.

        Not asserted *exactly* at the threshold on purpose: embeddings are
        float32, so a vector built to have similarity 0.32 actually scores
        0.31999999 — 7e-9 low — and lands in the band below. That is a property
        of the arithmetic, not of the decision rule, and it is equally true of
        real embeddings, where frame-to-frame noise is many orders of magnitude
        larger than the gap. `margin` is comfortably above float32 epsilon at
        this magnitude (~6e-8) and far below any difference that would matter.
        """
        eng = make_engine("Ada")
        margin = 1e-5

        for target, expected in (
            (config.STRONG_MATCH + margin, MATCH_PRESENT),
            (config.STRONG_MATCH - margin, MATCH_UNCERTAIN),
            (config.WEAK_MATCH + margin, MATCH_UNCERTAIN),
            (config.WEAK_MATCH - margin, MATCH_UNKNOWN),
        ):
            with self.subTest(similarity=target):
                self.assertEqual(eng._decide_face(embedding_at(target))[0], expected)

    def test_empty_gallery_yields_unknown_without_indexing_anything(self):
        eng = make_engine()
        decision, name, sim = eng._decide_face(embedding_at(0.99))

        self.assertEqual(decision, MATCH_UNKNOWN)
        self.assertIsNone(name)
        self.assertEqual(sim, -1.0)

    def test_correct_identity_is_chosen_from_a_multi_person_gallery(self):
        eng = stub_marks(AttendanceEngine())
        eng._gallery_loaded = True
        eng._append_known_locked(axis(0), template_id=1, person_id=1, name="Ada")
        eng._append_known_locked(axis(5), template_id=2, person_id=2, name="Grace")
        eng._person_template_counts.update({1: 1, 2: 1})

        decision, name, _ = eng._decide_face(axis(5))
        self.assertEqual(decision, MATCH_PRESENT)
        self.assertEqual(name, "Grace")
        self.assertEqual(marked_names(eng), ["Grace"])


class MultiTemplateMatchingTests(unittest.TestCase):
    """One person spanning several matrix rows still resolves to one person.

    This is the property the schema change exists for: a matrix row is a
    TEMPLATE, and the argmax has to be mapped through the row->person table
    before anything is decided.
    """

    def _two_people_uneven_templates(self):
        """Ada holds three templates on three axes; Grace holds one."""
        eng = stub_marks(AttendanceEngine())
        eng._gallery_loaded = True
        eng._append_known_locked(axis(0), template_id=1, person_id=1, name="Ada")
        eng._append_known_locked(axis(1), template_id=2, person_id=1, name="Ada")
        eng._append_known_locked(axis(2), template_id=3, person_id=1, name="Ada")
        eng._append_known_locked(axis(9), template_id=4, person_id=2, name="Grace")
        eng._person_template_counts.update({1: 3, 2: 1})
        return eng

    def test_any_of_a_persons_templates_identifies_that_person(self):
        for template_axis in (0, 1, 2):
            with self.subTest(axis=template_axis):
                eng = self._two_people_uneven_templates()
                decision, name, sim = eng._decide_face(axis(template_axis))

                self.assertEqual(decision, MATCH_PRESENT)
                self.assertEqual(name, "Ada")
                self.assertAlmostEqual(sim, 1.0, places=5)
                # One person marked, not one per matching template.
                self.assertEqual(len(eng.marked), 1)
                self.assertEqual(eng.marked[0]["person_id"], 1)

    def test_a_second_person_is_not_shadowed_by_the_multi_template_one(self):
        eng = self._two_people_uneven_templates()
        decision, name, _ = eng._decide_face(axis(9))

        self.assertEqual(decision, MATCH_PRESENT)
        self.assertEqual(name, "Grace")
        self.assertEqual(eng.marked[0]["person_id"], 2)

    def test_the_best_template_wins_even_when_it_is_not_the_first(self):
        """Ada's third template is the closest; the argmax must find it."""
        eng = self._two_people_uneven_templates()
        # Much closer to axis 2 (Ada's third template) than to anything else.
        probe = unit(0.95 * axis(2) + 0.05 * axis(9))
        decision, name, _ = eng._decide_face(probe)

        self.assertEqual(decision, MATCH_PRESENT)
        self.assertEqual(name, "Ada")

    def test_a_multi_template_match_is_flagged_for_the_attendance_row(self):
        """The flag that records "written under an uncalibrated threshold"."""
        eng = self._two_people_uneven_templates()

        eng._decide_face(axis(0))                       # Ada: 3 templates
        self.assertTrue(eng.marked[-1]["multi_template"])

        eng._decide_face(axis(9))                       # Grace: 1 template
        self.assertFalse(eng.marked[-1]["multi_template"])

    def test_calibration_warning_fires_only_when_someone_has_two_templates(self):
        eng = make_engine("Ada", "Grace")
        self.assertIsNone(
            eng._calibration_warning_locked(),
            "one template each is the calibrated configuration",
        )

        eng._append_known_locked(axis(3), template_id=99, person_id=1, name="Ada")
        eng._person_template_counts[1] = 2

        warning = eng._calibration_warning_locked()
        self.assertIsNotNone(warning)
        self.assertIn("Ada", warning)
        self.assertIn("0.370", warning)      # names the constant that is now stale
        self.assertIn("0.412", warning)      # and what the eval measured at N=5


class GalleryMatrixTests(unittest.TestCase):
    def test_matrix_grows_in_place_and_stays_normalized(self):
        eng = AttendanceEngine()
        eng._gallery_loaded = True
        for i in range(20):                     # past the initial capacity of 8
            vec = np.zeros(512, np.float32)
            vec[i] = 7.0                        # deliberately not unit length
            eng._append_known_locked(
                vec, template_id=100 + i, person_id=i, name=f"p{i}"
            )

        gallery = eng._gallery()
        self.assertEqual(gallery.shape, (20, 512))
        np.testing.assert_allclose(
            np.linalg.norm(gallery, axis=1), np.ones(20), rtol=1e-5)
        self.assertEqual(len(eng._person_names), 20)

    def test_the_id_arrays_grow_with_the_matrix_and_stay_aligned(self):
        """A regression guard: the parallel arrays must not fall out of step.

        If `_template_persons` were not grown alongside the matrix, a match on a
        row past the original capacity would resolve to person 0 — silently
        marking the wrong person present, which is the exact failure this
        system's threshold work exists to avoid.
        """
        eng = AttendanceEngine()
        eng._gallery_loaded = True
        for i in range(20):
            eng._append_known_locked(
                axis(i), template_id=500 + i, person_id=1000 + i, name=f"p{i}"
            )

        self.assertEqual(eng._known_count, 20)
        np.testing.assert_array_equal(
            eng._template_ids[:20], np.arange(500, 520, dtype=np.int64)
        )
        np.testing.assert_array_equal(
            eng._template_persons[:20], np.arange(1000, 1020, dtype=np.int64)
        )

        # And a match on a late row resolves to the right person.
        _row, person_id, sim = eng._match(axis(17))
        self.assertEqual(person_id, 1017)
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_gallery_is_none_when_nothing_is_enrolled(self):
        eng = AttendanceEngine()
        eng._gallery_loaded = True
        self.assertIsNone(eng._gallery())

    def test_constructing_an_engine_has_no_side_effects(self):
        """eval/evaluate.py builds one purely to read thresholds off the app.

        It has no database and no camera, so construction must not open either.
        """
        eng = AttendanceEngine()
        self.assertFalse(eng._gallery_loaded)
        self.assertIsNone(eng._known_matrix)
        self.assertIsNone(eng._marked_date)
        self.assertEqual(eng._marked, {})


class ThresholdProvenanceTests(unittest.TestCase):
    """The thresholds are the deliverable; pin them against silent drift."""

    def test_strong_match_is_the_evaluated_operating_point(self):
        self.assertAlmostEqual(config.STRONG_MATCH, 0.370, places=6)

    def test_weak_match_is_below_strong_and_display_only(self):
        self.assertLess(config.WEAK_MATCH, config.STRONG_MATCH)

    def test_engine_re_exports_what_the_eval_harness_reads(self):
        """eval/evaluate.py does `import engine; engine.STRONG_MATCH`."""
        import engine

        for attr in ("MODEL_NAME", "DET_SIZE", "STRONG_MATCH", "WEAK_MATCH"):
            self.assertTrue(hasattr(engine, attr), f"engine.{attr} is missing")
            self.assertEqual(getattr(engine, attr), getattr(config, attr))

    def test_the_matching_site_documents_the_multi_template_caveat(self):
        """The comment the threshold's validity depends on must actually be there.

        STRONG_MATCH = 0.370 was derived at one template per person. Anyone
        adding templates has to find that out at `_match`, which is where the
        max-over-every-row that causes the problem is written.
        """
        import inspect

        import engine

        doc = inspect.getdoc(engine.AttendanceEngine._match) or ""
        self.assertIn("one template", doc.lower())
        self.assertIn("0.412", doc)
        self.assertIn("eval/", doc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
