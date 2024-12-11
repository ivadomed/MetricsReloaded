#######################################################################
#
# Tests for the `compute_metrics_reloaded.py` script
#
# RUN BY:
#   python -m unittest test/test_metrics/test_pairwise_measures_neuropoly.py
#
# Authors: NeuroPoly team
#
#######################################################################

import unittest
import os
import numpy as np
import nibabel as nib
from compute_metrics_reloaded import compute_metrics_single_subject, get_images, fetch_bids_compatible_keys
import tempfile

METRICS = ['dsc', 'fbeta', 'nsd', 'vol_diff', 'rel_vol_error', 'lesion_ppv', 'lesion_sensitivity', 'lesion_f1_score',
           'ref_count', 'pred_count']


class TestComputeMetricsReloaded(unittest.TestCase):
    def setUp(self):
        # Use tempfile.NamedTemporaryFile to create temporary nifti files
        self.ref_file = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
        self.pred_file = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
        self.metrics = METRICS

    def create_dummy_nii(self, file_obj, data):
        img = nib.Nifti1Image(data, np.eye(4))
        nib.save(img, file_obj.name)
        file_obj.seek(0)  # Move back to the beginning of the file

    def tearDown(self):
        # Close and remove temporary files
        self.ref_file.close()
        os.unlink(self.ref_file.name)
        self.pred_file.close()
        os.unlink(self.pred_file.name)

    def assert_metrics(self, metrics_dict, expected_metrics):
        for metric in self.metrics:
            # Loop over labels/classes (e.g., 1, 2, ...)
            for label in expected_metrics.keys():
                # if value is nan, use np.isnan to check
                if np.isnan(expected_metrics[label][metric]):
                    self.assertTrue(np.isnan(metrics_dict[label][metric]))
                # if value is inf, use np.isinf to check
                elif np.isinf(expected_metrics[label][metric]):
                    self.assertTrue(np.isinf(metrics_dict[label][metric]))
                else:
                    self.assertAlmostEqual(metrics_dict[label][metric], expected_metrics[label][metric])

    def test_empty_ref_and_pred(self):
        """
        Empty reference and empty prediction
        """

        expected_metrics = {1.0: {'EmptyPred': True,
                                  'EmptyRef': True,
                                  'dsc': 1,
                                  'fbeta': 1,
                                  'nsd': np.nan,
                                  'rel_vol_error': 0,
                                  'vol_diff': np.nan,
                                  'lesion_ppv': 1.0,
                                  'lesion_sensitivity': 1.0,
                                  'lesion_f1_score': 1.0,
                                  'ref_count': 0,
                                  'pred_count': 0}}

        # Create empty reference
        self.create_dummy_nii(self.ref_file, np.zeros((10, 10, 10)))
        # Create empty prediction
        self.create_dummy_nii(self.pred_file, np.zeros((10, 10, 10)))
        # Compute metrics
        metrics_dict = compute_metrics_single_subject(self.pred_file.name, self.ref_file.name, self.metrics)
        # Assert metrics
        self.assert_metrics(metrics_dict, expected_metrics)

    def test_empty_ref(self):
        """
        Empty reference and non-empty prediction
        """

        expected_metrics = {1.0: {'EmptyPred': False,
                                  'EmptyRef': True,
                                  'dsc': 0.0,
                                  'fbeta': 0,
                                  'nsd': 0.0,
                                  'rel_vol_error': 100,
                                  'vol_diff': np.inf,
                                  'lesion_ppv': 0.0,
                                  'lesion_sensitivity': 1.0,
                                  'lesion_f1_score': 0.0,
                                  'ref_count': 0,
                                  'pred_count': 1}}

        # Create empty reference
        self.create_dummy_nii(self.ref_file, np.zeros((10, 10, 10)))
        # Create non-empty prediction
        pred = np.zeros((10, 10, 10))
        pred[5:7, 2:5] = 1
        self.create_dummy_nii(self.pred_file, pred)
        # Compute metrics
        metrics_dict = compute_metrics_single_subject(self.pred_file.name, self.ref_file.name, self.metrics)
        # Assert metrics
        self.assert_metrics(metrics_dict, expected_metrics)

    def test_empty_pred(self):
        """
        Non-empty reference and empty prediction
        """

        expected_metrics = {1.0: {'EmptyPred': True,
                                  'EmptyRef': False,
                                  'dsc': 0.0,
                                  'fbeta': 0,
                                  'nsd': 0.0,
                                  'rel_vol_error': -100.0,
                                  'vol_diff': 1.0,
                                  'lesion_ppv': 0.0,
                                  'lesion_sensitivity': 0.0,
                                  'lesion_f1_score': 0.0,
                                  'ref_count': 1,
                                  'pred_count': 0}}

        # Create non-empty reference
        ref = np.zeros((10, 10, 10))
        ref[5:7, 2:5] = 1
        self.create_dummy_nii(self.ref_file, ref)
        # Create empty prediction
        self.create_dummy_nii(self.pred_file, np.zeros((10, 10, 10)))
        # Compute metrics
        metrics_dict = compute_metrics_single_subject(self.pred_file.name, self.ref_file.name, self.metrics)
        # Assert metrics
        self.assert_metrics(metrics_dict, expected_metrics)

    def test_non_empty_ref_and_pred(self):
        """
        Non-empty reference and non-empty prediction with partial overlap
        """

        expected_metrics = {1.0: {'EmptyPred': False,
                                  'EmptyRef': False,
                                  'dsc': 0.26666666666666666,
                                  'fbeta': 0.26666667461395266,
                                  'nsd': 0.5373134328358209,
                                  'rel_vol_error': 300.0,
                                  'vol_diff': 3.0,
                                  'lesion_ppv': 1.0,
                                  'lesion_sensitivity': 1.0,
                                  'lesion_f1_score': 1.0,
                                  'ref_count': 1,
                                  'pred_count': 1}}

        # Create non-empty reference
        ref = np.zeros((10, 10, 10))
        ref[4:5, 3:6] = 1
        self.create_dummy_nii(self.ref_file, ref)
        # Create non-empty prediction
        pred = np.zeros((10, 10, 10))
        pred[4:8, 2:5] = 1
        self.create_dummy_nii(self.pred_file, pred)
        # Compute metrics
        metrics_dict = compute_metrics_single_subject(self.pred_file.name, self.ref_file.name, self.metrics)
        # Assert metrics
        self.assert_metrics(metrics_dict, expected_metrics)

    def test_non_empty_ref_and_pred_multi_lesion(self):
        """
        Non-empty reference (2 lesions) and non-empty prediction (2 lesions)
        Multi-lesion (i.e., there are multiple regions (lesions) with voxel values 1)
        Lesion #1: complete overlap; Lesion #2: partial overlap
        """

        expected_metrics = {1.0: {'EmptyPred': False,
                                  'EmptyRef': False,
                                  'dsc': 0.8571428571428571,
                                  'fbeta': 0.8571428571428571,
                                  'nsd': 1.0,
                                  'rel_vol_error': -25.0,
                                  'vol_diff': 0.25,
                                  'lesion_ppv': 1.0,
                                  'lesion_sensitivity': 1.0,
                                  'lesion_f1_score': 1.0,
                                  'ref_count': 2,
                                  'pred_count': 2}}

        # Create non-empty reference
        ref = np.zeros((10, 10, 10))
        # Lesion #1
        ref[1:3, 3:6] = 1
        # Lesion #2
        ref[7:9, 2:5] = 1
        self.create_dummy_nii(self.ref_file, ref)
        # Create non-empty prediction
        pred = np.zeros((10, 10, 10))
        # Lesion #1 -- complete overlap
        pred[1:3, 3:6] = 1
        # Lesion #2 -- partial overlap
        pred[7:8, 2:5] = 1
        self.create_dummy_nii(self.pred_file, pred)
        # Compute metrics
        metrics_dict = compute_metrics_single_subject(self.pred_file.name, self.ref_file.name, self.metrics)
        # Assert metrics
        self.assert_metrics(metrics_dict, expected_metrics)

    def test_non_empty_ref_and_pred_multi_lesion_one_lesion_not_predicted(self):
        """
        Non-empty reference (2 lesions) and non-empty prediction (1 lesion)
        Multi-lesion (i.e., there are multiple regions (lesions) with voxel values 1)
        Lesion #1: complete overlap; Lesion #2: only in reference
        """

        expected_metrics = {1.0: {'EmptyPred': False,
                                  'EmptyRef': False,
                                  'dsc': 0.6666666666666666,
                                  'fbeta': 0.6666666666666666,
                                  'nsd': 0.6666666666666666,
                                  'rel_vol_error': -50.0,
                                  'vol_diff': 0.5,
                                  'lesion_ppv': 1.0,
                                  'lesion_sensitivity': 0.5,
                                  'lesion_f1_score': 0.6666666666666666,
                                  'ref_count': 2,
                                  'pred_count': 1}}

        # Create non-empty reference
        ref = np.zeros((10, 10, 10))
        # Lesion #1
        ref[1:3, 3:6] = 1
        # Lesion #2
        ref[7:9, 2:5] = 1
        self.create_dummy_nii(self.ref_file, ref)
        # Create non-empty prediction
        pred = np.zeros((10, 10, 10))
        # Lesion #1 -- complete overlap
        pred[1:3, 3:6] = 1
        # Note: there is no Lesion #2 in prediction
        self.create_dummy_nii(self.pred_file, pred)
        # Compute metrics
        metrics_dict = compute_metrics_single_subject(self.pred_file.name, self.ref_file.name, self.metrics)
        # Assert metrics
        self.assert_metrics(metrics_dict, expected_metrics)

    def test_non_empty_ref_and_pred_multi_lesion_no_lesion_predicted(self):
        """
        Non-empty reference (2 lesions) and empty prediction (0 lesions)
        Multi-lesion (i.e., there are multiple regions (lesions) with voxel values 1)
        Lesion #1: only in reference; Lesion #2: only in reference
        """

        expected_metrics = {1.0: {'EmptyPred': False,
                                  'EmptyRef': True,
                                  'dsc': 0.0,
                                  'fbeta': 0,
                                  'nsd': 0,
                                  'rel_vol_error': -100.0,
                                  'vol_diff': 1.0,
                                  'lesion_ppv': 0.0,
                                  'lesion_sensitivity': 0.0,
                                  'lesion_f1_score': 0.0,
                                  'ref_count': 2,
                                  'pred_count': 0}}

        # Create non-empty reference
        ref = np.zeros((10, 10, 10))
        # Lesion #1
        ref[1:3, 3:6] = 1
        # Lesion #2
        ref[7:9, 2:5] = 1
        self.create_dummy_nii(self.ref_file, ref)
        # Create non-empty prediction
        pred = np.zeros((10, 10, 10))
        # Note: there is no lesion in prediction
        self.create_dummy_nii(self.pred_file, pred)
        # Compute metrics
        metrics_dict = compute_metrics_single_subject(self.pred_file.name, self.ref_file.name, self.metrics)
        # Assert metrics
        self.assert_metrics(metrics_dict, expected_metrics)

    def test_non_empty_ref_and_pred_multi_class(self):
        """
        Non-empty reference and non-empty prediction with partial overlap
        Multi-class (i.e., voxels with values 1 and 2, e.g., region-based nnUNet training)
        """

        expected_metrics = {1.0: {'dsc': 0.6521739130434783,
                                  'fbeta': 0.5769230751596257,
                                  'nsd': 0.23232323232323232,
                                  'vol_diff': 2.6,
                                  'rel_vol_error': 260.0,
                                  'EmptyRef': False,
                                  'EmptyPred': False,
                                  'lesion_ppv': 1.0,
                                  'lesion_sensitivity': 1.0,
                                  'lesion_f1_score': 1.0,
                                  'ref_count': 1,
                                  'pred_count': 1},
                            2.0: {'dsc': 0.26666666666666666,
                                  'fbeta': 0.26666667461395266,
                                  'nsd': 0.5373134328358209,
                                  'vol_diff': 3.0,
                                  'rel_vol_error': 300.0,
                                  'EmptyRef': False,
                                  'EmptyPred': False,
                                  'lesion_ppv': 1.0,
                                  'lesion_sensitivity': 1.0,
                                  'lesion_f1_score': 1.0,
                                  'ref_count': 1,
                                  'pred_count': 1}}

        # Create non-empty reference
        ref = np.zeros((10, 10, 10))
        ref[4:5, 3:10] = 1
        ref[4:5, 3:6] = 2  # e.g., lesion within spinal cord
        self.create_dummy_nii(self.ref_file, ref)
        # Create non-empty prediction
        pred = np.zeros((10, 10, 10))
        pred[4:8, 2:8] = 1
        pred[4:8, 2:5] = 2  # e.g., lesion within spinal cord
        self.create_dummy_nii(self.pred_file, pred)
        # Compute metrics
        metrics_dict = compute_metrics_single_subject(self.pred_file.name, self.ref_file.name, self.metrics)
        # Assert metrics
        self.assert_metrics(metrics_dict, expected_metrics)

    def test_non_empty_ref_and_pred_with_full_overlap(self):
        """
        Non-empty reference and non-empty prediction with full overlap
        """

        expected_metrics = {1.0: {'EmptyPred': False,
                                  'EmptyRef': False,
                                  'dsc': 1.0,
                                  'fbeta': 1.0,
                                  'nsd': 1.0,
                                  'rel_vol_error': 0.0,
                                  'vol_diff': 0.0,
                                  'lesion_ppv': 1.0,
                                  'lesion_sensitivity': 1.0,
                                  'lesion_f1_score': 1.0,
                                  'ref_count': 1,
                                  'pred_count': 1}}

        # Create non-empty reference
        ref = np.zeros((10, 10, 10))
        ref[4:8, 2:5] = 1
        self.create_dummy_nii(self.ref_file, ref)
        # Create non-empty prediction
        pred = np.zeros((10, 10, 10))
        pred[4:8, 2:5] = 1
        self.create_dummy_nii(self.pred_file, pred)
        # Compute metrics
        metrics_dict = compute_metrics_single_subject(self.pred_file.name, self.ref_file.name, self.metrics)
        # Assert metrics
        self.assert_metrics(metrics_dict, expected_metrics)

class TestGetImages(unittest.TestCase):
    def setUp(self):
        """
        Create temporary directories and files for testing.
        """
        self.pred_dir = tempfile.TemporaryDirectory()
        self.ref_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        """
        Cleanup temporary directories and files after tests.
        """
        self.pred_dir.cleanup()
        self.ref_dir.cleanup()

    def create_temp_file(self, directory, filename):
        """
        Create a temporary file in the given directory with the specified filename.
        """
        file_path = os.path.join(directory, filename)
        with open(file_path, 'w') as f:
            f.write('dummy content')
        return file_path

    def test_matching_files(self):
        """
        Test matching files based on participant_id, acq_id, and run_id.
        """
        self.create_temp_file(self.pred_dir.name, "sub-01_ses-01_acq-01_chunk-1_run-01_pred.nii.gz")
        self.create_temp_file(self.ref_dir.name, "sub-01_ses-01_acq-01_chunk-1_run-01_ref.nii.gz")

        pred_files, ref_files = get_images(self.pred_dir.name, self.ref_dir.name)
        self.assertEqual(len(pred_files), 1)
        self.assertEqual(len(ref_files), 1)

    def test_mismatched_files(self):
        """
        Test when no files match based on the criteria.
        """
        self.create_temp_file(self.pred_dir.name, "sub-01_ses-01_acq-01_chunk-1_run-01_pred.nii.gz")
        self.create_temp_file(self.ref_dir.name, "sub-02_ses-01_acq-02_chunk-1_run-02_ref.nii.gz")

        pred_files, ref_files = get_images(self.pred_dir.name, self.ref_dir.name)
        self.assertEqual(len(pred_files), 0)
        self.assertEqual(len(ref_files), 0)

    def test_ses_id_empty(self):
        """
        Test when ses_id is empty.
        """
        self.create_temp_file(self.pred_dir.name, "sub-01_acq-01_chunk-1_run-01_pred.nii.gz")
        self.create_temp_file(self.ref_dir.name, "sub-01_acq-01_chunk-1_run-01_ref.nii.gz")

        pred_files, ref_files = get_images(self.pred_dir.name, self.ref_dir.name)
        self.assertEqual(len(pred_files), 1)
        self.assertEqual(len(ref_files), 1)
        self.assertIn("sub-01_acq-01_chunk-1_run-01_pred.nii.gz", pred_files[0])
        self.assertIn("sub-01_acq-01_chunk-1_run-01_ref.nii.gz", ref_files[0])

    def test_acq_id_empty(self):
        """
        Test when acq_id is empty.
        """
        self.create_temp_file(self.pred_dir.name, "sub-01_ses-01_chunk-1_run-01_pred.nii.gz")
        self.create_temp_file(self.ref_dir.name, "sub-01_ses-01_chunk-1_run-01_ref.nii.gz")

        pred_files, ref_files = get_images(self.pred_dir.name, self.ref_dir.name)
        self.assertEqual(len(pred_files), 1)
        self.assertEqual(len(ref_files), 1)
        self.assertIn("sub-01_ses-01_chunk-1_run-01_pred.nii.gz", pred_files[0])
        self.assertIn("sub-01_ses-01_chunk-1_run-01_ref.nii.gz", ref_files[0])

    def test_chunk_id_empty(self):
        """
        Test when chunk_id is empty in the filenames.
        """
        self.create_temp_file(self.pred_dir.name, "sub-01_ses-01_acq-01_run-01_pred.nii.gz")
        self.create_temp_file(self.ref_dir.name, "sub-01_ses-01_acq-01_run-01_ref.nii.gz")

        pred_files, ref_files = get_images(self.pred_dir.name, self.ref_dir.name)

        # Assert the matched files
        self.assertEqual(len(pred_files), 1)
        self.assertEqual(len(ref_files), 1)
        self.assertIn("sub-01_ses-01_acq-01_run-01_pred.nii.gz", pred_files[0])
        self.assertIn("sub-01_ses-01_acq-01_run-01_ref.nii.gz", ref_files[0])

    def test_run_id_empty(self):
        """
        Test when run_id is empty in the filenames.
        """
        self.create_temp_file(self.pred_dir.name, "sub-01_ses-01_acq-01_chunk-1_pred.nii.gz")
        self.create_temp_file(self.ref_dir.name, "sub-01_ses-01_acq-01_chunk-1_ref.nii.gz")

        pred_files, ref_files = get_images(self.pred_dir.name, self.ref_dir.name)

        # Assert the matched files
        self.assertEqual(len(pred_files), 1)
        self.assertEqual(len(ref_files), 1)
        self.assertIn("sub-01_ses-01_acq-01_chunk-1_pred.nii.gz", pred_files[0])
        self.assertIn("sub-01_ses-01_acq-01_chunk-1_ref.nii.gz", ref_files[0])

    def test_no_files(self):
        """
        Test when there are no files in the directories.
        Ensure that FileNotFoundError is raised.
        """
        with self.assertRaises(FileNotFoundError) as context:
            get_images(self.pred_dir.name, self.ref_dir.name)
        # Check the exception message
        self.assertIn(f'No prediction files found in {self.pred_dir.name}', str(context.exception))

    def test_partial_matching(self):
        """
        Test when some files match and some do not.
        """
        self.create_temp_file(self.pred_dir.name, "sub-01_acq-01_run-01_pred.nii.gz")
        self.create_temp_file(self.ref_dir.name, "sub-01_acq-01_run-01_ref.nii.gz")
        # The following file will not be included in the lists below as there is no matching reference (GT) file
        self.create_temp_file(self.pred_dir.name, "sub-02_acq-02_run-02_pred.nii.gz")

        pred_files, ref_files = get_images(self.pred_dir.name, self.ref_dir.name)
        self.assertEqual(len(pred_files), 1)
        self.assertEqual(len(ref_files), 1)


if __name__ == '__main__':
    unittest.main()
