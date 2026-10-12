# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Tests for the reconstruction functional API."""

import inspect
import os
import tempfile
import pytest
import subprocess
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np
import h5py
import shutil

from lauelab.reconstruct.reconstruct import _find_executable, _validate_executable
from lauelab.reconstruct import (
    reconstruct,
    find_executable,
    ReconstructionResult,
    # GPU functions
    reconstruct_gpu,
    find_gpu_executable,
    gpu_available
)


def test_reconstruction_geometry_parameter_is_unified():
    assert "geometry" in inspect.signature(reconstruct).parameters
    assert "geometry_file" not in inspect.signature(reconstruct).parameters
    assert "geometry" in inspect.signature(reconstruct_gpu).parameters
    assert "geometry_file" not in inspect.signature(reconstruct_gpu).parameters


@pytest.fixture
def mock_subprocess():
    with patch("subprocess.run") as mock:
        help_response = MagicMock(
            returncode=1,
            stdout="Usage: WireScan -i <file> -o <file> -g <file>",
            stderr="",
        )
        run_response = MagicMock(
            returncode=0,
            stdout="Reconstruction complete",
            stderr="",
        )
        mock.side_effect = lambda args, **kwargs: (
            help_response if "--help" in args else run_response
        )
        yield mock


@pytest.fixture
def mock_executable():
    with patch("lauelab.reconstruct.reconstruct._find_executable") as mock:
        mock.side_effect = lambda name="reconstructN_cpu": f"/path/to/{name}"
        yield mock


class TestReconstruct:
    """Test reconstruction functions."""

    def test_reconstruction_result_type(self):
        """Test that ReconstructionResult is properly defined."""
        result = ReconstructionResult(
            success=True,
            output_files=['file1.h5', 'file2.h5'],
            log='Success',
            error=None,
            command='reconstructN_cpu ...',
            return_code=0
        )
        assert result.success is True
        assert len(result.output_files) == 2
        assert result.error is None
    
    @pytest.mark.parametrize(
        ("reconstruct_function", "executable"),
        [(reconstruct, "reconstructN_cpu"), (reconstruct_gpu, "reconstructN_gpu")],
    )
    def test_reconstruct_basic(
        self, mock_subprocess, mock_executable, reconstruct_function, executable
    ):
        """Test basic CPU and GPU reconstruction calls."""
        result = reconstruct_function(
            'input.h5',
            'output_',
            'geo.xml',
            (0.0, 10.0),
            resolution=1.0
        )
        
        assert isinstance(result, ReconstructionResult)
        assert result.success is True
        assert result.return_code == 0
        
        # Verify subprocess was called
        mock_subprocess.assert_called()
        call_args = mock_subprocess.call_args[0][0]
        
        # Check basic arguments
        assert call_args[0] == f"/path/to/{executable}"
        assert '-i' in call_args
        assert 'input.h5' in call_args
        assert '-o' in call_args
        assert 'output_' in call_args
        assert '-g' in call_args
        assert 'geo.xml' in call_args
        assert '-s' in call_args
        assert '0.0' in call_args
        assert '-e' in call_args
        assert '10.0' in call_args
        assert mock_subprocess.call_args.kwargs["env"]["OPENBLAS_NUM_THREADS"] == "1"

    @pytest.mark.parametrize("reconstruct_function", [reconstruct, reconstruct_gpu])
    def test_reconstruct_preserves_openblas_thread_setting(
        self, monkeypatch, mock_subprocess, mock_executable, reconstruct_function
    ):
        monkeypatch.setenv("OPENBLAS_NUM_THREADS", "4")

        reconstruct_function('input.h5', 'output_', 'geo.xml', (0.0, 10.0))

        assert mock_subprocess.call_args.kwargs["env"]["OPENBLAS_NUM_THREADS"] == "4"

    def test_reconstruct_reports_only_new_or_updated_outputs_in_numeric_order(
        self, tmp_path, mock_executable
    ):
        output_base = tmp_path / "recon_"
        stale = tmp_path / "recon_1.h5"
        stale.write_text("stale")

        def run(args, **kwargs):
            if "--help" in args:
                return MagicMock(
                    returncode=1, stdout="Usage: WireScan -i <file>", stderr=""
                )
            (tmp_path / "recon_10.h5").write_text("new")
            (tmp_path / "recon_2.h5").write_text("new")
            return MagicMock(returncode=0, stdout="complete", stderr="")

        _validate_executable.cache_clear()
        with patch("subprocess.run", side_effect=run):
            result = reconstruct(
                "input.h5", str(output_base), "geo.xml", (0.0, 10.0)
            )

        assert result.output_files == [
            str(tmp_path / "recon_2.h5"),
            str(tmp_path / "recon_10.h5"),
        ]

    @pytest.mark.parametrize(
        ("reconstruct_function", "wire_depths_flag"),
        [(reconstruct, "--wireDepths"), (reconstruct_gpu, "-W")],
    )
    def test_reconstruct_with_all_options(
        self, mock_subprocess, mock_executable, reconstruct_function, wire_depths_flag
    ):
        """Test shared CPU and GPU reconstruction options."""
        result = reconstruct_function(
            'input.h5',
            'output_',
            'geo.xml',
            (-5.0, 5.0),
            resolution=0.5,
            image_range=(1, 100),
            verbose=2,
            percent_brightest=50.0,
            wire_edge='both',
            memory_limit_mb=256,
            normalization='norm_tag',
            output_pixel_type=3,
            distortion_map='distortion.map',
            detector_number=1,
            wire_depths_file='depths.txt'
        )
        
        assert result.success is True
        
        # Check all parameters in command
        call_args = mock_subprocess.call_args[0][0]
        assert '-r' in call_args
        assert '0.5' in call_args
        assert '-f' in call_args
        assert '1' in call_args
        assert '-l' in call_args
        assert '100' in call_args
        assert '-v' in call_args
        assert '2' in call_args
        assert '-p' in call_args
        assert '50.0' in call_args
        assert '-w' in call_args
        assert 'b' in call_args  # 'both' maps to 'b'
        assert '-m' in call_args
        assert '256' in call_args
        assert '-n' in call_args
        assert 'norm_tag' in call_args
        assert '-t' in call_args
        assert '3' in call_args
        assert '-d' in call_args
        assert 'distortion.map' in call_args
        assert '-D' in call_args
        assert '1' in call_args
        assert wire_depths_flag in call_args
        assert 'depths.txt' in call_args
    
    @pytest.mark.parametrize("reconstruct_function", [reconstruct, reconstruct_gpu])
    def test_wire_edge_mapping(
        self, mock_subprocess, mock_executable, reconstruct_function
    ):
        """Test that wire edge names are properly mapped."""
        for user_name, expected in [
            ('leading', 'l'),
            ('trailing', 't'),
            ('both', 'b'),
            ('LEADING', 'l'),  # Case insensitive
            ('l', 'l'),  # Also accept short form
            ('t', 't'),
            ('b', 'b')
        ]:
            reconstruct_function(
                'input.h5', 'output_', 'geo.xml', (0, 10),
                wire_edge=user_name
            )
            call_args = mock_subprocess.call_args[0][0]
            idx = call_args.index('-w')
            assert call_args[idx + 1] == expected
    
    @pytest.mark.parametrize("reconstruct_function", [reconstruct, reconstruct_gpu])
    def test_invalid_wire_edge(self, mock_executable, reconstruct_function):
        """Test that invalid wire edge raises ValueError."""
        with patch('lauelab.reconstruct.reconstruct._validate_executable'):
            with pytest.raises(ValueError, match="Invalid wire_edge"):
                reconstruct_function(
                    'input.h5', 'output_', 'geo.xml', (0, 10),
                    wire_edge='invalid'
                )
    
    @pytest.mark.parametrize("reconstruct_function", [reconstruct, reconstruct_gpu])
    def test_invalid_depth_range(self, mock_executable, reconstruct_function):
        """Test that invalid depth range raises ValueError."""
        with patch('lauelab.reconstruct.reconstruct._validate_executable'):
            with pytest.raises(ValueError, match="Invalid depth range"):
                reconstruct_function(
                    'input.h5', 'output_', 'geo.xml', (10, 5)  # Start > end
                )
    
    
    def test_find_executable_function(self):
        """Test the public find_executable function."""
        with patch('lauelab.reconstruct.reconstruct._find_executable') as mock:
            mock.return_value = '/path/to/exe'
            
            path = find_executable()
            assert path == '/path/to/exe'
            mock.assert_called_once_with()
    
    def test_executable_not_found(self):
        """Test behavior when executable is not found."""
        # An earlier test may have cached the real executable's path.
        _find_executable.cache_clear()
        with patch('lauelab.reconstruct.reconstruct.shutil.which', return_value=None):
            with patch('lauelab.reconstruct.reconstruct.resources.files', side_effect=ModuleNotFoundError):
                with pytest.raises(FileNotFoundError, match="not found"):
                    reconstruct('in.h5', 'out_', 'geo.xml', (0, 10))
    
    def test_subprocess_timeout(self, mock_executable):
        """Test handling of subprocess timeout."""
        with patch('lauelab.reconstruct.reconstruct._validate_executable'):
            with patch('subprocess.run', side_effect=subprocess.TimeoutExpired('cmd', 30)):
                result = reconstruct(
                    'input.h5', 'output_', 'geo.xml', (0, 10),
                    timeout=30
                )
                
                assert result.success is False
                assert 'timed out' in result.error
                assert result.return_code == -1
    
    def test_executable_validation_wraps_oserror(self):
        from lauelab.reconstruct.reconstruct import _validate_executable

        with patch('subprocess.run', side_effect=OSError("Exec format error")):
            with pytest.raises(RuntimeError, match="Exec format error"):
                _validate_executable("broken-executable")

    def test_subprocess_exception(self, mock_executable):
        """Test handling of subprocess exceptions."""
        with patch('lauelab.reconstruct.reconstruct._validate_executable'):
            with patch('subprocess.run', side_effect=Exception("Test error")):
                result = reconstruct(
                    'input.h5', 'output_', 'geo.xml', (0, 10)
                )
                
                assert result.success is False
                assert "Test error" in result.error
                assert result.return_code == -1
    
    @pytest.mark.integration
    def test_real_executable_validation(self):
        """Test with real executable if available."""
        try:
            exe_path = find_executable()
            # If we get here, executable was found
            
            # Test that validation doesn't raise
            from lauelab.reconstruct.reconstruct import _validate_executable
            _validate_executable(exe_path)
            
        except FileNotFoundError:
            pytest.skip("Reconstruction executable not available")
    
    @pytest.mark.integration
    def test_smoke_test_with_synthetic_data(self):
        """Integration test with minimal synthetic data."""
        try:
            exe_path = find_executable()
        except FileNotFoundError:
            pytest.skip("Reconstruction executable not available")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal test data
            image_size = 128
            num_images = 5
            
            # Create synthetic HDF5 file
            input_file = os.path.join(tmpdir, "test_wire_scan.h5")
            with h5py.File(input_file, 'w') as f:
                # Create minimal required structure
                facility = f.create_group('Facility')
                facility.create_dataset('facility_name', data=b'TEST')
                facility.create_dataset('facility_beamline', data=b'34ID-E')
                
                entry = f.create_group('entry1')
                data_group = entry.create_group('data')
                
                # Create small synthetic images
                for i in range(num_images):
                    image_data = np.zeros((image_size, image_size), dtype=np.float32)
                    stripe_pos = int(i * image_size / num_images)
                    image_data[:, stripe_pos:stripe_pos+10] = 1000.0 * np.random.rand(image_size, 10)
                    image_data += 10.0 * np.random.rand(image_size, image_size)
                    data_group.create_dataset(f'data_{i:04d}', data=image_data)
                
                data_group.create_dataset('data', data=image_data)
                entry.create_dataset('depth', data=[0.0])
                
                detector = entry.create_group('detector')
                detector.create_dataset('Nx', data=[image_size])
                detector.create_dataset('Ny', data=[image_size])
                detector.create_dataset('binx', data=[1])
                detector.create_dataset('biny', data=[1])
            
            # Create or copy geometry file
            geo_file = os.path.join(tmpdir, "test_geo.xml")
            geo_source = "tests/data/geo/geoN_2022-03-29_14-15-05.xml"
            if os.path.exists(geo_source):
                shutil.copy(geo_source, geo_file)
            else:
                with open(geo_file, 'w') as f:
                    f.write("""<?xml version="1.0"?>
<geometry>
    <detector>
        <distance>100.0</distance>
        <pixelSize>0.1</pixelSize>
        <centerX>64</centerX>
        <centerY>64</centerY>
    </detector>
</geometry>""")
            
            # Run reconstruction
            output_base = os.path.join(tmpdir, "output", "recon_")
            result = reconstruct(
                input_file,
                output_base,
                geo_file,
                (-2.0, 2.0),
                resolution=1.0,
                image_range=(0, num_images-1),
                verbose=1,
                percent_brightest=50.0,
                memory_limit_mb=50
            )
            
            # Check results
            assert isinstance(result, ReconstructionResult)
            assert 'command' in result._asdict()
            
            if not result.success:
                print(f"Command: {result.command}")
                print(f"Error: {result.error}")
                print(f"Log: {result.log[:500]}")

class TestReconstructGPU:
    """Test GPU-specific reconstruction behavior."""

    def test_gpu_specific_command_options(self, mock_subprocess, mock_executable):
        result = reconstruct_gpu(
            'input.h5',
            'output_',
            'geo.xml',
            (0.0, 10.0),
            cuda_rows=16,
        )

        assert result.success is True
        call_args = mock_subprocess.call_args[0][0]
        assert call_args[call_args.index('-R') + 1] == '16'
        assert '-C' not in call_args
        assert '-E' not in call_args
        assert '-T' not in call_args
        assert '-N' not in call_args

    def test_gpu_cuda_rows_defaults_to_eight(self, mock_subprocess, mock_executable):
        reconstruct_gpu('input.h5', 'output_', 'geo.xml', (0.0, 10.0))
        call_args = mock_subprocess.call_args[0][0]
        assert call_args[call_args.index('-R') + 1] == '8'

    def test_find_gpu_executable_function(self):
        """Test the public find_gpu_executable function."""
        with patch('lauelab.reconstruct.reconstruct._find_executable') as mock:
            mock.return_value = '/path/to/gpu/exe'
            
            path = find_gpu_executable()
            assert path == '/path/to/gpu/exe'
            mock.assert_called_once_with('reconstructN_gpu')
    
    def test_gpu_available_true(self):
        """Test gpu_available when GPU is available."""
        with patch('lauelab.reconstruct.reconstruct._find_executable') as mock_find:
            mock_find.return_value = '/path/to/reconstructN_gpu'
            with patch('lauelab.reconstruct.reconstruct._validate_executable'):
                assert gpu_available() is True
    
    def test_gpu_available_false_not_found(self):
        """Test gpu_available when GPU executable not found."""
        with patch('lauelab.reconstruct.reconstruct._find_executable') as mock_find:
            mock_find.side_effect = FileNotFoundError("Not found")
            assert gpu_available() is False
    
    def test_gpu_available_false_validation_fails(self):
        """Test gpu_available when GPU executable validation fails."""
        with patch('lauelab.reconstruct.reconstruct._find_executable') as mock_find:
            mock_find.return_value = '/path/to/reconstructN_gpu'
            with patch('lauelab.reconstruct.reconstruct._validate_executable') as mock_validate:
                mock_validate.side_effect = RuntimeError("Validation failed")
                assert gpu_available() is False
    
    def test_gpu_executable_not_found(self):
        """Test behavior when GPU executable is not found."""
        with patch('lauelab.reconstruct.reconstruct.shutil.which', return_value=None):
            with patch('lauelab.reconstruct.reconstruct.resources.files', side_effect=ModuleNotFoundError):
                with pytest.raises(FileNotFoundError, match="reconstructN_gpu"):
                    reconstruct_gpu('in.h5', 'out_', 'geo.xml', (0, 10))
    
    @pytest.mark.integration
    def test_real_gpu_executable_validation(self):
        """Test with real GPU executable if available."""
        if not gpu_available():
            pytest.skip("GPU reconstruction executable not available")
        
        try:
            exe_path = find_gpu_executable()
            # If we get here, GPU executable was found
            
            # Test that validation doesn't raise
            from lauelab.reconstruct.reconstruct import _validate_executable
            _validate_executable(exe_path)
            
        except FileNotFoundError:
            pytest.skip("GPU reconstruction executable not available")


class TestReconstructCPUReference:
    """Numerical parity against the recorded CPU reconstruction reference.

    The reference in ``tests/data/reconstruction`` is an acceptance contract
    for the native build (see ``BUILD_DEPLOYMENT_PLAN.md``). It must not be
    regenerated or its tolerance loosened to make a build change pass.
    """

    REFERENCE_DIR = Path(__file__).resolve().parent / "data" / "reconstruction"

    @pytest.fixture(scope="class")
    @classmethod
    def reference_module(cls):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "cpu_reference_generator", cls.REFERENCE_DIR / "generate_reference.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @pytest.mark.integration
    @pytest.mark.parametrize("variant", [None, "trailing", "both", "cosmic", "norm_vector", "norm_exponent", "out_uint16", "out_int16"])
    def test_cpu_reconstruction_matches_reference(
        self, reference_module, tmp_path, variant
    ):
        try:
            exe_path = find_executable()
        except FileNotFoundError:
            pytest.skip("Reconstruction executable not available")

        import json

        suffix = f"_{variant}" if variant else ""
        expected = np.load(self.REFERENCE_DIR / f"cpu_reference{suffix}.npz")
        provenance = json.loads(
            (self.REFERENCE_DIR / f"cpu_reference{suffix}.json").read_text()
        )

        input_file = tmp_path / "synthetic_wire_scan.h5"
        options = reference_module.VARIANTS.get(variant, {})
        reference_module.write_input_file(
            input_file,
            write_mA=options.get("write_mA", False),
            write_microdiffraction=options.get("write_microdiffraction", False),
        )
        import hashlib

        assert hashlib.sha256(input_file.read_bytes()).hexdigest() == provenance["input_sha256"], (
            "synthetic input changed; the generator or its dependencies no longer reproduce the reference input"
        )

        result = reference_module.run_reconstruction(
            exe_path, input_file, tmp_path / "out" / "recon_", variant=variant
        )
        assert result.success, f"{result.command}\n{result.log}\n{result.error}"

        images, depths = reference_module.load_outputs(tmp_path / "out" / "recon_")
        tolerance = provenance["comparison"]
        np.testing.assert_allclose(depths, expected["depth_um"], rtol=0, atol=0)
        np.testing.assert_allclose(
            images, expected["images"], rtol=tolerance["rtol"], atol=tolerance["atol"]
        )
        assert images.shape == tuple(provenance["output_shape"])
