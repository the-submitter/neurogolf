from __future__ import annotations

import contextlib
import io
import math
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from hypothesis import given
from hypothesis import strategies as st

from neurogolf.errors import CandidateError
from neurogolf.judge.candidate_runner import CandidateRunner
from neurogolf.judge.legality import inspect_legality
from neurogolf.judge.official_adapter import OfficialAdapter
from neurogolf.onnx_lab.builder import GraphBuilder
from neurogolf.onnx_lab.constants import constant_node
from neurogolf.onnx_lab.templates.basic import identity_model, single_conv_model
from neurogolf.tasks.models import ArcExample

GRID_SHAPE = (1, 10, 30, 30)


def _save(model: onnx.ModelProto, path: Path) -> Path:
    onnx.save(model, path)
    return path


def _official_score(
    adapter: OfficialAdapter, model: onnx.ModelProto, tmp_path: Path
) -> tuple[int, int]:
    copied = onnx.load_from_string(model.SerializeToString())
    sanitized = adapter.module.sanitize_model(copied)
    assert sanitized is not None
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(tmp_path / "direct-official")
    session = ort.InferenceSession(sanitized.SerializeToString(), options)
    adapter.module.run_network(session, np.zeros(GRID_SHAPE, dtype=np.float32))
    with contextlib.redirect_stdout(io.StringIO()):
        memory, parameters = adapter.module.score_network(sanitized, session.end_profiling())
    assert memory is not None and parameters is not None
    return int(memory), int(parameters)


@st.composite
def _arc_grid(draw: st.DrawFn) -> list[list[int]]:
    height = draw(st.integers(1, 30))
    width = draw(st.integers(1, 30))
    return draw(
        st.lists(
            st.lists(st.integers(0, 9), min_size=width, max_size=width),
            min_size=height,
            max_size=height,
        )
    )


@given(grid=_arc_grid())
def test_grid_conversion_is_byte_exact(
    grid: list[list[int]], official_adapter: OfficialAdapter
) -> None:
    example = {"input": grid, "output": grid}
    expected = official_adapter.module.convert_to_numpy(example)
    actual = official_adapter.convert_example(example)
    assert expected is not None
    assert actual["input"].dtype == np.float32
    assert actual["input"].tobytes() == expected["input"].tobytes()
    assert official_adapter.tensor_to_grid(actual["output"]) == grid


def test_conversion_rejection_and_malformed_one_hot(official_adapter: OfficialAdapter) -> None:
    oversized = [[0] * 31]
    assert (
        official_adapter.module.convert_to_numpy({"input": oversized, "output": oversized}) is None
    )
    with pytest.raises(CandidateError, match="larger than 30x30"):
        official_adapter.convert_example({"input": oversized, "output": oversized})

    tensor = np.zeros(GRID_SHAPE, dtype=np.float32)
    tensor[0, 1, 0, 0] = 1
    tensor[0, 2, 0, 0] = 1
    tensor[0, 3, 0, 1] = 1
    expected = official_adapter.module.convert_from_numpy(tensor)
    assert official_adapter.tensor_to_grid(tensor) == expected == [[11, 3]]


def test_threshold_edges_match_official_runtime(official_adapter: OfficialAdapter) -> None:
    model = official_adapter.sanitize(identity_model())
    session = official_adapter.create_session(model)
    values = np.zeros(GRID_SHAPE, dtype=np.float32)
    values[0, 0, 0, 0] = -np.finfo(np.float32).tiny
    values[0, 1, 0, 0] = -0.0
    values[0, 2, 0, 0] = 0.0
    values[0, 3, 0, 0] = np.finfo(np.float32).tiny
    direct = official_adapter.module.run_network(session, values)
    assert np.array_equal(official_adapter.run_session(session, values), direct)
    assert np.array_equal(official_adapter.threshold_logits(values), direct)
    assert direct[0, :, 0, 0].tolist() == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("model", [identity_model(), single_conv_model()])
def test_identity_and_single_conv_scoring_matches_official(
    model: onnx.ModelProto, tmp_path: Path, official_adapter: OfficialAdapter
) -> None:
    direct_memory, direct_parameters = _official_score(official_adapter, model, tmp_path)
    score = official_adapter.score_model(model)
    assert (score.memory_bytes, score.parameter_elements) == (
        direct_memory,
        direct_parameters,
    )
    assert score.objective == direct_memory + direct_parameters
    assert score.points == max(1.0, 25.0 - math.log(max(1, score.objective)))


def test_official_starter_conv_runtime_matches_local_template(
    official_adapter: OfficialAdapter,
) -> None:
    def weight(out_channel: int, in_channel: int, offset: tuple[int, int]) -> float:
        return float(out_channel == in_channel and offset == (0, 0))

    official = official_adapter.module.single_layer_conv2d_network(weight, 3)
    local = single_conv_model(weight, kernel_size=3)
    tensor = official_adapter.grid_to_tensor([[1, 2, 3], [4, 5, 6]])
    official_session = official_adapter.create_session(official_adapter.sanitize(official))
    local_session = official_adapter.create_session(official_adapter.sanitize(local))
    assert np.array_equal(
        official_adapter.run_session(official_session, tensor),
        official_adapter.run_session(local_session, tensor),
    )
    assert official_adapter.calculate_params(official) == official_adapter.calculate_params(local)


def test_initializer_and_constant_parameter_count_parity(official_adapter: OfficialAdapter) -> None:
    builder = GraphBuilder.standard("parameters")
    initial = builder.initializer(np.ones((4, 5), dtype=np.float32))
    constant = constant_node(builder, np.ones((2, 3), dtype=np.float32))
    # Keep both parameter sources live while broadcasting through scalar reductions.
    sum_initial = str(
        builder.node(
            "ReduceSum",
            [initial],
            axes=[0, 1],
            keepdims=0,
            dtype=onnx.TensorProto.FLOAT,
            shape=(),
        )
    )
    sum_constant = str(
        builder.node(
            "ReduceSum",
            [constant],
            axes=[0, 1],
            keepdims=0,
            dtype=onnx.TensorProto.FLOAT,
            shape=(),
        )
    )
    total = str(
        builder.node("Add", [sum_initial, sum_constant], dtype=onnx.TensorProto.FLOAT, shape=())
    )
    output = str(
        builder.node("Add", ["input", total], dtype=onnx.TensorProto.FLOAT, shape=GRID_SHAPE)
    )
    builder.direct_output(output, dtype=onnx.TensorProto.FLOAT, shape=GRID_SHAPE)
    model = builder.build()
    assert official_adapter.calculate_params(model) == 26


@pytest.mark.parametrize(
    ("intermediate_dtype", "expected_bytes"),
    [(onnx.TensorProto.BOOL, 9_000), (onnx.TensorProto.FLOAT16, 18_000)],
)
def test_bool_and_float16_intermediate_memory_parity(
    intermediate_dtype: int,
    expected_bytes: int,
    tmp_path: Path,
    official_adapter: OfficialAdapter,
) -> None:
    builder = GraphBuilder.standard("cheap_intermediate")
    if intermediate_dtype == onnx.TensorProto.BOOL:
        zero = builder.initializer(np.asarray(0.0, dtype=np.float32))
        intermediate = str(
            builder.node(
                "Greater",
                ["input", zero],
                dtype=onnx.TensorProto.BOOL,
                shape=GRID_SHAPE,
            )
        )
    else:
        intermediate = str(
            builder.node(
                "Cast",
                ["input"],
                to=onnx.TensorProto.FLOAT16,
                dtype=onnx.TensorProto.FLOAT16,
                shape=GRID_SHAPE,
            )
        )
    builder.node(
        "Cast",
        [intermediate],
        output="output",
        to=onnx.TensorProto.FLOAT,
        dtype=onnx.TensorProto.FLOAT,
        shape=GRID_SHAPE,
    )
    builder.add_output("output", onnx.TensorProto.FLOAT, GRID_SHAPE)
    model = builder.build()
    direct_memory, direct_parameters = _official_score(official_adapter, model, tmp_path)
    score = official_adapter.score_model(model)
    assert direct_memory == score.memory_bytes == expected_bytes
    assert (
        direct_parameters
        == score.parameter_elements
        == (1 if intermediate_dtype == onnx.TensorProto.BOOL else 0)
    )


def _identity_with_io(
    inputs: list[onnx.ValueInfoProto], outputs: list[onnx.ValueInfoProto]
) -> onnx.ModelProto:
    nodes = [onnx.helper.make_node("Identity", [inputs[0].name], [outputs[0].name])]
    graph = onnx.helper.make_graph(nodes, "invalid_fixture", inputs, outputs)
    return onnx.helper.make_model(
        graph, ir_version=10, opset_imports=[onnx.helper.make_opsetid("", 10)]
    )


def test_dynamic_and_multiple_io_models_are_rejected(
    tmp_path: Path, official_adapter: OfficialAdapter
) -> None:
    dynamic_input = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, [1, 10, "height", 30]
    )
    dynamic_output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1, 10, "height", 30]
    )
    dynamic_path = _save(
        _identity_with_io([dynamic_input], [dynamic_output]), tmp_path / "dynamic.onnx"
    )
    result = inspect_legality(dynamic_path, official_adapter)
    assert not result.legal
    assert any("dynamic/nonpositive" in error for error in result.errors)

    standard_input = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, GRID_SHAPE)
    extra_input = onnx.helper.make_tensor_value_info("extra", onnx.TensorProto.FLOAT, GRID_SHAPE)
    standard_output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, GRID_SHAPE
    )
    multiple_path = _save(
        _identity_with_io([standard_input, extra_input], [standard_output]),
        tmp_path / "multiple.onnx",
    )
    result = inspect_legality(multiple_path, official_adapter)
    assert not result.legal
    assert any("Exactly one" in error for error in result.errors)


def test_excluded_op_duplicate_metadata_and_oversize_rejection(
    tmp_path: Path, official_adapter: OfficialAdapter
) -> None:
    input_info = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, GRID_SHAPE)
    output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.INT64, [4, 9000])
    nonzero = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("NonZero", ["input"], ["output"])],
            "excluded",
            [input_info],
            [output_info],
        ),
        ir_version=10,
        opset_imports=[onnx.helper.make_opsetid("", 10)],
    )
    excluded_path = _save(nonzero, tmp_path / "excluded.onnx")
    result = inspect_legality(excluded_path, official_adapter)
    assert not result.legal
    assert any("Prohibited operator" in error for error in result.errors)

    duplicate = identity_model()
    duplicate.graph.value_info.extend(
        [
            onnx.helper.make_tensor_value_info("duplicate", onnx.TensorProto.FLOAT, [1]),
            onnx.helper.make_tensor_value_info("duplicate", onnx.TensorProto.FLOAT, [1]),
        ]
    )
    duplicate_path = _save(duplicate, tmp_path / "duplicate.onnx")
    result = inspect_legality(duplicate_path, official_adapter)
    assert not result.legal
    assert any("Duplicate graph type metadata" in error for error in result.errors)

    oversized = _save(identity_model(), tmp_path / "oversized.onnx")
    with oversized.open("ab") as stream:
        stream.write(b"0" * (math.ceil(official_adapter.file_size_limit) + 1))
    valid, diagnostic = official_adapter.check_file(oversized)
    assert not valid
    assert "Filesize" in diagnostic


def test_candidate_runner_reports_zero_and_multi_hot_outputs(
    tmp_path: Path, official_adapter: OfficialAdapter
) -> None:
    builder = GraphBuilder.standard("zero_output")
    zero = builder.initializer(np.asarray(0.0, dtype=np.float32))
    builder.node(
        "Mul",
        ["input", zero],
        output="output",
        dtype=onnx.TensorProto.FLOAT,
        shape=GRID_SHAPE,
    )
    builder.add_output("output", onnx.TensorProto.FLOAT, GRID_SHAPE)
    path = _save(builder.build(), tmp_path / "zero.onnx")
    comparison = CandidateRunner(path, official_adapter).compare(
        ArcExample(input=[[1]], output=[[1]]), family="fixture"
    )
    assert not comparison.passed
    assert comparison.actual == []
    assert comparison.shape_difference == {
        "expected": [1, 1],
        "actual": [0, 0],
        "expected_row_widths": [1],
        "actual_row_widths": [],
    }
    assert comparison.malformed_one_hot[0]["positive_channels"] == 0
