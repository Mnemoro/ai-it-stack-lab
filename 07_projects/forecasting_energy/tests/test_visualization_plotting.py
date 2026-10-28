import ast
import inspect
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "visualization" / "plotting.py"
PUBLIC_APIS = {
    "plot_training_history", "plot_prediction_window", "plot_error_distribution",
    "plot_evaluation_metrics", "plot_future_forecast", "plot_grid_exchange",
    "plot_cumulative_grid_exchange",
}
FORBIDDEN_IMPORT_PARTS = {"tensorflow", "keras", "sklearn", "forecasting", "monitoring", "energy_balance"}


def _tree():
    return ast.parse(MODULE_PATH.read_text(encoding="utf-8"))


def _load_plotting():
    pytest.importorskip("numpy")
    pytest.importorskip("pandas")
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg", force=True)
    pytest.importorskip("seaborn")
    import sys
    src_path = str(MODULE_PATH.parents[1])
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from visualization import plotting
    return plotting


def _close(fig):
    import matplotlib.pyplot as plt
    plt.close(fig)


def _time_series():
    pd = pytest.importorskip("pandas")
    idx = pd.date_range("2026-01-01", periods=3, freq="h")
    return idx, pd.Series([1.0, 2.0, 3.0], index=idx), pd.Series([0.5, 1.5, 2.5], index=idx)


def test_public_apis_and_exact_parameter_contracts():
    plotting = _load_plotting()
    expected = {
        "plot_training_history": ["history", "title", "figsize", "colors", "show_grid"],
        "plot_prediction_window": ["y_true", "y_pred", "target_name", "window_index", "timestamps", "unit", "title", "figsize", "colors", "show_grid"],
        "plot_error_distribution": ["errors", "target_name", "bins", "kde", "unit", "title", "figsize", "color", "show_grid"],
        "plot_evaluation_metrics": ["metrics", "target_name", "unit", "title", "figsize", "colors", "show_grid"],
        "plot_future_forecast": ["consumption", "production", "unit", "title", "figsize", "colors", "show_grid"],
        "plot_grid_exchange": ["grid_import", "grid_export", "unit", "title", "figsize", "colors", "show_grid"],
        "plot_cumulative_grid_exchange": ["cumulative_grid_import", "cumulative_grid_export", "unit", "title", "figsize", "colors", "show_grid"],
    }
    expected_defaults = {
        "plot_training_history": ("required", None, (8, 5), None, True),
        "plot_prediction_window": ("required", "required", "required", 0, None, "kWh", None, (10, 5), None, True),
        "plot_error_distribution": ("required", "required", 50, True, "kWh", None, (8, 5), None, True),
        "plot_evaluation_metrics": ("required", "required", "kWh", None, (6, 4), None, False),
        "plot_future_forecast": ("required", "required", "kWh", None, (14, 5), None, True),
        "plot_grid_exchange": ("required", "required", "kWh", None, (14, 5), None, True),
        "plot_cumulative_grid_exchange": ("required", "required", "kWh", None, (14, 5), None, True),
    }
    assert set(plotting.__all__) == PUBLIC_APIS
    assert {n for n, obj in vars(plotting).items() if inspect.isfunction(obj) and not n.startswith("_")} == PUBLIC_APIS
    for name, parameters in expected.items():
        signature = inspect.signature(getattr(plotting, name))
        assert list(signature.parameters) == parameters
        defaults = tuple(p.default if p.default is not inspect.Parameter.empty else "required" for p in signature.parameters.values())
        assert defaults == expected_defaults[name]


def test_dependencies_and_no_io_or_recalculation():
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Import):
            imports = {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports = {node.module.split(".")[0]}
        else:
            continue
        assert imports.isdisjoint(FORBIDDEN_IMPORT_PARTS)
        assert imports <= {"collections", "numbers", "typing", "matplotlib", "numpy", "pandas", "seaborn"}
    source = MODULE_PATH.read_text(encoding="utf-8")
    for snippet in (".show(", ".savefig(", "open(", "Path(", "to_csv(", "to_excel(", "cumsum(", "mean_squared_error", "mean_absolute_error", "sqrt("):
        assert snippet not in source


def test_training_history_output_configuration_and_validation():
    plotting = _load_plotting()
    original = {"loss": [3.0, 2.0], "val_loss": [4.0, 3.0]}
    fig, ax = plotting.plot_training_history(original, title="Training", colors=("red", "blue"), show_grid=False)
    try:
        assert isinstance(fig, plotting.Figure) and isinstance(ax, plotting.Axes)
        assert ax.get_title() == "Training" and ax.get_xlabel() == "Epoch" and ax.get_ylabel() == "Loss (MSE)"
        assert len(ax.lines) == 2 and ax.get_legend() is not None and not ax.xaxis._major_tick_kw["gridOn"]
        assert original == {"loss": [3.0, 2.0], "val_loss": [4.0, 3.0]}
    finally:
        _close(fig)
    bad = [({}, ValueError), ({"loss": [1]}, ValueError), ({"loss": [], "val_loss": []}, ValueError), ({"loss": [1], "val_loss": [1, 2]}, ValueError), ({"loss": ["x"], "val_loss": [1]}, ValueError), ({"loss": [float("inf")], "val_loss": [1]}, ValueError)]
    for data, error in bad:
        with pytest.raises(error):
            plotting.plot_training_history(data)
    with pytest.raises(TypeError):
        plotting.plot_training_history({"loss": [1], "val_loss": [1]}, show_grid=1)


def test_prediction_window_shapes_timestamps_and_validation():
    plotting = _load_plotting()
    np, pd = pytest.importorskip("numpy"), pytest.importorskip("pandas")
    actual = np.array([[[1.0], [2.0]], [[3.0], [4.0]]])
    pred = np.array([[[1.1], [1.9]], [[3.2], [3.8]]])
    actual_copy, pred_copy = actual.copy(), pred.copy()
    stamps = pd.date_range("2026-01-01", periods=2, freq="h")
    fig, ax = plotting.plot_prediction_window(actual, pred, target_name="consumption", window_index=1, timestamps=stamps, unit="MWh", show_grid=False)
    try:
        assert isinstance(fig, plotting.Figure) and isinstance(ax, plotting.Axes)
        assert len(ax.lines) == 2 and ax.get_ylabel() == "consumption (MWh)"
        assert ax.get_legend() is not None and list(ax.lines[0].get_xdata()) == list(stamps)
        assert np.array_equal(actual, actual_copy) and np.array_equal(pred, pred_copy)
    finally:
        _close(fig)
    # The two contracted representations are each valid when both arrays use the same shape.
    fig, ax = plotting.plot_prediction_window(actual[:, :, 0], pred[:, :, 0], target_name="consumption")
    try:
        assert len(ax.lines) == 2
    finally:
        _close(fig)
    with pytest.raises(ValueError):
        plotting.plot_prediction_window(actual[:, :, 0], pred, target_name="consumption")
    calls = [
        (([1, 2], pred), TypeError), ((np.array([[1, 2]]), pred), ValueError),
        ((actual, pred[:, :1]), ValueError),
        ((np.array([[np.nan]]), np.array([[1.0]])), ValueError),
    ]
    for (left, right), error in calls:
        with pytest.raises(error):
            plotting.plot_prediction_window(left, right, target_name="load")
    with pytest.raises(ValueError):
        plotting.plot_prediction_window(actual, pred, target_name="load", window_index=2)
    with pytest.raises(TypeError):
        plotting.plot_prediction_window(actual, pred, target_name="load", window_index=0.5)
    with pytest.raises(TypeError):
        plotting.plot_prediction_window(actual, pred, target_name="load", timestamps=["a", "b"])
    with pytest.raises(ValueError):
        plotting.plot_prediction_window(actual, pred, target_name="load", timestamps=pd.date_range("2026-01-01", periods=1))


def test_prediction_window_rejects_empty_arrays():
    plotting = _load_plotting()
    np = pytest.importorskip("numpy")
    empty_windows = np.empty((0, 2))
    valid_windows = np.ones((1, 2))
    with pytest.raises(ValueError):
        plotting.plot_prediction_window(empty_windows, valid_windows, target_name="load")
    with pytest.raises(ValueError):
        plotting.plot_prediction_window(valid_windows, empty_windows, target_name="load")


def test_error_distribution_histogram_kde_zero_and_validation():
    plotting = _load_plotting()
    np = pytest.importorskip("numpy")
    data = np.array([[1.0, -1.0], [0.5, -0.5]])
    saved = data.copy()
    fig, ax = plotting.plot_error_distribution(data, target_name="load", bins=3, kde=False, color="green")
    try:
        assert isinstance(fig, plotting.Figure) and isinstance(ax, plotting.Axes)
        assert len(ax.patches) == 3 and len(ax.lines) == 1
        assert ax.lines[0].get_xdata()[0] == 0 and ax.get_xlabel() == "Error (kWh)"
        assert [text.get_text() for text in ax.get_legend().get_texts()] == ["Zero error"]
        assert np.array_equal(data, saved)
    finally:
        _close(fig)
    kde_fig, kde_ax = plotting.plot_error_distribution(data.ravel(), target_name="load", bins=3, kde=True)
    try:
        assert len(kde_ax.lines) >= 2  # KDE curve and explicit zero reference.
        assert [text.get_text() for text in kde_ax.get_legend().get_texts()] == ["Zero error"]
    finally:
        _close(kde_fig)
    constant_fig, constant_ax = plotting.plot_error_distribution([2.0, 2.0, 2.0], target_name="load", bins=3, kde=True)
    try:
        assert len(constant_ax.patches) == 3
        assert len(constant_ax.lines) == 1
        assert constant_ax.lines[0].get_xdata()[0] == 0
        assert [text.get_text() for text in constant_ax.get_legend().get_texts()] == ["Zero error"]
    finally:
        _close(constant_fig)
    for errors, error in [([], ValueError), (["x"], TypeError), ([np.nan], ValueError), ([np.inf], ValueError)]:
        with pytest.raises(error):
            plotting.plot_error_distribution(errors, target_name="load")
    for nested_errors in ([[1.0, 2.0], [3.0, 4.0]], ((1.0, 2.0), (3.0, 4.0))):
        with pytest.raises(ValueError):
            plotting.plot_error_distribution(nested_errors, target_name="load", kde=False)
    for bins, error in [(0, ValueError), (-1, ValueError), (2.5, TypeError), (True, TypeError)]:
        with pytest.raises(error):
            plotting.plot_error_distribution([1, 2], target_name="load", bins=bins)
    with pytest.raises(TypeError):
        plotting.plot_error_distribution([1, 2], target_name="load", kde=1)


def test_evaluation_metrics_bars_and_validation():
    plotting = _load_plotting()
    fig, ax = plotting.plot_evaluation_metrics({"mae": 1.2, "rmse": 2.3}, target_name="load", unit="MWh", show_grid=True)
    try:
        assert isinstance(fig, plotting.Figure) and isinstance(ax, plotting.Axes)
        assert len(ax.patches) == 2 and ax.get_ylabel() == "MWh"
        assert [p.get_height() for p in ax.patches] == [1.2, 2.3]
    finally:
        _close(fig)
    np = pytest.importorskip("numpy")
    for metrics, error in [({}, ValueError), ({"mae": "1", "rmse": 2}, ValueError), ({"mae": np.nan, "rmse": 2}, ValueError), ({"mae": -1, "rmse": 2}, ValueError)]:
        with pytest.raises(error):
            plotting.plot_evaluation_metrics(metrics, target_name="load")


def test_evaluation_metrics_does_not_mutate_input_mapping():
    plotting = _load_plotting()
    metrics = {"mae": 1.2, "rmse": 2.3}
    metrics_before = metrics.copy()
    fig, ax = plotting.plot_evaluation_metrics(metrics, target_name="load")
    try:
        assert len(ax.patches) == 2
        assert metrics == metrics_before
    finally:
        _close(fig)


@pytest.mark.parametrize("function_name", ["plot_future_forecast", "plot_grid_exchange", "plot_cumulative_grid_exchange"])
def test_time_series_graphs_validate_alignment_contents_and_inputs(function_name):
    plotting = _load_plotting()
    idx, left, right = _time_series()
    func = getattr(plotting, function_name)
    left_copy, right_copy = left.copy(deep=True), right.copy(deep=True)
    fig, ax = func(left, right, title="Energy", unit="MWh", colors=("black", "gray"), show_grid=False)
    try:
        assert isinstance(fig, plotting.Figure) and isinstance(ax, plotting.Axes)
        assert len(ax.lines) == 2 and ax.get_title() == "Energy" and ax.get_ylabel() == "MWh"
        assert ax.get_legend() is not None
        assert left.equals(left_copy) and right.equals(right_copy)
    finally:
        _close(fig)
    pd = pytest.importorskip("pandas")
    with pytest.raises(ValueError):
        func(left.iloc[:0], right.iloc[:0])
    with pytest.raises(ValueError):
        func(left, right.iloc[:2])
    with pytest.raises(ValueError):
        func(left, pd.Series([1, 2, 3], index=pd.date_range("2026-02-01", periods=3, freq="h")))
    with pytest.raises(TypeError):
        func(pd.Series(["x", "y", "z"], index=idx), right)
    with pytest.raises(ValueError):
        func(pd.Series([1, float("nan"), 3], index=idx), right)
    with pytest.raises(ValueError):
        func(pd.Series([1, 2, 3], index=[1, 2, 3]), pd.Series([1, 2, 3], index=[1, 2, 3]))
    with pytest.raises(TypeError):
        func([1, 2, 3], right)
    unordered = pd.to_datetime(["2026-01-02", "2026-01-01"])
    duplicate = pd.to_datetime(["2026-01-01", "2026-01-01"])
    nat_index = pd.DatetimeIndex([pd.Timestamp("2026-01-01"), pd.NaT])
    for bad_index in (unordered, duplicate, nat_index):
        bad = pd.Series([1.0, 2.0], index=bad_index)
        with pytest.raises(ValueError):
            func(bad, bad)


@pytest.mark.parametrize("function_name", ["plot_training_history", "plot_prediction_window", "plot_error_distribution", "plot_evaluation_metrics", "plot_future_forecast", "plot_grid_exchange", "plot_cumulative_grid_exchange"])
def test_common_configuration_validation(function_name):
    plotting = _load_plotting()
    idx, a, b = _time_series()
    args = {
        "plot_training_history": (({"loss": [1], "val_loss": [1]},), {}),
        "plot_prediction_window": ((pytest.importorskip("numpy").array([[[1.0]]]), pytest.importorskip("numpy").array([[[1.0]]])), {"target_name": "load"}),
        "plot_error_distribution": (([1],), {"target_name": "load"}),
        "plot_evaluation_metrics": (({"mae": 1, "rmse": 1},), {"target_name": "load"}),
        "plot_future_forecast": ((a, b), {}),
        "plot_grid_exchange": ((a, b), {}),
        "plot_cumulative_grid_exchange": ((a, b), {}),
    }
    positional, kwargs = args[function_name]
    fn = getattr(plotting, function_name)
    for bad_size, error in [((0, 5), ValueError), ((1, float("inf")), ValueError), ((1,), TypeError), (("1", 2), TypeError)]:
        with pytest.raises(error):
            fn(*positional, **kwargs, figsize=bad_size)
    with pytest.raises(TypeError):
        fn(*positional, **kwargs, show_grid="yes")
    color_parameter = "color" if function_name == "plot_error_distribution" else "colors"
    invalid_color = "not-a-color" if color_parameter == "color" else ("not-a-color", "red")
    with pytest.raises(ValueError):
        fn(*positional, **kwargs, **{color_parameter: invalid_color})
    if color_parameter == "colors":
        with pytest.raises(ValueError):
            fn(*positional, **kwargs, colors=("red",))
    with pytest.raises(TypeError):
        fn(*positional, **kwargs, title=4)
