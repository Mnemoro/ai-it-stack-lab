import ast
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "visualization" / "plotting.py"
PUBLIC_APIS = {
    "plot_training_history",
    "plot_prediction_window",
    "plot_error_distribution",
    "plot_evaluation_metrics",
    "plot_future_forecast",
    "plot_grid_exchange",
    "plot_cumulative_grid_exchange",
}
FORBIDDEN_IMPORT_PARTS = {
    "tensorflow",
    "keras",
    "sklearn",
    "forecasting",
    "monitoring",
    "energy_balance",
}


def _tree():
    return ast.parse(MODULE_PATH.read_text())


def _load_plotting():
    pytest.importorskip("numpy")
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import sys

    src_path = str(MODULE_PATH.parents[1])
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from visualization import plotting

    return plotting


def test_public_apis_exist_and_only_expected_are_exported():
    plotting = _load_plotting()
    assert set(plotting.__all__) == PUBLIC_APIS
    for name in PUBLIC_APIS:
        assert callable(getattr(plotting, name))


def test_module_has_no_forbidden_dependencies():
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Import):
            imports = {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports = {node.module.split(".")[0]}
        else:
            continue
        assert imports.isdisjoint(FORBIDDEN_IMPORT_PARTS)


def test_no_plt_show_or_file_io_or_forbidden_recalculations():
    source = MODULE_PATH.read_text()
    forbidden_snippets = [
        ".show(",
        ".savefig(",
        "open(",
        "Path(",
        "to_csv(",
        "to_excel(",
        "read_csv(",
        "read_excel(",
        "mean_absolute_error",
        "mean_squared_error",
        "sqrt(",
        "cumsum(",
        "net_balance",
        "to_buy",
        "to_grid",
        "bilancio",
    ]
    for snippet in forbidden_snippets:
        assert snippet not in source


def test_no_subtraction_inside_plot_error_distribution():
    tree = _tree()
    func = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "plot_error_distribution")
    assert not any(isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub) for node in ast.walk(func))


def test_plot_training_history_validates_and_returns_figure_axes():
    plotting = _load_plotting()
    fig, ax = plotting.plot_training_history({"loss": [3, 2], "val_loss": [4, 3]}, colors=("red", "blue"))
    assert fig.__class__.__name__ == "Figure"
    assert ax.__class__.__name__ == "Axes"
    with pytest.raises(ValueError):
        plotting.plot_training_history({"loss": [1]})
    with pytest.raises(ValueError):
        plotting.plot_training_history({"loss": [], "val_loss": []})
    with pytest.raises(ValueError):
        plotting.plot_training_history({"loss": [1], "val_loss": [1, 2]})
    with pytest.raises(ValueError):
        plotting.plot_training_history({"loss": [float("nan")], "val_loss": [1]})
    with pytest.raises(ValueError):
        plotting.plot_training_history({"loss": [1], "val_loss": [1]}, colors=("red",))


def test_plot_prediction_window_validation_and_output():
    plotting = _load_plotting()
    np = pytest.importorskip("numpy")
    y_true = np.array([[[1.0], [2.0]], [[3.0], [4.0]]])
    y_pred = np.array([[[1.1], [1.9]], [[3.2], [3.8]]])
    fig, ax = plotting.plot_prediction_window(y_true, y_pred, target_name="consumption", window_index=1, timestamps=["t1", "t2"])
    assert fig.__class__.__name__ == "Figure"
    assert ax.__class__.__name__ == "Axes"
    with pytest.raises(IndexError):
        plotting.plot_prediction_window(y_true, y_pred, target_name="consumption", window_index=2)
    with pytest.raises(ValueError):
        plotting.plot_prediction_window(np.array([1, 2]), y_pred, target_name="consumption")
    with pytest.raises(ValueError):
        plotting.plot_prediction_window(y_true, y_pred[:, :1], target_name="consumption")
    with pytest.raises(ValueError):
        plotting.plot_prediction_window(y_true, y_pred, target_name="consumption", timestamps=["only-one"])
    with pytest.raises(ValueError):
        plotting.plot_prediction_window(np.array([[np.inf]]), np.array([[1.0]]), target_name="consumption")


def test_plot_error_distribution_validation_and_output():
    plotting = _load_plotting()
    fig, ax = plotting.plot_error_distribution([1.0, -1.0, 0.5], target_name="production", bins=3, kde=False, color="green")
    assert fig.__class__.__name__ == "Figure"
    assert ax.__class__.__name__ == "Axes"
    with pytest.raises(ValueError):
        plotting.plot_error_distribution([], target_name="production")
    with pytest.raises(ValueError):
        plotting.plot_error_distribution([float("inf")], target_name="production")


def test_plot_evaluation_metrics_validation_and_output():
    plotting = _load_plotting()
    fig, ax = plotting.plot_evaluation_metrics({"mae": 1.2, "rmse": 2.3}, target_name="consumption", colors=("cyan", "magenta"))
    assert fig.__class__.__name__ == "Figure"
    assert ax.__class__.__name__ == "Axes"
    with pytest.raises(ValueError):
        plotting.plot_evaluation_metrics({"mae": 1.0}, target_name="consumption")
    with pytest.raises(ValueError):
        plotting.plot_evaluation_metrics({"mae": -1.0, "rmse": 2.0}, target_name="consumption")
    with pytest.raises(ValueError):
        plotting.plot_evaluation_metrics({"mae": 1.0, "rmse": float("nan")}, target_name="consumption")


def test_time_series_plots_validate_indices_lengths_and_output():
    plotting = _load_plotting()
    pd = pytest.importorskip("pandas")
    idx = pd.date_range("2026-01-01", periods=3, freq="h")
    a = pd.Series([1.0, 2.0, 3.0], index=idx)
    b = pd.Series([0.5, 1.5, 2.5], index=idx)
    for func in (plotting.plot_future_forecast, plotting.plot_grid_exchange, plotting.plot_cumulative_grid_exchange):
        fig, ax = func(a, b, colors=("black", "gray"))
        assert fig.__class__.__name__ == "Figure"
        assert ax.__class__.__name__ == "Axes"
        with pytest.raises(ValueError):
            func(a.iloc[:0], b.iloc[:0])
        with pytest.raises(ValueError):
            func(a, b.iloc[:2])
        with pytest.raises(ValueError):
            func(a, pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2026-02-01", periods=3, freq="h")))
        with pytest.raises(ValueError):
            func(pd.Series([1.0, float("nan"), 3.0], index=idx), b)


def test_cumulative_grid_exchange_requires_ordered_unique_index():
    plotting = _load_plotting()
    pd = pytest.importorskip("pandas")
    unordered_idx = pd.to_datetime(["2026-01-02", "2026-01-01"])
    dup_idx = pd.to_datetime(["2026-01-01", "2026-01-01"])
    with pytest.raises(ValueError):
        plotting.plot_cumulative_grid_exchange(pd.Series([2.0, 1.0], index=unordered_idx), pd.Series([1.0, 2.0], index=unordered_idx))
    with pytest.raises(ValueError):
        plotting.plot_cumulative_grid_exchange(pd.Series([1.0, 2.0], index=dup_idx), pd.Series([1.0, 2.0], index=dup_idx))
