"""Best-metric tracking keeps a new low and ignores a worse score."""
from host_finetune.best_metric_checkpoints import improved_metrics, macro_average, medium_name


def test_medium_names():
    assert medium_name("youtube_jason") == "talk"
    assert medium_name("youtube_saastr") == "talk"
    assert medium_name("blog") == "blog"
    assert medium_name("x") == "x"


def test_only_improvements_are_recorded():
    best: dict[str, float] = {}
    assert improved_metrics(best, {"eval_loss": 1.5, "eval_loss_blog": 1.2}) == [
        "eval_loss",
        "eval_loss_blog",
    ]
    assert improved_metrics(best, {"eval_loss": 1.4, "eval_loss_blog": 1.3}) == ["eval_loss"]
    assert best == {"eval_loss": 1.4, "eval_loss_blog": 1.2}


def test_macro_average_uses_the_four_mediums():
    score = macro_average({
        "eval_loss": 9.0,
        "eval_loss_blog": 1.0,
        "eval_loss_linkedin": 2.0,
        "eval_loss_x": 3.0,
        "eval_loss_talk": 4.0,
    })
    assert score == 2.5
