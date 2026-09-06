from src.utils.config import Config, load_config


def test_load_default_config():
    cfg = load_config()
    assert cfg.project.seed == 42
    assert cfg.data.dataset == "cifar10"
    assert cfg.model.architecture == "resnet18"


def test_attribute_and_dict_access_agree():
    cfg = load_config()
    assert cfg.training.lr == cfg["training"]["lr"]


def test_overrides_apply():
    cfg = load_config(overrides={"training.epochs": 3, "project.seed": 7})
    assert cfg.training.epochs == 3
    assert cfg.project.seed == 7
    # unrelated fields remain untouched
    assert cfg.model.architecture == "resnet18"


def test_config_is_mutable_copy_not_shared_reference():
    cfg1 = load_config()
    cfg1.training.epochs = 999
    cfg2 = load_config()
    assert cfg2.training.epochs != 999
