import pytest
from src.splitter import split_dataset
from src.data_loader import get_image_files


@pytest.fixture
def mock_data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    # Create dummy images
    for i in range(10):
        p = d / f"img_{i}.jpg"
        p.write_text("content")
    return d


def test_get_image_files(mock_data_dir):
    files = get_image_files(str(mock_data_dir))
    assert len(files) == 10


def test_split_dataset(mock_data_dir):
    files = get_image_files(str(mock_data_dir))
    train, test = split_dataset(files, 0.8)
    assert len(train) == 8
    assert len(test) == 2

    # Test disjoint
    assert set(train).isdisjoint(set(test))

    # Test union
    assert set(train).union(set(test)) == set(files)


def test_split_dataset_invalid_ratio(mock_data_dir):
    files = get_image_files(str(mock_data_dir))
    with pytest.raises(ValueError):
        split_dataset(files, 1.5)
