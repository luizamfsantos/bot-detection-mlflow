import pytest
import tarfile
import xml.etree.ElementTree as ET
import numpy as np
from src.read_raw import (
    extract_tar_gz,
    extract_coordinates,
    read_data_block,
    split_string_to_floats,
    parse_xml,
)


@pytest.fixture
def sample_tar_gz(tmp_path):
    tar_path = tmp_path / "sample.tar.gz"
    xml_context = """<?xml version="1.0"?>
    <root>
        <head>
            <Coordinate>
                <RA>123.456</RA>
                <Dec>78.90</Dec>
            </Coordinate>
        </head>
        <DataBlock min="0" max="255">00 01 02 03 04 05 06 07</DataBlock>
        <DataBlock min="0" max="255">08 09 0A 0B 0C 0D 0E 0F</DataBlock>
    </root>"""
    xml_file = tmp_path / "sample.xml"
    xml_file.write_text(xml_context)
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(xml_file, arcname="sample.xml")
    return tar_path


@pytest.fixture
def sample_xml():
    xml_string = """<?xml version="1.0"?>
    <root>
        <head>
            <Coordinate>
                <RA>123.456</RA>
                <Dec>78.90</Dec>
            </Coordinate>
        </head>
        <DataBlock min="-0.000211" max="0.000546">61147D55095B52502B3D635
        76345DF72BE938198776E9EB8BABE6EC4FFD5BCB7A7F1AFFDD3C1B0B5
    B6AB4071749499956A688E6955345F5127211F00553E9B53</DataBlock>
        <DataBlock min="0" max="255">61147D55095B5
        2502B3D63576345DF72BE938198776E9EB8BABE6EC4FFD5BCB7A7F1AFFDD3C1B0B5
    B6AB4071749499956A688E6955345F5127211F00553E9B53</DataBlock>
    </root>"""
    return ET.ElementTree(ET.fromstring(xml_string)).getroot()


def test_extract_tar_gz(sample_tar_gz, tmp_path):
    file_list = extract_tar_gz(sample_tar_gz)
    assert len(file_list) == 1
    assert file_list[0] == "sample.xml"
    extracted_file = tmp_path / file_list[0]
    assert extracted_file.exists()


def test_extract_coordinates(sample_xml):
    coords = extract_coordinates(sample_xml)
    assert coords['rajd'] == 123.456
    assert coords['decjd'] == 78.90


def test_read_data_block(sample_xml):
    data_blocks = sample_xml.findall('DataBlock')
    data_arrays = [read_data_block(block) for block in data_blocks]
    assert len(data_arrays) == 2
    assert isinstance(data_arrays[0], np.ndarray)


def test_split_string_to_floats():
    result = split_string_to_floats("1.0 2.0 3.0")
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert np.all(result == np.array([1.0, 2.0, 3.0]))


def test_parse_xml(sample_xml, tmp_path):
    xml_path = tmp_path / "parsed.xml"
    ET.ElementTree(sample_xml).write(xml_path)
    parsed_root = parse_xml(xml_path)
    assert parsed_root is not None
    assert 'rajd' in parsed_root
    assert parsed_root['rajd'] == 123.456
