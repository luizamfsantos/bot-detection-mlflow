import pytest
import tarfile
import xml.etree.ElementTree as ET
import numpy as np
import os
import pickle
from pathlib import Path
import sys
from src.read_raw import (
    extract_tar_gz,
    extract_coordinates,
    read_data_block,
    split_string_to_floats,
    parse_pdmp_section,
    parse_fft_section,
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
        <DataBlock min="0" max="255">00 01 02 03 04 05 06 07</DataBlock>
        <DataBlock min="0" max="255">08 09 0A 0B 0C 0D 0E 0F</DataBlock>
    </root>"""
    return ET.ElementTree(ET.fromstring(xml_string)).getroot()


def test_extract_tar_gz(sample_tar_gz, tmp_path):
    file_list = extract_tar_gz(sample_tar_gz)
    assert len(file_list) == 1
    assert file_list[0] == "sample.xml"
    extracted_file = tmp_path / file_list[0]
    assert extracted_file.exists()

