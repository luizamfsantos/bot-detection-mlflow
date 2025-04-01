# STEP DONE MANUALLY: download from drive: https://drive.google.com/drive/
# folders/0B3-6QHgPuBInTkNzLWREX3ZzUTg?resourcekey=0-VR5maZBUPhgEyjQ8GODSpw
# Source: https://astronomy.swin.edu.au/~vmorello/

# ################################################################################
import tarfile
import xml.etree.cElementTree as ET
import logging
import numpy as np
from pathlib import Path
import os
import pickle


def extract_tar_gz(file_path: str) -> list[str]:
    with tarfile.open(file_path, mode="r:gz") as tarobj:
        tarobj.extractall()
        file_list = tarobj.getnames()
    logging.info(f"Extracted {file_path} to current directory.")
    return file_list


def extract_coordinates(root: ET.Element) -> dict:
    """
    Extract coordinates from the XML root element.
    """
    try:
        coord_node = root.find('head').find('Coordinate')
        return {
            'rajd': float(coord_node.find('RA').text),
            'decjd': float(coord_node.find('Dec').text)
        }
    except AttributeError:
        logging.info("Coordinate node not found in the XML file.")
        return {'rajd': None, 'decjd': None}
    except ValueError:
        logging.info("Error converting coordinate values to float.")
        return {'rajd': None, 'decjd': None}


def read_data_block(xmlnode: ET.Element) -> np.ndarray:
    """
    Turn any 'DataBlock' XML node into a numpy array of floats.
    """
    try:
        value_min = float(xmlnode.get('min'))
        value_max = float(xmlnode.get('max'))
        string = xmlnode.text
        string = string.replace("\t", "").replace(" ", "").replace("\n", "")
        data = np.frombuffer(bytearray.fromhex(string), dtype=float)
        return data * (value_max - value_min) / 255.0 + value_min
    except ValueError:
        logging.info("Error converting DataBlock to numpy array.")
        return np.array([])
    except AttributeError:
        logging.info("DataBlock node not found in the XML file.")
        return np.array([])


def split_string_to_floats(string: str) -> np.ndarray:
    """
    Convert a space-separated string of floats into a numpy array.
    """
    try:
        values = np.array([float(x) for x in string.split()])
        return values
    except ValueError:
        logging.info("Error converting string to floats.")
        return np.array([])
    except AttributeError:
        logging.info("String is None or empty.")
        return np.array([])


def parse_pdmp_section(opt_section: ET.Element) -> dict:
    """
    Parse the PDMP section of the XML file and return relevant data.
    """
    data = {}
    # Best values as returned by PDMP
    opt_values = {
        node.tag: float(node.text)
        for node in opt_section.find('BestValues')
    }
    data['bary_period'] = opt_values.get('BaryPeriod')
    data['topo_period'] = opt_values.get('TopoPeriod')
    data['dm'] = opt_values.get('Dm')
    data['snr'] = opt_values.get('Snr')
    data['width'] = opt_values.get('Width')

    # Sub-Integrations
    subints_node = opt_section.find('SubIntegrations')
    data['subints'] = read_data_block(subints_node)

    # Sub-Bands
    subbands_node = opt_section.find('SubBands')
    data['subbands'] = read_data_block(subbands_node)

    # Profile
    profile_node = opt_section.find('Profile')
    data['profile'] = read_data_block(profile_node)

    # P-DM plane
    pdm_node = opt_section.find('SnrBlock')
    if pdm_node is None:
        logging.info("SnrBlock node not found in the XML file.")
        data['dm_index'] = np.array([])
        data['period_index'] = np.array([])
        data['pdm_plane'] = np.array([])
        return data

    # DmIndex
    try:
        dm_index_string = pdm_node.find('DmIndex').text
        data['dm_index'] = split_string_to_floats(dm_index_string)
    except AttributeError:
        logging.info("DmIndex node not found in the XML file.")
        data['dm_index'] = np.array([])

    # PeriodIndex
    try:
        period_index_string = pdm_node.find('PeriodIndex').text
        data['period_index'] = split_string_to_floats(period_index_string)
    except AttributeError:
        logging.info("PeriodIndex node not found in the XML file.")
        data['period_index'] = np.array([])

    # S/N data
        pdm_data = pdm_node.find('DataBlock')
        data['pdm_plane'] = read_data_block(pdm_data)

    return data


def parse_fft_section(fft_section: ET.Element) -> dict:
    """
    Parse the FFT section of the XML file and return relevant data.
    """
    data = {}
    # Parse FFT Section (PEASOUP Data)
    fft_values = {
        node.tag: float(node.text)
        for node in fft_section.find('BestValues')
    }
    data['accn'] = fft_values.get('Accn')
    data['hits'] = fft_values.get('Hits')
    data['rank'] = fft_values.get('Rank')
    data['fftsnr'] = fft_values.get('SpectralSnr')

    # DmCurve
    try:
        dmcurve_node = fft_section.find('DmCurve')
        dmcurve_string = dmcurve_node.find('DmValues').text
        data['dm_values'] = split_string_to_floats(dmcurve_string)
        snr_string = dmcurve_node.find('SnrValues').text
        data['dm_curve_snr_values'] = split_string_to_floats(snr_string)
    except AttributeError:
        logging.info("DmCurve node not found in the XML file.")
        data['dm_values'] = np.array([])
        data['dm_curve_snr_values'] = np.array([])

    # AccnCurve
    try:
        accncurve_node = fft_section.find('AccnCurve')
        accncurve_string = accncurve_node.find('AccnValues').text
        data['accn_values'] = split_string_to_floats(accncurve_string)
        snr_string = accncurve_node.find('SnrValues').text
        data['accn_curve_snr_values'] = split_string_to_floats(snr_string)
    except AttributeError:
        logging.info("AccnCurve node not found in the XML file.")
        data['accn_values'] = np.array([])
        data['accn_curve_snr_values'] = np.array([])

    return data


def parse_xml(file):
    """
    Parse a PHCX file and return the data as a dictionary.
    """
    # Get the root of the XML tree
    root = ET.parse(file).getroot()

    # Read Coordinates
    data = extract_coordinates(root)

    # Parse PDMP & FFT sections
    for section in root.findall('Section'):
        if 'pdmp' in section.get('name', '').lower():
            data.extend(parse_pdmp_section(section))
        elif 'fft' in section.get('name', '').lower():
            data.extend(parse_fft_section(section))

    return data


def main():
    folder_path = Path('data/raw')
    # loop over tar gz files
    for tar_file in folder_path.glob('**/*.tar.gz'):
        # extract files inside tar gz
        file_list = extract_tar_gz(tar_file)
        # initialize list of dicts
        data_list = []
        # loop over extracted files
        for xml_file in file_list:
            # parse xml files
            data = parse_xml(xml_file)
            # delete parsed xml file
            os.remove(xml_file)
            # append data to list
            data_list.append(data)
        # export dict
        with open(tar_file.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(data_list, f)
