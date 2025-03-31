# STEP DONE MANUALLY: download from drive: https://drive.google.com/drive/folders/0B3-6QHgPuBInTkNzLWREX3ZzUTg?resourcekey=0-VR5maZBUPhgEyjQ8GODSpw
# Source: https://astronomy.swin.edu.au/~vmorello/

# ################################################################################
import tarfile
import xml.etree.cElementTree as ET
import logging
import numpy as np


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


def read_data_block(xmlnode: ET.Element) -> np.ndarray:
    """
    Turn any 'DataBlock' XML node into a numpy array of floats.
    """
    value_min = float(xmlnode.get('min'))
    value_max = float(xmlnode.get('max'))
    string = xmlnode.text
    string = string.replace("\t", "").replace(" ", "").replace("\n", "")
    data = np.frombuffer(bytearray.fromhex(string), dtype=float)
    return data * (value_max - value_min) / 255.0 + value_min


def parse_pdmp_section(section: ET.Element) -> dict:
    """
    Parse the PDMP section of the XML file and return relevant data.
    """
    data = {}
    # Best values as returned by PDMP
    opt_values = {
        node.tag: float(node.text)
        for node in section.find('BestValues')
    }
    data['bary_period'] = opt_values['BaryPeriod']
    data['topo_period'] = opt_values['TopoPeriod']
    data['dm'] = opt_values['Dm']
    data['snr'] = opt_values['Snr']
    data['width'] = opt_values['Width']

    # P-DM plane
    pdm_node = section.find('SnrBlock')

    # DmIndex
    dm_index_string = pdm_node.find('DmIndex').text
    dm_index = np.array([float(x) for x in dm_index_string.split()])
    data['dm_index'] = dm_index

    # PeriodIndex
    period_index_string = pdm_node.find('PeriodIndex').text
    period_index = np.array([float(x) for x in period_index_string.split()]) / 1.0e12  # Picoseconds to seconds
    data['period_index'] = period_index

    # S/N data
    pdm_data = pdm_node.find('DataBlock')
    pdm_plane = read_data_block(pdm_data)
    data['pdm_plane'] = pdm_plane

    return data


def parse_fft_section(section: ET.Element) -> dict:
    """
    Parse the FFT section of the XML file and return relevant data.
    """
    data = {}
    # Best values as returned by FFT
    fft_values = {
        node.tag: float(node.text)
        for node in section.find('BestValues')
    }
    data['accn'] = fft_values['Accn']
    data['hits'] = fft_values['Hits']
    data['rank'] = fft_values['Rank']
    data['fftsnr'] = fft_values['SpectralSnr']

    # DmCurve
    dmcurve_node = section.find('DmCurve')
    dmcurve_string = dmcurve_node.find('DmValues').text
    dm_values = np.array([float(x) for x in dmcurve_string.split()])
    data['dm_values'] = dm_values
    snr_string = dmcurve_node.find('SnrValues').text
    snr_values = np.array([float(x) for x in snr_string.split()])
    data['snr_values'] = snr_values

    # AccnCurve
    accncurve_node = section.find('AccnCurve')
    accncurve_string = accncurve_node.find('AccnValues').text
    accn_values = np.array([float(x) for x in accncurve_string.split()])
    data['accn_values'] = accn_values
    snr_string = accncurve_node.find('SnrValues').text
    snr_values = np.array([float(x) for x in snr_string.split()])
    data['snr_values'] = snr_values

    return data


def parse_raw(file):
    """
    Parse a PHCX file and return the data as a dictionary.
    """
    # Get the root of the XML tree
    root = ET.parse(file).getroot()

    # Read Coordinates
    data = extract_coordinates(root)

    # Separate PDMP & FFT sections
    for section in root.findall('Section'):
        if 'pdmp' in section.get('name').lower():
            opt_section = section
        else:
            fft_section = section

    # Best values as returned by PDMP
    opt_values = {
        node.tag: float(node.text)
        for node in opt_section.find('BestValues')
    }
    data['bary_period'] = opt_values['BaryPeriod']
    data['topo_period'] = opt_values['TopoPeriod']
    data['dm'] = opt_values['Dm']
    data['snr'] = opt_values['Snr']
    data['width'] = opt_values['Width']

    # P-DM plane
    pdm_node = opt_section.find('SnrBlock')

    # DmIndex
    dm_index_string = pdm_node.find('DmIndex').text
    dm_index = [float(x) for x in dm_index_string.split()]
    dm_index = np.array(dm_index)
    data['dm_index'] = dm_index

    # PeriodIndex
    period_index_string = pdm_node.find('PeriodIndex').text
    period_index = [float(x) for x in period_index_string.split()]
    period_index = np.array(period_index) / 1.0e12  # Picoseconds to seconds
    data['period_index'] = period_index

    # S/N data
    pdm_data = pdm_node.find('DataBlock')
    pdm_plane = read_data_block(pdm_data)
    data['pdm_plane'] = pdm_plane

    # Sub-Integrations
    subints_node = opt_section.find('SubIntegrations')
    data['subints'] = read_data_block(subints_node)

    # Sub-Bands
    subbands_node = opt_section.find('SubBands')
    data['subbands'] = read_data_block(subbands_node)

    # Profile 
    profile_node = opt_section.find('Profile')
    data['profile'] = read_data_block(profile_node)

    # Parse FFT Section (PEASOUP Data)
    fft_values = {
        node.tag: float(node.text)
        for node in fft_section.find('BestValues')
    }
    data['accn'] = fft_values['Accn']
    data['hits'] = fft_values['Hits']
    data['rank'] = fft_values['Rank']
    data['fftsnr'] = fft_values['SpectralSnr']

    # DmCurve
    dmcurve_node = fft_section.find('DmCurve')
    dmcurve_string = dmcurve_node.find('DmValues').text
    dm_values = [float(x) for x in dmcurve_string.split()]
    data['dm_values'] = np.array(dm_values)
    snr_string = dmcurve_node.find('SnrValues').text
    snr_values = [float(x) for x in snr_string.split()]
    data['snr_values'] = np.array(snr_values)

    # AccnCurve
    accncurve_node = fft_section.find('AccnCurve')
    accncurve_string = accncurve_node.find('AccnValues').text
    accn_values = [float(x) for x in accncurve_string.split()]
    data['accn_values'] = np.array(accn_values)
    snr_string = accncurve_node.find('SnrValues').text
    snr_values = [float(x) for x in snr_string.split()]
    data['snr_values'] = np.array(snr_values)
    return data