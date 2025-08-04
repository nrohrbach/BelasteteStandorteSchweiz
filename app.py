from streamlit_folium import folium_static
import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import xml.etree.ElementTree as ET
import requests
import zipfile
from io import BytesIO
import os
from typing import Optional # Keep Optional as it's used in function signatures
from xml.etree.ElementTree import Element # Keep Element as it's used in function signatures


# --- Data Fetching and Processing Functions ---
def get_zip(from_url: str, output_dir: str):
    """Downloads a zip file from a URL and extracts its contents."""
    try:
        response = requests.get(from_url, timeout=10) # Added timeout
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        zip_file = zipfile.ZipFile(BytesIO(response.content))
        zip_file.extractall(output_dir)
        # print(f"Extracted to: {output_dir}") # Avoid printing in Streamlit apps unless necessary for debugging
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading or extracting zip file from {from_url}: {e}")
        raise # Re-raise the exception to be caught by fetch_data
    except zipfile.BadZipFile as e:
        st.error(f"Error extracting bad zip file from {from_url}: {e}")
        raise # Re-raise the exception

def get_xtf(dir: str):
    """Finds the first .xtf file in a directory."""
    xtf_file = None
    for name in os.listdir(dir):
        if name.lower().endswith('.xtf'):
            xtf_file = os.path.join(dir, name)
            break
    return xtf_file

def get_ns(root: ET.Element):
    """Extracts the XML namespace from the root element."""
    return {'ili': root.tag.split('}')[0].strip('{')}

def get_elements(xtf_file: str):
    """Parses the XTF file and finds relevant elements."""
    try:
        tree = ET.parse(xtf_file)
        root = tree.getroot()
        ns = get_ns(root)
        tag_name = "KbS_V1_5.Belastete_Standorte.Belasteter_Standort"
        return root.findall(f".//ili:{tag_name}", ns)
    except ET.ParseError as e:
        st.error(f"Error parsing XTF file {xtf_file}: {e}")
        return [] # Return empty list if parsing fails

def extract_geometry_interlis(element: Element, ns: dict[str, str]) -> Optional[dict]:
    """Extracts geometry information from an Interlis XML element."""
    # Geo_Lage_Punkt (single point)
    punkt = element.find('.//ili:Geo_Lage_Punkt/ili:COORD', ns)
    if punkt is not None:
        try:
            x = float(punkt.findtext('ili:C1', default='0', namespaces=ns))
            y = float(punkt.findtext('ili:C2', default='0', namespaces=ns))
            return {
                "type": "Point",
                "coordinates": [x, y],
                "spatialReference": {"wkid": 2056}
            }
        except (ValueError, TypeError):
            return None # Handle cases where C1 or C2 are not valid numbers

    # Geo_Lage_Polygon (list of coords forming a ring)
    coords = []
    for coord in element.findall(".//ili:Geo_Lage_Polygon//ili:POLYLINE/ili:COORD", ns):
        x_text = coord.findtext('ili:C1', default=None, namespaces=ns)
        y_text = coord.findtext('ili:C2', default=None, namespaces=ns)
        try:
            if x_text and y_text:
                coords.append([float(x_text), float(y_text)])
        except (ValueError, TypeError):
            continue # Skip invalid coordinates

    if coords:
        # Berechnung des Schwerpunkts (Centroid) eines Polygons
        centroid_x = sum(p[0] for p in coords) / len(coords)
        centroid_y = sum(p[1] for p in coords) / len(coords)
        return {
            "type": "Point",
            "coordinates": [centroid_x, centroid_y],
            "spatialReference": {"wkid": 2056}
        }

    return None

def extract_text(parent: Element, tag: str, ns: dict[str, str]) -> str | None:
    """Extracts text content from a child element."""
    el = parent.find(f'.//ili:{tag}', ns)
    return el.text if el is not None else None

def extract_localised_urls(uri_element: Element, ns: dict[str, str]):
    """Extracts localised URLs from a MultilingualUri element."""
    urls = {}
    for loc_uri in uri_element.findall('.//ili:KbS_V1_5.Belastete_Standorte.LocalisedUri', ns):
        lang = extract_text(loc_uri, 'Language', ns)
        text = extract_text(loc_uri, 'Text', ns)
        if lang and text:
            urls[lang] = text
    return urls


def get_data(elements: list[Element], ns: dict[str, str], quelle: str):
    """Extracts data from a list of XML elements into a DataFrame."""
    data = []

    for elem in elements:
        eintrag = {
            'TID': elem.attrib.get('TID'),
            'quelle': quelle,
            'Katasternummer': extract_text(elem, 'Katasternummer', ns),
            'Standorttyp': extract_text(elem, 'Standorttyp', ns),
            'InBetrieb': extract_text(elem, 'InBetrieb', ns),
            'Nachsorge': extract_text(elem, 'Nachsorge', ns),
            'StatusAltlV': extract_text(elem, 'StatusAltlV', ns),
            'Ersteintrag': extract_text(elem, 'Ersteintrag', ns),
            'LetzteAnpassung': extract_text(elem, 'LetzteAnpassung', ns),
            'ZustaendigkeitKataster_REF': elem.find('ili:ZustaendigkeitKataster', ns).attrib.get('REF') if elem.find('ili:ZustaendigkeitKataster', ns) is not None else None,
        }
        eintrag['geom'] = extract_geometry_interlis(elem, ns)

        # URLs
        url_standort_elem = elem.find('ili:URL_Standort/ili:KbS_V1_5.Belastete_Standorte.MultilingualUri', ns)
        url_kbs_elem = elem.find('ili:URL_KbS_Auszug/ili:KbS_V1_5.Belastete_Standorte.MultilingualUri', ns)
        eintrag['URL_Standort'] = extract_localised_urls(url_standort_elem, ns) if url_standort_elem is not None else {}
        eintrag['URL_KbS_Auszug'] = extract_localised_urls(url_kbs_elem, ns) if url_kbs_elem is not None else {}

        # Untersuchungsmassnahmen
        massnahme = elem.find('.//ili:KbS_V1_5.UntersMassn_', ns)
        eintrag['Untersuchungsmassnahme'] = extract_text(massnahme, 'value', ns) if massnahme is not None else None


        data.append(eintrag)
    return pd.DataFrame(data)


def get_shp_pt(path: str) -> pd.DataFrame:
    """Reads a shapefile (or GeoJSON) and extracts point data into a DataFrame."""
    try:
        gdf = gpd.read_file(path)

        # Ensure the GeoDataFrame has a CRS, set to 2056 if missing
        if gdf.crs is None:
             gdf.set_crs(epsg=2056, inplace=True, allow_override=True)
        else:
             gdf = gdf.to_crs(epsg=2056) # Reproject if necessary


        gdf['geom'] = gdf['geometry'].apply(lambda geom: {
            "type": "Point",
            "coordinates": [geom.x, geom.y],
            "spatialReference": {"wkid": 2056}
        } if geom and geom.geom_type == 'Point' else None) # Check if geom is not None


        gdf.rename(columns={
            'katasternu': 'Katasternummer',
            'standortty': 'Standorttyp',
            'url_kbs_au': 'URL_de', # Assuming this is the German URL column from WFS
            'kanton': 'quelle' # Assuming 'kanton' column exists and is the source
        }, inplace=True)

        # Keep relevant columns and the 'geom' column
        # Identify columns to keep: all original columns except geometry, plus the new 'geom'
        cols_to_keep = [col for col in gdf.columns if col != 'geometry']


        # Convert GeoDataFrame to DataFrame, keeping the 'geom' column
        df = pd.DataFrame(gdf[cols_to_keep])

        return df
    except Exception as e:
        st.error(f"Error reading or processing shapefile/GeoJSON from {path}: {e}")
        return pd.DataFrame() # Return empty DataFrame on error


def fetch_data(selected_xtf_sources, selected_wfs_source, wfs_url):
    """Fetches data from selected sources and combines them into a single DataFrame."""
    all_dataframes = []

    for source in selected_xtf_sources:
        out_dir = source["out_dir"]
        url = source["url"]
        st.info(f"Fetching data from XTF source: {out_dir.capitalize()}")
        try:
            # Clean up previous extraction directory if it exists
            if os.path.exists(out_dir):
                 import shutil
                 shutil.rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=True)

            get_zip(url, out_dir)
            xtf_file = get_xtf(out_dir)
            if xtf_file:
                tree = ET.parse(xtf_file)
                root = tree.getroot()
                ns = get_ns(root)
                elements = get_elements(xtf_file)
                df_source = get_data(elements, ns, quelle=out_dir)
                all_dataframes.append(df_source)
            else:
                st.warning(f"No .xtf file found for source: {out_dir}")
        except Exception as e:
            st.error(f"Error processing data from {out_dir}: {e}")


    if selected_wfs_source:
        st.info(f"Fetching data from WFS source: Kanton Basel-Landschaft")
        try:
            # Fetch WFS data - assuming get_shp_pt can handle the URL directly or needs a local file path
            # For this example, fetch GeoJSON and save to a temp file
            response = requests.get(wfs_url, timeout=30) # Added timeout
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

            temp_geojson_path = "temp_wfs.geojson"
            with open(temp_geojson_path, "wb") as f:
                f.write(response.content)

            df_wfs = get_shp_pt(temp_geojson_path)
            df_wfs['quelle'] = 'wfs' # Add source identifier
            all_dataframes.append(df_wfs)
            os.remove(temp_geojson_path) # Clean up temp file
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching data from WFS: {e}")
        except Exception as e:
             st.error(f"Error processing data from WFS: {e}")


    if all_dataframes:
        # Harmonize columns before concatenating
        # Find all unique columns across all dataframes
        all_columns = list(set(col for df in all_dataframes for col in df.columns))

        # Add missing columns with None to each dataframe
        for i in range(len(all_dataframes)):
            for col in all_columns:
                if col not in all_dataframes[i].columns:
                    all_dataframes[i][col] = None

        combined_df = pd.concat(all_dataframes, ignore_index=True)

        # Ensure 'geom' column is the last column
        if 'geom' in combined_df.columns:
            geom_column = combined_df.pop('geom')
            combined_df['geom'] = geom_column

        return combined_df
    else:
        return pd.DataFrame() # Return empty DataFrame if no data was fetched


# --- Streamlit App Layout and Logic ---
st.set_page_config(page_title="Belastete Standorte Data Aggregator", layout="wide") # Set layout to wide
st.title("Aggregation of Contaminated Sites Data")

# Define data sources
xtf_urls = [
    { "out_dir": 'zivil', "url": 'https://data.geo.admin.ch/ch.bazl.kataster-belasteter-standorte-zivilflugplaetze/data.zip' },
    { "out_dir": 'mil', "url": 'https://data.geo.admin.ch/ch.vbs.kataster-belasteter-standorte-militaer/data.zip' },
    { "out_dir": 'oev', "url": 'https://data.geo.admin.ch/ch.bav.kataster-belasteter-standorte-oev/data.zip' }
]
wfs_url = "https://geowfs.bl.ch/wfs/kbs?service=WFS&version=1.1.0&request=GetFeature&typename=kbs:belastete_standorte&outputFormat=application%2Fjson"

st.sidebar.header("Select Data Sources")

# Initialize session state variables if they don't exist
if 'selected_xtf_sources' not in st.session_state:
    st.session_state['selected_xtf_sources'] = xtf_urls # Default to selecting all XTF
if 'selected_wfs_source' not in st.session_state:
    st.session_state['selected_wfs_source'] = True # Default to selecting WFS
if 'combined_df' not in st.session_state:
    st.session_state['combined_df'] = pd.DataFrame()

# Create checkboxes for XTF sources
selected_xtf_sources = []
for source in xtf_urls:
    checkbox_state = st.sidebar.checkbox(f"XTF: {source['out_dir'].capitalize()}", value=source in st.session_state['selected_xtf_sources'], key=f"xtf_{source['out_dir']}")
    if checkbox_state:
        selected_xtf_sources.append(source)
st.session_state['selected_xtf_sources'] = selected_xtf_sources

# Create checkbox for WFS source
selected_wfs_source = st.sidebar.checkbox("WFS: Kanton Basel-Landschaft", value=st.session_state['selected_wfs_source'], key="wfs_bl")
st.session_state['selected_wfs_source'] = selected_wfs_source

# Create Fetch Data button
fetch_button = st.sidebar.button("Fetch Data")

if fetch_button:
    st.session_state['combined_df'] = pd.DataFrame() # Clear previous data on fetch
    with st.spinner("Fetching data..."):
        st.session_state['combined_df'] = fetch_data(
            st.session_state['selected_xtf_sources'],
            st.session_state['selected_wfs_source'],
            wfs_url
            )
    if not st.session_state['combined_df'].empty:
        st.success("Data fetching complete!")
    else:
        st.warning("No data was fetched based on the selections.")

# Display KPIs, Data Table, and Map if data is available
if not st.session_state['combined_df'].empty:
    # Display KPIs
    st.header("Key Performance Indicators")
    total_objects = st.session_state['combined_df'].shape[0]
    st.metric(label="Total Objects", value=total_objects)

    st.subheader("Objects per Source")
    source_counts = st.session_state['combined_df']['quelle'].value_counts()
    st.write(source_counts)


    # Display the combined data table
    st.header("Combined Data Table")
    st.dataframe(st.session_state['combined_df'])

    # Integrate the interactive map visualization
    st.header("Interactive Map")
    # Create a GeoDataFrame from the combined_df
    # Convert the dictionary representation of geometry to Point objects, handling potential None or invalid entries
    geometry = st.session_state['combined_df']['geom'].apply(lambda x: Point(x['coordinates']) if isinstance(x, dict) and 'coordinates' in x and x['type'] == 'Point' else None)

    # Create the GeoDataFrame, dropping rows where geometry could not be created
    gdf_combined = gpd.GeoDataFrame(st.session_state['combined_df'], geometry=geometry).dropna(subset=['geometry'])

    # Set the coordinate reference system (CRS) to match the data (EPSG:2056)
    gdf_combined.set_crs(epsg=2056, inplace=True)

    # Display the GeoDataFrame interactively
    # Use a placeholder to render the map
    map_placeholder = st.empty()
    with map_placeholder:
         st.write(gdf_combined.explore())


    # Add a download button for the combined data
    st.download_button(
        label="Download Data as CSV",
        data=st.session_state['combined_df'].to_csv(index=False).encode('utf-8'), # Encode to utf-8
        file_name="combined_data.csv",
        mime="text/csv"
    )

