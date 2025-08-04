import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import folium
import requests
import zipfile
import io
import os
import xml.etree.ElementTree as ET
import shutil
from owslib.wfs import WebFeatureService
from streamlit_folium import folium_static

# Define the URLs for the XTF files and the WFS service
xtf_urls = [
    { "out_dir": 'zivil', "url": 'https://data.geo.admin.ch/ch.bazl.kataster-belasteter-standorte-zivilflugplaetze/data.zip' },
    { "out_dir": 'mil', "url": 'https://data.geo.admin.ch/ch.vbs.kataster-belasteter-standorte-militaer/data.zip' },
    { "out_dir": 'oev', "url": 'https://data.geo.admin.ch/ch.bav.kataster-belasteter-standorte-oev/data.zip' }
]
wfs_url = "https://geodienste.ch/db/kataster_belasteter_standorte_v1_5_0/deu?"

# Helper functions (assuming these are defined elsewhere or will be included)
# You will need to include the definitions of get_zip, get_xtf, get_ns, get_elements, get_data, and create_folium_map here

# Placeholder for helper functions - replace with actual code
def get_zip(url, out_dir):
    """Downloads and extracts a zip file from a URL."""
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    try:
        response = requests.get(url)
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(out_dir)
    except Exception as e:
        st.error(f"Error downloading or extracting zip file from {url}: {e}")
        shutil.rmtree(out_dir) # Clean up if extraction fails

def get_xtf(out_dir):
    """Finds the first .xtf file in a directory."""
    for file in os.listdir(out_dir):
        if file.endswith(".xtf"):
            return os.path.join(out_dir, file)
    return None

def get_ns(root):
    """Extracts namespace from XML root."""
    ns = dict([node for _, node in ET.iterparse(io.StringIO(ET.tostring(root, encoding='unicode')), events=['start-ns'])])
    # Add any missing but expected namespaces if necessary
    if 'belasteteStandorte' not in ns:
         ns['belasteteStandorte'] = 'http://www.interlis.ch/xtf/belasteteStandorte' # Example
    if 'ili' not in ns:
        ns['ili'] = 'http://www.interlis.ch/ILI1/01' # Example
    return ns


def get_elements(xtf_file):
    """Parses XML and returns a list of elements."""
    try:
        tree = ET.parse(xtf_file)
        return tree.getroot()
    except Exception as e:
        st.error(f"Error parsing XTF file {xtf_file}: {e}")
        return None

def get_data(elements, ns, quelle):
    """Extracts data from XML elements into a DataFrame."""
    data = []
    for element in elements.findall('.//belasteteStandorte:BelasteterStandort', ns):
        geom_element = element.find('.//belasteteStandorte:geometrie', ns)
        geom_coords = None
        if geom_element is not None:
            point_element = geom_element.find('.//ili:Coord', {'ili': 'http://www.interlis.ch/ILI1/01'})
            if point_element is not None:
                c1_element = point_element.find('ili:C1', {'ili': 'http://www.interlis.ch/ILI1/01'})
                c2_element = point_element.find('ili:C2', {'ili': 'http://www.interlis.ch/ILI1/01'})
                if c1_element is not None and c2_element is not None and c1_element.text and c2_element.text:
                    try:
                        geom_coords = {
                            "type": "Point",
                            "coordinates": [float(c1_element.text), float(c2_element.text)],
                            "spatialReference": {"wkid": 2056} # Assuming EPSG:2056
                        }
                    except ValueError:
                        geom_coords = None # Handle cases where C1 or C2 are not valid numbers
        
        data.append({
            'TID': element.get('{http://www.interlis.ch/ILI1/01}TID'),
            'Katasternummer': element.findtext('.//belasteteStandorte:katasternummer', default='N/A', namespaces=ns),
            'Standorttyp': element.findtext('.//belasteteStandorte:standorttyp', default='N/A', namespaces=ns),
            'InBetrieb': element.findtext('.//belasteteStandorte:inbetrieb', default='N/A', namespaces=ns),
            'Nachsorge': element.findtext('.//belasteteStandorte:nachsorge', default='N/A', namespaces=ns),
            'StatusAltlV': element.findtext('.//belasteteStandorte:statusaltlv', default='N/A', namespaces=ns),
            'Ersteintrag': element.findtext('.//belasteteStandorte:ersteintrag', default='N/A', namespaces=ns),
            'LetzteAnpassung': element.findtext('.//belasteteStandorte:letzteanpassung', default='N/A', namespaces=ns),
            'URL_Standort_de': element.findtext('.//belasteteStandorte:url_standort_de', default='N/A', namespaces=ns),
            'URL_KbS_Auszug_de': element.findtext('.//belasteteStandorte:url_kbs_auszug_de', default='N/A', namespaces=ns),
            'geom': geom_coords,
            'quelle': quelle
        })
    return pd.DataFrame(data)

def create_folium_map(gdf):
    """
    Transforms a GeoDataFrame to WGS84 and creates a Folium map.

    Args:
        gdf (geopandas.GeoDataFrame): The input GeoDataFrame.

    Returns:
        folium.Map: The generated Folium map.
    """
    # Transform to WGS84 (EPSG:4326) if not already in that CRS
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf_wgs84 = gdf.to_crs(epsg=4326)
    else:
        gdf_wgs84 = gdf.copy()

    # Create a Folium map centered at a reasonable location (e.g., Switzerland)
    # Use the centroid of the data for centering if available, otherwise a default location
    if not gdf_wgs84.empty and not gdf_wgs84.geometry.unary_union.is_empty:
        try:
            # Check if the unary_union is a valid geometry before accessing centroid
            if gdf_wgs84.geometry.unary_union.is_valid:
                center_lat, center_lon = gdf_wgs84.geometry.unary_union.centroid.y, gdf_wgs84.geometry.unary_union.centroid.x
                m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
            else:
                 m = folium.Map(location=[46.8182, 8.2242], zoom_start=8) # Default to Switzerland if centroid is invalid
        except Exception as e: # Catch potential errors if centroid calculation fails for complex geometries
             st.warning(f"Could not determine centroid for centering the map: {e}")
             m = folium.Map(location=[46.8182, 8.2242], zoom_start=8) # Default to Switzerland
    else:
        m = folium.Map(location=[46.8182, 8.2242], zoom_start=8) # Default to Switzerland


    # Add points from the GeoDataFrame to the map
    for idx, row in gdf_wgs84.iterrows():
        if row['geometry'] is not None and row['geometry'].geom_type == 'Point':
            folium.Marker(
                location=[row['geometry'].y, row['geometry'].x],
                popup=f"Katasternummer: {row.get('Katasternummer', 'N/A')}<br>Standorttyp: {row.get('Standorttyp', 'N/A')}"
            ).add_to(m)

    return m


st.title("Aggregator für Daten zu belasteten Standorten")

st.sidebar.header("Datenquellen auswählen")

# Checkboxes for source selection
selected_sources = []
if st.sidebar.checkbox("XTF: zivil"):
    selected_sources.append({"url": 'https://data.geo.admin.ch/ch.bazl.kataster-belasteter-standorte-zivilflugplaetze/data.zip', "out_dir": 'zivil'})
if st.sidebar.checkbox("XTF: mil"):
    selected_sources.append({"url": 'https://data.geo.admin.ch/ch.vbs.kataster-belasteter-standorte-militaer/data.zip', "out_dir": 'mil'})
if st.sidebar.checkbox("XTF: oev"):
    selected_sources.append({"url": 'https://data.geo.admin.ch/ch.bav.kataster-belasteter-standorte-oev/data.zip', "out_dir": 'oev'})
if st.sidebar.checkbox("WFS: kataster_belasteter_standorte_v1_5_0"):
     selected_sources.append({"url": wfs_url, "out_dir": "wfs"})


if st.sidebar.button("Daten abfragen"):
    if not selected_sources:
        st.warning("Bitte wählen Sie mindestens eine Datenquelle aus.")
    else:
        all_data = []
        for source in selected_sources:
            if source["out_dir"] == "wfs":
                st.info(f"Lade Daten von WFS: {source['url']}")
                try:
                    wfs = WebFeatureService(url=source["url"], version='2.0.0')
                    feature_type_name = list(wfs.contents)[0]
                    response = wfs.getfeature(typename=feature_type_name, outputFormat='GeoJSON')
                    geojson_data = response.read()
                    gdf_wfs = gpd.read_file(io.BytesIO(geojson_data))

                    # Rename columns to match expected schema
                    gdf_wfs.rename(columns={
                        't_id': 'TID',
                        'katasternummer': 'Katasternummer',
                        'standorttyp': 'Standorttyp',
                        'inbetrieb': 'InBetrieb',
                        'nachsorge': 'Nachsorge',
                        'statusaltlv': 'StatusAltlV',
                        'ersteintrag': 'Ersteintrag',
                        'letzteanpassung': 'LetzteAnpassung',
                        'url_standort': 'URL_Standort_de',
                        'url_kbs_auszug': 'URL_KbS_Auszug_de'
                    }, inplace=True)
                    gdf_wfs['quelle'] = 'wfs'

                    # Extract geometry as centroid points and format for combined_df if not already Point
                    geometry = gdf_wfs['geometry'].apply(lambda geom: geom.centroid if geom and geom.geom_type == 'Polygon' else geom if geom and geom.geom_type == 'Point' else None)
                    gdf_wfs['geom'] = geometry.apply(lambda geom: {
                        "type": geom.geom_type,
                        "coordinates": list(geom.coords)[0],
                        "spatialReference": {"wkid": 4326} # WFS is in WGS84 (EPSG:4326)
                    } if geom else None)

                    # Drop original geometry column
                    gdf_wfs = gdf_wfs.drop(columns=['geometry', 'untersuchungsmassnahmen', 'deponietyp', 'zustaendigkeitkataster_ref'], errors='ignore')

                    all_data.append(gdf_wfs)

                except Exception as e:
                    st.error(f"Fehler beim Laden der WFS-Daten: {e}")

            else:
                st.info(f"Lade Daten von XTF: {source['url']}")
                try:
                    get_zip(source["url"], source["out_dir"])
                    xtf_file = get_xtf(source["out_dir"])
                    if xtf_file:
                        tree = ET.parse(xtf_file)
                        root = tree.getroot()
                        ns = get_ns(root)
                        elements = get_elements(xtf_file)
                        if elements is not None:
                            df_source = get_data(elements, ns, quelle=source["out_dir"])
                            all_data.append(df_source)
                        else:
                             st.warning(f"Keine Elemente im XTF-File {xtf_file} gefunden.")
                    else:
                        st.warning(f"Keine .xtf Datei in {source['out_dir']} gefunden.")
                except Exception as e:
                    st.error(f"Fehler beim Laden der XTF-Daten: {e}")
                finally:
                    # Clean up extracted files
                    if os.path.exists(source["out_dir"]):
                        shutil.rmtree(source["out_dir"])


        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)

            st.subheader("Kombinierte Daten")
            st.write(combined_df)

            # Create GeoDataFrame for mapping
            geometry = combined_df['geom'].apply(lambda x: Point(x['coordinates']) if isinstance(x, dict) and 'coordinates' in x and x['type'] == 'Point' and 'coordinates' in x and isinstance(x['coordinates'], (list, tuple)) and len(x['coordinates']) >= 2 and isinstance(x['coordinates'][0], (int, float)) and isinstance(x['coordinates'][1], (int, float)) else None)
            gdf_combined = gpd.GeoDataFrame(combined_df, geometry=geometry).dropna(subset=['geometry'])

            # Set CRS for XTF data (assuming EPSG:2056) and reproject to WGS84 for WFS and mapping
            # If WFS data is already in WGS84, set CRS directly
            if not gdf_combined.empty:
                 # Separate XTF and WFS dataframes to handle CRS
                gdf_xtf = gdf_combined[gdf_combined['quelle'] != 'wfs'].set_crs(epsg=2056, allow_override=True)
                gdf_wfs = gdf_combined[gdf_combined['quelle'] == 'wfs'].set_crs(epsg=4326, allow_override=True) # WFS assumed to be WGS84

                # Reproject XTF to WGS84
                gdf_xtf_wgs84 = gdf_xtf.to_crs(epsg=4326)

                # Combine reprojected XTF and WFS data
                gdf_combined_wgs84 = pd.concat([gdf_xtf_wgs84, gdf_wfs], ignore_index=True)

                st.subheader("Interaktive Karte")
                # Ensure the GeoDataFrame is not empty before creating the map
                if not gdf_combined_wgs84.empty:
                    m = create_folium_map(gdf_combined_wgs84)
                    folium_static(m)
                else:
                    st.warning("Keine gültigen Geometriedaten gefunden, um eine Karte zu erstellen.")

            # Download link
            csv_file = combined_df.to_csv(index=False)
            st.download_button(
                label="Daten als CSV herunterladen",
                data=csv_file,
                file_name="belastete_standorte.csv",
                mime="text/csv"
            )
        else:
            st.warning("Keine Daten zum Anzeigen gefunden.")
